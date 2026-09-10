"""
skill_memory.py

Avalanche-integrated port of the task-free Skill Memory strategy prototyped
in notebooks/skill_memory.py. The decision logic here is the SAME dynamic,
probe-based, gap-clustering selection as the notebook -- nothing here uses a
fixed/universal reuse or clone threshold.

The only things this file adds on top of the notebook's design are the bits
that are strictly required to run inside Avalanche on real backbone models
(SlimResNet18 / resnet18 / resnet50 + a growing IncrementalClassifier head):

  * skills are stored as full model state_dicts (a plain linear "skill bank"
    like the notebook's isn't possible here -- there is no shared backbone
    outside the skill, the skill *is* the whole network), addressed by a
    fixed-capacity integer slot exactly like the notebook's SkillClassifierBank
  * IncrementalClassifier resizing + avalanche_model_adaptation when loading
    a stored skill into the live model (classes seen per skill can differ)
  * sub-experience (online / task-free) hooks and optimizer resets

Everything else -- the forgetting guard, the gap-based clustering with an
auto-derived floor, the binary reuse-vs-allocate decision, the always-train
behaviour, and the optional replay while continuing a reused skill -- mirrors
the notebook function-for-function.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from avalanche.models.dynamic_modules import (
    IncrementalClassifier,
    avalanche_model_adaptation,
)
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from torch import Tensor, nn
from torch.utils.data import ConcatDataset, DataLoader

logger = logging.getLogger(__name__)


# ============================================================
# SKILL STORAGE
#
# Fixed-capacity, index-addressed registry -- the same shape as the
# notebook's SkillClassifierBank (allocate() claims a free slot, raises
# once full, no eviction). The only difference forced by this codebase's
# models is *what* a slot holds: a full model state_dict instead of a bare
# nn.Linear, since there is no shared backbone living outside the skill.
# ============================================================


class SkillMemory:
    """Bounded, index-addressed registry of independent skill states."""

    def __init__(self, max_skills: int = 20):
        if max_skills < 1:
            raise ValueError("max_skills must be positive")
        self.max_skills = max_skills
        self._states: dict[int, dict] = {}
        self._metadata: dict[int, dict] = {}

    def allocate(self) -> int:
        for slot in range(self.max_skills):
            if slot not in self._states:
                return slot
        raise RuntimeError(f"skill memory is at capacity ({self.max_skills})")

    def store(
        self, slot: int, state_dict: Mapping[str, Tensor], metadata: dict | None = None
    ) -> None:
        self._states[slot] = {
            k: v.detach().cpu().clone() for k, v in state_dict.items()
        }
        self._metadata[slot] = dict(metadata or {})

    def state(self, slot: int) -> dict:
        return self._states[slot]

    def metadata(self, slot: int) -> dict:
        return self._metadata.get(slot, {})

    def slots(self) -> set[int]:
        return set(self._states)

    def __len__(self) -> int:
        return len(self._states)


# ============================================================
# AVALANCHE-SPECIFIC PLUMBING
# (strictly required to load a stored skill into a live, growing model)
# ============================================================


def _resize_incremental_classifiers_for_state(
    model: nn.Module, state_dict: Mapping[str, Tensor]
) -> None:
    for module_name, module in model.named_modules():
        if not isinstance(module, IncrementalClassifier):
            continue
        prefix = f"{module_name}." if module_name else ""
        target_weight = state_dict.get(f"{prefix}classifier.weight")
        if target_weight is None or target_weight.ndim != 2:
            continue
        if module.classifier.out_features == target_weight.shape[0]:
            continue
        device = module.classifier.weight.device
        dtype = module.classifier.weight.dtype
        module.classifier = nn.Linear(
            module.classifier.in_features, target_weight.shape[0]
        ).to(device=device, dtype=dtype)
        active_key = f"{prefix}active_units"
        if active_key in state_dict:
            module.active_units = state_dict[active_key].to(device=device).clone()


def _incremental_out_features(
    model: nn.Module, state_dict: Mapping[str, Tensor]
) -> int | None:
    """Read the class-count a stored skill was saved with, straight from its weights."""
    for module_name, module in model.named_modules():
        if not isinstance(module, IncrementalClassifier):
            continue
        prefix = f"{module_name}." if module_name else ""
        weight = state_dict.get(f"{prefix}classifier.weight")
        if weight is not None and weight.ndim == 2:
            return int(weight.shape[0])
    return None


def _restore_initial_state(
    model: nn.Module, initial_state: Mapping[str, Tensor]
) -> None:
    current = model.state_dict()
    for name, initial in initial_state.items():
        if name not in current:
            continue
        target = current[name]
        if target.shape == initial.shape:
            target.copy_(initial.to(device=target.device, dtype=target.dtype))
        elif name.endswith("classifier.weight") and target.ndim == 2:
            rows = min(target.shape[0], initial.shape[0])
            target[:rows].copy_(
                initial[:rows].to(device=target.device, dtype=target.dtype)
            )
        elif name.endswith("classifier.bias") and target.ndim == 1:
            rows = min(target.shape[0], initial.shape[0])
            target[:rows].copy_(
                initial[:rows].to(device=target.device, dtype=target.dtype)
            )
        elif name.endswith("active_units"):
            continue
        else:
            raise RuntimeError(f"Cannot restore initial state for {name}")


def _origin_experience(experience):
    return getattr(experience, "origin_experience", experience)


def _apply_skill_state(
    model: nn.Module, state_dict: Mapping[str, Tensor], experience
) -> None:
    """Load a stored skill onto `model` and adapt it (grow the head) to `experience`."""
    _resize_incremental_classifiers_for_state(model, state_dict)
    model.load_state_dict(state_dict, strict=False)
    avalanche_model_adaptation(model, _origin_experience(experience))


# ============================================================
# PROBING  (same "concatenate a few shuffled batches" recipe as the notebook)
# ============================================================


def _probe(experience, batch_size: int, n_batches: int, seed: int | None = None):
    if len(experience.dataset) == 0:
        raise RuntimeError("Cannot probe an empty experience")
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    loader = DataLoader(
        experience.dataset,
        batch_size=min(batch_size, len(experience.dataset)),
        shuffle=True,
        generator=generator,
    )
    xs, ys = [], []
    it = iter(loader)
    for _ in range(max(1, n_batches)):
        try:
            batch = next(it)
        except StopIteration:
            break
        xs.append(batch[0])
        ys.append(batch[1])
    if not xs:
        raise RuntimeError("Probe loader produced no batches")
    return torch.cat(xs), torch.cat(ys)


def score_from_loss(loss_value: float) -> float:
    """Geometric-mean true-class probability induced by cross entropy."""
    return float(np.exp(-loss_value))


def _evaluate_state(
    model_factory: Callable[[], nn.Module], state_dict, experience, x, y, criterion
):
    model = model_factory()
    _apply_skill_state(model, state_dict, experience)
    model.eval()
    device = next(model.parameters()).device
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        logits = model(x)
        loss = float(criterion(logits, y).item())
        accuracy = float((logits.argmax(dim=1) == y).float().mean().item())
    return loss, score_from_loss(loss), accuracy


# ============================================================
# DECISION LOGIC -- ported directly from the notebook, unchanged in spirit.
# No fixed reuse/clone threshold anywhere below: candidates are found by
# clustering (largest gap in sorted values) plus an auto-derived floor.
# ============================================================


def find_best_skill(
    imagination_results: list[dict[str, Any]],
    forgetting_margin: float,
    score_floor: float = 0.9,
):
    """
    Select an existing skill only when there is evidence it is BOTH
    safe to reuse (forgetting guard on old data) AND compatible with the
    new experience (relative clustering + absolute reuse threshold).

    Each entry in `imagination_results` must have: skill, chance,
    old_score, old_accuracy, new_score, new_accuracy.
    """
    if not imagination_results:
        return None

    # Forgetting guard: drop any skill whose grip on old data is already
    # weak -- reusing it would trivially "forget" further.
    safe_results = [
        r
        for r in imagination_results
        if r["old_accuracy"] > r["chance"] + forgetting_margin
    ]
    if not safe_results:
        return None

    def strongest_candidates(results, key, floor):
        ranked = sorted(results, key=lambda r: r[key], reverse=True)
        if len(ranked) == 1:
            return (
                {ranked[0]["skill"]}
                if ranked[0][key] > floor
                else set()
            )
        values = [r[key] for r in ranked]
        gaps = [
            values[i] - values[i + 1]
            for i in range(len(values) - 1)
        ]
        split = max(range(len(gaps)), key=lambda i: gaps[i])
        if gaps[split] <= 0:
            return set()
        candidates = ranked[: split + 1]
        candidates = [r for r in candidates if r[key] > floor]
        return {r["skill"] for r in candidates}

    # Reuse requires an absolute score threshold on the new experience.
    # This prevents a relative winner from being reused when its absolute
    # compatibility with the new experience is still too weak.
    floor_score = (
        max(r["chance"] for r in safe_results)
        if score_floor is None
        else score_floor
    )

    floor_accuracy = max(r["chance"] for r in safe_results)

    score_candidates = strongest_candidates(
        safe_results,
        "new_score",
        floor_score,
    )
    accuracy_candidates = strongest_candidates(
        safe_results,
        "new_accuracy",
        floor_accuracy,
    )

    intersection = score_candidates & accuracy_candidates
    if not intersection:
        return None

    candidates = [
        r for r in safe_results if r["skill"] in intersection
    ]
    return max(
        candidates,
        key=lambda r: (r["new_score"], r["new_accuracy"]),
    )


# ============================================================
# STRATEGY PLUGIN
# ============================================================


class SkillMemoryPlugin(SupervisedPlugin):
    """Probe-based Skill Memory: reuse-and-keep-training an existing skill,
    or allocate and train a fresh one from scratch. Binary decision, exactly
    like the notebook -- no REUSE/CLONE/SCRATCH three-way split."""

    REUSE, SCRATCH = "reuse", "scratch"

    def __init__(
        self,
        memory: SkillMemory | None = None,
        *,
        max_skills: int = 20,
        forgetting_margin: float = 0.05,
        score_floor: float | None = None,
        probe_batch_size: int = 64,
        probe_batches: int = 5,
        probe_seed: int | None = None,
        replay_old_during_reuse: bool = False,
        replay_batches_per_epoch: int = 1,
        skill_name: Callable | None = None,
        force_decision: str | None = None,
        verbose: bool = True,
    ):
        super().__init__()
        if force_decision not in (None, self.REUSE, self.SCRATCH):
            raise ValueError("invalid force_decision")

        self.memory = (
            memory if memory is not None else SkillMemory(max_skills=max_skills)
        )
        self.forgetting_margin = forgetting_margin
        self.score_floor = score_floor
        self.probe_batch_size = probe_batch_size
        self.probe_batches = probe_batches
        self.probe_seed = probe_seed
        self.replay_old_during_reuse = replay_old_during_reuse
        self.replay_batches_per_epoch = replay_batches_per_epoch
        self.skill_name = skill_name
        self.force_decision = force_decision
        self.verbose = verbose

        self.last_decision = self.SCRATCH
        self.last_selected_skill: int | None = None
        self.last_compatibility_score = 0.0
        self.last_old_accuracy = 0.0
        self.last_new_accuracy = 0.0

        self._initial_state: dict | None = None
        self._active_slot: int | None = None
        self._task_active = False
        self._seen_experiences: list = []

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)
        else:
            logger.info(msg)

    # ------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------

    def _reset_optimizer(self, strategy) -> None:
        optimizer = strategy.optimizer
        if optimizer is None:
            return
        params = list(strategy.model.parameters())
        if not optimizer.param_groups:
            optimizer.add_param_group({"params": params})
        else:
            optimizer.param_groups[0]["params"] = params
        for group in optimizer.param_groups[1:]:
            group["params"] = []
        optimizer.state.clear()

    def _scratch(self, strategy) -> None:
        _restore_initial_state(strategy.model, self._initial_state)
        self._reset_optimizer(strategy)

    @staticmethod
    def _is_first_subexp(experience) -> bool:
        return getattr(experience, "is_first_subexp", True)

    @staticmethod
    def _is_last_subexp(experience) -> bool:
        return getattr(experience, "is_last_subexp", True)

    def _score_slots(self, strategy, experience) -> list[dict[str, Any]]:
        """Mirrors the notebook's imagine(): evaluate every stored skill on
        both an old probe (previously seen experiences) and a new probe
        (the current experience)."""
        new_x, new_y = _probe(
            experience, self.probe_batch_size, self.probe_batches, self.probe_seed
        )
        model_factory = lambda: deepcopy(strategy.model)
        results = []
        for slot in self.memory.slots():
            state_dict = self.memory.state(slot)
            meta = self.memory.metadata(slot)

            old_index = meta.get("experience_index")
            if isinstance(old_index, int) and 0 <= old_index < len(
                self._seen_experiences
            ):
                old_experience = self._seen_experiences[old_index]
            else:
                old_experience = self._seen_experiences[0]
            seed = (
                None
                if self.probe_seed is None
                else self.probe_seed + 100003 + len(results)
            )
            old_x, old_y = _probe(
                old_experience, self.probe_batch_size, self.probe_batches, seed
            )

            old_loss, old_score, old_accuracy = _evaluate_state(
                model_factory,
                state_dict,
                old_experience,
                old_x,
                old_y,
                nn.functional.cross_entropy,
            )
            new_loss, new_score, new_accuracy = _evaluate_state(
                model_factory,
                state_dict,
                experience,
                new_x,
                new_y,
                nn.functional.cross_entropy,
            )

            chance = 1.0 / (_incremental_out_features(strategy.model, state_dict) or 2)

            results.append(
                {
                    "skill": slot,
                    "chance": chance,
                    "old_loss": old_loss,
                    "old_score": old_score,
                    "old_accuracy": old_accuracy,
                    "new_loss": new_loss,
                    "new_score": new_score,
                    "new_accuracy": new_accuracy,
                }
            )
        return results

    def _build_replay_dataset(self, experience):
        if not self._seen_experiences:
            return experience.dataset
        old_dataset = ConcatDataset(
            [_origin_experience(e).dataset for e in self._seen_experiences]
        )
        return ConcatDataset([experience.dataset, old_dataset])

    # ------------------------------------------------------------
    # avalanche hooks
    # ------------------------------------------------------------

    def before_training_exp(self, strategy, **kwargs):
        experience = strategy.experience
        if self._task_active and not self._is_first_subexp(experience):
            return
        self._task_active = True

        if self._initial_state is None:
            self._initial_state = {
                k: v.detach().cpu().clone()
                for k, v in strategy.model.state_dict().items()
            }

        self.last_decision = self.SCRATCH
        self.last_selected_skill = None
        self.last_compatibility_score = 0.0
        self.last_old_accuracy = 0.0
        self.last_new_accuracy = 0.0

        if len(self.memory) == 0:
            self._active_slot = self.memory.allocate()
            self._scratch(strategy)
            self._log(f"No existing skills -> allocated skill {self._active_slot}")
            return

        if not self._seen_experiences:
            raise RuntimeError(
                "Skill Memory has skills but no previous experiences to probe"
            )

        results = self._score_slots(strategy, experience)
        self._log("\nImagination:")
        for r in results:
            self._log(
                f"  skill {r['skill']}: "
                f"old_score={r['old_score']:.3f}, "
                f"old_acc={r['old_accuracy']:.3f}, "
                f"new_score={r['new_score']:.3f}, "
                f"new_acc={r['new_accuracy']:.3f}"
            )

        best = None
        if self.force_decision is None:
            best = find_best_skill(results, self.forgetting_margin, self.score_floor)
        elif self.force_decision == self.REUSE and results:
            best = max(results, key=lambda r: (r["new_score"], r["new_accuracy"]))

        if best is not None:
            self._active_slot = best["skill"]
            self.last_decision = self.REUSE
            self.last_selected_skill = best["skill"]
            self.last_compatibility_score = best["new_score"]
            self.last_old_accuracy = best["old_accuracy"]
            self.last_new_accuracy = best["new_accuracy"]

            _apply_skill_state(
                strategy.model, self.memory.state(best["skill"]), experience
            )
            self._reset_optimizer(strategy)
            self._log(
                f"\nBest compatible skill: {best['skill']} "
                f"(new_score={best['new_score']:.3f}, "
                f"new_accuracy={best['new_accuracy']:.3f})"
            )

            if self.replay_old_during_reuse:
                strategy.adapted_dataset = self._build_replay_dataset(experience)
        else:
            self._active_slot = self.memory.allocate()
            self._scratch(strategy)
            self.last_decision = self.SCRATCH
            self._log(
                f"\nNo compatible existing skill -> allocated skill {self._active_slot}"
            )

    def after_training_exp(self, strategy, **kwargs):
        experience = strategy.experience
        if not self._is_last_subexp(experience):
            return

        slot = self._active_slot
        self.memory.store(
            slot,
            strategy.model.state_dict(),
            metadata={
                "acquisition_decision": self.last_decision,
                "selected_skill": self.last_selected_skill,
                "compatibility_score": self.last_compatibility_score,
                "old_accuracy": self.last_old_accuracy,
                "new_accuracy": self.last_new_accuracy,
                "probe_batch_size": self.probe_batch_size,
                "probe_batches": self.probe_batches,
                "probe_seed": self.probe_seed,
                "experience_index": len(self._seen_experiences),
            },
        )

        self._seen_experiences.append(_origin_experience(experience))
        self._active_slot = None
        self._task_active = False
