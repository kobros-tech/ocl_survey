"""
skill_memory.py

Avalanche-integrated port of the task-free Skill Memory strategy prototyped
in notebooks/skill_memory.py.

The decision logic is the same dynamic, probe-based, gap-clustering selection
used by the notebook. Skill reuse is decided before training an experience.

In addition to the training-time Skill Memory behaviour, this implementation
adds evaluation-time skill routing:

  * every completed training experience is mapped to the skill slot that
    learned it;
  * before Avalanche evaluates an experience, the corresponding stored skill
    is loaded;
  * after the complete evaluation pass, the exact pre-evaluation training
    state is restored.

This is required for class-incremental benchmarks such as Split-CIFAR100,
where Avalanche evaluates the cumulative test stream after every training
experience. Without evaluation-time routing, the last trained skill remains
active for the entire evaluation pass and older experiences are evaluated
with the wrong skill.
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
        self,
        slot: int,
        state_dict: Mapping[str, Tensor],
        metadata: dict | None = None,
    ) -> None:
        self._states[slot] = {
            key: value.detach().cpu().clone() for key, value in state_dict.items()
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
# ============================================================


def _resize_incremental_classifiers_for_state(
    model: nn.Module,
    state_dict: Mapping[str, Tensor],
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
            module.classifier.in_features,
            target_weight.shape[0],
        ).to(device=device, dtype=dtype)
        active_key = f"{prefix}active_units"
        if active_key in state_dict:
            module.active_units = state_dict[active_key].to(device=device).clone()


def _incremental_out_features(
    model: nn.Module,
    state_dict: Mapping[str, Tensor],
) -> int | None:
    """Read the class count from a stored skill state."""
    for module_name, module in model.named_modules():
        if not isinstance(module, IncrementalClassifier):
            continue
        prefix = f"{module_name}." if module_name else ""
        weight = state_dict.get(f"{prefix}classifier.weight")
        if weight is not None and weight.ndim == 2:
            return int(weight.shape[0])
    return None


def _restore_initial_state(
    model: nn.Module,
    initial_state: Mapping[str, Tensor],
) -> None:
    current = model.state_dict()
    for name, initial in initial_state.items():
        if name not in current:
            continue
        target = current[name]
        if target.shape == initial.shape:
            target.copy_(
                initial.to(
                    device=target.device,
                    dtype=target.dtype,
                )
            )
        elif name.endswith("classifier.weight") and target.ndim == 2:
            rows = min(target.shape[0], initial.shape[0])
            target[:rows].copy_(
                initial[:rows].to(
                    device=target.device,
                    dtype=target.dtype,
                )
            )
        elif name.endswith("classifier.bias") and target.ndim == 1:
            rows = min(target.shape[0], initial.shape[0])
            target[:rows].copy_(
                initial[:rows].to(
                    device=target.device,
                    dtype=target.dtype,
                )
            )
        elif name.endswith("active_units"):
            continue
        else:
            raise RuntimeError(f"Cannot restore initial state for {name}")


def _origin_experience(experience):
    return getattr(
        experience,
        "origin_experience",
        experience,
    )


def _apply_skill_state(
    model: nn.Module,
    state_dict: Mapping[str, Tensor],
    experience,
) -> None:
    """
    Load a stored skill during training.

    Training-time loading may adapt the classifier to the current
    experience because Avalanche may need the head to grow.
    """

    _resize_incremental_classifiers_for_state(
        model,
        state_dict,
    )

    model.load_state_dict(
        state_dict,
        strict=False,
    )

    avalanche_model_adaptation(
        model,
        _origin_experience(experience),
    )


def _apply_skill_state_exact(
    model: nn.Module,
    state_dict: Mapping[str, Tensor],
) -> None:
    """
    Restore a stored skill exactly for evaluation.

    Unlike _apply_skill_state(), this does not call
    avalanche_model_adaptation() because evaluation must not modify the
    stored skill according to the currently evaluated experience.
    """

    _resize_incremental_classifiers_for_state(
        model,
        state_dict,
    )

    model.load_state_dict(
        state_dict,
        strict=False,
    )


# ============================================================
# PROBING
# ============================================================


def _probe(
    experience,
    batch_size: int,
    n_batches: int,
    seed: int | None = None,
):
    if len(experience.dataset) == 0:
        raise RuntimeError("Cannot probe an empty experience")
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    loader = DataLoader(
        experience.dataset,
        batch_size=min(
            batch_size,
            len(experience.dataset),
        ),
        shuffle=True,
        generator=generator,
    )
    xs, ys = [], []
    iterator = iter(loader)
    for _ in range(max(1, n_batches)):
        try:
            batch = next(iterator)
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
    model_factory: Callable[[], nn.Module],
    state_dict,
    experience,
    x,
    y,
    criterion,
):
    model = model_factory()
    _apply_skill_state(
        model,
        state_dict,
        experience,
    )
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
        logits = model(x)
        loss = float(criterion(logits, y).item())
        accuracy = float((logits.argmax(dim=1) == y).float().mean().item())
    return (
        loss,
        score_from_loss(loss),
        accuracy,
    )


# ============================================================
# DECISION LOGIC
# ============================================================


def find_best_skill(
    imagination_results: list[dict[str, Any]],
    forgetting_margin: float,
    score_floor: float = 0.9,
):
    """
    Select an existing skill only when there is evidence it is BOTH
    safe to reuse and compatible with the new experience.

    Each entry must contain:

      skill
      chance
      old_score
      old_accuracy
      new_score
      new_accuracy
    """
    if not imagination_results:
        return None

    safe_results = [
        result
        for result in imagination_results
        if result["old_accuracy"] > result["chance"] + forgetting_margin
    ]
    if not safe_results:
        return None

    def strongest_candidates(
        results,
        key,
        floor,
    ):
        ranked = sorted(
            results,
            key=lambda result: result[key],
            reverse=True,
        )

        if len(ranked) == 1:
            return {ranked[0]["skill"]} if ranked[0][key] > floor else set()

        values = [result[key] for result in ranked]

        gaps = [values[i] - values[i + 1] for i in range(len(values) - 1)]

        split = max(
            range(len(gaps)),
            key=lambda index: gaps[index],
        )

        if gaps[split] <= 0:
            return set()
        candidates = ranked[: split + 1]
        candidates = [result for result in candidates if result[key] > floor]
        return {result["skill"] for result in candidates}

    floor_score = (
        max(result["chance"] for result in safe_results)
        if score_floor is None
        else score_floor
    )

    floor_accuracy = max(result["chance"] for result in safe_results)

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

    candidates = [result for result in safe_results if result["skill"] in intersection]
    return max(
        candidates,
        key=lambda result: (
            result["new_score"],
            result["new_accuracy"],
        ),
    )


# ============================================================
# STRATEGY PLUGIN
# ============================================================


class SkillMemoryPlugin(SupervisedPlugin):
    """
    Probe-based Skill Memory.

    A training experience either:

      * reuses an existing compatible skill and keeps training it, or
      * allocates a new skill and trains it from scratch.

    Evaluation is routed independently: each training experience remembers
    the skill that learned it, and before_eval_exp loads that skill when
    Avalanche evaluates the corresponding experience.
    """

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
        if force_decision not in (
            None,
            self.REUSE,
            self.SCRATCH,
        ):
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

        # --------------------------------------------------------
        # Evaluation routing
        # --------------------------------------------------------

        # training experience index -> skill slot
        self._experience_to_skill: dict[int, int] = {}

        # Exact state of the model before evaluation started.
        self._pre_eval_state: dict | None = None

        # Evaluation routing is based on the original benchmark
        # experience index, not on the order in which hooks happen.
        self._eval_active = False

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
        if self._initial_state is None:
            raise RuntimeError("Initial model state has not been captured")
        _restore_initial_state(
            strategy.model,
            self._initial_state,
        )
        self._reset_optimizer(strategy)

    @staticmethod
    def _is_first_subexp(experience) -> bool:
        return getattr(
            experience,
            "is_first_subexp",
            True,
        )

    @staticmethod
    def _is_last_subexp(experience) -> bool:
        return getattr(
            experience,
            "is_last_subexp",
            True,
        )

    @staticmethod
    def _experience_index(experience) -> int | None:
        """
        Return Avalanche's benchmark experience index.

        In Avalanche 0.6.0 benchmark experiences expose
        `current_experience`. Keep the fallback for custom experiences
        used by task-free/online wrappers.
        """
        index = getattr(
            experience,
            "current_experience",
            None,
        )
        if index is not None:
            return int(index)
        index = getattr(
            experience,
            "experience_id",
            None,
        )
        if index is not None:
            return int(index)
        return None

    def _score_slots(
        self,
        strategy,
        experience,
    ) -> list[dict[str, Any]]:
        """
        Evaluate every stored skill on an old probe and the new probe.
        """
        new_x, new_y = _probe(
            experience,
            self.probe_batch_size,
            self.probe_batches,
            self.probe_seed,
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
                else (self.probe_seed + 100003 + len(results))
            )
            old_x, old_y = _probe(
                old_experience,
                self.probe_batch_size,
                self.probe_batches,
                seed,
            )

            (
                old_loss,
                old_score,
                old_accuracy,
            ) = _evaluate_state(
                model_factory,
                state_dict,
                old_experience,
                old_x,
                old_y,
                nn.functional.cross_entropy,
            )
            (
                new_loss,
                new_score,
                new_accuracy,
            ) = _evaluate_state(
                model_factory,
                state_dict,
                experience,
                new_x,
                new_y,
                nn.functional.cross_entropy,
            )
            chance = 1.0 / (
                _incremental_out_features(
                    strategy.model,
                    state_dict,
                )
                or 2
            )
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
        return ConcatDataset(
            [
                experience.dataset,
                old_dataset,
            ]
        )

    # ------------------------------------------------------------
    # TRAINING HOOKS
    # ------------------------------------------------------------

    def before_training_exp(
        self,
        strategy,
        **kwargs,
    ):
        experience = strategy.experience
        if self._task_active and not self._is_first_subexp(experience):
            return
        self._task_active = True

        if self._initial_state is None:
            self._initial_state = {
                key: value.detach().cpu().clone()
                for key, value in strategy.model.state_dict().items()
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

        results = self._score_slots(
            strategy,
            experience,
        )
        self._log("\nImagination:")
        for result in results:
            self._log(
                f"  skill {result['skill']}: "
                f"old_score={result['old_score']:.3f}, "
                f"old_acc={result['old_accuracy']:.3f}, "
                f"new_score={result['new_score']:.3f}, "
                f"new_acc={result['new_accuracy']:.3f}"
            )

        best = None
        if self.force_decision is None:
            best = find_best_skill(
                results,
                self.forgetting_margin,
                self.score_floor,
            )

        elif self.force_decision == self.REUSE and results:
            best = max(
                results,
                key=lambda result: (
                    result["new_score"],
                    result["new_accuracy"],
                ),
            )

        if best is not None:
            self._active_slot = best["skill"]
            self.last_decision = self.REUSE
            self.last_selected_skill = best["skill"]
            self.last_compatibility_score = best["new_score"]
            self.last_old_accuracy = best["old_accuracy"]
            self.last_new_accuracy = best["new_accuracy"]

            _apply_skill_state(
                strategy.model,
                self.memory.state(best["skill"]),
                experience,
            )
            self._reset_optimizer(strategy)
            self._log(
                f"\nBest compatible skill: "
                f"{best['skill']} "
                f"(new_score="
                f"{best['new_score']:.3f}, "
                f"new_accuracy="
                f"{best['new_accuracy']:.3f})"
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

    def after_training_exp(
        self,
        strategy,
        **kwargs,
    ):
        experience = strategy.experience
        if not self._is_last_subexp(experience):
            return

        if self._active_slot is None:
            raise RuntimeError("No active Skill Memory slot after training experience")
        slot = self._active_slot
        experience_index = self._experience_index(experience)
        # Store the exact trained skill.
        self.memory.store(
            slot,
            strategy.model.state_dict(),
            metadata={
                "acquisition_decision": (self.last_decision),
                "selected_skill": (self.last_selected_skill),
                "compatibility_score": (self.last_compatibility_score),
                "old_accuracy": (self.last_old_accuracy),
                "new_accuracy": (self.last_new_accuracy),
                "probe_batch_size": (self.probe_batch_size),
                "probe_batches": (self.probe_batches),
                "probe_seed": (self.probe_seed),
                "experience_index": (
                    experience_index
                    if experience_index is not None
                    else len(self._seen_experiences)
                ),
            },
        )

        # Explicit routing table.
        #
        # IMPORTANT:
        # If a skill is reused by another training experience, this mapping
        # does NOT rewrite the previous experience's entry. Each experience
        # keeps the skill that actually trained it.
        if experience_index is not None:
            self._experience_to_skill[experience_index] = slot

        self._seen_experiences.append(_origin_experience(experience))
        self._active_slot = None
        self._task_active = False

    # ------------------------------------------------------------
    # EVALUATION HOOKS
    # ------------------------------------------------------------

    def before_eval(
        self,
        strategy,
        **kwargs,
    ):
        """
        Snapshot the exact model before Avalanche starts evaluating.

        Evaluation will temporarily swap stored skills in and out of the
        live model. after_eval() restores this snapshot.
        """
        self._pre_eval_state = {
            key: value.detach().cpu().clone()
            for key, value in strategy.model.state_dict().items()
        }
        self._eval_active = True

    def before_eval_exp(
        self,
        strategy,
        **kwargs,
    ):
        """
        Load the skill associated with the experience being evaluated.
        """
        if not self._eval_active:
            return
        experience = strategy.experience
        experience_index = self._experience_index(experience)
        if experience_index is None:
            self._log(
                "Evaluation experience has no "
                "current_experience index; keeping "
                "current model."
            )
            return
        slot = self._experience_to_skill.get(experience_index)
        if slot is None:
            self._log(
                "No stored skill for evaluation "
                f"experience {experience_index}; "
                "keeping current model."
            )
            return
        _apply_skill_state_exact(
            strategy.model,
            self.memory.state(slot),
        )

        # The classifier may have been replaced by
        # _resize_incremental_classifiers_for_state().
        # Optimizer references therefore need to be rebuilt.
        self._reset_optimizer(strategy)
        self._log(f"Evaluation routing: experience {experience_index} -> skill {slot}")

    def after_eval(
        self,
        strategy,
        **kwargs,
    ):
        """
        Restore the exact model that existed before evaluation began.
        """
        if not self._eval_active:
            return
        try:
            if self._pre_eval_state is not None:
                _restore_initial_state(
                    strategy.model,
                    self._pre_eval_state,
                )
                self._reset_optimizer(strategy)
        finally:
            self._pre_eval_state = None
            self._eval_active = False
