"""
skill_memory.py

Avalanche-integrated port of the task-free Skill Memory strategy prototyped
in notebooks/skill_memory.py.

The decision logic is dynamic and probe-based: compatibility/accuracy
selection uses gap clustering with an automatically derived floor. There is
no fixed/universal reuse or clone threshold.

In addition to REUSE and SCRATCH, the strategy supports CLONE. CLONE does not
merge or overwrite stored skills. Instead, it creates an initialization from
copies of two existing skills: the strongest compatibility-oriented skill and
the strongest accuracy-oriented skill. A grid search interpolates their copied
weights in both directions and selects an initialization that balances the
best compatibility and accuracy observed in that search. The selected
initialization is then trained normally on the new experience and stored as a
new skill.

The Avalanche-specific additions are the bits required to run on real
backbones (SlimResNet18 / resnet18 / resnet50 + a growing
IncrementalClassifier head):

  * skills are stored as full model state_dicts
  * fixed-capacity integer slots are used exactly like the notebook bank
  * IncrementalClassifier resizing + avalanche_model_adaptation are applied
    when loading a stored skill
  * sub-experience (online / task-free) hooks and optimizer resets are handled
  * CLONE operates on copied state_dicts, so existing skills are never changed
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
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
        if not 0 <= slot < self.max_skills:
            raise ValueError(f"invalid skill slot: {slot}")
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
    """Read the class-count a stored skill was saved with."""
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
    return getattr(experience, "origin_experience", experience)


def _apply_skill_state(
    model: nn.Module,
    state_dict: Mapping[str, Tensor],
    experience,
) -> None:
    """Load a stored skill and adapt its head to the current experience."""
    _resize_incremental_classifiers_for_state(model, state_dict)
    model.load_state_dict(state_dict, strict=False)
    avalanche_model_adaptation(
        model,
        _origin_experience(experience),
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
    model_factory: Callable[[], nn.Module],
    state_dict: Mapping[str, Tensor],
    experience,
    x: Tensor,
    y: Tensor,
    criterion,
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
# CLONE
#
# CLONE uses copies of existing skills only to find a better starting
# point. The original skills remain untouched. The resulting state is
# loaded into the live model and trained normally.
# ============================================================


@dataclass
class CloneResult:
    state_dict: dict[str, Tensor]
    compatibility: float
    accuracy: float
    overall: float
    alpha: float
    direction: str
    score_skill: int
    accuracy_skill: int


def _interpolate_state_dicts(
    state_1: Mapping[str, Tensor],
    state_2: Mapping[str, Tensor],
    alpha: float,
) -> dict[str, Tensor]:
    """
    alpha=0 -> state_1
    alpha=1 -> state_2.

    Floating-point tensors are interpolated. Non-floating buffers are taken
    from the closer endpoint because they are not meaningful to interpolate.
    """
    if state_1.keys() != state_2.keys():
        raise ValueError("State dictionaries must have identical keys.")

    merged = {}

    for key in state_1:
        a = state_1[key]
        b = state_2[key]

        if a.shape != b.shape:
            raise ValueError(
                f"Cannot interpolate '{key}': "
                f"shape {tuple(a.shape)} != {tuple(b.shape)}"
            )

        if torch.is_floating_point(a):
            merged[key] = ((1.0 - alpha) * a.float() + alpha * b.float()).to(
                dtype=a.dtype
            )
        else:
            merged[key] = a.clone() if alpha < 0.5 else b.clone()

    return merged


def find_best_weight_clone(
    score_state: Mapping[str, Tensor],
    accuracy_state: Mapping[str, Tensor],
    evaluate: Callable[
        [Mapping[str, Tensor]],
        tuple[float, float],
    ],
    *,
    score_skill: int,
    accuracy_skill: int,
    n_steps: int = 21,
    overall_fn: Callable[[float, float], float] | None = None,
) -> CloneResult | None:
    """
    Find a cloned initialization between two existing skills.

    The two inputs are treated as immutable source skills. Every candidate is
    built from copies of their tensors.

    Both interpolation directions are evaluated:

        score -> accuracy
        accuracy -> score

    The best observed compatibility and accuracy are used as empirical peaks.
    Candidates are then compared by their normalized distance to those peaks;
    no universal reuse/clone threshold is introduced.
    """
    if n_steps < 2:
        raise ValueError("n_steps must be >= 2.")

    if score_state.keys() != accuracy_state.keys():
        return None

    for key in score_state:
        if score_state[key].shape != accuracy_state[key].shape:
            return None

    if overall_fn is None:

        def overall_fn(score: float, accuracy: float) -> float:
            return 0.5 * score + 0.5 * accuracy

    candidates: list[CloneResult] = []
    alphas = np.linspace(0.0, 1.0, n_steps)

    directions = (
        (score_state, accuracy_state, "score -> accuracy"),
        (accuracy_state, score_state, "accuracy -> score"),
    )

    for state_1, state_2, direction in directions:
        for alpha in alphas:
            candidate_state = _interpolate_state_dicts(
                state_1,
                state_2,
                float(alpha),
            )

            compatibility, accuracy = evaluate(candidate_state)

            candidates.append(
                CloneResult(
                    state_dict=candidate_state,
                    compatibility=compatibility,
                    accuracy=accuracy,
                    overall=overall_fn(
                        compatibility,
                        accuracy,
                    ),
                    alpha=float(alpha),
                    direction=direction,
                    score_skill=score_skill,
                    accuracy_skill=accuracy_skill,
                )
            )

    if not candidates:
        return None

    best_compatibility = max(c.compatibility for c in candidates)
    best_accuracy = max(c.accuracy for c in candidates)

    # Dynamic normalization against the actual peaks found in this search.
    # This does not impose a fixed/universal threshold.
    def normalized_overall(candidate: CloneResult) -> float:
        score_ratio = (
            candidate.compatibility / best_compatibility
            if best_compatibility > 0.0
            else 0.0
        )
        accuracy_ratio = (
            candidate.accuracy / best_accuracy if best_accuracy > 0.0 else 0.0
        )
        return overall_fn(score_ratio, accuracy_ratio)

    return max(
        candidates,
        key=normalized_overall,
    )


# ============================================================
# DECISION LOGIC
#
# Existing-skill selection remains the same dynamic, gap-based logic.
# If no existing skill is compatible, CLONE is attempted from the strongest
# compatibility and accuracy skills. If CLONE cannot produce an initialization,
# the caller falls back to SCRATCH.
# ============================================================


def _strongest_candidates(
    results: list[dict[str, Any]],
    key: str,
    floor: float,
) -> set[int]:
    ranked = sorted(
        results,
        key=lambda r: r[key],
        reverse=True,
    )

    values = [r[key] for r in ranked]
    gaps = [values[i] - values[i + 1] for i in range(len(values) - 1)]

    if not gaps:
        return set()

    split = max(
        range(len(gaps)),
        key=lambda i: gaps[i],
    )

    if gaps[split] <= 0:
        return set()

    candidates = ranked[: split + 1]
    candidates = [r for r in candidates if r[key] > floor]

    return {r["skill"] for r in candidates}


def find_best_skill(
    imagination_results: list[dict[str, Any]],
    forgetting_margin: float,
    score_floor: float = 0.9,
):
    """
    Select an existing skill only when there is evidence it is BOTH
    safe to reuse and compatible with the new experience.

    Selection uses dynamic gap clustering plus an absolute score floor.
    This function does not perform CLONE because CLONE needs access to the
    actual stored state_dicts and the model/probe evaluator.
    """
    if not imagination_results:
        return None

    safe_results = [
        r
        for r in imagination_results
        if r["old_accuracy"] > r["chance"] + forgetting_margin
    ]

    if not safe_results:
        return None

    if len(safe_results) == 1:
        result = safe_results[0]
        if (
            result["new_score"] > score_floor
            and result["new_accuracy"] > result["chance"]
        ):
            return result
        return None

    # Reuse requires an absolute score threshold on the new experience.
    # This prevents a relative winner from being reused when its absolute
    # compatibility with the new experience is still too weak.
    floor_score = score_floor
    floor_accuracy = max(r["chance"] for r in safe_results)

    score_candidates = _strongest_candidates(
        safe_results,
        "new_score",
        floor_score,
    )

    accuracy_candidates = _strongest_candidates(
        safe_results,
        "new_accuracy",
        floor_accuracy,
    )

    intersection = score_candidates & accuracy_candidates

    if not intersection:
        return None

    candidates = [r for r in safe_results if r["skill"] in intersection]

    return max(
        candidates,
        key=lambda r: (
            r["new_score"],
            r["new_accuracy"],
        ),
    )


# ============================================================
# STRATEGY PLUGIN
# ============================================================


class SkillMemoryPlugin(SupervisedPlugin):
    """
    Probe-based Skill Memory.

    Decision paths:

        REUSE  -> continue training an existing skill
        CLONE  -> initialize a new skill from interpolated copies of two
                  existing skills, then train normally
        SCRATCH -> initialize a new skill from the original model state

    CLONE never modifies either source skill.
    """

    REUSE, CLONE, SCRATCH = "reuse", "clone", "scratch"

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
        clone_steps: int = 21,
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
            self.CLONE,
            self.SCRATCH,
        ):
            raise ValueError("invalid force_decision")

        if clone_steps < 2:
            raise ValueError("clone_steps must be >= 2.")

        self.memory = (
            memory if memory is not None else SkillMemory(max_skills=max_skills)
        )
        self.forgetting_margin = forgetting_margin
        self.score_floor = score_floor
        self.probe_batch_size = probe_batch_size
        self.probe_batches = probe_batches
        self.probe_seed = probe_seed
        self.clone_steps = clone_steps
        self.replay_old_during_reuse = replay_old_during_reuse
        self.replay_batches_per_epoch = replay_batches_per_epoch
        self.skill_name = skill_name
        self.force_decision = force_decision
        self.verbose = verbose

        self.last_decision = self.SCRATCH
        self.last_selected_skill: int | None = None
        self.last_clone_score_skill: int | None = None
        self.last_clone_accuracy_skill: int | None = None
        self.last_clone_alpha: float | None = None
        self.last_clone_direction: str | None = None
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

    def _score_slots(
        self,
        strategy,
        experience,
    ) -> list[dict[str, Any]]:
        """
        Mirrors the notebook's imagine(): evaluate every stored skill on
        both old and new probes.
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
                else self.probe_seed + 100003 + len(results)
            )

            old_x, old_y = _probe(
                old_experience,
                self.probe_batch_size,
                self.probe_batches,
                seed,
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

    def _find_clone(
        self,
        strategy,
        experience,
        results: list[dict[str, Any]],
    ) -> CloneResult | None:
        """
        Find the two source skills for CLONE and search their copied
        interpolation space.
        """
        if len(results) < 2:
            return None

        safe_results = [
            r
            for r in results
            if r["old_accuracy"] > r["chance"] + self.forgetting_margin
        ]

        if len(safe_results) < 2:
            return None

        # Strongest compatibility and strongest accuracy are selected
        # dynamically from the actual imagination results.
        score_result = max(
            safe_results,
            key=lambda r: r["new_score"],
        )
        accuracy_result = max(
            safe_results,
            key=lambda r: r["new_accuracy"],
        )

        if score_result["skill"] == accuracy_result["skill"]:
            # There is no meaningful two-skill clone if the same skill is
            # already strongest on both dimensions.
            return None

        score_skill = score_result["skill"]
        accuracy_skill = accuracy_result["skill"]

        score_state = self.memory.state(score_skill)
        accuracy_state = self.memory.state(accuracy_skill)

        new_x, new_y = _probe(
            experience,
            self.probe_batch_size,
            self.probe_batches,
            self.probe_seed,
        )

        model_factory = lambda: deepcopy(strategy.model)

        def evaluate(candidate_state):
            _, score, accuracy = _evaluate_state(
                model_factory,
                candidate_state,
                experience,
                new_x,
                new_y,
                nn.functional.cross_entropy,
            )
            return score, accuracy

        return find_best_weight_clone(
            score_state,
            accuracy_state,
            evaluate,
            score_skill=score_skill,
            accuracy_skill=accuracy_skill,
            n_steps=self.clone_steps,
        )

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
        self.last_clone_score_skill = None
        self.last_clone_accuracy_skill = None
        self.last_clone_alpha = None
        self.last_clone_direction = None
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
            best = find_best_skill(
                results,
                self.forgetting_margin,
                self.score_floor,
            )

        elif self.force_decision == self.REUSE and results:
            best = max(
                results,
                key=lambda r: (
                    r["new_score"],
                    r["new_accuracy"],
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
                f"\nBest compatible skill: {best['skill']} "
                f"(new_score={best['new_score']:.3f}, "
                f"new_accuracy={best['new_accuracy']:.3f})"
            )

            if self.replay_old_during_reuse:
                strategy.adapted_dataset = self._build_replay_dataset(experience)

            return

        # --------------------------------------------------------
        # No existing skill passed the reuse decision.
        # Try CLONE as a new initialization.
        # --------------------------------------------------------
        clone = None

        if self.force_decision in (
            None,
            self.CLONE,
        ):
            clone = self._find_clone(
                strategy,
                experience,
                results,
            )

        if clone is not None:
            self._active_slot = self.memory.allocate()
            self.last_decision = self.CLONE
            self.last_selected_skill = None
            self.last_clone_score_skill = clone.score_skill
            self.last_clone_accuracy_skill = clone.accuracy_skill
            self.last_clone_alpha = clone.alpha
            self.last_clone_direction = clone.direction
            self.last_compatibility_score = clone.compatibility
            self.last_new_accuracy = clone.accuracy

            # The source skills remain untouched. Only the selected copied
            # initialization is loaded into the live model.
            _apply_skill_state(
                strategy.model,
                clone.state_dict,
                experience,
            )
            self._reset_optimizer(strategy)

            self._log(
                f"\nCLONE -> allocated skill {self._active_slot} "
                f"from score_skill={clone.score_skill}, "
                f"accuracy_skill={clone.accuracy_skill}, "
                f"alpha={clone.alpha:.3f}, "
                f"direction={clone.direction}, "
                f"new_score={clone.compatibility:.3f}, "
                f"new_accuracy={clone.accuracy:.3f}"
            )

            return

        # --------------------------------------------------------
        # Neither REUSE nor CLONE produced a usable initialization.
        # Start from scratch.
        # --------------------------------------------------------
        self._active_slot = self.memory.allocate()
        self._scratch(strategy)
        self.last_decision = self.SCRATCH

        self._log(
            f"\nNo compatible existing skill and no useful clone "
            f"-> allocated skill {self._active_slot}"
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
                "clone_score_skill": (self.last_clone_score_skill),
                "clone_accuracy_skill": (self.last_clone_accuracy_skill),
                "clone_alpha": self.last_clone_alpha,
                "clone_direction": (self.last_clone_direction),
                "compatibility_score": (self.last_compatibility_score),
                "old_accuracy": self.last_old_accuracy,
                "new_accuracy": self.last_new_accuracy,
                "probe_batch_size": (self.probe_batch_size),
                "probe_batches": self.probe_batches,
                "probe_seed": self.probe_seed,
                "clone_steps": self.clone_steps,
                "experience_index": (len(self._seen_experiences)),
            },
        )

        self._seen_experiences.append(_origin_experience(experience))
        self._active_slot = None
        self._task_active = False
