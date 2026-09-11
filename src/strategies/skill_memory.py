"""
skill_memory.py

Avalanche-integrated Skill Memory strategy.

Skill allocation is decided at CLASS level:

    experience
        -> class 1 -> probe existing skills -> decide
        -> class 2 -> probe existing skills -> decide
        -> ...
        -> class N -> probe existing skills -> decide

The Avalanche experience remains the outer continual-learning unit.

Training semantics:

  * SCRATCH allocates a new skill slot and stores the trained result.
  * REUSE loads an existing stored skill but never overwrites it.
  * Class -> skill assignments are bookkeeping only.
  * Training-time decisions use the current TRAINING experience data.
  * Evaluation does not use class labels or experience identifiers for
    routing unless the explicit oracle mode is requested.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any, Literal

import numpy as np
import torch
from avalanche.models.dynamic_modules import (
    IncrementalClassifier,
    avalanche_model_adaptation,
)
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from torch import Tensor, nn
from torch.utils.data import ConcatDataset, DataLoader, Subset

logger = logging.getLogger(__name__)

EvalRouting = Literal["none", "probe", "oracle"]
_VALID_EVAL_ROUTINGS = ("none", "probe", "oracle")


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
    _resize_incremental_classifiers_for_state(
        model,
        state_dict,
    )

    model.load_state_dict(
        state_dict,
        strict=False,
    )


def _probe(
    experience,
    batch_size: int,
    n_batches: int,
    seed: int | None = None,
    target_class: int | None = None,
):
    """Probe an experience, optionally restricted to one class."""

    dataset = experience.dataset

    if target_class is not None:
        indices = []

        for index in range(len(dataset)):
            sample = dataset[index]
            target = int(sample[1])

            if target == target_class:
                indices.append(index)

        if not indices:
            raise RuntimeError(f"No samples found for class {target_class}")

        dataset = Subset(dataset, indices)

    if len(dataset) == 0:
        raise RuntimeError("Cannot probe an empty experience")

    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
    )

    xs = []
    ys = []

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


def _classes_in_experience(experience) -> list[int]:
    """Return the distinct target classes in an experience."""

    classes = set()

    for index in range(len(experience.dataset)):
        sample = experience.dataset[index]
        classes.add(int(sample[1]))

    return sorted(classes)


def score_from_loss(loss_value: float) -> float:
    return float(np.exp(-loss_value))


def _evaluate_state(
    model_factory: Callable[[], nn.Module],
    state_dict,
    experience,
    x,
    y,
    criterion,
):
    """Evaluate a stored skill on a training-time probe."""

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


def _predictive_entropy(
    model_factory: Callable[[], nn.Module],
    state_dict,
    experience,
    x,
) -> float:
    """Blind input-only compatibility signal for evaluation."""

    model = model_factory()

    _apply_skill_state(
        model,
        state_dict,
        experience,
    )

    model.eval()

    device = next(model.parameters()).device
    x = x.to(device)

    with torch.no_grad():
        logits = model(x)

        probs = torch.softmax(
            logits,
            dim=-1,
        )

        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).mean()

    return float(entropy.item())


def find_best_skill(
    imagination_results: list[dict[str, Any]],
    forgetting_margin: float,
    score_floor: float | None = 0.9,
):
    """Select an existing skill only when evidence is strong.

    The results passed here belong to ONE CLASS.

    REUSE requires:

      1. The skill remains safe on its own previously learned data.
      2. It is a strong candidate on the current class by score.
      3. It is a strong candidate on the current class by accuracy.
      4. The score and accuracy candidate sets intersect.

    This keeps REUSE conservative.
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
            if ranked[0][key] > floor:
                return {ranked[0]["skill"]}
            return set()

        values = [result[key] for result in ranked]

        gaps = [values[index] - values[index + 1] for index in range(len(values) - 1)]

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


class SkillMemoryPlugin(SupervisedPlugin):
    """Class-level, probe-based Skill Memory plugin.

    Experiences remain the outer continual-learning loop.

    Within every experience, the plugin now loops over the distinct
    classes and independently asks:

        "Which stored skill, if any, is compatible with this class?"

    The class -> skill mapping is internal bookkeeping. The experience
    identifier is never used to select a training skill.

    REUSE remains immutable: loading and training from a stored skill does
    not overwrite the stored state.
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
        eval_routing: EvalRouting = "none",
        verbose: bool = True,
    ):
        super().__init__()

        if force_decision not in (
            None,
            self.REUSE,
            self.SCRATCH,
        ):
            raise ValueError("invalid force_decision")

        if eval_routing not in _VALID_EVAL_ROUTINGS:
            raise ValueError(
                f"invalid eval_routing={eval_routing!r}; "
                f"must be one of {_VALID_EVAL_ROUTINGS}"
            )

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
        self.eval_routing = eval_routing
        self.verbose = verbose

        if self.eval_routing == "oracle":
            logger.warning(
                "SkillMemoryPlugin: eval_routing='oracle' "
                "is enabled. This is an explicit diagnostic "
                "and must not be used for the headline "
                "comparison."
            )

        self.last_decision = self.SCRATCH
        self.last_selected_skill: int | None = None
        self.last_compatibility_score = 0.0
        self.last_old_accuracy = 0.0
        self.last_new_accuracy = 0.0

        self._initial_state: dict | None = None
        self._active_slot: int | None = None
        self._task_active = False

        self._seen_experiences: list = []

        self._training_experience_count = 0
        self._current_training_experience_index: int | None = None

        # Experience -> class -> skill.
        #
        # This is bookkeeping only. It is not consulted when selecting
        # a skill for a new training class.
        self._class_to_skill: dict[int, dict[int, int]] = {}

        # Kept for backwards-compatible audit information.
        self._experience_to_skill: dict[int, int] = {}

        self._class_decisions: dict[
            int,
            dict[int, dict[str, Any]],
        ] = {}

        self._pre_eval_state: dict | None = None
        self._eval_active = False

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)
        else:
            logger.info(msg)

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

    def _score_slots_for_class(
        self,
        strategy,
        experience,
        target_class: int,
    ) -> list[dict[str, Any]]:
        """Score every stored skill on ONE current class."""

        new_x, new_y = _probe(
            experience,
            self.probe_batch_size,
            self.probe_batches,
            self.probe_seed,
            target_class=target_class,
        )

        model_factory = lambda: deepcopy(strategy.model)

        results = []

        for slot in self.memory.slots():
            state_dict = self.memory.state(slot)
            meta = self.memory.metadata(slot)

            old_experience = None

            # Prefer the skill's stored class provenance.
            old_class = meta.get("class")

            if old_class is not None:
                for previous_experience in self._seen_experiences:
                    try:
                        if old_class in _classes_in_experience(previous_experience):
                            old_experience = previous_experience
                            break
                    except Exception:
                        continue

            if old_experience is None:
                old_index = meta.get("experience_index")

                if isinstance(old_index, int) and 0 <= old_index < len(
                    self._seen_experiences
                ):
                    old_experience = self._seen_experiences[old_index]

            if old_experience is None:
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
                target_class=old_class,
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

            out_features = _incremental_out_features(
                strategy.model,
                state_dict,
            )

            chance = 1.0 / out_features if out_features else 0.0

            results.append(
                {
                    "skill": slot,
                    "class": target_class,
                    "old_class": old_class,
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

    def _decide_class(
        self,
        strategy,
        experience,
        target_class: int,
    ):
        """Run the complete imagination decision for one class."""

        results = self._score_slots_for_class(
            strategy,
            experience,
            target_class,
        )

        self._log(f"\nImagination for class {target_class}:")

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

        decision = {
            "class": target_class,
            "decision": self.SCRATCH,
            "skill": None,
            "new_score": 0.0,
            "old_accuracy": 0.0,
            "new_accuracy": 0.0,
            "results": results,
        }

        if best is not None:
            decision.update(
                {
                    "decision": self.REUSE,
                    "skill": best["skill"],
                    "new_score": best["new_score"],
                    "old_accuracy": best["old_accuracy"],
                    "new_accuracy": best["new_accuracy"],
                }
            )

            self._log(
                f"Class {target_class}: "
                f"REUSE skill {best['skill']} "
                f"(score={best['new_score']:.3f}, "
                f"accuracy={best['new_accuracy']:.3f})"
            )

        else:
            self._log(f"Class {target_class}: no compatible skill -> SCRATCH")

        return decision

    def before_training_exp(
        self,
        strategy,
        **kwargs,
    ):
        experience = strategy.experience

        if self._task_active and not self._is_first_subexp(experience):
            return

        self._task_active = True

        self._current_training_experience_index = self._training_experience_count
        self._training_experience_count += 1

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

        classes = _classes_in_experience(experience)

        if not classes:
            raise RuntimeError("Current experience contains no classes")

        experience_index = self._current_training_experience_index

        self._class_decisions[experience_index] = {}

        # ------------------------------------------------------------
        # FIRST EXPERIENCE
        # ------------------------------------------------------------

        if len(self.memory) == 0:
            for target_class in classes:
                slot = self.memory.allocate()

                self._class_decisions[experience_index][target_class] = {
                    "decision": self.SCRATCH,
                    "skill": slot,
                }

                self._log(
                    f"Class {target_class}: "
                    f"no existing skills -> "
                    f"allocated skill {slot}"
                )

            # The actual model training still happens through Avalanche
            # for the complete experience.
            #
            # The first class allocation is used as the active experience
            # slot for backwards-compatible training/storage semantics.
            first_class = classes[0]
            self._active_slot = self._class_decisions[experience_index][first_class][
                "skill"
            ]

            self._scratch(strategy)
            return

        if not self._seen_experiences:
            raise RuntimeError(
                "Skill Memory has skills but no previous experiences to probe"
            )

        # ------------------------------------------------------------
        # CLASS-LEVEL DECISION LOOP
        # ------------------------------------------------------------

        for target_class in classes:
            decision = self._decide_class(
                strategy,
                experience,
                target_class,
            )

            self._class_decisions[experience_index][target_class] = decision

        # ------------------------------------------------------------
        # EXPERIENCE TRAINING STATE
        #
        # Avalanche trains the experience as a normal unit. Therefore
        # the model state used by that training is selected from the
        # strongest class decision rather than silently pretending that
        # one model can be swapped independently for every class inside
        # the same minibatch.
        # ------------------------------------------------------------

        decisions = list(self._class_decisions[experience_index].values())

        reuse_decisions = [
            decision for decision in decisions if decision["decision"] == self.REUSE
        ]

        if reuse_decisions:
            best = max(
                reuse_decisions,
                key=lambda decision: (
                    decision["new_score"],
                    decision["new_accuracy"],
                ),
            )

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
                f"Experience {experience_index}: "
                f"using strongest class-compatible "
                f"skill {best['skill']} for training"
            )

            if self.replay_old_during_reuse:
                strategy.adapted_dataset = self._build_replay_dataset(experience)

        else:
            self._active_slot = self.memory.allocate()

            self._scratch(strategy)

            self.last_decision = self.SCRATCH

            self._log(
                f"Experience {experience_index}: "
                f"no class found with a compatible "
                f"existing skill -> allocated "
                f"skill {self._active_slot}"
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

        experience_index = self._current_training_experience_index

        if experience_index is None:
            raise RuntimeError("Missing current training experience index")

        # ------------------------------------------------------------
        # STORE ONLY SCRATCH RESULTS.
        #
        # REUSE is immutable. The loaded stored skill remains unchanged.
        # ------------------------------------------------------------

        if self.last_decision == self.SCRATCH:
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
                    "experience_index": (experience_index),
                },
            )

            self._log(f"Stored trained scratch model as skill {slot}")

        else:
            self._log(f"Reused skill {slot}; stored skill left unchanged")

        # ------------------------------------------------------------
        # CLASS BOOKKEEPING
        # ------------------------------------------------------------

        class_mapping = self._class_to_skill.setdefault(
            experience_index,
            {},
        )

        for target_class, decision in self._class_decisions[experience_index].items():
            selected_skill = decision.get("skill")

            if selected_skill is None:
                selected_skill = slot

            class_mapping[target_class] = selected_skill

        # Backwards-compatible experience-level audit.
        self._experience_to_skill[experience_index] = slot

        self._seen_experiences.append(_origin_experience(experience))

        self._active_slot = None
        self._current_training_experience_index = None
        self._task_active = False

    def before_eval(
        self,
        strategy,
        **kwargs,
    ):
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
        if not self._eval_active:
            return

        if self.eval_routing == "none":
            return

        experience = strategy.experience

        if self.eval_routing == "oracle":
            experience_index = self._experience_index(experience)

            if experience_index is None:
                return

            slot = self._experience_to_skill.get(experience_index)

            if slot is None:
                return

            _apply_skill_state_exact(
                strategy.model,
                self.memory.state(slot),
            )

            self._reset_optimizer(strategy)

            self._log(f"[ORACLE eval] experience {experience_index} -> skill {slot}")

            return

        if self.eval_routing == "probe":
            if len(self.memory) == 0:
                return

            eval_x, _ = _probe(
                experience,
                self.probe_batch_size,
                self.probe_batches,
                self.probe_seed,
            )

            model_factory = lambda: deepcopy(strategy.model)

            best_slot = None
            best_entropy = None

            for slot in self.memory.slots():
                entropy = _predictive_entropy(
                    model_factory,
                    self.memory.state(slot),
                    experience,
                    eval_x,
                )

                if best_entropy is None or entropy < best_entropy:
                    best_entropy = entropy
                    best_slot = slot

            if best_slot is None:
                return

            _apply_skill_state_exact(
                strategy.model,
                self.memory.state(best_slot),
            )

            self._reset_optimizer(strategy)

            self._log(
                f"[PROBE eval] selected skill {best_slot} (entropy={best_entropy:.4f})"
            )

    def after_eval(
        self,
        strategy,
        **kwargs,
    ):
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
