"""Avalanche plugin for class-level, probe-based Skill Memory.

An Avalanche experience is only a container.  The strategy extracts the
classes actually present in that experience and handles them independently:

    experience -> class -> REUSE/SCRATCH -> train only that class

Bookkeeping is explicit:

    experience -> [(skill, {classes})]
    class      -> canonical skill
    skill      -> all classes it currently masters

A class that has already been mastered is never re-assigned by the generic
probe heuristic.  It deterministically returns to its canonical skill, and
REUSE is mutable: training updates the same reserved skill slot.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import logging
import time
from typing import Any, Literal

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

from .decision import decide_class
from .probing import (
    apply_skill_state,
    apply_skill_state_exact,
    classes_in_experience,
    origin_experience,
    prepare_for_experience,
    predictive_entropy,
    probe_whole_experience,
    restore_initial_state,
)
from .skill_registry import ClassRecord, ExperienceClassMap, SkillMemory
from .training import train_on_class

logger = logging.getLogger(__name__)
EvalRouting = Literal["none", "probe", "oracle"]
_VALID_EVAL_ROUTINGS = ("none", "probe", "oracle")


class SkillMemoryPlugin(SupervisedPlugin):
    """Class-level Skill Memory plugin with explicit class bookkeeping."""

    REUSE, SCRATCH = "reuse", "scratch"

    def __init__(
        self,
        memory: SkillMemory | None = None,
        *,
        max_skills: int = 200,
        forgetting_margin: float = 0.05,
        score_floor: float | None = 0.9,
        probe_batch_size: int = 64,
        probe_batches: int = 5,
        probe_seed: int | None = None,
        max_safety_candidates: int = 5,
        class_train_epochs: int = 1,
        class_train_batch_size: int = 64,
        reuse_is_mutable: bool = True,
        skill_name: Callable | None = None,
        force_decision: str | None = None,
        eval_routing: EvalRouting = "none",
        verbose: bool = True,
    ):
        super().__init__()
        if force_decision not in (None, self.REUSE, self.SCRATCH):
            raise ValueError("invalid force_decision")
        if eval_routing not in _VALID_EVAL_ROUTINGS:
            raise ValueError(
                f"invalid eval_routing={eval_routing!r}; "
                f"must be one of {_VALID_EVAL_ROUTINGS}"
            )

        self.memory = memory if memory is not None else SkillMemory(max_skills)
        self.class_map = ExperienceClassMap()
        self.forgetting_margin = forgetting_margin
        self.score_floor = score_floor
        self.probe_batch_size = probe_batch_size
        self.probe_batches = probe_batches
        self.probe_seed = probe_seed
        self.max_safety_candidates = max(1, int(max_safety_candidates))
        self.class_train_epochs = class_train_epochs
        self.class_train_batch_size = class_train_batch_size
        self.reuse_is_mutable = reuse_is_mutable
        self.skill_name = skill_name
        self.force_decision = force_decision
        self.eval_routing = eval_routing
        self.verbose = verbose

        self.last_class_decisions: dict[int, dict[int, dict[str, Any]]] = {}
        self._initial_state: dict | None = None
        self._task_active = False
        self._seen_experiences: list = []
        self._training_experience_count = 0
        self._current_training_experience_index: int | None = None
        self._original_train_epochs: int | None = None
        self._pre_eval_state: dict | None = None
        self._eval_active = False

        if eval_routing == "oracle":
            logger.warning(
                "eval_routing='oracle' is diagnostic only and should not be "
                "used for the headline benchmark."
            )

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)
        else:
            logger.info(message)

    @staticmethod
    def _snapshot(model) -> dict:
        return {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

    @staticmethod
    def _is_first_subexp(experience) -> bool:
        return getattr(experience, "is_first_subexp", True)

    @staticmethod
    def _is_last_subexp(experience) -> bool:
        return getattr(experience, "is_last_subexp", True)

    def _reset_optimizer(self, strategy) -> None:
        optimizer = getattr(strategy, "optimizer", None)
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

    def _scratch_reset(self, strategy, experience) -> None:
        if self._initial_state is None:
            raise RuntimeError("Initial model state has not been captured")
        restore_initial_state(strategy.model, self._initial_state)
        # A fresh skill must have the head required by the current
        # Avalanche experience before its single-class training pass.
        prepare_for_experience(strategy.model, experience)
        self._reset_optimizer(strategy)

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------

    def before_training_exp(self, strategy, **kwargs) -> None:
        experience = strategy.experience
        first_subexp = self._is_first_subexp(experience)

        # A logical Avalanche experience can be split into sub-experiences.
        # The old implementation processed ONLY the first sub-experience,
        # which silently dropped classes that lived in later sub-experiences.
        # Keep one logical experience index, but process every sub-experience.
        if first_subexp:
            if self._task_active:
                raise RuntimeError(
                    "A new first sub-experience arrived while the previous "
                    "logical experience is still active"
                )

            self._task_active = True
            self._current_training_experience_index = self._training_experience_count
            self._training_experience_count += 1
            experience_index = self._current_training_experience_index

            if self._initial_state is None:
                self._initial_state = self._snapshot(strategy.model)

            self.last_class_decisions[experience_index] = {}

            # Never let Avalanche's normal mixed-experience loop retrain the
            # data after our explicit class-by-class loop.
            self._original_train_epochs = getattr(strategy, "train_epochs", None)
            if self._original_train_epochs is not None:
                strategy.train_epochs = 0
        else:
            if not self._task_active or self._current_training_experience_index is None:
                raise RuntimeError(
                    "Received a non-first sub-experience without an active "
                    "logical training experience"
                )
            experience_index = self._current_training_experience_index

        classes = classes_in_experience(experience)
        self._log(
            f"Experience {experience_index} "
            f"subexp(first={first_subexp}, last={self._is_last_subexp(experience)}): "
            f"classes={classes}"
        )

        if not classes:
            self._log(
                f"Experience {experience_index}: empty sub-experience; nothing to train"
            )
            return

        for target_class in classes:
            decision_start = time.perf_counter()
            decision = decide_class(
                strategy,
                experience,
                target_class,
                self.memory,
                self.class_map,
                self.probe_batch_size,
                self.probe_batches,
                self.probe_seed,
                self._seen_experiences,
                self.forgetting_margin,
                self.score_floor,
                self.force_decision,
                self._log,
                max_safety_candidates=self.max_safety_candidates,
            )
            self._log(
                f"Class {target_class}: imagination+decision "
                f"time={time.perf_counter() - decision_start:.2f}s"
            )

            if decision["decision"] == self.REUSE:
                skill = decision["skill"]
                self._log(
                    f"Class {target_class}: REUSE skill {skill} "
                    f"(mutable={self.reuse_is_mutable})"
                )
                apply_skill_state(strategy.model, self.memory.state(skill), experience)
                self._reset_optimizer(strategy)

                if self.reuse_is_mutable:
                    train_on_class(
                        strategy,
                        experience,
                        target_class,
                        self.class_train_epochs,
                        self.class_train_batch_size,
                    )
                    self.memory.store(
                        skill,
                        strategy.model.state_dict(),
                        metadata={
                            **self.memory.metadata(skill),
                            "last_updated_class": target_class,
                            "last_updated_experience": experience_index,
                        },
                    )
                    self._log(
                        f"Class {target_class}: skill {skill} updated in place"
                    )
                else:
                    self._log(f"Class {target_class}: skill {skill} left unchanged")
            else:
                skill = self.memory.allocate()
                self._log(
                    f"Class {target_class}: SCRATCH -> new skill {skill}"
                )
                self._scratch_reset(strategy, experience)
                train_on_class(
                    strategy,
                    experience,
                    target_class,
                    self.class_train_epochs,
                    self.class_train_batch_size,
                )
                self.memory.store(
                    skill,
                    strategy.model.state_dict(),
                    metadata={
                        "acquisition_decision": self.SCRATCH,
                        "experience_index": experience_index,
                        "last_updated_class": target_class,
                        "probe_batch_size": self.probe_batch_size,
                        "probe_batches": self.probe_batches,
                        "probe_seed": self.probe_seed,
                    },
                )
                decision["skill"] = skill

            self.last_class_decisions[experience_index][target_class] = decision
            self.class_map.record(
                ClassRecord(
                    experience_index=experience_index,
                    class_id=target_class,
                    decision=decision["decision"],
                    skill=decision["skill"],
                    new_score=decision.get("new_score", 0.0),
                    old_score=decision.get("old_score", 0.0),
                    old_accuracy=decision.get("old_accuracy", 0.0),
                    new_accuracy=decision.get("new_accuracy", 0.0),
                )
            )

    def after_training_exp(self, strategy, **kwargs) -> None:
        experience = strategy.experience
        if not self._is_last_subexp(experience):
            return

        experience_index = self._current_training_experience_index
        if experience_index is None:
            raise RuntimeError("Missing current training experience index")

        grouped = self.class_map.skills_for_experience(experience_index)
        for skill, classes in grouped:
            self._log(
                f"Experience {experience_index}: skill {skill} covers "
                f"classes {sorted(classes)}"
            )

        # Explicit class -> skill view.  This is intentionally printed for
        # every class in the logical experience, including classes that were
        # attached to an already-existing skill.  It makes it impossible to
        # mistake a skill index for a class index.
        assignments = self.class_map.class_skill_for_experience(experience_index)
        self._log(
            f"Experience {experience_index}: class->skill "
            + ", ".join(
                f"{class_id}->{skill}" for class_id, skill in sorted(assignments.items())
            )
        )

        if self._original_train_epochs is not None:
            strategy.train_epochs = self._original_train_epochs
        self._original_train_epochs = None
        self._seen_experiences.append(origin_experience(experience))
        self._current_training_experience_index = None
        self._task_active = False

    # ------------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------------

    def before_eval(self, strategy, **kwargs) -> None:
        self._pre_eval_state = self._snapshot(strategy.model)
        self._eval_active = True

    def before_eval_exp(self, strategy, **kwargs) -> None:
        if not self._eval_active or self.eval_routing == "none":
            return

        experience = strategy.experience

        if self.eval_routing == "oracle":
            # This is intentionally only a coarse diagnostic.  A mixed
            # eval experience may contain classes belonging to several
            # skills, so a single global swap is not a true class oracle.
            experience_index = getattr(experience, "current_experience", None)
            if experience_index is None:
                experience_index = getattr(experience, "experience_id", None)
            if experience_index is None:
                return

            grouped = self.class_map.skills_for_experience(int(experience_index))
            if not grouped:
                return
            skill = max(grouped, key=lambda item: len(item[1]))[0]
            apply_skill_state_exact(strategy.model, self.memory.state(skill))
            self._reset_optimizer(strategy)
            self._log(
                f"[ORACLE eval diagnostic] experience {experience_index} -> skill {skill}"
            )
            return

        if not self.memory:
            return

        eval_x, _ = probe_whole_experience(
            experience,
            self.probe_batch_size,
            self.probe_batches,
            self.probe_seed,
        )
        probe_model = deepcopy(strategy.model)  # reused across all candidates
        best_slot, best_entropy = None, None
        for slot in sorted(self.memory.slots()):
            entropy = predictive_entropy(
                probe_model, self.memory.state(slot), experience, eval_x
            )
            if best_entropy is None or entropy < best_entropy:
                best_entropy, best_slot = entropy, slot

        if best_slot is not None:
            apply_skill_state_exact(strategy.model, self.memory.state(best_slot))
            self._reset_optimizer(strategy)
            self._log(
                f"[PROBE eval diagnostic] selected skill {best_slot} "
                f"(entropy={best_entropy:.4f})"
            )

    def after_eval(self, strategy, **kwargs) -> None:
        if not self._eval_active:
            return
        try:
            if self._pre_eval_state is not None:
                restore_initial_state(strategy.model, self._pre_eval_state)
                self._reset_optimizer(strategy)
        finally:
            self._pre_eval_state = None
            self._eval_active = False
