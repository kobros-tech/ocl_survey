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

import logging
import time
from collections.abc import Callable
from copy import deepcopy
from typing import Any, Literal

import torch
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from torch import Tensor

from .decision import decide_class
from .probing import (
    apply_skill_state_exact,
    classes_in_experience,
    expand_skill_logits,
    origin_experience,
    predict_logits,
    prepare_for_experience,
    restore_initial_state,
    route_probe_logits,
)
from .skill_registry import ClassRecord, ExperienceClassMap, SkillMemory
from .training import train_on_class

logger = logging.getLogger(__name__)
EvalRouting = Literal["none", "probe", "class_oracle", "oracle"]
_VALID_EVAL_ROUTINGS = ("none", "probe", "class_oracle", "oracle")


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
        eval_routing: EvalRouting = "probe",
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

        if eval_routing in ("oracle", "class_oracle"):
            logger.warning(
                "eval_routing=%r uses ground-truth labels for routing and is "
                "diagnostic only, not a task-free headline result.",
                eval_routing,
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
                # A canonical class is already known to belong to this skill.
                # Restore its snapshot exactly: re-adapting the model to the
                # current sub-experience can traverse incompatible FlatData
                # indices and is unnecessary for deterministic class reuse.
                apply_skill_state_exact(
                    strategy.model,
                    self.memory.state(skill),
                )
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
                    self._log(f"Class {target_class}: skill {skill} updated in place")
                else:
                    self._log(f"Class {target_class}: skill {skill} left unchanged")
            else:
                skill = self.memory.allocate()
                self._log(f"Class {target_class}: SCRATCH -> new skill {skill}")
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
                f"{class_id}->{skill}"
                for class_id, skill in sorted(assignments.items())
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
            # Backwards-compatible coarse diagnostic: one skill for the
            # complete evaluation experience. This is intentionally not the
            # main class-aware evaluator because one experience can contain
            # classes owned by several skills.
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
                ("[ORACLE eval diagnostic] experience ")(
                    f"{experience_index} -> skill {skill}"
                )
            )
            return

        if self.eval_routing in ("probe", "class_oracle") and self.memory:
            self._log(
                f"[{self.eval_routing.upper()} eval] per-sample routing active "
                f"for experience {getattr(experience, 'current_experience', '?')} "
                f"({len(self.memory)} skills known)"
            )

    def after_eval_forward(self, strategy, **kwargs) -> None:
        """Route each evaluation sample to a stored skill before metrics.

        ``probe`` is label-free: it compares each sample's prediction across
        all stored skills and uses a normalized confidence score.
        ``class_oracle`` is diagnostic only and uses the true label to select
        the canonical skill. Both operate on the actual minibatch, so samples
        from different skills may coexist in one Avalanche batch.

        The confidence used by ``probe`` is deliberately based on the
        classifier output margin rather than raw entropy. Raw entropy is
        incomparable when skill snapshots have different numbers of output
        units, and a one-class head has identically zero entropy for every
        input. The margin is normalized by the L2 norm of the classifier
        weights when that structure is available.
        """
        if not self._eval_active or self.eval_routing not in ("probe", "class_oracle"):
            return
        if len(self.memory) == 0:
            return

        x, y = strategy.mbatch[0], strategy.mbatch[1]
        device = x.device
        slot_ids = sorted(self.memory.slots())
        probe_model = deepcopy(strategy.model)

        raw_skill_logits = [
            predict_logits(probe_model, self.memory.state(slot), x) for slot in slot_ids
        ]
        output_dim = strategy.mb_output.shape[-1]
        per_skill_logits = [
            expand_skill_logits(
                logits,
                self.memory.state(slot),
                self.class_map.classes_for_skill(slot),
                output_dim,
            )
            for slot, logits in zip(slot_ids, raw_skill_logits, strict=False)
        ]
        batch_size = x.shape[0]

        if self.eval_routing == "class_oracle":
            chosen = []
            for label in y.detach().cpu().tolist():
                skill = self.class_map.find_skill_for_class_anywhere(int(label))
                chosen.append(slot_ids.index(skill) if skill in slot_ids else 0)
            chosen = torch.tensor(chosen, device=device, dtype=torch.long)
        else:
            chosen = route_probe_logits(
                raw_skill_logits,
                [self.memory.state(slot) for slot in slot_ids],
                [self.class_map.classes_for_skill(slot) for slot in slot_ids],
            )
            self._log_probe_routing_diagnostic(y, chosen, slot_ids)

        strategy.mb_output = torch.stack(per_skill_logits, dim=0)[
            chosen, torch.arange(batch_size, device=device)
        ]

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

    def _log_probe_routing_diagnostic(
        self, y: Tensor, chosen: Tensor, slot_ids: list[int]
    ) -> None:
        """Log probe-vs-oracle skill-selection agreement for this batch.

        This measures routing quality in isolation, separate from the
        combined probe accuracy number. It answers: "even ignoring
        whether the final prediction was correct, did probe pick the
        SAME skill that class_oracle would have picked?" That number
        distinguishes a routing problem (skills are fine, wrong one
        got picked) from a skill-quality problem (right skill picked,
        but its prediction was still wrong).

        This is purely additive logging -- it does not change routing,
        predictions, or metrics. Safe to run alongside a normal
        `eval_routing="probe"` pass.
        """
        labels = y.detach().cpu().tolist()
        chosen_list = chosen.detach().cpu().tolist()
        agree = 0
        mismatches = []
        for label, chosen_idx in zip(labels, chosen_list, strict=False):
            oracle_skill = self.class_map.find_skill_for_class_anywhere(int(label))
            oracle_idx = (
                slot_ids.index(oracle_skill) if oracle_skill in slot_ids else None
            )
            probe_skill = slot_ids[chosen_idx]
            if oracle_idx is not None and chosen_idx == oracle_idx:
                agree += 1
            else:
                mismatches.append((int(label), oracle_skill, probe_skill))
        total = len(labels)
        routing_acc = agree / total if total else float("nan")
        self._log(
            ("[PROBE routing diagnostic] batch")(
                f" routing_accuracy={routing_acc:.4f} "
            )(f"({agree}/{total} samples ")(
                "routed to the same skill class_oracle would pick)"
            )
        )
        if mismatches:
            sample = mismatches[:5]
            self._log(
                f"[PROBE routing diagnostic] sample mismatches "
                f"(label, oracle_skill, probe_skill): {sample}"
            )
