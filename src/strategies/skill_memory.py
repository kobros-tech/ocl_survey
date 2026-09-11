"""
skill_memory.py

Avalanche-integrated port of the task-free Skill Memory strategy prototyped
in notebooks/skill_memory.py.

The decision logic is the same dynamic, probe-based, gap-clustering selection
used by the notebook. Skill reuse is decided before training an experience.

Training semantics are intentionally explicit:

  * SCRATCH allocates a new skill slot, trains it, and stores the result.
  * REUSE starts from an existing skill, but does not overwrite that stored
    skill after training.
  * Every completed training experience is recorded in an explicit
    experience -> skill routing table (`_experience_to_skill`).

That routing table is bookkeeping only. It is never used to decide which
skill should be selected for a *new training* experience, and — after the
correction below — it is also no longer used to decide which skill should
be loaded at *evaluation* time by default.

--------------------------------------------------------------------------
CORRECTION (see PR review): evaluation-time routing must not be an oracle
--------------------------------------------------------------------------
A previous revision of this file implemented `before_eval_exp` by looking
up the *ground-truth* Avalanche experience index (`current_experience`)
in `_experience_to_skill` and loading the exact skill that had been frozen
for that experience during training:

    slot = self._experience_to_skill.get(experience_index)
    _apply_skill_state_exact(strategy.model, self.memory.state(slot))

This is a genuine evaluation oracle. It does not ask the model to infer
anything about the test batch; it asks the benchmark "which training
experience is this officially from?" and answers with a table lookup.
Because stored skills are immutable once written, this makes forgetting
close to 0% by construction — not because the mechanism resists
interference, but because interference was never possible: nothing had
touched the frozen skill since training. That is not comparable to a
single continually-updated model such as ER, and reporting it as though
it were is not a fair "forgetting" comparison.

The fix below removes that default behaviour and replaces it with three
explicit, clearly-labeled `eval_routing` modes:

  * "none"  (default) — no swapping at all. Evaluation scores whatever
    state is currently sitting in `strategy.model`, exactly like ER. This
    is the mode that should be used for the headline Skill-Memory-vs-ER
    comparison and forgetting metric.

  * "probe" — a legitimate inference-time retrieval mode. For each eval
    experience, every stored skill's *predictive confidence* (softmax
    entropy) is measured on the current eval batch's *inputs only* — no
    labels, and no access to the ground-truth experience index — and the
    most confident skill is loaded. This is a real statistical decision
    the model makes about which skill fits, comparable to a mixture-of-
    experts gate. It is a different mechanism from a single continual
    model, so it should be reported as a separate result, not folded into
    the ER-style forgetting comparison.

  * "oracle" — the old ground-truth lookup, kept only as an explicit,
    opt-in upper-bound / mechanism-validation diagnostic. It must never be
    used to produce the number compared against ER. A loud warning is
    logged whenever it is enabled.
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
from torch.utils.data import ConcatDataset, DataLoader

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
    model: nn.Module,
    state_dict: Mapping[str, Tensor],
    experience,
) -> None:
    _resize_incremental_classifiers_for_state(model, state_dict)
    model.load_state_dict(state_dict, strict=False)
    avalanche_model_adaptation(model, _origin_experience(experience))


def _apply_skill_state_exact(
    model: nn.Module,
    state_dict: Mapping[str, Tensor],
) -> None:
    _resize_incremental_classifiers_for_state(model, state_dict)
    model.load_state_dict(state_dict, strict=False)


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
    return float(np.exp(-loss_value))


def _evaluate_state(
    model_factory: Callable[[], nn.Module],
    state_dict,
    experience,
    x,
    y,
    criterion,
):
    """Score a candidate skill using inputs AND labels.

    This is appropriate during *training-time* decisions, where the probe
    data comes from the current training experience and using its labels
    to compute a compatibility score is completely legitimate (it is no
    different from training on that data). It must never be used to route
    at evaluation time, because that would mean selecting a model using
    the test set's ground-truth labels.
    """
    model = model_factory()
    _apply_skill_state(model, state_dict, experience)
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
        logits = model(x)
        loss = float(criterion(logits, y).item())
        accuracy = float((logits.argmax(dim=1) == y).float().mean().item())
    return loss, score_from_loss(loss), accuracy


def _predictive_entropy(
    model_factory: Callable[[], nn.Module],
    state_dict,
    experience,
    x,
) -> float:
    """Blind compatibility signal for eval-time retrieval.

    Uses only the *inputs* of the current eval batch — never labels, and
    never the ground-truth experience index — to measure how confidently a
    candidate skill's model predicts on this batch. Lower mean predictive
    entropy means the skill is more confident on this input distribution,
    which is used as a stand-in for "this skill is a good match."

    This mirrors what a real deployed system could do at inference time:
    it has the input, not an answer key telling it which training
    experience the input came from.
    """
    model = model_factory()
    _apply_skill_state(model, state_dict, experience)
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).mean()
    return float(entropy.item())


def find_best_skill(
    imagination_results: list[dict[str, Any]],
    forgetting_margin: float,
    score_floor: float = 0.9,
):
    """Select an existing skill only when it is safe and compatible.

    Used exclusively for the *training-time* REUSE/SCRATCH decision. Never
    used for evaluation routing.
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

    def strongest_candidates(results, key, floor):
        ranked = sorted(results, key=lambda result: result[key], reverse=True)
        if len(ranked) == 1:
            return {ranked[0]["skill"]} if ranked[0][key] > floor else set()

        values = [result[key] for result in ranked]
        gaps = [values[i] - values[i + 1] for i in range(len(values) - 1)]
        split = max(range(len(gaps)), key=lambda index: gaps[index])
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
        key=lambda result: (result["new_score"], result["new_accuracy"]),
    )


class SkillMemoryPlugin(SupervisedPlugin):
    """Probe-based, task-free Skill Memory plugin.

    `eval_routing` controls what happens to `strategy.model` during
    evaluation:

      * "none"   (default, safe): no swapping. Evaluation uses whatever
        state training left behind, exactly like ER. Use this for the
        headline comparison against other OCL methods.
      * "probe":  blind, statistics-only retrieval using only the current
        eval batch's inputs (see `_predictive_entropy`). A legitimate
        mixture-of-experts-style mechanism, but a different mechanism from
        a single continual model — report it separately, not as the same
        "forgetting" number as ER.
      * "oracle": ground-truth `experience_index -> skill` lookup. Upper
        bound / mechanism-validation only. Never use this to produce the
        number compared against ER.
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
        if force_decision not in (None, self.REUSE, self.SCRATCH):
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
                "SkillMemoryPlugin: eval_routing='oracle' is enabled. "
                "This loads the skill frozen for each experience using the "
                "ground-truth experience index and MUST NOT be used to "
                "produce the number compared against ER or other single-"
                "model OCL baselines. Use it only as an explicit upper-"
                "bound / mechanism-validation diagnostic."
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

        # Internal chronological bookkeeping. This is deliberately separate
        # from Avalanche's experience identifier.
        self._training_experience_count = 0
        self._current_training_experience_index: int | None = None

        # Training experience index -> skill slot.
        # Bookkeeping / audit trail. Read at evaluation time ONLY when
        # eval_routing == "oracle" (an explicit, non-default diagnostic).
        self._experience_to_skill: dict[int, int] = {}

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
        _restore_initial_state(strategy.model, self._initial_state)
        self._reset_optimizer(strategy)

    @staticmethod
    def _is_first_subexp(experience) -> bool:
        return getattr(experience, "is_first_subexp", True)

    @staticmethod
    def _is_last_subexp(experience) -> bool:
        return getattr(experience, "is_last_subexp", True)

    @staticmethod
    def _experience_index(experience) -> int | None:
        """Return Avalanche's benchmark experience index.

        Only ever consulted when eval_routing == 'oracle'.
        """
        index = getattr(experience, "current_experience", None)
        if index is not None:
            return int(index)
        index = getattr(experience, "experience_id", None)
        if index is not None:
            return int(index)
        return None

    def _score_slots(self, strategy, experience) -> list[dict[str, Any]]:
        """Training-time compatibility scoring. Uses the current TRAINING
        experience's data (inputs and labels) — never eval/test data."""
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

            out_features = _incremental_out_features(strategy.model, state_dict)
            chance = 1.0 / out_features if out_features else 0.0

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

    def before_training_exp(self, strategy, **kwargs):
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

        if self._active_slot is None:
            raise RuntimeError("No active Skill Memory slot after training experience")

        slot = self._active_slot
        experience_index = self._current_training_experience_index
        if experience_index is None:
            raise RuntimeError("Missing current training experience index")

        # SCRATCH creates a new independent skill. Store the trained state.
        # REUSE deliberately does NOT store the post-training state back into
        # the existing slot. The stored skill remains immutable.
        if self.last_decision == self.SCRATCH:
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
                    "experience_index": experience_index,
                },
            )
            self._log(
                f"Stored trained scratch model as skill {slot} "
                f"for experience {experience_index}"
            )
        else:
            self._log(
                f"Reused skill {slot} for experience {experience_index}; "
                "stored skill left unchanged"
            )

        # This is bookkeeping / audit trail. It is never used to bias which
        # skill is selected for a NEW training experience, and — as of the
        # fix documented at the top of this file — it is also not consulted
        # at evaluation time unless eval_routing == "oracle" is explicitly
        # enabled for a clearly-labeled diagnostic run.
        self._experience_to_skill[experience_index] = slot
        self._seen_experiences.append(_origin_experience(experience))

        self._active_slot = None
        self._current_training_experience_index = None
        self._task_active = False

    def before_eval(self, strategy, **kwargs):
        self._pre_eval_state = {
            key: value.detach().cpu().clone()
            for key, value in strategy.model.state_dict().items()
        }
        self._eval_active = True

    def before_eval_exp(self, strategy, **kwargs):
        if not self._eval_active:
            return
        if self.eval_routing == "none":
            # Safe default: no swapping. Evaluate whatever the continually
            # trained model currently holds, exactly like ER.
            return

        experience = strategy.experience

        if self.eval_routing == "oracle":
            experience_index = self._experience_index(experience)
            if experience_index is None:
                self._log(
                    "Evaluation experience has no current_experience index; "
                    "keeping current model."
                )
                return
            slot = self._experience_to_skill.get(experience_index)
            if slot is None:
                self._log(
                    f"No stored skill for evaluation experience "
                    f"{experience_index}; keeping current model."
                )
                return
            _apply_skill_state_exact(strategy.model, self.memory.state(slot))
            self._reset_optimizer(strategy)
            self._log(
                f"[ORACLE eval_routing] experience {experience_index} -> "
                f"skill {slot}. Do not use this run for the ER comparison."
            )
            return

        if self.eval_routing == "probe":
            if len(self.memory) == 0:
                return
            # Blind retrieval: use only the current eval batch's INPUTS.
            # No labels, no ground-truth experience index.
            eval_x, _eval_y = _probe(
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
            _apply_skill_state_exact(strategy.model, self.memory.state(best_slot))
            self._reset_optimizer(strategy)
            self._log(
                f"[PROBE eval_routing] selected skill {best_slot} "
                f"(mean predictive entropy={best_entropy:.4f}) using only "
                "eval-batch inputs; this is a retrieval mechanism, report "
                "it separately from the ER-style forgetting comparison."
            )
            return

    def after_eval(self, strategy, **kwargs):
        if not self._eval_active:
            return
        try:
            if self._pre_eval_state is not None:
                _restore_initial_state(strategy.model, self._pre_eval_state)
                self._reset_optimizer(strategy)
        finally:
            self._pre_eval_state = None
            self._eval_active = False
