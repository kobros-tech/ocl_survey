"""Skill Memory extension using a CL-independent normal ML router."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any

import torch
from torch import Tensor

from .behavior import (
    BehaviorFingerprintCache,
    ClassBehaviorRecord,
    build_weight_behavior_statistics,
)
from .probing import apply_skill_state_exact, predict_logits, probe_class
from .reverse_engineering import NormalMLReverseEngineer
from .skill_memory_plugin import SkillMemoryPlugin


class PersistentFingerprintSkillMemoryPlugin(SkillMemoryPlugin):
    """Skill Memory with anonymous class routing learned by standalone ML.

    The router is trained only after training experiences, from frozen skill
    snapshots and deterministic reference samples. Evaluation only performs
    inference. No evaluation label, task ID, or experience ID is an input to the
    routing model.
    """

    def __init__(
        self,
        *args,
        reverse_engineer_y_fn: Callable | None = None,
        reverse_hidden_size: int = 64,
        reverse_epochs: int = 200,
        reverse_learning_rate: float = 1e-3,
        reverse_seed: int = 0,
        reverse_batch_size: int = 256,
        reverse_training_mode: str = "listwise",
        **kwargs,
    ):
        kwargs.setdefault("reuse_is_mutable", False)
        super().__init__(*args, **kwargs)
        self.behavior = BehaviorFingerprintCache()
        self._custom_reverse_engineer_y = reverse_engineer_y_fn
        self.reverse_engineer = NormalMLReverseEngineer(
            hidden_size=reverse_hidden_size,
            epochs=reverse_epochs,
            learning_rate=reverse_learning_rate,
            seed=reverse_seed,
            batch_size=reverse_batch_size,
            training_mode=reverse_training_mode,
        )
        self._behavior_initialized = False
        self._pending_reference_inputs: dict[int, Tensor] = {}
        self._reverse_output_dim: int | None = None
        self._reverse_candidate_dim: int | None = None
        self.last_fingerprint_routes: list[dict] = []
        self.fingerprint_route_history: list[dict] = []
        self._fingerprint_batch_index = 0
        self._evaluation_experience_index: int | None = None

    @staticmethod
    def _pad_logits(logits: Tensor, output_dim: int) -> Tensor:
        """Pad an older frozen skill response to the current class space."""
        result = logits.new_full((logits.shape[0], output_dim), -1e4)
        width = min(logits.shape[-1], output_dim)
        result[:, :width] = logits[:, :width]
        return result

    @classmethod
    def _make_features(
        cls,
        x: Tensor,
        logits: Tensor,
        candidate_weight: Tensor,
        candidate_bias: float,
        output_dim: int,
        candidate_class_id: int | None = None,
    ) -> Tensor:
        """Build normal-ML inputs for one candidate class.

        Candidate identity is represented only through model-derived quantities:
        the candidate classifier parameters and, when available, that
        candidate's logit/probability in the frozen response. The integer class
        ID itself is never included as a feature.
        """
        samples = x.detach().float().cpu().reshape(x.shape[0], -1)
        padded = cls._pad_logits(logits.detach().float().cpu(), output_dim)
        probabilities = torch.softmax(padded, dim=-1)
        if candidate_class_id is None or not 0 <= candidate_class_id < output_dim:
            candidate_logit = torch.zeros((samples.shape[0], 1))
            candidate_probability = torch.zeros((samples.shape[0], 1))
        else:
            candidate_logit = padded[:, candidate_class_id].reshape(-1, 1)
            candidate_probability = probabilities[:, candidate_class_id].reshape(
                -1, 1
            )
        weight = candidate_weight.detach().float().cpu().reshape(1, -1)
        weight = weight.expand(samples.shape[0], -1)
        bias = torch.full((samples.shape[0], 1), float(candidate_bias))

        if samples.shape[1] == weight.shape[1]:
            interaction = samples * weight
            sample_norm = samples.norm(dim=1, keepdim=True).clamp_min(1e-8)
            weight_norm = weight.norm(dim=1, keepdim=True).clamp_min(1e-8)
            cosine = interaction.sum(dim=1, keepdim=True) / (
                sample_norm * weight_norm
            )
            dot_product = interaction.sum(dim=1, keepdim=True)
        else:
            interaction = torch.zeros_like(weight)
            dot_product = torch.zeros((samples.shape[0], 1))
            cosine = torch.zeros((samples.shape[0], 1))

        return torch.cat(
            (
                samples,
                padded,
                probabilities,
                candidate_logit,
                candidate_probability,
                weight,
                bias,
                interaction,
                dot_product,
                cosine,
            ),
            dim=1,
        )

    def _frozen_logits(
        self,
        strategy,
        skill_id: int,
        x: Tensor,
    ) -> Tensor:
        """Evaluate one immutable stored skill without changing the live model."""
        records = self.behavior.records_for_skill(skill_id)
        if records:
            version = records[0].version
            state_dict = self.behavior.skill_state(skill_id, version)
        else:
            state_dict = None
        if state_dict is None:
            state_dict = self.memory.state(skill_id)
        model = deepcopy(strategy.model)
        return predict_logits(model, state_dict, x).detach().cpu()

    def _build_record(
        self,
        strategy,
        skill_id: int,
        class_id: int,
        x: Tensor,
        version: int,
        state_dict: dict | None = None,
    ) -> ClassBehaviorRecord:
        """Create diagnostics from the frozen skill-generation state."""
        if state_dict is None:
            state_dict = self.memory.state(skill_id)
        model = deepcopy(strategy.model)
        apply_skill_state_exact(model, state_dict)
        logits = predict_logits(model, state_dict, x)
        reference_y = logits[:, class_id].gt(0).detach().cpu()
        statistics = build_weight_behavior_statistics(model, x, class_id)
        return ClassBehaviorRecord(
            class_id=class_id,
            skill_id=skill_id,
            version=version,
            reference_inputs=x.detach().cpu().clone(),
            reference_y=reference_y,
            reference_feature_mean=statistics["feature_mean"],
            reference_feature_std=statistics["feature_std"],
            reference_margin_mean=statistics["margin_mean"],
            reference_margin_std=statistics["margin_std"],
            reference_weight=statistics["weight"],
            reference_bias=statistics["bias"],
        )

    def _capture_new_class_inputs(self, experience, experience_index: int) -> None:
        """Keep deterministic reference inputs for newly introduced classes."""
        decisions = self.last_class_decisions.get(experience_index, {})
        for class_id, item in decisions.items():
            skill_id = item.get("skill")
            if skill_id is None:
                continue
            if self.behavior.get(class_id, int(skill_id)) is not None:
                continue
            if class_id in self._pending_reference_inputs:
                continue
            x, _ = probe_class(
                experience,
                class_id,
                self.probe_batch_size,
                self.probe_batches,
                self.probe_seed,
            )
            self._pending_reference_inputs[class_id] = x.detach().cpu().clone()

    def _refresh_skill(self, strategy, skill_id: int, experience) -> None:
        """Refresh diagnostics for one immutable skill generation."""
        version = self.behavior.skill_version(skill_id)
        state_dict = self.memory.state(skill_id)
        self.behavior.put_skill_state(skill_id, version, state_dict)
        classes = sorted(self.class_map.classes_for_skill(skill_id))
        existing = {
            record.class_id: record
            for record in self.behavior.all_records_for_skill(skill_id)
        }
        for class_id in classes:
            record = existing.get(class_id)
            x = (
                record.reference_inputs
                if record is not None
                else self._pending_reference_inputs.get(class_id)
            )
            if x is None:
                x, _ = probe_class(
                    experience,
                    class_id,
                    self.probe_batch_size,
                    self.probe_batches,
                    self.probe_seed,
                )
            self.behavior.put(
                self._build_record(strategy, skill_id, class_id, x, version, state_dict)
            )

    def _collect_changed_skills(self, experience_index: int) -> set[int]:
        """Return skill generations whose frozen response actually changed."""
        decisions = self.last_class_decisions.get(experience_index, {})
        changed: set[int] = set()
        for item in decisions.values():
            decision = item.get("decision")
            skill_id = item.get("skill")
            if skill_id is None:
                continue
            if decision == self.SCRATCH:
                changed.add(int(skill_id))
            elif decision == self.REUSE and self.reuse_is_mutable:
                changed.add(int(skill_id))
        return changed

    def before_training_exp(self, strategy, **kwargs) -> None:
        super().before_training_exp(strategy, **kwargs)

    def after_training_exp(self, strategy, **kwargs) -> None:
        """Update frozen references, then retrain the standalone router once."""
        experience = strategy.experience
        experience_index = self._current_training_experience_index
        is_last = self._is_last_subexp(experience)
        if experience_index is not None:
            self._capture_new_class_inputs(experience, experience_index)
        changed = (
            self._collect_changed_skills(experience_index)
            if is_last and experience_index is not None
            else set()
        )
        pending = dict(self._pending_reference_inputs)
        super().after_training_exp(strategy, **kwargs)
        if experience_index is None or not is_last:
            return
        for skill_id in sorted(changed):
            self.behavior.bump_skill(skill_id)
            self._refresh_skill(strategy, skill_id, experience)
        for class_id, x in sorted(pending.items()):
            skill_id = self.class_map.find_skill_for_class_anywhere(class_id)
            if skill_id is None or self.behavior.get(class_id, int(skill_id)) is not None:
                continue
            skill_id = int(skill_id)
            version = self.behavior.skill_version(skill_id)
            state_dict = self.behavior.skill_state(skill_id, version)
            if state_dict is None:
                state_dict = self.memory.state(skill_id)
                self.behavior.put_skill_state(skill_id, version, state_dict)
            self.behavior.put(
                self._build_record(
                    strategy, skill_id, int(class_id), x, version, state_dict
                )
            )
        self._pending_reference_inputs.clear()
        self._behavior_initialized = bool(self.behavior.state_dict()["records"])
        if self._behavior_initialized:
            self._fit_reverse_router(strategy)

    def _fit_reverse_router(self, strategy) -> None:
        """Fit from candidate sets using the same candidate behavior as routing."""
        slot_ids = sorted(self.memory.slots())
        records = [
            record
            for slot in slot_ids
            for record in self.behavior.records_for_skill(slot)
        ]
        if not records:
            self.reverse_engineer.fit_candidate_sets([])
            self._reverse_output_dim = None
            self._reverse_candidate_dim = None
            return

        output_dim = 0
        candidate_dim = 0
        for record in records:
            candidate_dim = max(
                candidate_dim,
                int(record.reference_weight.numel())
                if record.reference_weight is not None
                else 0,
            )
            for slot in slot_ids:
                logits = self._frozen_logits(strategy, slot, record.reference_inputs)
                output_dim = max(output_dim, int(logits.shape[-1]))
        if candidate_dim == 0:
            raise RuntimeError("reverse router requires stored candidate class weights")

        candidates = [
            record
            for record in records
            if record.reference_weight is not None
            and record.reference_weight.numel() == candidate_dim
        ]
        class_to_candidate = {
            record.class_id: index for index, record in enumerate(candidates)
        }
        if len(class_to_candidate) != len(candidates):
            raise RuntimeError("reverse router requires one candidate record per class")

        self._reverse_output_dim = output_dim
        self._reverse_candidate_dim = candidate_dim
        candidate_logits: dict[tuple[int, int], Tensor] = {}
        for record in records:
            for candidate in candidates:
                key = (record.class_id, candidate.skill_id)
                if key not in candidate_logits:
                    candidate_logits[key] = self._frozen_logits(
                        strategy, candidate.skill_id, record.reference_inputs
                    )

        candidate_sets: list[tuple[Tensor, int]] = []
        for record in records:
            target_index = class_to_candidate.get(record.class_id)
            if target_index is None:
                continue
            feature_rows: list[Tensor] = []
            for sample_index in range(record.reference_inputs.shape[0]):
                rows = []
                for candidate in candidates:
                    logits = candidate_logits[(record.class_id, candidate.skill_id)]
                    rows.append(
                        self._make_features(
                            record.reference_inputs[sample_index : sample_index + 1],
                            logits[sample_index : sample_index + 1],
                            candidate.reference_weight,
                            candidate.reference_bias,
                            output_dim,
                            candidate.class_id,
                        ).squeeze(0)
                    )
                feature_rows.append(torch.stack(rows, dim=0))
            candidate_sets.extend((rows, target_index) for rows in feature_rows)
        self.reverse_engineer.fit_candidate_sets(candidate_sets)

    def _route(self, strategy, x: Tensor, slot_ids: list[int]) -> tuple[Tensor, list[int]]:
        """Route anonymously by listwise scoring of the complete candidate set."""
        if self.reverse_engineer.model is None or self._reverse_output_dim is None:
            raise RuntimeError("reverse router has not been fitted")
        records = [
            record
            for slot in slot_ids
            for record in self.behavior.records_for_skill(slot)
            if record.reference_weight is not None
            and record.reference_weight.numel() == self._reverse_candidate_dim
        ]
        if not records:
            return (
                torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device),
                [-1] * x.shape[0],
            )
        if len({record.class_id for record in records}) != len(records):
            raise RuntimeError("reverse router requires one candidate record per class")

        route_candidates: list[tuple[ClassBehaviorRecord, Tensor]] = []
        for record in records:
            logits = self._frozen_logits(strategy, record.skill_id, x)
            features = self._make_features(
                x,
                logits,
                record.reference_weight,
                record.reference_bias,
                self._reverse_output_dim,
                record.class_id,
            )
            score = self.reverse_engineer.predict_scores_features(features)
            route_candidates.append((record, score))

        scores = torch.stack([item[1] for item in route_candidates], dim=1)
        probabilities = torch.softmax(scores, dim=1)
        best = probabilities.argmax(dim=1)
        chosen_skills: list[int] = []
        chosen_classes: list[int] = []
        routes: list[dict[str, Any]] = []
        for sample_index in range(x.shape[0]):
            candidate_index = int(best[sample_index].item())
            record, score = route_candidates[candidate_index]
            chosen_skills.append(record.skill_id)
            chosen_classes.append(record.class_id)
            routes.append(
                {
                    "sample_index": sample_index,
                    "status": "IDENTIFIED",
                    "class": record.class_id,
                    "skill": record.skill_id,
                    "score": float(score[sample_index].item()),
                    "probability": float(
                        probabilities[sample_index, candidate_index].item()
                    ),
                    "candidates": [
                        {
                            "class": candidate.class_id,
                            "skill": candidate.skill_id,
                            "score": float(candidate_score[sample_index].item()),
                            "probability": float(
                                probabilities[sample_index, index].item()
                            ),
                        }
                        for index, (candidate, candidate_score) in enumerate(
                            route_candidates
                        )
                    ],
                }
            )
        self.last_fingerprint_routes = routes
        return (
            torch.tensor(chosen_skills, dtype=torch.long, device=x.device),
            chosen_classes,
        )

    def before_eval(self, strategy, **kwargs) -> None:
        super().before_eval(strategy, **kwargs)
        self.fingerprint_route_history = []
        self._fingerprint_batch_index = 0
        self._evaluation_experience_index = None

    def before_eval_exp(self, strategy, **kwargs) -> None:
        super().before_eval_exp(strategy, **kwargs)
        experience = strategy.experience
        index = getattr(experience, "current_experience", None)
        if index is None:
            index = getattr(experience, "experience_id", None)
        self._evaluation_experience_index = None if index is None else int(index)

    def after_eval_forward(self, strategy, **kwargs) -> None:
        if not self._eval_active or self.eval_routing != "probe":
            return super().after_eval_forward(strategy, **kwargs)
        if len(self.memory) == 0 or not self._behavior_initialized:
            return super().after_eval_forward(strategy, **kwargs)
        x = strategy.mbatch[0]
        y = strategy.mbatch[1]
        slot_ids = sorted(self.memory.slots())
        chosen, class_matches = self._route(strategy, x, slot_ids)
        for route, label in zip(
            self.last_fingerprint_routes,
            y.detach().cpu().tolist(),
            strict=False,
        ):
            route["batch_index"] = self._fingerprint_batch_index
            route["evaluation_experience"] = self._evaluation_experience_index
            route["evaluation_y"] = int(label)
        self.fingerprint_route_history.extend(self.last_fingerprint_routes)
        self._fingerprint_batch_index += 1

        output_model = deepcopy(strategy.model)
        per_skill_logits = [
            predict_logits(output_model, self.memory.state(slot), x)
            for slot in slot_ids
        ]
        output_dim = strategy.mb_output.shape[-1]
        padded = [self._pad_logits(logits, output_dim) for logits in per_skill_logits]
        valid = chosen.ge(0)
        if valid.any():
            positions = torch.nonzero(valid, as_tuple=False).squeeze(-1)
            skill_to_row = {skill_id: row for row, skill_id in enumerate(slot_ids)}
            rows = torch.tensor(
                [skill_to_row[int(skill)] for skill in chosen[valid].tolist()],
                device=chosen.device,
                dtype=torch.long,
            )
            stacked = torch.stack(padded, dim=0)
            strategy.mb_output[positions] = stacked[rows, positions]

        final_predictions = strategy.mb_output.detach().argmax(dim=-1).cpu().tolist()
        labels = y.detach().cpu().tolist()
        for route, prediction, label in zip(
            self.last_fingerprint_routes,
            final_predictions,
            labels,
            strict=False,
        ):
            route["model_predicted_class"] = int(prediction)
            route["model_correct"] = int(prediction) == int(label)

        self._log(
            "[NORMAL ML routing] "
            f"eval_exp={self._evaluation_experience_index} "
            f"samples={x.shape[0]} identified={len(self.last_fingerprint_routes)} "
            f"matched_classes={class_matches[:5]}"
        )

    def state_dict(self) -> dict:
        """Serialize persistent references and the fitted reverse router."""
        return {
            "behavior": self.behavior.state_dict(),
            "reverse_engineer": self.reverse_engineer.state_dict(),
            "reverse_output_dim": self._reverse_output_dim,
            "reverse_candidate_dim": self._reverse_candidate_dim,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore persistent references and a previously fitted router."""
        behavior_state = state.get("behavior", {})
        self.behavior.load_state_dict(behavior_state)
        self.reverse_engineer.load_state_dict(state.get("reverse_engineer", {}))
        output_dim = state.get("reverse_output_dim")
        self._reverse_output_dim = None if output_dim is None else int(output_dim)
        candidate_dim = state.get("reverse_candidate_dim")
        self._reverse_candidate_dim = (
            None if candidate_dim is None else int(candidate_dim)
        )
        self._behavior_initialized = bool(behavior_state.get("records"))
