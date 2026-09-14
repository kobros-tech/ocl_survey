"""Persistent anonymous routing with a standalone ML reverse model."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import Tensor

from .global_fingerprint_refresh import refresh_all_fingerprints
from .persistent_skill_memory_plugin import (
    PersistentFingerprintSkillMemoryPlugin as _BaseFingerprintPlugin,
)
from .probing import predict_logits
from .reverse_engineering import NormalMLReverseEngineer


class PersistentFingerprintSkillMemoryPlugin(_BaseFingerprintPlugin):
    """Route from frozen-skill behavior using an independent ML model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reverse_engineer_model = NormalMLReverseEngineer()

    @staticmethod
    def _behavior_features(model, state_dict, x: Tensor, class_id: int) -> Tensor:
        """Extract fixed-size behavior signals from one frozen candidate skill."""
        logits = predict_logits(model, state_dict, x).detach().float().cpu()
        class_id = int(class_id)
        if not 0 <= class_id < logits.shape[1]:
            raise ValueError("candidate class is outside the model output")
        candidate_logit = logits[:, class_id]
        max_logit = logits.max(dim=1).values
        if logits.shape[1] > 1:
            masked = logits.clone()
            masked[:, class_id] = float("-inf")
            other_max = masked.max(dim=1).values
        else:
            other_max = candidate_logit
        margin = candidate_logit - other_max
        probabilities = torch.softmax(logits, dim=1)
        candidate_probability = probabilities[:, class_id]
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=1)
        return torch.stack(
            (
                candidate_logit,
                max_logit,
                margin,
                candidate_probability,
                entropy,
            ),
            dim=1,
        )

    def _fit_normal_reverse_model(self, strategy) -> None:
        """Train only on behavior emitted by frozen skill generations."""
        records = [
            record
            for slot in sorted(self.memory.slots())
            for record in self.behavior.records_for_skill(slot)
        ]
        if len(records) < 2:
            self.reverse_engineer_model.fit_feature_pairs([])
            return
        model = deepcopy(strategy.model)
        pairs = []
        for source in records:
            for candidate in records:
                state = self.behavior.skill_state(
                    candidate.skill_id,
                    candidate.version,
                )
                if state is None:
                    state = self.memory.state(candidate.skill_id)
                features = self._behavior_features(
                    model,
                    state,
                    source.reference_inputs,
                    candidate.class_id,
                )
                pairs.append(
                    (
                        features,
                        float(candidate.class_id == source.class_id),
                    )
                )
        self.reverse_engineer_model.fit_feature_pairs(pairs)

    def after_training_exp(self, strategy, **kwargs) -> None:
        """Refresh frozen references and retrain the independent ML model."""
        super().after_training_exp(strategy, **kwargs)
        experience = strategy.experience
        if not self._is_last_subexp(experience):
            return
        if self._current_training_experience_index is None:
            return
        self.last_fingerprint_refresh = refresh_all_fingerprints(self, strategy)
        self._fit_normal_reverse_model(strategy)

    def _fingerprint_route(
        self,
        strategy,
        x: Tensor,
        slot_ids: list[int],
    ) -> tuple[Tensor, list[int]]:
        """Infer anonymous class identity from frozen-skill behavior."""
        records = [
            record
            for slot in slot_ids
            for record in self.behavior.records_for_skill(slot)
        ]
        if self.reverse_engineer_model.model is None:
            self._fit_normal_reverse_model(strategy)
        if self.reverse_engineer_model.model is None or not records:
            self.last_fingerprint_routes = [
                {"sample_index": i, "status": "FAILED", "candidates": []}
                for i in range(x.shape[0])
            ]
            return (
                torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device),
                [-1] * x.shape[0],
            )

        model = deepcopy(strategy.model)
        probabilities = []
        candidate_metadata = []
        for record in records:
            state = self.behavior.skill_state(record.skill_id, record.version)
            if state is None:
                state = self.memory.state(record.skill_id)
            features = self._behavior_features(
                model,
                state,
                x,
                record.class_id,
            )
            probabilities.append(
                self.reverse_engineer_model.predict_proba_features(features)
            )
            candidate_metadata.append(record)

        probability_matrix = torch.stack(probabilities, dim=1)
        best_indices = probability_matrix.argmax(dim=1)
        chosen_skills = []
        chosen_classes = []
        routes = []
        for sample_index, best_index in enumerate(best_indices.tolist()):
            selected = candidate_metadata[best_index]
            row = probability_matrix[sample_index]
            top_probability = float(row[best_index].item())
            second_probability = (
                float(torch.topk(row, k=2).values[1].item())
                if row.numel() > 1
                else 0.0
            )
            candidates = []
            for candidate_index, record in enumerate(candidate_metadata):
                probability = float(row[candidate_index].item())
                candidates.append(
                    {
                        "class": record.class_id,
                        "skill": record.skill_id,
                        "predicted_y": probability >= 0.5,
                        "binary_compatible": probability >= 0.5,
                        "reverse_engineering_probability": probability,
                        "expected_y": record.expected_y,
                        "reference_accuracy": record.reference_accuracy,
                        "correct": probability >= 0.5,
                        "class_score": None,
                        "continuous_evidence": probability,
                    }
                )
            chosen_skills.append(int(selected.skill_id))
            chosen_classes.append(int(selected.class_id))
            routes.append(
                {
                    "sample_index": sample_index,
                    "status": "IDENTIFIED",
                    "class": int(selected.class_id),
                    "skill": int(selected.skill_id),
                    "binary_compatible_candidates": int((row >= 0.5).sum().item()),
                    "top_probability": top_probability,
                    "second_probability": second_probability,
                    "candidates": candidates,
                }
            )

        self.last_fingerprint_routes = routes
        return (
            torch.tensor(chosen_skills, dtype=torch.long, device=x.device),
            chosen_classes,
        )
