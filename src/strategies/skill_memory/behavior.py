"""Persistent binary class-behavior fingerprints for anonymous routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from avalanche.models.dynamic_modules import IncrementalClassifier
from torch import Tensor


@dataclass
class ClassBehaviorRecord:
    """Persistent behavior and continuous evidence for one canonical class."""

    class_id: int
    skill_id: int
    version: int
    reference_inputs: Tensor
    reference_y: Tensor
    expected_y: bool = True
    valid: bool = True
    reference_feature_mean: Tensor | None = None
    reference_feature_std: Tensor | None = None
    reference_margin_mean: float = 0.0
    reference_margin_std: float = 1.0
    reference_weight: Tensor | None = None
    reference_bias: float = 0.0

    @property
    def reference_accuracy(self) -> float:
        result = compare_binary_behavior(self.reference_y, self.expected_y)
        return float(result["accuracy"])

    def state_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "skill_id": self.skill_id,
            "version": self.version,
            "reference_inputs": self.reference_inputs.detach().cpu(),
            "reference_y": self.reference_y.detach().cpu().bool(),
            "expected_y": self.expected_y,
            "valid": self.valid,
            "reference_feature_mean": (
                None
                if self.reference_feature_mean is None
                else self.reference_feature_mean.detach().cpu()
            ),
            "reference_feature_std": (
                None
                if self.reference_feature_std is None
                else self.reference_feature_std.detach().cpu()
            ),
            "reference_margin_mean": self.reference_margin_mean,
            "reference_margin_std": self.reference_margin_std,
            "reference_weight": (
                None
                if self.reference_weight is None
                else self.reference_weight.detach().cpu()
            ),
            "reference_bias": self.reference_bias,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "ClassBehaviorRecord":
        return cls(
            class_id=int(state["class_id"]),
            skill_id=int(state["skill_id"]),
            version=int(state["version"]),
            reference_inputs=state["reference_inputs"].detach().cpu(),
            reference_y=state["reference_y"].detach().cpu().bool(),
            expected_y=bool(state.get("expected_y", True)),
            valid=bool(state.get("valid", True)),
            reference_feature_mean=state.get("reference_feature_mean"),
            reference_feature_std=state.get("reference_feature_std"),
            reference_margin_mean=float(state.get("reference_margin_mean", 0.0)),
            reference_margin_std=max(
                float(state.get("reference_margin_std", 1.0)), 1e-6
            ),
            reference_weight=state.get("reference_weight"),
            reference_bias=float(state.get("reference_bias", 0.0)),
        )


class BehaviorFingerprintCache:
    """Version-aware persistent cache of class behavior references.

    A fingerprint must not be recomputed from the live model merely because a
    later experience changed that model. The binary behavior records therefore
    share a frozen skill snapshot for each skill generation. A mutable REUSE
    creates a new generation and deliberately replaces that snapshot.
    """

    def __init__(self) -> None:
        self._records: dict[int, ClassBehaviorRecord] = {}
        self._skill_versions: dict[int, int] = {}
        self._skill_states: dict[int, dict[str, Tensor]] = {}
        self._skill_state_versions: dict[int, int] = {}

    def skill_version(self, skill_id: int) -> int:
        return self._skill_versions.get(int(skill_id), 0)

    def bump_skill(self, skill_id: int) -> int:
        skill_id = int(skill_id)
        version = self.skill_version(skill_id) + 1
        self._skill_versions[skill_id] = version
        self.invalidate_skill(skill_id)
        self._skill_states.pop(skill_id, None)
        self._skill_state_versions.pop(skill_id, None)
        return version

    def invalidate_skill(self, skill_id: int) -> None:
        skill_id = int(skill_id)
        for record in self._records.values():
            if record.skill_id == skill_id:
                record.valid = False

    def put_skill_state(
        self, skill_id: int, version: int, state_dict: dict[str, Tensor]
    ) -> None:
        skill_id = int(skill_id)
        version = int(version)
        self._skill_states[skill_id] = {
            key: value.detach().cpu().clone() for key, value in state_dict.items()
        }
        self._skill_state_versions[skill_id] = version

    def skill_state(self, skill_id: int, version: int) -> dict[str, Tensor] | None:
        skill_id = int(skill_id)
        version = int(version)
        if self._skill_state_versions.get(skill_id) != version:
            return None
        state = self._skill_states.get(skill_id)
        if state is None:
            return None
        return {key: value.clone() for key, value in state.items()}

    def put(self, record: ClassBehaviorRecord) -> None:
        self._records[record.class_id] = record
        self._skill_versions[record.skill_id] = max(
            self.skill_version(record.skill_id), record.version
        )

    def get(self, class_id: int, skill_id: int) -> ClassBehaviorRecord | None:
        record = self._records.get(int(class_id))
        if record is None or not record.valid:
            return None
        if record.skill_id != int(skill_id):
            return None
        if record.version != self.skill_version(skill_id):
            return None
        return record

    def records_for_skill(self, skill_id: int) -> list[ClassBehaviorRecord]:
        return [
            record
            for record in self._records.values()
            if self.get(record.class_id, int(skill_id)) is not None
        ]

    def all_records_for_skill(self, skill_id: int) -> list[ClassBehaviorRecord]:
        skill_id = int(skill_id)
        return [
            record for record in self._records.values() if record.skill_id == skill_id
        ]

    def state_dict(self) -> dict[str, Any]:
        return {
            "skill_versions": dict(self._skill_versions),
            "skill_states": {
                int(skill_id): {
                    key: value.detach().cpu() for key, value in state.items()
                }
                for skill_id, state in self._skill_states.items()
            },
            "skill_state_versions": dict(self._skill_state_versions),
            "records": [record.state_dict() for record in self._records.values()],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._skill_versions = {
            int(key): int(value)
            for key, value in state.get("skill_versions", {}).items()
        }
        self._skill_states = {
            int(skill_id): {
                key: value.detach().cpu().clone() for key, value in skill_state.items()
            }
            for skill_id, skill_state in state.get("skill_states", {}).items()
        }
        self._skill_state_versions = {
            int(key): int(value)
            for key, value in state.get("skill_state_versions", {}).items()
        }
        self._records = {}
        for record_state in state.get("records", []):
            record = ClassBehaviorRecord.from_state_dict(record_state)
            self._records[record.class_id] = record


def _find_classifier(model) -> IncrementalClassifier:
    """Return the model's Avalanche incremental classifier head."""
    for module in model.modules():
        if isinstance(module, IncrementalClassifier):
            return module
    raise ValueError("reverse engineering requires an IncrementalClassifier head")


def extract_features_from_weights(model, x: Tensor) -> Tensor:
    """Capture the representation entering the learned classifier head."""
    classifier = _find_classifier(model).classifier
    captured: dict[str, Tensor] = {}

    def capture_input(_module, inputs, _output) -> None:
        if not inputs:
            raise RuntimeError("classifier forward hook received no input")
        captured["features"] = inputs[0].detach()

    handle = classifier.register_forward_hook(capture_input)
    try:
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            model(x.to(device))
    finally:
        handle.remove()

    features = captured.get("features")
    if features is None:
        raise RuntimeError("could not capture classifier input features")
    if features.ndim != 2:
        raise ValueError(
            "classifier input must have shape [batch, features] for "
            "weight-based reverse engineering"
        )
    return features


def _classifier_scores(classifier, features: Tensor) -> Tensor:
    """Compute scores with the same active-unit semantics as Avalanche."""
    scores = F.linear(
        features,
        classifier.weight.detach(),
        classifier.bias.detach() if classifier.bias is not None else None,
    )
    active_units = getattr(classifier, "active_units", None)
    if active_units is not None:
        active_units = active_units.to(device=scores.device)
        if active_units.numel() != scores.shape[-1]:
            raise ValueError(
                "active_units must have one entry per classifier output unit"
            )
        active_mask = active_units.to(torch.bool)
        if not bool(active_mask.any()):
            raise ValueError("classifier has no active output units")
        mask_value = getattr(classifier, "mask_value", -1000.0)
        scores = scores.masked_fill(~active_mask.unsqueeze(0), mask_value)
    return scores


def reverse_engineer_scores_from_weights(model, x: Tensor) -> Tensor:
    """Reconstruct classifier scores directly from learned head weights."""
    classifier = _find_classifier(model).classifier
    features = extract_features_from_weights(model, x)
    return _classifier_scores(classifier, features)


def reverse_engineer_y_from_weights(model, x: Tensor, target_class: int) -> Tensor:
    """Produce one-vs-rest binary ``y`` from the candidate class score."""
    scores = reverse_engineer_scores_from_weights(model, x)
    return reverse_engineer_y(scores, target_class)


def reverse_engineer_y(logits: Tensor, target_class: int) -> Tensor:
    """Return binary ``y`` from a candidate class score threshold.

    ``y`` is intentionally not defined as ``argmax(logits) == target_class``.
    Argmax is a multiclass decision and therefore makes exactly one candidate
    compatible for every sample, including the class produced by a
    misclassification. A class fingerprint instead tests the candidate's own
    reconstructed score against the zero decision boundary, allowing zero,
    one, or multiple candidates to be compatible.
    """
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    if not 0 <= int(target_class) < logits.shape[-1]:
        raise ValueError("target_class is outside the classifier output")
    return logits[:, int(target_class)].gt(0)


def compare_binary_behavior(
    predicted_y: Tensor,
    expected_y: bool,
) -> dict[str, Any]:
    """Compare predicted binary ``y`` values with an expected value."""
    predicted_y = predicted_y.detach().cpu().bool()
    expected = torch.full_like(predicted_y, expected_y, dtype=torch.bool)
    correct = predicted_y.eq(expected)
    return {
        "predicted_y": predicted_y,
        "expected_y": bool(expected_y),
        "correct": correct,
        "accuracy": float(correct.float().mean().item()) if correct.numel() else 0.0,
        "all_correct": bool(correct.all().item()) if correct.numel() else False,
    }


def identify_binary_behavior(
    predicted_y: Tensor,
    expected_y: bool = True,
) -> bool:
    """Return whether every anonymous probe agrees with expected ``y``."""
    return bool(compare_binary_behavior(predicted_y, expected_y)["all_correct"])


def build_weight_behavior_statistics(model, x: Tensor, class_id: int) -> dict[str, Any]:
    """Build continuous, weight-derived statistics for a persistent class."""
    classifier = _find_classifier(model).classifier
    features = extract_features_from_weights(model, x)
    scores = _classifier_scores(classifier, features)
    class_id = int(class_id)
    if not 0 <= class_id < scores.shape[-1]:
        raise ValueError("class_id is outside the classifier output")
    own = scores[:, class_id]
    if scores.shape[-1] > 1:
        other = scores.clone()
        other[:, class_id] = -torch.inf
        margin = own - other.max(dim=-1).values
    else:
        margin = own
    return {
        "feature_mean": features.mean(dim=0).detach().cpu(),
        "feature_std": features.std(unbiased=False)
        .clamp_min(1e-6)
        .detach()
        .cpu(),
        "margin_mean": float(margin.mean().item()),
        "margin_std": max(float(margin.std(unbiased=False).item()), 1e-6),
        "weight": classifier.weight[class_id].detach().cpu().clone(),
        "bias": (
            float(classifier.bias[class_id].item())
            if classifier.bias is not None
            else 0.0
        ),
    }
