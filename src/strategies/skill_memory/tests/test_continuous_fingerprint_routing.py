"""Tests for continuous evidence used after binary fingerprint matching."""

import torch

from skill_memory import ClassBehaviorRecord
from skill_memory.persistent_skill_memory_plugin import (
    PersistentFingerprintSkillMemoryPlugin,
)


def _record(class_id: int) -> ClassBehaviorRecord:
    return ClassBehaviorRecord(
        class_id=class_id,
        skill_id=class_id,
        version=0,
        reference_inputs=torch.zeros(1, 2),
        reference_y=torch.ones(1, dtype=torch.bool),
        reference_feature_mean=torch.tensor([1.0, 0.0]),
        reference_feature_std=torch.ones(2),
        reference_margin_mean=1.0,
        reference_margin_std=0.5,
    )


def test_continuous_candidate_selects_clear_top_cluster():
    candidates = [
        {"class": 1, "skill": 1, "continuous_evidence": 0.90},
        {"class": 2, "skill": 2, "continuous_evidence": 0.40},
        {"class": 3, "skill": 3, "continuous_evidence": 0.39},
    ]

    selected = PersistentFingerprintSkillMemoryPlugin._select_continuous_candidate(
        candidates
    )

    assert selected is candidates[0]


def test_continuous_candidate_keeps_non_top_cluster_ambiguous():
    candidates = [
        {"class": 1, "skill": 1, "continuous_evidence": 0.90},
        {"class": 2, "skill": 2, "continuous_evidence": 0.89},
        {"class": 3, "skill": 3, "continuous_evidence": 0.20},
    ]

    selected = PersistentFingerprintSkillMemoryPlugin._select_continuous_candidate(
        candidates
    )

    assert selected is None


def test_continuous_evidence_uses_persistent_reference_statistics():
    record = _record(1)
    features = torch.tensor([[1.0, 0.0]])
    scores = torch.tensor([[0.0, 2.0, 1.0]])

    evidence = PersistentFingerprintSkillMemoryPlugin._continuous_evidence(
        record,
        features,
        scores,
        0,
    )

    assert evidence["feature_similarity"] > 0.99
    assert evidence["margin_similarity"] > 0.99
    assert evidence["evidence"] > 0.99
