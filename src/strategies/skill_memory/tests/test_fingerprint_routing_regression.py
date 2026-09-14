"""Regression tests for evidence-first anonymous fingerprint routing."""

from types import SimpleNamespace

import torch

from skill_memory import ClassBehaviorRecord
from skill_memory.fingerprint_routing import PersistentFingerprintSkillMemoryPlugin


class _Behavior:
    def __init__(self, records):
        self.records = records

    def records_for_skill(self, skill_id):
        return [record for record in self.records if record.skill_id == skill_id]

    def skill_state(self, skill_id, version):
        del skill_id, version
        return {}


class _Memory:
    def state(self, skill_id):
        del skill_id
        return {}


def _record(class_id, skill_id, reference_mean, margin_mean):
    return ClassBehaviorRecord(
        class_id=class_id,
        skill_id=skill_id,
        version=0,
        reference_inputs=torch.zeros(1, 1),
        reference_y=torch.ones(1, dtype=torch.bool),
        reference_feature_mean=torch.tensor(reference_mean),
        reference_margin_mean=margin_mean,
        reference_margin_std=0.5,
    )


def test_wrong_argmax_does_not_block_stronger_continuous_class_evidence(monkeypatch):
    """A one-class skill must not win merely because its argmax is positive."""
    class_four = _record(4, 0, [1.0, 0.0], -1.5)
    class_one = _record(1, 1, [0.0, 1.0], 0.0)

    plugin = PersistentFingerprintSkillMemoryPlugin.__new__(
        PersistentFingerprintSkillMemoryPlugin
    )
    plugin.behavior = _Behavior([class_four, class_one])
    plugin.memory = _Memory()
    plugin._custom_reverse_engineer_y = None
    plugin.last_fingerprint_routes = []

    monkeypatch.setattr(
        "skill_memory.fingerprint_routing.extract_features_from_weights",
        lambda model, x: torch.tensor([[1.0, 0.0]]),
    )
    monkeypatch.setattr(
        "skill_memory.fingerprint_routing.reverse_engineer_scores_from_weights",
        lambda model, x: torch.tensor([[0.5, 2.0, -1.0, -1.0, -0.5]]),
    )
    monkeypatch.setattr(
        "skill_memory.fingerprint_routing.apply_skill_state_exact",
        lambda model, state: None,
    )

    strategy = SimpleNamespace(model=object())
    chosen, classes = plugin._fingerprint_route(strategy, torch.zeros(1, 1), [0, 1])

    assert classes == [4]
    assert chosen.tolist() == [0]
    route = plugin.last_fingerprint_routes[0]
    assert route["status"] == "IDENTIFIED"
    assert route["class"] == 4
    assert route["binary_compatible_candidates"] == 1
    assert route["candidates"][0]["class"] == 4
    assert route["candidates"][0]["binary_compatible"] is False
    assert route["candidates"][1]["class"] == 1
    assert route["candidates"][1]["binary_compatible"] is True
