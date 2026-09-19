"""Regression tests for the current listwise reverse router."""

from types import SimpleNamespace

import torch

from skill_memory import ClassBehaviorRecord
from skill_memory.fingerprint_routing import PersistentFingerprintSkillMemoryPlugin


def _record(class_id: int, skill_id: int) -> ClassBehaviorRecord:
    return ClassBehaviorRecord(
        class_id=class_id,
        skill_id=skill_id,
        version=0,
        reference_inputs=torch.zeros(1, 1),
        reference_y=torch.ones(1, dtype=torch.bool),
        reference_weight=torch.tensor([1.0]),
        reference_bias=0.0,
    )


def test_router_scores_complete_candidate_set_without_argmax_shortcut(monkeypatch):
    # diagnose=True: this test asserts on the per-candidate breakdown, which
    # is only populated when diagnostics are requested (see
    # fingerprint_routing.PersistentFingerprintSkillMemoryPlugin).
    plugin = PersistentFingerprintSkillMemoryPlugin(
        verbose=False, reverse_epochs=1, diagnose=True
    )
    plugin.behavior.put(_record(0, 0))
    plugin.behavior.put(_record(1, 1))
    plugin._reverse_output_dim = 2
    plugin._reverse_candidate_dim = 1

    def frozen_logits(_strategy, skill_id, x):
        logits = torch.full((x.shape[0], 2), -2.0)
        target = skill_id
        logits[:, target] = 3.0
        return logits

    class FakeReverse:
        model = object()
        training_mode = "listwise"

        @staticmethod
        def predict_scores_candidate_sets(features):
            return features[:, :, 5]

    plugin.reverse_engineer = FakeReverse()
    monkeypatch.setattr(plugin, "_frozen_logits", frozen_logits)

    chosen, classes = plugin._route(
        SimpleNamespace(model=object()),
        torch.zeros(1, 1),
        [0, 1],
    )

    assert chosen.tolist() == [0]
    assert classes == [0]
    assert len(plugin.last_fingerprint_routes[0]["candidates"]) == 2


def test_router_skips_candidate_breakdown_when_not_diagnosing(monkeypatch):
    """diagnose=False (the default) must still route correctly, but must not
    pay for building the per-candidate diagnostic breakdown that only
    routing_rank_diagnostics consumes."""
    plugin = PersistentFingerprintSkillMemoryPlugin(verbose=False, reverse_epochs=1)
    assert plugin.diagnose is False
    plugin.behavior.put(_record(0, 0))
    plugin.behavior.put(_record(1, 1))
    plugin._reverse_output_dim = 2
    plugin._reverse_candidate_dim = 1

    def frozen_logits(_strategy, skill_id, x):
        logits = torch.full((x.shape[0], 2), -2.0)
        target = skill_id
        logits[:, target] = 3.0
        return logits

    class FakeReverse:
        model = object()
        training_mode = "listwise"

        @staticmethod
        def predict_scores_candidate_sets(features):
            return features[:, :, 5]

    plugin.reverse_engineer = FakeReverse()
    monkeypatch.setattr(plugin, "_frozen_logits", frozen_logits)

    chosen, classes = plugin._route(
        SimpleNamespace(model=object()),
        torch.zeros(1, 1),
        [0, 1],
    )

    assert chosen.tolist() == [0]
    assert classes == [0]
    assert "candidates" not in plugin.last_fingerprint_routes[0]
