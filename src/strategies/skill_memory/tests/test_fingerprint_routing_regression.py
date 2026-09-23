"""Regression tests for anonymous listwise routing."""

from types import SimpleNamespace

import torch

from skill_memory import ClassBehaviorRecord
from skill_memory.evaluation.fingerprint_routing import (
    PersistentFingerprintSkillMemoryPlugin,
)


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


def test_listwise_router_uses_candidate_behavior(monkeypatch):
    plugin = PersistentFingerprintSkillMemoryPlugin(verbose=False, reverse_epochs=1)
    plugin.behavior.put(_record(0, 0))
    plugin.behavior.put(_record(1, 1))
    plugin._reverse_output_dim = 2
    plugin._reverse_candidate_dim = 1

    def frozen_logits(_strategy, skill_id, x):
        del x
        logits = torch.full((1, 2), -1.0)
        logits[:, skill_id] = 4.0
        return logits

    class FakeReverse:
        model = object()
        training_mode = "listwise"

        @staticmethod
        def predict_scores_candidate_sets(features):
            return features[:, :, 6]

    plugin.reverse_engineer = FakeReverse()
    monkeypatch.setattr(plugin, "_frozen_logits", frozen_logits)

    chosen, classes = plugin._route(
        SimpleNamespace(model=object()),
        torch.zeros(1, 1),
        [0, 1],
    )

    assert chosen.tolist() == [0]
    assert classes == [0]
    assert plugin.last_fingerprint_routes[0]["status"] == "IDENTIFIED"
