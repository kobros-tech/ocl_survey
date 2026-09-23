"""Tests for persistent anonymous listwise routing."""

from types import SimpleNamespace

import torch

from skill_memory import ClassBehaviorRecord, SkillMemory
from skill_memory.cl.persistent_skill_memory_plugin import (
    PersistentFingerprintSkillMemoryPlugin,
)


def _record(class_id: int, skill_id: int, weight: float) -> ClassBehaviorRecord:
    return ClassBehaviorRecord(
        class_id=class_id,
        skill_id=skill_id,
        version=0,
        reference_inputs=torch.ones(2, 1),
        reference_y=torch.ones(2, dtype=torch.bool),
        reference_weight=torch.tensor([weight]),
        reference_bias=0.0,
    )


def _plugin() -> PersistentFingerprintSkillMemoryPlugin:
    plugin = PersistentFingerprintSkillMemoryPlugin(verbose=False, reverse_epochs=1)
    plugin.memory = SkillMemory(max_skills=10)
    plugin.behavior = plugin.behavior.__class__()
    return plugin


def test_listwise_route_identifies_complete_candidate_set(monkeypatch):
    plugin = _plugin()
    plugin.memory.store(0, {"slot": torch.tensor([0.0])})
    plugin.memory.store(1, {"slot": torch.tensor([1.0])})
    plugin.behavior.put(_record(0, 0, 1.0))
    plugin.behavior.put(_record(1, 1, 1.0))
    plugin._reverse_output_dim = 2
    plugin._reverse_candidate_dim = 1

    def frozen_logits(_strategy, skill_id, x):
        logits = torch.full((x.shape[0], 2), -1.0)
        if skill_id == 0:
            logits[:, 0] = torch.where(x[:, 0] == 10, 5.0, -2.0)
        else:
            logits[:, 1] = torch.where(x[:, 0] == 20, 5.0, -2.0)
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
        torch.tensor([[10.0], [20.0]]),
        [0, 1],
    )

    assert chosen.tolist() == [0, 1]
    assert classes == [0, 1]
    assert [item["status"] for item in plugin.last_fingerprint_routes] == [
        "IDENTIFIED",
        "IDENTIFIED",
    ]
    assert all(len(item["candidates"]) == 2 for item in plugin.last_fingerprint_routes)


def test_route_keeps_skill_id_distinct_from_slot_position(monkeypatch):
    plugin = _plugin()
    plugin.memory.store(2, {"slot": torch.tensor([2.0])})
    plugin.memory.store(7, {"slot": torch.tensor([7.0])})
    plugin.behavior.put(_record(0, 2, 1.0))
    plugin.behavior.put(_record(1, 7, 1.0))
    plugin._reverse_output_dim = 2
    plugin._reverse_candidate_dim = 1

    def frozen_logits(_strategy, skill_id, x):
        if skill_id == 2:
            first = torch.where(x[:, 0] == 1, 6.0, -1.0)
            second = torch.where(x[:, 0] == 1, -1.0, 6.0)
            return torch.stack((first, second), dim=1)
        first = torch.where(x[:, 0] == 2, -1.0, 6.0)
        second = torch.where(x[:, 0] == 2, 6.0, -1.0)
        return torch.stack((first, second), dim=1)

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
        torch.tensor([[1.0], [2.0]]),
        [2, 7],
    )

    assert chosen.tolist() == [2, 7]
    assert classes == [0, 1]


def test_immutable_reuse_does_not_mark_skill_changed():
    plugin = _plugin()
    plugin.last_class_decisions = {
        0: {
            10: {"decision": plugin.REUSE, "skill": 0},
            20: {"decision": plugin.SCRATCH, "skill": 1},
        }
    }
    assert plugin._collect_changed_skills(0) == {1}


def test_skill_generation_state_is_frozen_and_serializable():
    plugin = _plugin()
    state = {"weight": torch.tensor([1.0, 2.0])}
    plugin.behavior.put_skill_state(3, 0, state)
    state["weight"][0] = 99.0

    frozen = plugin.behavior.skill_state(3, 0)
    assert frozen is not None
    assert torch.equal(frozen["weight"], torch.tensor([1.0, 2.0]))

    restored = plugin.behavior.__class__()
    restored.load_state_dict(plugin.behavior.state_dict())
    restored_state = restored.skill_state(3, 0)
    assert restored_state is not None
    assert torch.equal(restored_state["weight"], torch.tensor([1.0, 2.0]))
