"""Regression tests for binary behavior inside persistent routing."""

from types import SimpleNamespace

import torch
from torch import nn

import skill_memory.fingerprint_routing as fingerprint_routing
from skill_memory import ClassBehaviorRecord, SkillMemory
from skill_memory.fingerprint_routing import PersistentFingerprintSkillMemoryPlugin


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = nn.Parameter(torch.tensor(0.0))


def test_router_uses_candidate_score_not_argmax(monkeypatch):
    """A positive candidate score remains compatible below another class."""
    plugin = PersistentFingerprintSkillMemoryPlugin(
        memory=SkillMemory(max_skills=1),
        verbose=False,
    )
    state = {"parameter": torch.tensor(0.0)}
    plugin.memory.store(0, state)
    plugin.behavior.put_skill_state(0, 0, state)
    plugin.behavior.put(
        ClassBehaviorRecord(
            class_id=0,
            skill_id=0,
            version=0,
            reference_inputs=torch.zeros(1, 1),
            reference_y=torch.ones(1, dtype=torch.bool),
            reference_feature_mean=torch.tensor([1.0]),
            reference_feature_std=torch.ones(1),
            reference_margin_mean=1.0,
            reference_margin_std=1.0,
        )
    )

    monkeypatch.setattr(
        fingerprint_routing,
        "apply_skill_state_exact",
        lambda model, state_dict: None,
    )
    monkeypatch.setattr(
        fingerprint_routing,
        "extract_features_from_weights",
        lambda model, x: torch.tensor([[1.0]]),
    )
    monkeypatch.setattr(
        fingerprint_routing,
        "reverse_engineer_scores_from_weights",
        lambda model, x: torch.tensor([[0.5, 2.0]]),
    )

    strategy = SimpleNamespace(model=DummyModel())
    plugin._behavior_initialized = True
    chosen, classes = plugin._fingerprint_route(
        strategy,
        torch.zeros(1, 1),
        [0],
    )

    route = plugin.last_fingerprint_routes[0]
    candidate = route["candidates"][0]
    assert route["status"] == "IDENTIFIED"
    assert candidate["predicted_class"] == 1
    assert candidate["predicted_y"] is True
    assert candidate["binary_compatible"] is True
    assert chosen.tolist() == [0]
    assert classes == [0]
