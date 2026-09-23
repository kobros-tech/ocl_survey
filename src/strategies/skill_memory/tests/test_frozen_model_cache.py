"""Tests for the frozen-model cache in PersistentFingerprintSkillMemoryPlugin.

Review comment (pasted from an external thread, referencing
kobros-tech/cl_strategy PR #7) pointed out that `_fit_reverse_router` used
to deepcopy(strategy.model) once per (record, slot) pair - `records x
slots` deepcopies, e.g. ~10,000 for a 100-class / 100-skill setup - and that
`_route` re-deepcopied the same candidate's model on every evaluation
batch. These tests confirm the cache actually collapses that to one
deepcopy per skill_id, reused for the rest of the fit and every subsequent
`_route` call, with identical routing results.
"""

from types import SimpleNamespace

import torch
from torch import nn

import skill_memory.cl.persistent_skill_memory_plugin as psmp
from skill_memory import ClassBehaviorRecord, SkillMemory
from skill_memory.evaluation.fingerprint_routing import (
    PersistentFingerprintSkillMemoryPlugin,
)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(2, 6)

    def forward(self, x):
        return self.classifier(x)


def _record(class_id: int, skill_id: int, weight: float) -> ClassBehaviorRecord:
    return ClassBehaviorRecord(
        class_id=class_id,
        skill_id=skill_id,
        version=0,
        reference_inputs=torch.ones(2, 2),
        reference_y=torch.ones(2, dtype=torch.bool),
        reference_weight=torch.tensor([weight]),
        reference_bias=0.0,
    )


def _plugin_with_skills(n_slots: int, classes_per_slot: int):
    plugin = PersistentFingerprintSkillMemoryPlugin(verbose=False, reverse_epochs=1)
    plugin.memory = SkillMemory(max_skills=n_slots)
    plugin.behavior = plugin.behavior.__class__()
    class_id = 0
    for slot in range(n_slots):
        plugin.memory.store(
            slot,
            {
                "classifier.weight": torch.randn(6, 2),
                "classifier.bias": torch.randn(6),
            },
        )
        for _ in range(classes_per_slot):
            plugin.behavior.put(_record(class_id, slot, float(class_id + 1)))
            class_id += 1
    return plugin


def test_fit_reverse_router_deepcopies_once_per_skill_not_per_record_x_slot(
    monkeypatch,
):
    n_slots, classes_per_slot = 4, 3  # 12 records x 4 slots = 48 pairs, 4 skills
    plugin = _plugin_with_skills(n_slots, classes_per_slot)

    deepcopy_calls: list[int] = []
    real_deepcopy = psmp.deepcopy

    def counting_deepcopy(obj):
        deepcopy_calls.append(1)
        return real_deepcopy(obj)

    monkeypatch.setattr(psmp, "deepcopy", counting_deepcopy)

    plugin._fit_reverse_router(SimpleNamespace(model=_TinyModel()))

    # Without the cache this would be 12 records x 4 slots = 48 deepcopies.
    assert len(deepcopy_calls) == n_slots


def test_route_reuses_the_cache_populated_by_the_preceding_fit(monkeypatch):
    """Once `_fit_reverse_router` has populated the cache, a subsequent
    `_route` call (as happens on every evaluation batch) must not
    deepcopy again."""
    n_slots, classes_per_slot = 3, 2
    plugin = _plugin_with_skills(n_slots, classes_per_slot)
    strategy = SimpleNamespace(model=_TinyModel())
    plugin._fit_reverse_router(strategy)

    deepcopy_calls: list[int] = []
    real_deepcopy = psmp.deepcopy

    def counting_deepcopy(obj):
        deepcopy_calls.append(1)
        return real_deepcopy(obj)

    monkeypatch.setattr(psmp, "deepcopy", counting_deepcopy)

    x = torch.ones(3, 2)
    slot_ids = sorted(plugin.memory.slots())
    for _ in range(5):  # simulate several evaluation batches in one epoch
        plugin._route(strategy, x, slot_ids)

    assert deepcopy_calls == []


def test_refit_after_a_skill_mutation_does_not_serve_a_stale_cached_model():
    """The cache is cleared at the start of every `_fit_reverse_router` call,
    so if a skill's stored weights changed (e.g. a mutable REUSE) between
    two fits, the second fit must reflect the *new* weights, not a cached
    copy of the old ones."""
    plugin = _plugin_with_skills(n_slots=1, classes_per_slot=1)
    strategy = SimpleNamespace(model=_TinyModel())

    # First fit/route with the original weights.
    plugin._fit_reverse_router(strategy)
    original_logits = plugin._frozen_logits(strategy, 0, torch.ones(1, 2))

    # Simulate a mutable REUSE: the skill's stored weights change, and its
    # behavior version is bumped (mirroring behavior.bump_skill's contract).
    plugin.memory.store(
        0,
        {
            "classifier.weight": torch.full((6, 2), 100.0),
            "classifier.bias": torch.full((6,), 100.0),
        },
    )
    plugin.behavior.bump_skill(0)
    plugin.behavior.put(_record(0, 0, 1.0))

    plugin._fit_reverse_router(strategy)
    refreshed_logits = plugin._frozen_logits(strategy, 0, torch.ones(1, 2))

    assert not torch.allclose(original_logits, refreshed_logits)


def test_reference_logits_are_reused_across_router_refits(monkeypatch):
    """Old record/skill pairs should not run frozen inference again."""
    plugin = _plugin_with_skills(n_slots=2, classes_per_slot=2)
    strategy = SimpleNamespace(model=_TinyModel())

    calls: list[tuple[int, int]] = []
    real_frozen_logits = plugin._frozen_logits

    def counting_frozen_logits(strategy_arg, skill_id, x):
        calls.append((skill_id, x.shape[0]))
        return real_frozen_logits(strategy_arg, skill_id, x)

    monkeypatch.setattr(plugin, "_frozen_logits", counting_frozen_logits)

    plugin._fit_reverse_router(strategy)
    first_fit_calls = len(calls)

    # The first fit evaluates every reference record against every candidate
    # skill. There are 4 records total (2 per skill) and 2 candidate skills,
    # so this is 4 x 2 = 8 frozen-inference calls. Each call receives the
    # record's two reference samples as one batch.
    assert first_fit_calls == 8
    assert all(batch_size == 2 for _, batch_size in calls)

    calls.clear()
    plugin._fit_reverse_router(strategy)

    # The second fit has the same skill generations and reference inputs, so
    # every frozen reference response comes from the persistent cache.
    assert calls == []
