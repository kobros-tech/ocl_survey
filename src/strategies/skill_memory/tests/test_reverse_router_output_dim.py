"""Regression test for a global-class-id vs observed-logit-width mismatch.

Avalanche's IncrementalClassifier indexes its output units by the raw class
label, not a compacted per-skill position. `_fit_reverse_router` used to set
`_reverse_output_dim` from the *observed* width of frozen responses alone.
If every currently-observed response happened to be narrower than a
candidate's own (larger) global class_id - e.g. a stale/otherwise-narrow
skill capping the max seen so far - `_pad_logits` would silently truncate
away exactly the column holding that candidate's own logit, and every
candidate whose class_id fell outside the observed width would be routed
using an all-zero feature instead of a real signal. This looks exactly like
a routing failure (e.g. near chance-level accuracy) rather than a bug.
"""

from types import SimpleNamespace

import torch

from skill_memory import ClassBehaviorRecord, SkillMemory
from skill_memory.evaluation.fingerprint_routing import (
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


def test_fit_reverse_router_widens_output_dim_to_cover_every_candidate_class_id(
    monkeypatch,
):
    plugin = _plugin()
    # Two skills. Skill 0 only ever reports a 5-wide response (as an
    # IncrementalClassifier would for a skill whose own classes never
    # required more units). Skill 1's candidate class_id is 37, far outside
    # that observed width.
    plugin.memory.store(0, {"slot": torch.tensor([0.0])})
    plugin.memory.store(1, {"slot": torch.tensor([1.0])})
    plugin.behavior.put(_record(2, 0, 1.0))
    plugin.behavior.put(_record(37, 1, 1.0))

    def frozen_logits(_strategy, skill_id, x):
        # Every currently observed frozen response is only 5-wide,
        # regardless of which skill/class it belongs to.
        logits = torch.zeros(x.shape[0], 5)
        logits[:, skill_id] = 3.0
        return logits

    monkeypatch.setattr(plugin, "_frozen_logits", frozen_logits)

    plugin._fit_reverse_router(SimpleNamespace(model=object()))

    # output_dim must be widened to at least cover class_id 37, not just the
    # 5-wide width actually observed on any single frozen response.
    assert plugin._reverse_output_dim is not None
    assert plugin._reverse_output_dim >= 38


def test_fit_reverse_router_does_not_recompute_the_same_frozen_response(monkeypatch):
    """`_frozen_logits` deepcopies the live model; fitting must call it at
    most once per (class_id, slot) pair rather than recomputing the same
    response again while building candidate_logits."""
    plugin = _plugin()
    plugin.memory.store(0, {"slot": torch.tensor([0.0])})
    plugin.memory.store(1, {"slot": torch.tensor([1.0])})
    plugin.behavior.put(_record(0, 0, 1.0))
    plugin.behavior.put(_record(1, 1, 1.0))

    calls: list[tuple[int, int]] = []

    def frozen_logits(_strategy, skill_id, x):
        calls.append((skill_id, x.shape[0]))
        logits = torch.zeros(x.shape[0], 2)
        logits[:, skill_id] = 3.0
        return logits

    monkeypatch.setattr(plugin, "_frozen_logits", frozen_logits)

    plugin._fit_reverse_router(SimpleNamespace(model=object()))

    # 2 records x 2 slots = 4 distinct (class_id, slot) pairs; each should be
    # computed exactly once, not recomputed again for candidate_logits.
    assert len(calls) == 4


def test_make_features_raises_instead_of_silently_zeroing_out_of_range_candidate():
    # Once output_dim is (mis)computed too small for a candidate's own
    # class_id, extracting that candidate's feature must fail loudly rather
    # than silently return a zeroed-out (and therefore misleading) feature.
    x = torch.ones(1, 1)
    logits = torch.zeros(1, 5)
    weight = torch.tensor([1.0])

    try:
        PersistentFingerprintSkillMemoryPlugin._make_features(
            x,
            logits,
            weight,
            0.0,
            5,  # output_dim, too small for class_id 37
            37,  # candidate_class_id
        )
    except RuntimeError as exc:
        assert "outside the routed output space" in str(exc)
    else:
        raise AssertionError("expected a RuntimeError for the out-of-range class_id")


def test_make_features_still_zeros_when_class_id_is_genuinely_unknown():
    # candidate_class_id=None (no mapping available) remains a legitimate,
    # deliberate zero-feature fallback - only an out-of-range *known* class
    # id should raise.
    x = torch.ones(1, 1)
    logits = torch.zeros(1, 5)
    weight = torch.tensor([1.0])

    features = PersistentFingerprintSkillMemoryPlugin._make_features(
        x, logits, weight, 0.0, 5, None
    )
    assert features.shape[0] == 1
