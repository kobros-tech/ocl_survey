import importlib.util
import sys
from pathlib import Path

import torch

root = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("behavior", root / "behavior.py")
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def _record(class_id=1, skill_id=2, version=0):
    return mod.ClassBehaviorRecord(
        class_id=class_id,
        skill_id=skill_id,
        version=version,
        reference_inputs=torch.ones(2, 3),
        output_class_ids=(0, 1),
        reference_output=torch.tensor([0.8, 0.6]),
        reference_summary=torch.ones(4),
    )


def test_fingerprint_is_reused_until_skill_changes():
    cache = mod.BehaviorFingerprintCache()
    cache.put(_record())
    assert cache.get(1, 2) is not None
    assert cache.get(1, 2) is cache.get(1, 2)
    cache.bump_skill(2)
    assert cache.get(1, 2) is None


def test_bump_invalidates_every_class_of_skill():
    cache = mod.BehaviorFingerprintCache()
    cache.put(_record(class_id=1))
    cache.put(_record(class_id=3))
    cache.put(_record(class_id=4, skill_id=7))
    cache.bump_skill(2)
    assert cache.get(1, 2) is None
    assert cache.get(3, 2) is None
    assert cache.get(4, 7) is not None


def test_state_roundtrip_preserves_valid_behavior():
    cache = mod.BehaviorFingerprintCache()
    cache.put(_record())
    restored = mod.BehaviorFingerprintCache()
    restored.load_state_dict(cache.state_dict())
    record = restored.get(1, 2)
    assert record is not None
    assert torch.equal(record.reference_inputs, torch.ones(2, 3))
    assert record.output_class_ids == (0, 1)


def test_probe_behavior_matches_class_aligned_reference():
    record = _record()
    logits = torch.tensor([[8.0, 6.0], [6.0, 8.0]])
    similarity, summary = mod.probe_behavior_fingerprint(
        logits,
        record.output_class_ids,
        record.reference_output,
        record.reference_summary,
    )
    assert similarity.shape == (2,)
    assert similarity[0] > similarity[1]
    assert summary.shape == (4,)


def test_generated_fingerprints_rank_their_source_classes():
    class_zero_reference = torch.tensor(
        [[10.0, 0.0, -2.0], [9.0, 1.0, -2.0]]
    )
    class_one_reference = torch.tensor(
        [[0.0, 10.0, -2.0], [1.0, 9.0, -2.0]]
    )

    zero_output, zero_summary, class_ids = mod.extract_reference_behavior(
        class_zero_reference,
        [0, 1, 2],
        target_class=0,
    )
    one_output, one_summary, _ = mod.extract_reference_behavior(
        class_one_reference,
        [0, 1, 2],
        target_class=1,
    )

    zero_on_zero, _ = mod.probe_behavior_fingerprint(
        class_zero_reference[:1], class_ids, zero_output, zero_summary
    )
    zero_on_one, _ = mod.probe_behavior_fingerprint(
        class_one_reference[:1], class_ids, zero_output, zero_summary
    )
    one_on_one, _ = mod.probe_behavior_fingerprint(
        class_one_reference[:1], class_ids, one_output, one_summary
    )
    one_on_zero, _ = mod.probe_behavior_fingerprint(
        class_zero_reference[:1], class_ids, one_output, one_summary
    )

    assert zero_on_zero.item() > zero_on_one.item()
    assert one_on_one.item() > one_on_zero.item()


def test_generated_class_fingerprints_are_not_uniform_for_distinct_classes():
    class_zero_reference = torch.tensor([[12.0, 0.0, -3.0]])
    class_one_reference = torch.tensor([[0.0, 12.0, -3.0]])
    zero_output, zero_summary, class_ids = mod.extract_reference_behavior(
        class_zero_reference,
        [0, 1, 2],
        target_class=0,
    )
    one_output, one_summary, _ = mod.extract_reference_behavior(
        class_one_reference,
        [0, 1, 2],
        target_class=1,
    )

    probe_zero = torch.tensor([[12.0, 0.0, -3.0]])
    probe_one = torch.tensor([[0.0, 12.0, -3.0]])
    zero_match, _ = mod.probe_behavior_fingerprint(
        probe_zero, class_ids, zero_output, zero_summary
    )
    zero_other, _ = mod.probe_behavior_fingerprint(
        probe_one, class_ids, zero_output, zero_summary
    )
    one_match, _ = mod.probe_behavior_fingerprint(
        probe_one, class_ids, one_output, one_summary
    )
    one_other, _ = mod.probe_behavior_fingerprint(
        probe_zero, class_ids, one_output, one_summary
    )

    assert zero_match.item() - zero_other.item() > 0.1
    assert one_match.item() - one_other.item() > 0.1


def test_fingerprint_uses_global_class_ids_not_local_columns():
    reference_logits = torch.zeros(1, 100)
    reference_logits[0, 42] = 100.0
    reference_output, reference_summary, output_class_ids = (
        mod.extract_reference_behavior(
            reference_logits,
            [42, 87],
            target_class=42,
        )
    )
    record = mod.ClassBehaviorRecord(
        class_id=42,
        skill_id=2,
        version=0,
        reference_inputs=torch.ones(1, 2),
        output_class_ids=output_class_ids,
        reference_output=reference_output,
        reference_summary=reference_summary,
    )
    logits = torch.zeros(1, 100)
    logits[0, 42] = 100.0
    similarity, _ = mod.probe_behavior_fingerprint(
        logits,
        record.output_class_ids,
        record.reference_output,
        record.reference_summary,
    )
    assert similarity.item() > 0.99
