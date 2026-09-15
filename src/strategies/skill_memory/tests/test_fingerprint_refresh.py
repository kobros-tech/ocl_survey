import importlib.util
from pathlib import Path

import torch

root = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("behavior", root / "behavior.py")
behavior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(behavior)


def test_invalidated_record_retains_reference_inputs_for_refresh():
    cache = behavior.BehaviorFingerprintCache()
    reference_inputs = torch.tensor([[1.0, 2.0]])
    cache.put(
        behavior.ClassBehaviorRecord(
            class_id=7,
            skill_id=3,
            version=0,
            reference_inputs=reference_inputs,
            output_class_ids=(7, 8),
            reference_output=torch.tensor([1.0, 0.0]),
            reference_summary=torch.ones(4),
        )
    )

    cache.bump_skill(3)

    assert cache.get(7, 3) is None
    records = cache.all_records_for_skill(3)
    assert len(records) == 1
    assert not records[0].valid
    assert torch.equal(records[0].reference_inputs, reference_inputs)
