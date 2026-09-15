import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parents[1]


def _load_behavior():
    spec = importlib.util.spec_from_file_location(
        "checkpoint_behavior", ROOT / "behavior.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_behavior_checkpoint_roundtrip_preserves_fingerprints():
    mod = _load_behavior()
    first = mod.BehaviorFingerprintCache()
    first.put(
        mod.ClassBehaviorRecord(
            class_id=42,
            skill_id=3,
            version=2,
            reference_inputs=torch.tensor([[1.0, 2.0]]),
            output_class_ids=(42, 87),
            reference_output=torch.tensor([0.8, 0.2]),
            reference_summary=torch.tensor([0.5, 0.5, 0.5, 0.5]),
        )
    )

    checkpoint = first.state_dict()
    second = mod.BehaviorFingerprintCache()
    second.load_state_dict(checkpoint)

    restored = second.get(42, 3)
    assert restored is not None
    assert restored.version == 2
    assert restored.output_class_ids == (42, 87)
    assert torch.equal(restored.reference_inputs, first.get(42, 3).reference_inputs)
    assert torch.equal(restored.reference_output, first.get(42, 3).reference_output)
    assert torch.equal(restored.reference_summary, first.get(42, 3).reference_summary)
    assert second.skill_version(3) == 2
