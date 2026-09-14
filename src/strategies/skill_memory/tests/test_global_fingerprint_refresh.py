import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "global_fingerprint_refresh",
    ROOT / "global_fingerprint_refresh.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _record(class_id, skill_id, value):
    return {
        "class_id": class_id,
        "skill_id": skill_id,
        "version": 0,
        "reference_inputs": torch.tensor([[value], [value + 1]]),
    }


def test_refresh_rebuilds_all_classes_from_current_skill_states():
    records = [_record(10, 0, 10.0), _record(20, 1, 20.0)]
    calls = []

    class Behavior:
        def state_dict(self):
            return {"records": records}

        def skill_version(self, skill_id):
            return 3 if skill_id == 0 else 7

        def put_skill_state(self, skill_id, version, state):
            calls.append(("state", skill_id, version, state["value"].item()))

        def put(self, record):
            calls.append(
                (
                    "record",
                    record.class_id,
                    record.skill_id,
                    record.version,
                    record.reference_inputs.clone(),
                )
            )

    class ClassMap:
        def find_skill_for_class_anywhere(self, class_id):
            return {10: 0, 20: 1}.get(class_id)

    class Memory:
        def state(self, skill_id):
            return {"value": torch.tensor(float(skill_id + 100))}

    def build_record(strategy, skill_id, class_id, x, version, state):
        del strategy
        return SimpleNamespace(
            class_id=class_id,
            skill_id=skill_id,
            version=version,
            reference_inputs=x,
            state_value=float(state["value"].item()),
        )

    plugin = SimpleNamespace(
        behavior=Behavior(),
        class_map=ClassMap(),
        memory=Memory(),
        _build_record=build_record,
        _behavior_initialized=False,
    )

    result = MODULE.refresh_all_fingerprints(plugin, object())

    assert result == {
        "records_seen": 2,
        "records_refreshed": 2,
        "records_skipped": 0,
        "skills_refreshed": 2,
    }
    refreshed = [item for item in calls if item[0] == "record"]
    assert [(item[1], item[2], item[3]) for item in refreshed] == [
        (10, 0, 3),
        (20, 1, 7),
    ]
    assert torch.equal(refreshed[0][4], torch.tensor([[10.0], [11.0]]))
    assert torch.equal(refreshed[1][4], torch.tensor([[20.0], [21.0]]))
    assert plugin._behavior_initialized is True


def test_refresh_skips_classes_without_a_current_canonical_skill():
    records = [_record(10, 0, 10.0), _record(99, 9, 99.0)]

    class Behavior:
        def state_dict(self):
            return {"records": records}

        def skill_version(self, skill_id):
            return 0

        def put_skill_state(self, skill_id, version, state):
            del skill_id, version, state

        def put(self, record):
            raise AssertionError("unmapped class must not be rebuilt")

    class ClassMap:
        def find_skill_for_class_anywhere(self, class_id):
            return 0 if class_id == 10 else None

    class Memory:
        def state(self, skill_id):
            return {"value": torch.tensor(float(skill_id))}

    plugin = SimpleNamespace(
        behavior=Behavior(),
        class_map=ClassMap(),
        memory=Memory(),
        _build_record=lambda *args: None,
        _behavior_initialized=True,
    )

    result = MODULE.refresh_all_fingerprints(plugin, object())

    assert result["records_seen"] == 2
    assert result["records_refreshed"] == 1
    assert result["records_skipped"] == 1
    assert plugin._behavior_initialized is True
