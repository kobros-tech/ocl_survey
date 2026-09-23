import torch

from skill_memory.cl.skill_registry import ClassRecord, ExperienceClassMap, SkillMemory


def test_experience_groups_multiple_skills_and_classes():
    mapping = ExperienceClassMap()
    mapping.record(ClassRecord(0, 10, "scratch", 0))
    mapping.record(ClassRecord(0, 20, "reuse", 0))
    mapping.record(ClassRecord(0, 30, "scratch", 1))

    assert mapping.find_skill_for_class_anywhere(10) == 0
    assert mapping.find_skill_for_class_anywhere(20) == 0
    assert mapping.classes_for_skill(0) == {10, 20}
    assert mapping.skills_for_experience(0) == [(0, {10, 20}), (1, {30})]


def test_class_cannot_be_remapped():
    mapping = ExperienceClassMap()
    mapping.record(ClassRecord(0, 10, "scratch", 0))
    try:
        mapping.record(ClassRecord(1, 10, "scratch", 1))
    except RuntimeError as exc:
        assert "already mapped" in str(exc)
    else:
        raise AssertionError("conflicting class mapping was accepted")


def test_skill_memory_snapshots_are_copied():
    memory = SkillMemory(max_skills=2)
    slot = memory.allocate()
    state = {"weight": torch.tensor([1.0])}
    memory.store(slot, state)
    state["weight"][0] = 99.0
    assert memory.state(slot)["weight"].item() == 1.0
