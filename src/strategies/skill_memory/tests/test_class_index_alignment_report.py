"""Tests for the per-skill global-class-id / classifier-width diagnostic.

This is the concrete, runnable version of the diagnostic PR review comment
issuecomment-5718049313 (Sept 2026) asked for: record, per skill,
owned_global_classes / logits_width / active_units, and determine whether a
skill's own classifier already has a column for every class it owns.
"""

from types import SimpleNamespace

from avalanche.models import IncrementalClassifier
from torch import nn

from skill_memory import class_index_alignment_report


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = IncrementalClassifier(in_features=4)


def test_alignment_holds_for_non_relabeled_global_class_ids():
    """A skill trained on raw (non-relabeled) global class ids must show up
    as aligned: its own classifier already grew to cover every class it
    owns, at exactly the matching column."""
    model = _Model()
    model.classifier.adaptation(
        SimpleNamespace(classes_in_this_experience=[35, 36, 37, 38, 39])
    )

    report = class_index_alignment_report(
        model,
        states=[model.state_dict()],
        slot_ids=[7],
        owned_classes_by_slot=[[35, 36, 37, 38, 39]],
    )

    assert report["any_misaligned"] is False
    skill = report["skills"][0]
    assert skill["skill_id"] == 7
    assert skill["logits_width"] == 40
    assert skill["active_units"] == [35, 36, 37, 38, 39]
    assert skill["out_of_range_classes"] == []


def test_alignment_breaks_when_classes_are_relabeled_from_zero():
    """If a benchmark relabels each experience's classes to start from zero,
    the skill's own classifier only ever grows to the local (small) width,
    while SkillMemory's own bookkeeping still uses the larger global ids -
    the report must flag this as misaligned rather than silently pass."""
    model = _Model()
    model.classifier.adaptation(
        SimpleNamespace(classes_in_this_experience=[0, 1, 2, 3, 4])
    )

    report = class_index_alignment_report(
        model,
        states=[model.state_dict()],
        slot_ids=[7],
        owned_classes_by_slot=[[35, 36, 37, 38, 39]],
    )

    assert report["any_misaligned"] is True
    skill = report["skills"][0]
    assert skill["logits_width"] == 5
    assert skill["out_of_range_classes"] == [35, 36, 37, 38, 39]


def test_alignment_report_covers_multiple_skills_independently():
    aligned_model = _Model()
    aligned_model.classifier.adaptation(
        SimpleNamespace(classes_in_this_experience=[0, 1, 2, 3, 4])
    )
    misaligned_model = _Model()
    misaligned_model.classifier.adaptation(
        SimpleNamespace(classes_in_this_experience=[5, 6, 7, 8, 9])
    )

    report = class_index_alignment_report(
        aligned_model,
        states=[aligned_model.state_dict(), misaligned_model.state_dict()],
        slot_ids=[0, 1],
        owned_classes_by_slot=[[0, 1, 2, 3, 4], [50, 51, 52, 53, 54]],
    )

    assert report["skills"][0]["misaligned"] is False
    assert report["skills"][1]["misaligned"] is True
    assert report["any_misaligned"] is True
