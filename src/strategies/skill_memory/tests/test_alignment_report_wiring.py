"""diagnose=True must also compute the class-index alignment report.

This is the concrete diagnostic PR review comment issuecomment-5718049313
asked for (per skill: owned_global_classes, logits_width, active_units),
now wired to run automatically once per completed training experience
whenever `diagnose=True`, alongside the existing routing-rank diagnostics.
"""

from types import SimpleNamespace

import torch
from torch import nn

from skill_memory import ClassRecord, SkillMemory
from skill_memory.cl.skill_registry import ExperienceClassMap
from skill_memory.evaluation.fingerprint_routing import (
    PersistentFingerprintSkillMemoryPlugin,
)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(2, 6)

    def forward(self, x):
        return self.classifier(x)


def _plugin_with_one_skill(*, diagnose: bool) -> PersistentFingerprintSkillMemoryPlugin:
    plugin = PersistentFingerprintSkillMemoryPlugin(
        verbose=False, reverse_epochs=1, diagnose=diagnose
    )
    plugin.memory = SkillMemory(max_skills=4)
    plugin.memory.store(
        0,
        {
            "classifier.weight": torch.randn(6, 2),
            "classifier.bias": torch.randn(6),
        },
    )
    plugin.class_map = ExperienceClassMap()
    plugin.class_map.record(
        ClassRecord(experience_index=0, class_id=2, decision="SCRATCH", skill=0)
    )
    return plugin


def test_diagnose_true_populates_last_alignment_report(monkeypatch):
    plugin = _plugin_with_one_skill(diagnose=True)
    # Isolate the new wiring from the (unrelated, heavier) base-class
    # training-experience bookkeeping, which needs a full Avalanche
    # experience object to drive correctly.
    monkeypatch.setattr(
        type(plugin).__mro__[1], "after_training_exp", lambda self, strategy, **kw: None
    )

    plugin.after_training_exp(SimpleNamespace(model=_TinyModel()))

    assert plugin.last_alignment_report != {}
    assert plugin.last_alignment_report["skills"][0]["skill_id"] == 0
    assert plugin.last_alignment_report["skills"][0]["owned_global_classes"] == [2]
    assert "verdict" in plugin.last_alignment_report


def test_diagnose_false_leaves_alignment_report_empty(monkeypatch):
    plugin = _plugin_with_one_skill(diagnose=False)
    monkeypatch.setattr(
        type(plugin).__mro__[1], "after_training_exp", lambda self, strategy, **kw: None
    )

    plugin.after_training_exp(SimpleNamespace(model=_TinyModel()))

    assert plugin.last_alignment_report == {}


def test_diagnose_true_with_no_skills_yet_leaves_alignment_report_empty(monkeypatch):
    plugin = PersistentFingerprintSkillMemoryPlugin(
        verbose=False, reverse_epochs=1, diagnose=True
    )
    monkeypatch.setattr(
        type(plugin).__mro__[1], "after_training_exp", lambda self, strategy, **kw: None
    )

    plugin.after_training_exp(SimpleNamespace(model=_TinyModel()))

    assert plugin.last_alignment_report == {}
