"""Regression test: `_route` must route each sample in a batch independently.

`_route` used to score the reverse-router's listwise Transformer once per
candidate, across the *entire* eval batch (`predict_scores_features` called
with `[batch_of_real_samples, feature_dim]`). That model was trained on
`[reference_points, candidates, feature_dim]` - candidates as the attention
sequence, real points as the batch. Calling it the old way fed
`[1, batch_of_real_samples, feature_dim]` instead, so the Transformer
attended across *unrelated eval samples* rather than across candidates, and
collapsed to one answer for the whole batch regardless of each sample's
true class.

This was invisible whenever an eval batch happened to be homogeneous (one
true class per experience, the only configuration exercised elsewhere in
this suite) and became a severe, silent accuracy collapse the moment a
batch mixed samples from more than one class/skill - exactly what happens
whenever an experience (or an evaluation batch straddling several
experiences) contains more than one class.
"""

import torch

from skill_memory import ClassBehaviorRecord, SkillMemory
from skill_memory.fingerprint_routing import PersistentFingerprintSkillMemoryPlugin


def _record(class_id: int, skill_id: int, weight: float) -> ClassBehaviorRecord:
    return ClassBehaviorRecord(
        class_id=class_id,
        skill_id=skill_id,
        version=0,
        reference_inputs=torch.tensor([[float(class_id)]] * 4),
        reference_y=torch.ones(4, dtype=torch.bool),
        reference_weight=torch.tensor([weight]),
        reference_bias=0.0,
    )


def _plugin_with_skills(n_skills: int) -> PersistentFingerprintSkillMemoryPlugin:
    plugin = PersistentFingerprintSkillMemoryPlugin(
        verbose=False, reverse_epochs=40, diagnose=False
    )
    plugin.memory = SkillMemory(max_skills=n_skills)
    plugin.behavior = plugin.behavior.__class__()
    for skill_id in range(n_skills):
        plugin.memory.store(skill_id, {"slot": torch.tensor([float(skill_id)])})
        plugin.behavior.put(_record(skill_id, skill_id, float(skill_id + 1)))
    return plugin


def test_route_scores_each_sample_in_a_heterogeneous_batch_independently(
    monkeypatch,
):
    """Every skill's frozen response cleanly identifies its own class (an
    obvious, near-noise-free signal): skill k reports logit +8 at column k
    and -8 elsewhere, driven directly by the *input value*, not by which
    skill's turn it is. A correctly-routing model must therefore identify
    every sample by its own value, independent of every other sample sharing
    its eval batch."""
    n_skills = 4
    plugin = _plugin_with_skills(n_skills)

    def frozen_logits(_strategy, skill_id, x):
        # Each skill "recognizes" exactly the input value equal to its own
        # class_id - a clean, per-sample-determined signal with no
        # dependence on batch position or other samples.
        logits = torch.full((x.shape[0], n_skills), -8.0)
        matches = x[:, 0] == float(skill_id)
        logits[matches, skill_id] = 8.0
        return logits

    monkeypatch.setattr(plugin, "_frozen_logits", frozen_logits)
    plugin._fit_reverse_router(object())
    assert plugin.reverse_engineer.model is not None

    # A single heterogeneous batch spanning every class, in a scrambled
    # order so no positional/majority shortcut could accidentally look
    # correct.
    x = torch.tensor([[2.0], [0.0], [3.0], [1.0], [0.0], [3.0], [2.0], [1.0]])
    true_classes = [2, 0, 3, 1, 0, 3, 2, 1]

    chosen_skills, chosen_classes = plugin._route(object(), x, list(range(n_skills)))

    assert chosen_classes == true_classes
    assert chosen_skills.tolist() == true_classes


def test_route_does_not_collapse_to_a_single_answer_for_the_whole_batch(monkeypatch):
    """A cheap smoke check on the failure signature itself: if routing were
    still batch-collapsed, every sample would get the *same* predicted
    class regardless of its true label. With >1 distinct true class in the
    batch, a non-collapsed router must produce more than one distinct
    predicted class."""
    n_skills = 4
    plugin = _plugin_with_skills(n_skills)

    def frozen_logits(_strategy, skill_id, x):
        logits = torch.full((x.shape[0], n_skills), -8.0)
        matches = x[:, 0] == float(skill_id)
        logits[matches, skill_id] = 8.0
        return logits

    monkeypatch.setattr(plugin, "_frozen_logits", frozen_logits)
    plugin._fit_reverse_router(object())

    x = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    _, chosen_classes = plugin._route(object(), x, list(range(n_skills)))

    assert len(set(chosen_classes)) > 1
