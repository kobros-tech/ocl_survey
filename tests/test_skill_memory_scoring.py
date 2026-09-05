import math

import pytest
import torch
from torch import nn

from src.strategies import skill_memory


class DummyExperience:
    current_experience = 0
    dataset = [(torch.tensor([0.0]), 0)]


def test_compatibility_is_normalized_probe_loss_not_probability(monkeypatch):
    monkeypatch.setattr(skill_memory, "avalanche_model_adaptation", lambda *_args, **_kwargs: None)

    model = nn.Linear(1, 2)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()

    scorer = skill_memory.ProbeCompatibilityScorer(
        model_factory=lambda: nn.Linear(1, 2),
        loss_fn=lambda logits, y: torch.tensor(0.5),
        probe_fn=lambda _experience: (torch.zeros(1, 1), torch.zeros(1, dtype=torch.long)),
        reference_fn=lambda _y: 1.0,
        probe_samples=1,
    )
    record = skill_memory.SkillRecord(
        name="skill-0",
        state_dict={k: v.detach().clone() for k, v in model.state_dict().items()},
    )

    result = scorer(record, DummyExperience())

    assert result.score == pytest.approx(math.exp(-0.5))
    assert result.accuracy == pytest.approx(1.0)
    assert 0.0 <= result.score <= 1.0


def test_zero_accuracy_cannot_have_a_high_score():
    """The score and accuracy must be mathematically consistent.

    If every prediction is wrong, every true-class probability is below 0.5,
    so CE > log(2). With the CIFAR-100 reference log(100), the normalized
    score must therefore be below exp(-log(2) / log(100)) ~= 0.86.
    """
    num_classes = 100
    upper_bound = math.exp(-math.log(2.0) / math.log(num_classes))

    # A score above this bound together with zero accuracy is impossible when
    # score and accuracy are computed from the same logits and labels.
    assert upper_bound < 0.90
    assert skill_memory.max_compatible_score_for_accuracy(
        accuracy=0.0, num_classes=num_classes
    ) == pytest.approx(upper_bound)


def test_score_upper_bound_decreases_with_lower_probe_accuracy():
    bound_high_accuracy = skill_memory.max_compatible_score_for_accuracy(
        accuracy=0.75, num_classes=100
    )
    bound_low_accuracy = skill_memory.max_compatible_score_for_accuracy(
        accuracy=0.25, num_classes=100
    )
    assert bound_low_accuracy < bound_high_accuracy
