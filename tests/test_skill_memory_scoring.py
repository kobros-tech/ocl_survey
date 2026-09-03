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

    score = scorer(record, DummyExperience())

    assert score == pytest.approx(0.5)
    assert 0.0 <= score <= 1.0
