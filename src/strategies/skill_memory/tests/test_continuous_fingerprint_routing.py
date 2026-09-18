"""Tests for input-only anonymous routing helpers."""

import pytest
import torch

from skill_memory.evaluation import find_best_routing_skill


def test_routing_selects_skill_with_highest_owned_class_probability():
    logits = [
        torch.tensor([[4.0, -2.0]]),
        torch.tensor([[-2.0, 3.0]]),
    ]
    result = find_best_routing_skill(
        logits,
        [{}, {}],
        [{0}, {1}],
    )

    assert result.skill_indices.tolist() == [0]
    assert result.best_probability.item() > result.second_probability.item()
    assert result.confidence_gap.item() > 0


def test_routing_returns_ambiguity_signal_for_equal_scores():
    logits = [
        torch.tensor([[1.0, 1.0]]),
        torch.tensor([[1.0, 1.0]]),
    ]
    result = find_best_routing_skill(
        logits,
        [{}, {}],
        [{0, 1}, {0, 1}],
    )

    assert result.best_probability.item() == pytest.approx(0.5)
    assert result.second_probability.item() == pytest.approx(0.5)
    assert result.confidence_gap.item() == pytest.approx(0.0)


def test_routing_uses_owned_classes_not_skill_position():
    logits = [
        torch.tensor([[-2.0, 5.0, -2.0]]),
        torch.tensor([[5.0, -2.0, -2.0]]),
    ]
    result = find_best_routing_skill(
        logits,
        [{}, {}],
        [{0}, {1}],
    )

    assert result.skill_indices.tolist() == [0]
    assert result.probabilities.shape == (2, 1)
