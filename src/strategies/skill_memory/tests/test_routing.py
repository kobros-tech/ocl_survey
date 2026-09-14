import importlib.util
import sys
import types
from pathlib import Path

import torch

# Lightweight Avalanche stubs.
avalanche = types.ModuleType("avalanche")
models = types.ModuleType("avalanche.models")
dynamic = types.ModuleType("avalanche.models.dynamic_modules")


class IncrementalClassifier:
    pass


dynamic.IncrementalClassifier = IncrementalClassifier
dynamic.avalanche_model_adaptation = lambda model, experience: None
models.dynamic_modules = dynamic
avalanche.models = models
sys.modules.update(
    {
        "avalanche": avalanche,
        "avalanche.models": models,
        "avalanche.models.dynamic_modules": dynamic,
    }
)

root = Path(__file__).parents[1]
pkg = types.ModuleType("routingpkg")
pkg.__path__ = [str(root)]
sys.modules["routingpkg"] = pkg

path = root / "probing.py"
spec = importlib.util.spec_from_file_location("routingpkg.probing", path)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def _inputs():
    states = [
        {"classifier.weight": torch.tensor([[1.0], [1.0]])},
        {"classifier.weight": torch.tensor([[1.0], [1.0]])},
        {"classifier.weight": torch.tensor([[1.0], [1.0]])},
    ]
    logits = [
        torch.tensor([[5.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
        torch.tensor([[1.0, 0.0], [4.0, 0.0], [0.0, 3.0]]),
        torch.tensor([[2.0, 0.0], [1.0, 0.0], [5.0, 0.0]]),
    ]
    classes = [{0}, {0}, {0}]
    return logits, states, classes


def test_find_best_routing_skill_selects_one_skill_per_sample():
    logits, states, classes = _inputs()

    result = mod.find_best_routing_skill(logits, states, classes)

    assert result.skill_indices.tolist() == [0, 1, 2]
    assert result.probabilities.shape == (3, 3)
    assert result.best_probability.shape == (3,)
    assert result.second_probability.shape == (3,)
    assert result.confidence_gap.shape == (3,)


def test_routing_probabilities_sum_to_one_per_sample():
    logits, states, classes = _inputs()

    result = mod.find_best_routing_skill(logits, states, classes)

    assert torch.allclose(
        result.probabilities.sum(dim=0),
        torch.ones(3),
    )


def test_routing_gap_is_best_minus_second_best():
    logits, states, classes = _inputs()

    result = mod.find_best_routing_skill(logits, states, classes)
    expected = result.best_probability - result.second_probability

    assert torch.allclose(result.confidence_gap, expected)
    assert torch.all(result.confidence_gap >= 0)


def test_single_skill_has_probability_one_and_zero_second_best():
    logits = [torch.tensor([[5.0], [1.0]])]
    states = [{"classifier.weight": torch.tensor([[1.0]])}]
    classes = [{0}]

    result = mod.find_best_routing_skill(logits, states, classes)

    assert result.skill_indices.tolist() == [0, 0]
    assert torch.equal(result.best_probability, torch.ones(2))
    assert torch.equal(result.second_probability, torch.zeros(2))
    assert torch.equal(result.confidence_gap, torch.ones(2))


def test_temperature_changes_sharpness_not_winner():
    logits, states, classes = _inputs()

    cold = mod.find_best_routing_skill(logits, states, classes, temperature=0.5)
    hot = mod.find_best_routing_skill(logits, states, classes, temperature=2.0)

    assert cold.skill_indices.tolist() == hot.skill_indices.tolist()
    assert not torch.allclose(cold.probabilities, hot.probabilities)


def test_invalid_temperature_is_rejected():
    logits, states, classes = _inputs()

    try:
        mod.find_best_routing_skill(logits, states, classes, temperature=0.0)
    except ValueError as exc:
        assert "temperature" in str(exc)
    else:
        raise AssertionError("non-positive temperature was accepted")


def test_routing_uses_owned_global_class_columns():
    class_a = 87
    class_b = 42
    n_classes = max(class_a, class_b) + 1

    first = torch.zeros(2, n_classes)
    second = torch.zeros(2, n_classes)
    first[:, class_a] = torch.tensor([5.0, 1.0])
    second[:, class_b] = torch.tensor([1.0, 5.0])

    states = [
        {"classifier.weight": torch.tensor([[1.0]])},
        {"classifier.weight": torch.tensor([[1.0]])},
    ]

    result = mod.find_best_routing_skill(
        [first, second],
        states,
        [{class_a}, {class_b}],
    )

    assert result.skill_indices.tolist() == [0, 1]


def test_routing_input_contract_has_no_labels():
    logits, states, classes = _inputs()

    # The public routing API is intentionally called without y/labels.
    result = mod.find_best_routing_skill(logits, states, classes)

    assert result.skill_indices.numel() == 3


def test_route_probe_logits_remains_compatible():
    logits, states, classes = _inputs()

    legacy = mod.route_probe_logits(logits, states, classes)
    richer = mod.find_best_routing_skill(logits, states, classes).skill_indices

    assert torch.equal(legacy, richer)
