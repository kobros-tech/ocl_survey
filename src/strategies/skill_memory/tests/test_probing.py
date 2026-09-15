import importlib.util
import sys
import types
from pathlib import Path

import torch
from torch.utils.data import Dataset

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
pkg = types.ModuleType("probepkg")
pkg.__path__ = [str(root)]
sys.modules["probepkg"] = pkg

path = root / "probing.py"
spec = importlib.util.spec_from_file_location("probepkg.probing", path)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


class TinyDataset(Dataset):
    def __init__(self):
        self.samples = [(torch.tensor([i]), i % 3) for i in range(9)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class Experience:
    def __init__(self):
        self.dataset = TinyDataset()


def test_classes_and_probe_are_based_on_dataset_content():
    exp = Experience()

    assert mod.classes_in_experience(exp) == [0, 1, 2]

    _x, y = mod.probe_class(
        exp,
        1,
        batch_size=10,
        n_batches=1,
        seed=1,
    )

    assert set(y.tolist()) == {1}


def test_probe_routing_selects_one_skill_per_sample_without_labels():
    states = [
        {"classifier.weight": torch.tensor([[2.0, 0.0]])},
        {"classifier.weight": torch.tensor([[2.0, 0.0]])},
    ]

    logits_by_skill = [
        torch.tensor([[4.0, 1.0], [1.0, 0.0]]),
        torch.tensor([[1.0, 0.0], [4.0, 1.0]]),
    ]

    chosen = mod.route_probe_logits(
        logits_by_skill,
        states,
        [{0}, {0}],
    )

    assert chosen.tolist() == [0, 1]


def test_find_best_routing_skill_reports_normalized_probabilities():
    states = [{}, {}]
    logits_by_skill = [
        torch.tensor([[5.0, 0.0], [2.0, 0.0]]),
        torch.tensor([[0.0, 5.0], [2.0, 0.0]]),
    ]

    result = mod.find_best_routing_skill(
        logits_by_skill,
        states,
        [{0}, {1}],
    )

    assert torch.allclose(
        result.probabilities.sum(dim=0),
        torch.ones(2),
    )
    assert result.skill_indices.tolist() == [0, 0]
    assert torch.all(result.best_probability >= result.second_probability)
    assert torch.allclose(
        result.confidence_gap,
        result.best_probability - result.second_probability,
    )
    assert torch.all(result.confidence_gap >= 0)


def test_find_best_routing_skill_uses_uniform_fallback_when_no_evidence():
    states = [{}, {}, {}]
    logits_by_skill = [
        torch.zeros(2, 3),
        torch.zeros(2, 3),
        torch.zeros(2, 3),
    ]

    result = mod.find_best_routing_skill(
        logits_by_skill,
        states,
        [set(), set(), set()],
    )

    assert torch.allclose(result.probabilities, torch.full((3, 2), 1 / 3))
    assert result.skill_indices.tolist() == [0, 0]


def test_probe_routing_does_not_use_raw_entropy_for_one_class_heads():
    states = [
        {"classifier.weight": torch.tensor([[1.0]])},
        {"classifier.weight": torch.tensor([[1.0]])},
    ]

    logits_by_skill = [
        torch.tensor([[5.0], [1.0]]),
        torch.tensor([[1.0], [5.0]]),
    ]

    chosen = mod.route_probe_logits(
        logits_by_skill,
        states,
        [{0}, {0}],
    )

    assert chosen.tolist() == [0, 1]


def test_probe_routing_uses_owned_global_class_columns():
    """Routing must score skills using their owned global class columns."""
    class_a = 87
    class_b = 42
    n_samples = 2
    n_classes = max(class_a, class_b) + 1

    def make_logits(class_values):
        logits = torch.zeros(n_samples, n_classes)

        for class_id, values in class_values.items():
            logits[:, class_id] = torch.tensor(values)

        return logits

    states = [
        {"classifier.weight": torch.tensor([[1.0]])},
        {"classifier.weight": torch.tensor([[1.0]])},
    ]

    # Each skill has meaningful logits only in its owned global
    # classifier column. This verifies that class 87 is read from
    # column 87 and class 42 is read from column 42, rather than
    # being remapped to local columns such as column 0.
    logits_by_skill = [
        make_logits(
            {
                class_a: [5.0, 1.0],
            }
        ),
        make_logits(
            {
                class_b: [1.0, 5.0],
            }
        ),
    ]

    chosen = mod.route_probe_logits(
        logits_by_skill,
        states,
        [{class_a}, {class_b}],
    )

    assert chosen.tolist() == [0, 1]
