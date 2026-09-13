from pathlib import Path
import importlib.util
import sys
import types

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
sys.modules.update({
    "avalanche": avalanche,
    "avalanche.models": models,
    "avalanche.models.dynamic_modules": dynamic,
})

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
    x, y = mod.probe_class(exp, 1, batch_size=10, n_batches=1, seed=1)
    assert set(y.tolist()) == {1}


def test_probe_routing_selects_one_skill_per_sample_without_labels():
    # Two skills, mixed minibatch. Skill 0 is more confident for sample 0,
    # skill 1 for sample 1. The router consumes only logits and stored states.
    states = [
        {"classifier.weight": torch.tensor([[2.0, 0.0]])},
        {"classifier.weight": torch.tensor([[2.0, 0.0]])},
    ]
    logits_by_skill = [
        torch.tensor([[4.0, 1.0], [1.0, 0.0]]),
        torch.tensor([[1.0, 0.0], [4.0, 1.0]]),
    ]
    chosen = mod.route_probe_logits(logits_by_skill, states)
    assert chosen.tolist() == [0, 1]


def test_probe_routing_does_not_use_raw_entropy_for_one_class_heads():
    # A one-class softmax has zero entropy for every input, so entropy-based
    # routing cannot distinguish these skills. The probe router instead uses
    # the stored single-class logit as its confidence signal.
    states = [
        {"classifier.weight": torch.tensor([[1.0]])},
        {"classifier.weight": torch.tensor([[1.0]])},
    ]
    logits_by_skill = [
        torch.tensor([[5.0], [1.0]]),
        torch.tensor([[1.0], [5.0]]),
    ]
    chosen = mod.route_probe_logits(logits_by_skill, states)
    assert chosen.tolist() == [0, 1]


def test_expand_skill_logits_handles_different_head_sizes():
    state = {
        "classifier.weight": torch.tensor([[1.0], [2.0]]),
        "active_units": torch.tensor([2, 7]),
    }
    logits = torch.tensor([[3.0, 4.0]])
    expanded = mod.expand_skill_logits(logits, state, {2, 7}, output_dim=10)
    assert expanded.shape == (1, 10)
    assert expanded[0, 2].item() == 3.0
    assert expanded[0, 7].item() == 4.0
    assert torch.isneginf(expanded[0, 0])
    assert torch.isneginf(expanded[0, 9])
