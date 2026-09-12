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
