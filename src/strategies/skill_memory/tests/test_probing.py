import torch
from torch.utils.data import Dataset

from skill_memory.utils import probing as mod


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
