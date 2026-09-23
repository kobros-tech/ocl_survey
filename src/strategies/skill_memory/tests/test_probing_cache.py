"""Regression test for the CI stall: class lookups must be cached.

Before the fix, `class_indices`/`classes_in_experience` called
`dataset[index]` (a full decode) for every sample on *every* call. Since
`score_class_against_skills` calls these once per (skill, mastered_class)
pair, and both the skill count and the seen-experience count grow every
training step, the uncached cost compounded until runs stalled out on
Split-CIFAR-100 (see the CI log: each newly-reached experience took
longer than the last -- 20s, 70s, 146s, 267s, ... -- while every
previously-seen one stayed ~0.5s).

This test doesn't need torch/Avalanche: it only exercises the pure
label-indexing path, with a fake dataset that counts how many times
`__getitem__` is called.
"""

from skill_memory.utils import probing as mod


class CountingDataset:
    """Fake dataset without a `.targets` shortcut, so every label read
    would otherwise require the (expensive, simulated) __getitem__."""

    def __init__(self, labels):
        self._labels = labels
        self.getitem_calls = 0

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index):
        self.getitem_calls += 1
        return (f"decoded_{index}", self._labels[index])


class Experience:
    def __init__(self, dataset):
        self.dataset = dataset


def test_repeated_class_queries_do_not_rescan_the_dataset():
    dataset = CountingDataset([i % 5 for i in range(500)])
    exp = Experience(dataset)

    # Simulate what score_class_against_skills does: querying several
    # different classes against the SAME experience many times over (once
    # per candidate skill).
    for _ in range(20):
        for target_class in range(5):
            mod.class_indices(exp, target_class)

    # Without caching this would be 20 * 5 * 500 = 50,000 calls. With
    # caching, the dataset is scanned exactly once, ever.
    assert dataset.getitem_calls == len(dataset)


def test_cache_is_keyed_per_dataset_not_global():
    a = CountingDataset([0, 1, 0, 1])
    b = CountingDataset([1, 1, 0, 0])
    exp_a, exp_b = Experience(a), Experience(b)

    assert mod.class_indices(exp_a, 0) == [0, 2]
    assert mod.class_indices(exp_b, 0) == [2, 3]
    assert a.getitem_calls == len(a)
    assert b.getitem_calls == len(b)
