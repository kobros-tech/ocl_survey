"""Synthetic benchmark-shape coverage for routing and weight reconstruction."""

import torch
import torch.nn as nn
import pytest
from avalanche.models.dynamic_modules import IncrementalClassifier

from skill_memory import reverse_engineer_scores_from_weights


class SyntheticBenchmarkClassifier(nn.Module):
    """Match the MLP + growing Avalanche classifier used by SplitMNIST demos."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
        )
        self.classifier = IncrementalClassifier(
            8, initial_out_features=n_classes
        )

    def forward(self, x):
        return self.classifier(self.features(x))


@pytest.mark.parametrize("n_classes", [1, 2, 5, 10])
def test_weight_reconstruction_matches_supported_head_widths(n_classes):
    torch.manual_seed(n_classes)
    model = SyntheticBenchmarkClassifier(n_classes)
    x = torch.randn(16, 4)

    with torch.no_grad():
        expected = model(x)
    reconstructed = reverse_engineer_scores_from_weights(model, x)

    torch.testing.assert_close(reconstructed, expected)


def test_synthetic_mixed_class_batch_preserves_per_sample_outputs():
    """A single batch may contain samples from multiple benchmark classes."""
    model = SyntheticBenchmarkClassifier(5)
    with torch.no_grad():
        model.classifier.classifier.weight.copy_(
            torch.eye(5, 8)
        )
        model.classifier.classifier.bias.zero_()

    x = torch.zeros(5, 4)
    x[:, 0] = torch.arange(5, dtype=torch.float32)

    expected = model(x)
    reconstructed = reverse_engineer_scores_from_weights(model, x)

    torch.testing.assert_close(reconstructed, expected)
