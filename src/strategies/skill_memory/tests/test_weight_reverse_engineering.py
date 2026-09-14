"""Tests for weight-based reconstruction of classifier decisions."""

import torch
import torch.nn as nn
from avalanche.models.dynamic_modules import IncrementalClassifier

from skill_memory import (
    reverse_engineer_scores_from_weights,
    reverse_engineer_y_from_weights,
)


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Linear(4, 3, bias=False)
        self.classifier = IncrementalClassifier(3, initial_out_features=3)

    def forward(self, x):
        return self.classifier(self.features(x))


def test_weight_reconstruction_matches_classifier_output():
    torch.manual_seed(0)
    model = TinyClassifier()
    x = torch.randn(8, 4)

    with torch.no_grad():
        expected = model(x)
    reconstructed = reverse_engineer_scores_from_weights(model, x)

    torch.testing.assert_close(reconstructed, expected)


def test_weight_reverse_engineering_uses_candidate_score_threshold():
    torch.manual_seed(1)
    model = TinyClassifier()
    x = torch.randn(8, 4)

    scores = model(x)
    for class_id in range(3):
        expected = scores[:, class_id] > 0
        actual = reverse_engineer_y_from_weights(model, x, class_id)
        torch.testing.assert_close(actual, expected)
