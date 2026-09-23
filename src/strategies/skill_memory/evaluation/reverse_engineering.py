"""Standalone ML reverse engineering from frozen model behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class CandidateParameters:
    """Frozen classifier parameters retained for compatibility."""

    weight: Tensor
    bias: float


class _FeatureReverseModel(nn.Module):
    """Cross-candidate Transformer scorer for listwise routing."""

    def __init__(
        self,
        feature_dim: int,
        hidden_size: int,
        num_heads: int = 8,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.input_projection = nn.Linear(feature_dim, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, features: Tensor) -> Tensor:
        tokens = self.input_projection(features)
        tokens = self.encoder(tokens)
        return self.output(tokens)


class _BinaryFeatureReverseModel(nn.Module):
    """Small MLP retained for the historical binary-pair API."""

    def __init__(self, feature_dim: int, hidden_size: int) -> None:
        super().__init__()
        width = max(16, min(int(hidden_size), 128))
        self.network = nn.Sequential(
            nn.Linear(feature_dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


class NormalMLReverseEngineer:
    """Learn anonymous candidate/class compatibility as a normal ML problem.

    Two training modes are supported:

    - ``"listwise"`` (default): for every reference sample, all currently
      known class candidates form one candidate set and cross-entropy trains
      the model to put the correct class at the top. Candidate tokens attend
      to one another, so the scorer reasons about the complete candidate set
      rather than scoring every candidate independently.
    - ``"binary"``: the legacy pairwise-compatibility API, retained for
      backward compatibility with callers built around ``fit_feature_pairs``.

    In both modes, candidate identity is represented only through
    model-derived behavior features. The integer class ID is never an input
    feature, so no evaluation-time ground-truth leakage is possible through
    this model.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        epochs: int = 60,
        learning_rate: float = 1e-3,
        seed: int = 0,
        batch_size: int = 256,
        training_mode: str = "listwise",
        num_heads: int = 8,
        num_layers: int = 3,
    ) -> None:
        """Configure training hyperparameters (nothing is fit until `.fit*`)."""
        if training_mode not in {"listwise", "binary"}:
            raise ValueError("training_mode must be 'listwise' or 'binary'")
        if hidden_size <= 0 or num_heads <= 0 or num_layers <= 0:
            raise ValueError("model dimensions must be positive")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = int(hidden_size)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.training_mode = training_mode
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.model: nn.Module | None = None
        self.feature_dim: int | None = None
        self.feature_mean: Tensor | None = None
        self.feature_std: Tensor | None = None

    @staticmethod
    def _fit_scaler(features: Tensor) -> tuple[Tensor, Tensor]:
        mean = features.mean(dim=0)
        std = features.std(dim=0, unbiased=False).clamp_min(1e-6)
        return mean, std

    def _fit_model(self, features: Tensor, targets: Tensor) -> None:
        """Fit either the listwise Transformer or binary compatibility MLP."""
        torch.manual_seed(self.seed)
        self.feature_dim = int(features.shape[-1])
        flat = features.reshape(-1, self.feature_dim)
        self.feature_mean, self.feature_std = self._fit_scaler(flat)
        normalized = (features - self.feature_mean) / self.feature_std

        if self.training_mode == "binary":
            model: nn.Module = _BinaryFeatureReverseModel(
                self.feature_dim, self.hidden_size
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
            positive_count = float(targets.sum().item())
            negative_count = float(targets.numel() - positive_count)
            pos_weight = (
                torch.tensor(
                    negative_count / positive_count,
                    dtype=torch.float32,
                )
                if positive_count > 0 and negative_count > 0
                else torch.tensor(1.0, dtype=torch.float32)
            )
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            model.train()
            with torch.enable_grad():
                for _ in range(self.epochs):
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(normalized).squeeze(-1)
                    loss = criterion(logits, targets.float().reshape(-1))
                    loss.backward()
                    optimizer.step()
        else:
            self.hidden_size = max(self.hidden_size, 128)
            if self.hidden_size % self.num_heads != 0:
                raise ValueError("hidden_size must be divisible by num_heads")
            model = _FeatureReverseModel(
                self.feature_dim,
                self.hidden_size,
                self.num_heads,
                self.num_layers,
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
            criterion = nn.CrossEntropyLoss()
            sample_count = normalized.shape[0]
            batch_size = max(1, min(self.batch_size, sample_count))
            model.train()
            with torch.enable_grad():
                for _ in range(self.epochs):
                    order = torch.randperm(sample_count)
                    for start in range(0, sample_count, batch_size):
                        indices = order[start : start + batch_size]
                        batch = normalized[indices]
                        batch_targets = targets[indices]
                        logits = model(batch).squeeze(-1)
                        logits = logits.reshape(-1, batch.shape[1])
                        optimizer.zero_grad(set_to_none=True)
                        loss = criterion(logits, batch_targets)
                        loss.backward()
                        optimizer.step()
        self.model = model.eval()

    def fit_candidate_sets(
        self,
        candidate_sets: list[tuple[Tensor, Tensor | int]],
    ) -> None:
        """Fit listwise compatibility scores over complete candidate sets.

        Each item is ``(features, target_index)`` where ``features`` has shape
        ``[candidates, feature_dim]`` and ``target_index`` identifies the
        correct candidate. Candidate order has no learned positional meaning.
        """
        if not candidate_sets:
            self.model = None
            self.feature_dim = None
            self.feature_mean = None
            self.feature_std = None
            return

        normalized_sets: list[Tensor] = []
        target_indices: list[int] = []
        for features, target in candidate_sets:
            features = features.detach().float().cpu()
            if features.ndim != 2:
                raise ValueError(
                    "candidate-set features must be [candidates, features]"
                )
            target_index = (
                int(target) if not isinstance(target, Tensor) else int(target.item())
            )
            if not 0 <= target_index < features.shape[0]:
                raise ValueError("candidate target index is outside candidate set")
            normalized_sets.append(features)
            target_indices.append(target_index)

        candidate_count = normalized_sets[0].shape[0]
        feature_dim = normalized_sets[0].shape[1]
        if any(
            item.shape != (candidate_count, feature_dim) for item in normalized_sets
        ):
            raise ValueError("candidate sets must have identical shapes")
        features = torch.stack(normalized_sets, dim=0)
        targets = torch.tensor(target_indices, dtype=torch.long)
        self.training_mode = "listwise"
        self._fit_model(features, targets)

    def fit_feature_pairs(
        self,
        pairs: list[tuple[Tensor, float]],
    ) -> None:
        """Fit from detached frozen-model features and binary targets."""
        if not pairs:
            self.model = None
            self.feature_dim = None
            self.feature_mean = None
            self.feature_std = None
            return
        features = torch.cat(
            [feature.detach().float().cpu() for feature, _ in pairs],
            dim=0,
        )
        targets = torch.cat(
            [
                torch.full(
                    (feature.shape[0], 1),
                    float(target),
                    dtype=torch.float32,
                )
                for feature, target in pairs
            ],
            dim=0,
        )
        self.training_mode = "binary"
        self._fit_model(features, targets)

    def predict_scores_features(self, features: Tensor) -> Tensor:
        """Return unnormalized compatibility scores for candidate features.

        In listwise mode, ``features`` must be *one* sample's full candidate
        set: shape ``[candidates, feature_dim]``. Every candidate in it is
        one token in a single attention sequence, matching how
        `fit_candidate_sets` trains the model (see its docstring). Scoring
        many real samples against one candidate at a time, by calling this
        once per candidate across a batch of samples, does NOT do that - it
        feeds the model a `[1, batch_of_samples, feature_dim]` tensor, so the
        transformer attends across unrelated samples instead of across
        candidates, silently producing one collapsed answer for the whole
        batch instead of a real per-sample decision. Batched routing of many
        real samples against the full candidate set must use
        `predict_scores_candidate_sets` instead.
        """
        if (
            self.model is None
            or self.feature_dim is None
            or self.feature_mean is None
            or self.feature_std is None
        ):
            raise RuntimeError("reverse-engineering model has not been fitted")
        features = features.detach().float().cpu()
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("reverse-engineering feature shape changed")
        normalized = (features - self.feature_mean) / self.feature_std
        with torch.no_grad():
            if self.training_mode == "binary":
                return self.model(normalized).squeeze(-1)
            return self.model(normalized.unsqueeze(0)).squeeze(0).squeeze(-1)

    def predict_scores_candidate_sets(self, features: Tensor) -> Tensor:
        """Score every candidate for a batch of real samples in one pass.

        ``features`` must have shape ``[batch, candidates, feature_dim]`` -
        the same convention `fit_candidate_sets` trains on, just with
        ``batch`` real samples instead of reference points. Each sample's
        ``candidates`` row is its own attention sequence; the batch
        dimension is a genuine torch batch dimension throughout, so
        real samples never attend to one another. Returns
        ``[batch, candidates]`` scores. This is the correct way to route
        many real samples against a full candidate set at once - see
        `predict_scores_features`'s docstring for the bug this avoids.
        """
        if self.training_mode != "listwise":
            raise RuntimeError(
                "predict_scores_candidate_sets requires training_mode='listwise'"
            )
        if (
            self.model is None
            or self.feature_dim is None
            or self.feature_mean is None
            or self.feature_std is None
        ):
            raise RuntimeError("reverse-engineering model has not been fitted")
        features = features.detach().float().cpu()
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                "candidate-set features must be [batch, candidates, feature_dim]"
            )
        normalized = (features - self.feature_mean) / self.feature_std
        with torch.no_grad():
            return self.model(normalized).squeeze(-1)

    def predict_proba_features(self, features: Tensor) -> Tensor:
        """Return independent binary probabilities for compatibility mode."""
        return torch.sigmoid(self.predict_scores_features(features))

    def fit(self, pairs: list[tuple[Tensor, CandidateParameters, float]]) -> None:
        """Backward-compatible binary-pair API."""
        if not pairs:
            self.fit_feature_pairs([])
            return
        feature_pairs = []
        for samples, params, target in pairs:
            samples = samples.detach().float().cpu().reshape(samples.shape[0], -1)
            weight = params.weight.detach().float().cpu().reshape(1, -1)
            weight = weight.expand(samples.shape[0], -1)
            bias = torch.full((samples.shape[0], 1), float(params.bias))
            feature_pairs.append((torch.cat((samples, weight, bias), dim=1), target))
        self.fit_feature_pairs(feature_pairs)

    def predict_proba(
        self,
        x: Tensor,
        weight: Tensor,
        bias: float,
    ) -> Tensor:
        """Backward-compatible binary prediction API."""
        samples = x.detach().float().cpu().reshape(x.shape[0], -1)
        weight = weight.detach().float().cpu().reshape(1, -1)
        weight = weight.expand(samples.shape[0], -1)
        bias_column = torch.full((samples.shape[0], 1), float(bias))
        return self.predict_proba_features(
            torch.cat((samples, weight, bias_column), dim=1)
        )

    def state_dict(self) -> dict[str, Any]:
        """Serialize the fitted reverse model without optimizer state."""
        return {
            "hidden_size": self.hidden_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "training_mode": self.training_mode,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "feature_dim": self.feature_dim,
            "feature_mean": (
                None if self.feature_mean is None else self.feature_mean.clone()
            ),
            "feature_std": (
                None if self.feature_std is None else self.feature_std.clone()
            ),
            "model": None
            if self.model is None
            else {
                key: value.detach().cpu().clone()
                for key, value in self.model.state_dict().items()
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a previously fitted reverse model."""
        self.hidden_size = int(state.get("hidden_size", self.hidden_size))
        self.epochs = int(state.get("epochs", self.epochs))
        self.learning_rate = float(state.get("learning_rate", self.learning_rate))
        self.seed = int(state.get("seed", self.seed))
        self.batch_size = int(state.get("batch_size", self.batch_size))
        self.training_mode = state.get("training_mode", self.training_mode)
        self.num_heads = int(state.get("num_heads", self.num_heads))
        self.num_layers = int(state.get("num_layers", self.num_layers))
        feature_dim = state.get("feature_dim")
        model_state = state.get("model")
        feature_mean = state.get("feature_mean")
        feature_std = state.get("feature_std")
        if (
            feature_dim is None
            or model_state is None
            or feature_mean is None
            or feature_std is None
        ):
            self.model = None
            self.feature_dim = None
            self.feature_mean = None
            self.feature_std = None
            return
        self.feature_dim = int(feature_dim)
        self.feature_mean = feature_mean.detach().cpu().clone()
        self.feature_std = feature_std.detach().cpu().clone()
        if self.training_mode == "binary":
            model: nn.Module = _BinaryFeatureReverseModel(
                self.feature_dim, self.hidden_size
            )
        else:
            model = _FeatureReverseModel(
                self.feature_dim,
                self.hidden_size,
                self.num_heads,
                self.num_layers,
            )
        model.load_state_dict(model_state)
        self.model = model.eval()
