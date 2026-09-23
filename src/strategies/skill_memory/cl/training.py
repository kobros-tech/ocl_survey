"""Single-class training loop."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ..utils.probing import class_subset


def train_on_class(
    strategy, experience, target_class: int, epochs: int, batch_size: int
) -> None:
    """Train only on samples whose label equals ``target_class``.

    This loop bypasses Avalanche's normal training-iteration machinery because
    Skill Memory trains one class at a time. The training clock therefore has
    to be advanced explicitly so evaluation checkpoints receive distinct
    ``mb_index`` values in the JSON logger.
    """
    dataset = class_subset(experience, target_class)
    if len(dataset) == 0:
        raise RuntimeError(f"class {target_class} has no samples to train on")
    if epochs < 1:
        return

    device = next(strategy.model.parameters()).device
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
    )

    criterion = getattr(strategy, "_criterion", None)
    if criterion is None:
        criterion = torch.nn.functional.cross_entropy

    strategy.model.train()
    for _ in range(epochs):
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            strategy.optimizer.zero_grad()
            logits = strategy.model(x)
            loss = criterion(logits, y)
            loss.backward()
            strategy.optimizer.step()

            # This custom loop bypasses Avalanche's normal training
            # iteration events, so BaseStrategy cannot advance the clock.
            # JSONLogger uses this clock to distinguish evaluation
            # checkpoints. Without it, later evaluations overwrite earlier
            # records because they receive the same mb_index.
            strategy.clock.train_iterations += 1
