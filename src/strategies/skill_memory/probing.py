"""Class-aware probing and model-state helpers.

This module only samples data and applies/evaluates stored model states.
Routing and decision logic belongs to evaluation and decision modules.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from avalanche.models.dynamic_modules import (
    IncrementalClassifier,
    avalanche_model_adaptation,
)
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset


def origin_experience(experience):
    return getattr(experience, "origin_experience", experience)


_CLASS_INDEX_CACHE: dict[int, tuple[object, dict[int, list[int]]]] = {}


def _class_index_map(experience) -> dict[int, list[int]]:
    """Build the class-to-sample index map from actual dataset samples."""
    dataset = experience.dataset
    cache_key = id(dataset)
    cached = _CLASS_INDEX_CACHE.get(cache_key)
    if cached is not None:
        cached_dataset, mapping = cached
        if cached_dataset is dataset:
            return mapping

    mapping: dict[int, list[int]] = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        if len(sample) < 2:
            raise RuntimeError(
                "Experience dataset samples must contain at least input and label"
            )
        mapping.setdefault(int(sample[1]), []).append(index)

    _CLASS_INDEX_CACHE[cache_key] = (dataset, mapping)
    return mapping


def classes_in_experience(experience) -> list[int]:
    """Return distinct labels actually present in the experience dataset."""
    return sorted(_class_index_map(experience))


def class_indices(experience, target_class: int) -> list[int]:
    indices = _class_index_map(experience).get(int(target_class))
    if not indices:
        raise RuntimeError(f"class {target_class} has no samples in this experience")
    return list(indices)


def class_subset(experience, target_class: int) -> Subset:
    """Return only samples belonging to ``target_class``."""
    return Subset(experience.dataset, class_indices(experience, target_class))


def _sample_batches(dataset, batch_size: int, n_batches: int, seed: int | None):
    if len(dataset) == 0:
        raise RuntimeError("Cannot probe an empty dataset")
    if batch_size < 1 or n_batches < 1:
        raise ValueError("batch_size and n_batches must be positive")

    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
    )
    xs, ys = [], []
    iterator = iter(loader)
    for _ in range(n_batches):
        try:
            x, y = next(iterator)[:2]
        except StopIteration:
            break
        xs.append(x)
        ys.append(y)
    if not xs:
        raise RuntimeError("Probe loader produced no batches")
    return torch.cat(xs), torch.cat(ys)


def probe_class(
    experience,
    target_class: int,
    batch_size: int,
    n_batches: int,
    seed=None,
):
    return _sample_batches(
        class_subset(experience, target_class), batch_size, n_batches, seed
    )


def probe_whole_experience(experience, batch_size: int, n_batches: int, seed=None):
    """Return an unfiltered input/label probe for evaluation callers."""
    return _sample_batches(experience.dataset, batch_size, n_batches, seed)


def score_from_loss(loss_value: float) -> float:
    return float(np.exp(-loss_value))


def resize_incremental_classifiers_for_state(
    model: nn.Module, state_dict: Mapping[str, Tensor]
) -> None:
    """Resize IncrementalClassifier heads to the stored snapshot shape."""
    for module_name, module in model.named_modules():
        if not isinstance(module, IncrementalClassifier):
            continue
        prefix = f"{module_name}." if module_name else ""
        target_weight = state_dict.get(f"{prefix}classifier.weight")
        if target_weight is None or target_weight.ndim != 2:
            continue
        if module.classifier.out_features == target_weight.shape[0]:
            continue

        device = module.classifier.weight.device
        dtype = module.classifier.weight.dtype
        module.classifier = nn.Linear(
            module.classifier.in_features,
            target_weight.shape[0],
        ).to(device=device, dtype=dtype)
        active_key = f"{prefix}active_units"
        if active_key in state_dict:
            module.active_units = state_dict[active_key].to(device=device).clone()


def incremental_out_features(
    model: nn.Module, state_dict: Mapping[str, Tensor]
) -> int | None:
    for module_name, module in model.named_modules():
        if not isinstance(module, IncrementalClassifier):
            continue
        prefix = f"{module_name}." if module_name else ""
        weight = state_dict.get(f"{prefix}classifier.weight")
        if weight is not None and weight.ndim == 2:
            return int(weight.shape[0])
    return None


def incremental_active_units(
    model: nn.Module, state_dict: Mapping[str, Tensor]
) -> Tensor | None:
    """Return the stored ``active_units`` mask for a skill's classifier head."""
    for module_name, module in model.named_modules():
        if not isinstance(module, IncrementalClassifier):
            continue
        prefix = f"{module_name}." if module_name else ""
        active = state_dict.get(f"{prefix}active_units")
        if active is not None:
            return active
    return None


def restore_initial_state(
    model: nn.Module, initial_state: Mapping[str, Tensor]
) -> None:
    current = model.state_dict()
    for name, initial in initial_state.items():
        if name not in current:
            continue
        target = current[name]
        value = initial.to(device=target.device, dtype=target.dtype)
        if target.shape == value.shape:
            target.copy_(value)
        elif (name.endswith("classifier.weight") and target.ndim == 2) or (
            name.endswith("classifier.bias") and target.ndim == 1
        ):
            rows = min(target.shape[0], value.shape[0])
            target[:rows].copy_(value[:rows])
        elif name.endswith("active_units"):
            continue
        else:
            raise RuntimeError(f"Cannot restore initial state for {name}")


def prepare_for_experience(model: nn.Module, experience) -> None:
    """Apply Avalanche's current-experience head adaptation."""
    avalanche_model_adaptation(model, origin_experience(experience))


def apply_skill_state(
    model: nn.Module, state_dict: Mapping[str, Tensor], experience
) -> None:
    """Load a skill, then adapt its head for the current experience."""
    resize_incremental_classifiers_for_state(model, state_dict)
    model.load_state_dict(state_dict, strict=False)
    prepare_for_experience(model, experience)


def apply_skill_state_exact(model: nn.Module, state_dict: Mapping[str, Tensor]) -> None:
    resize_incremental_classifiers_for_state(model, state_dict)
    model.load_state_dict(state_dict, strict=False)


def evaluate_state(
    model, state_dict, x, y, criterion, adaptation_experience=None
) -> tuple[float, float, float]:
    """Evaluate a stored snapshot on already-filtered class data."""
    apply_skill_state_exact(model, state_dict)
    if adaptation_experience is not None:
        prepare_for_experience(model, adaptation_experience)
    model.eval()
    device = next(model.parameters()).device
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        logits = model(x)
        loss = float(criterion(logits, y).item())
        predictions = logits.argmax(dim=1)
        accuracy = float((predictions == y).float().mean().item())
    return loss, score_from_loss(loss), accuracy


def predict_logits(model, state_dict: Mapping[str, Tensor], x: Tensor) -> Tensor:
    apply_skill_state_exact(model, state_dict)
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        return model(x.to(device))


def expand_skill_logits(
    logits: Tensor,
    state_dict: Mapping[str, Tensor],
    skill_classes: set[int],
    output_dim: int,
) -> Tensor:
    """Pad global classifier logits without remapping class columns."""
    del state_dict, skill_classes
    result = logits.new_full((logits.shape[0], output_dim), -1e4)
    width = min(logits.shape[1], output_dim)
    result[:, :width] = logits[:, :width]
    return result


# Compatibility re-exports only. The implementations live in evaluation.py;
# probing itself contains no routing/decision logic. The `as`-aliases below
# are required so linters treat these as intentional re-exports rather than
# unused imports; removing them silently breaks `skill_memory_plugin.py`'s
# `from .probing import find_best_routing_skill`.
from .evaluation import RoutingResult as RoutingResult  # noqa: E402
from .evaluation import find_best_routing_skill as find_best_routing_skill  # noqa: E402
from .evaluation import route_probe_logits as route_probe_logits  # noqa: E402
