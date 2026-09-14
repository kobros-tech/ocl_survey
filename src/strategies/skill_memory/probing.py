"""Class-aware probing and model-state helpers.

Training-time probing is always class-scoped.  The only intentionally
multi-class probe is ``probe_whole_experience``, reserved for blind
input-only evaluation routing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

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


# Cache of {id(dataset): (dataset, {class_id: [indices]})}. Keeping the
# dataset object in the value prevents Python from recycling its id while a
# stale cache entry is still alive. This matters for Avalanche sub-experiences:
# their dataset wrappers can be short-lived even though the logical experience
# continues across several sub-experiences.
#
# Labels are read from the actual dataset samples rather than an optional
# `.targets` attribute. Avalanche's FlatData/Subsets can expose metadata whose
# indexing semantics do not necessarily match the logical experience dataset.
_CLASS_INDEX_CACHE: dict[int, tuple[object, dict[int, list[int]]]] = {}


def _class_index_map(experience) -> dict[int, list[int]]:
    """Build the class -> sample-index map from actual dataset samples."""
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
    """Return the distinct labels actually present in the dataset.

    We intentionally inspect dataset content rather than trusting an
    experience-level class declaration.  That prevents routing decisions
    from being based on metadata that can disagree with the samples.
    """
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
    """Unfiltered probe; use only when labels are intentionally ignored."""
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
        elif name.endswith("classifier.weight") and target.ndim == 2:
            rows = min(target.shape[0], value.shape[0])
            target[:rows].copy_(value[:rows])
        elif name.endswith("classifier.bias") and target.ndim == 1:
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
    """Evaluate a stored snapshot on already-filtered class data.

    ``model`` is a scratch module the caller builds ONCE (e.g. one
    `deepcopy(strategy.model)` per decision) and reuses across every
    candidate skill being scored; this function just overwrites its
    weights in place via `load_state_dict` rather than deep-copying the
    whole module again per candidate. Deep-copying a full model is far
    more expensive than swapping a state_dict, and doing it once per
    skill instead of once per decision is what made per-class decisions
    scale with the total number of stored skills.

    If the snapshot predates the current class head, ``adaptation_experience``
    is used only to grow the Avalanche dynamic head.  The probe data itself
    must already be class-filtered; the experience is never sampled here.
    """
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


@dataclass(frozen=True)
class RoutingResult:
    """Input-only skill-routing result for one minibatch.

    ``probabilities`` are softmax-normalized routing scores across the
    available skills. They are useful as relative routing confidence, but
    they are not calibrated probabilities of correctness.
    """

    skill_indices: Tensor
    probabilities: Tensor
    best_probability: Tensor
    second_probability: Tensor
    confidence_gap: Tensor


def _routing_scores(
    raw_skill_logits: list[Tensor],
    states: list[Mapping[str, Tensor]],
    skill_classes: list[set[int]],
) -> Tensor:
    """Return normalized routing scores with shape ``[skills, batch]``."""
    if not raw_skill_logits:
        raise RuntimeError("No skills available for probe routing")
    if len(raw_skill_logits) != len(states) or len(states) != len(skill_classes):
        raise ValueError("routing inputs must contain the same number of skills")

    scores = []
    for logits, _state, owned_classes in zip(
        raw_skill_logits, states, skill_classes, strict=False
    ):
        if not owned_classes:
            scores.append(logits.new_full((logits.shape[0],), -1e4))
            continue

        valid_classes = sorted(
            class_id for class_id in owned_classes if 0 <= class_id < logits.shape[1]
        )
        if not valid_classes:
            scores.append(logits.new_full((logits.shape[0],), -1e4))
            continue

        # Keep the v0.1.4 routing semantics: a skill is scored by its
        # strongest owned global class logit. The new function adds ranking
        # and normalized diagnostics without changing that routing signal.
        owned_logits = logits[:, valid_classes]
        score = owned_logits.max(dim=1).values
        scores.append(score)

    return torch.stack(scores, dim=0)


def find_best_routing_skill(
    raw_skill_logits: list[Tensor],
    states: list[Mapping[str, Tensor]],
    skill_classes: list[set[int]],
    temperature: float = 1.0,
) -> RoutingResult:
    """Find the best stored skill for every unlabeled probe sample.

    ``raw_skill_logits`` must contain one ``[batch, output_dim]`` tensor per
    stored skill. The function never receives labels. For every sample it
    computes the existing v0.1.4 owned-class routing score for every skill,
    applies a softmax over skills, and returns the winning skill together
    with top-1/top-2 routing diagnostics.

    The softmax values are normalized routing probabilities, not calibrated
    probabilities of correctness.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    scores = _routing_scores(raw_skill_logits, states, skill_classes)
    probabilities = torch.softmax(scores / temperature, dim=0)
    skill_indices = probabilities.argmax(dim=0)

    if probabilities.shape[0] == 1:
        best_probability = probabilities[0]
        second_probability = torch.zeros_like(best_probability)
    else:
        top2 = torch.topk(probabilities, k=2, dim=0).values
        best_probability = top2[0]
        second_probability = top2[1]

    return RoutingResult(
        skill_indices=skill_indices,
        probabilities=probabilities,
        best_probability=best_probability,
        second_probability=second_probability,
        confidence_gap=best_probability - second_probability,
    )


def route_probe_logits(
    raw_skill_logits: list[Tensor],
    states: list[Mapping[str, Tensor]],
    skill_classes: list[set[int]],
) -> Tensor:
    """Select the best skill for each probe sample.

    This compatibility wrapper preserves the v0.1.4 return type while the
    richer ``find_best_routing_skill`` API exposes routing probabilities and
    diagnostics.
    """
    return find_best_routing_skill(
        raw_skill_logits, states, skill_classes
    ).skill_indices
