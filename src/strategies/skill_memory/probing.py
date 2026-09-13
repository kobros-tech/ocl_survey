"""Class-aware probing and model-state helpers.

Training-time probing is always class-scoped.  The only intentionally
multi-class probe is ``probe_whole_experience``, reserved for blind
input-only evaluation routing.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from avalanche.models.dynamic_modules import IncrementalClassifier, avalanche_model_adaptation
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset


def origin_experience(experience):
    return getattr(experience, "origin_experience", experience)


# Cache of {id(dataset): {class_id: [indices]}}, built once per dataset the
# first time it's touched and reused for every later query.
#
# WHY THIS EXISTS: `decide_class` probes every stored skill against every
# seen experience for every new class. Without caching, that means the
# WHOLE dataset gets re-decoded (every `__getitem__`, i.e. every image
# load + transform) from scratch on every single one of those comparisons
# -- an O(skills x seen_experiences x dataset_size) cost that compounds
# every experience and is what actually made runs stall/time out on
# Split-CIFAR-100 (see the CI log: each newly-reached experience took
# longer than the last -- 20s, 70s, 146s, 267s, ... -- while every
# previously-seen one stayed ~0.5s, which is exactly the signature of an
# uncached O(n^2)-ish re-scan).
#
# Safe to key by `id(dataset)`: every experience whose class this cache
# might be asked about is kept alive for the plugin's whole lifetime via
# `_seen_experiences`, so its id can't be recycled out from under us.
_CLASS_INDEX_CACHE: dict[int, dict[int, list[int]]] = {}


def _cheap_targets(dataset):
    """Best-effort O(1)-per-sample label access with no decoding.

    Most torchvision-style datasets (and Subset wrappers around them)
    expose a `.targets` list/array. Walk through Subset wrappers to find
    it. Returns None if no such cheap accessor exists anywhere in the
    chain, in which case the caller falls back to `dataset[i]` -- but
    still only once per dataset, thanks to the cache above.
    """
    current = dataset
    index_map = None  # composition of Subset .indices, outermost last
    while True:
        targets = getattr(current, "targets", None)
        if targets is not None:
            targets = list(targets)
            if index_map is None:
                return targets
            return [int(targets[i]) for i in index_map]
        if isinstance(current, Subset):
            index_map = current.indices if index_map is None else [
                current.indices[i] for i in index_map
            ]
            current = current.dataset
            continue
        return None


def _class_index_map(experience) -> dict[int, list[int]]:
    dataset = experience.dataset
    cache_key = id(dataset)
    cached = _CLASS_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    targets = _cheap_targets(dataset)
    mapping: dict[int, list[int]] = {}
    if targets is not None and len(targets) == len(dataset):
        for index, label in enumerate(targets):
            mapping.setdefault(int(label), []).append(index)
    else:
        # No cheap label accessor available anywhere in the chain: this
        # dataset type genuinely requires decoding each sample once to
        # read its label. Still only paid ONCE per dataset (cached),
        # never again for every later (skill, class) comparison.
        for index in range(len(dataset)):
            mapping.setdefault(int(dataset[index][1]), []).append(index)

    _CLASS_INDEX_CACHE[cache_key] = mapping
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
        raise RuntimeError(
            f"class {target_class} has no samples in this experience"
        )
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


def probe_class(experience, target_class: int, batch_size: int, n_batches: int, seed=None):
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


def incremental_out_features(model: nn.Module, state_dict: Mapping[str, Tensor]) -> int | None:
    for module_name, module in model.named_modules():
        if not isinstance(module, IncrementalClassifier):
            continue
        prefix = f"{module_name}." if module_name else ""
        weight = state_dict.get(f"{prefix}classifier.weight")
        if weight is not None and weight.ndim == 2:
            return int(weight.shape[0])
    return None


def restore_initial_state(model: nn.Module, initial_state: Mapping[str, Tensor]) -> None:
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


def apply_skill_state(model: nn.Module, state_dict: Mapping[str, Tensor], experience) -> None:
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
        accuracy = float((logits.argmax(dim=1) == y).float().mean().item())
    return loss, score_from_loss(loss), accuracy




def _classifier_prefix_and_active_units(state):
    """Find the classifier state prefix and its active class ids."""
    for key, value in state.items():
        if key.endswith("classifier.weight") and getattr(value, "ndim", 0) == 2:
            prefix = key[: -len("classifier.weight")]
            active = state.get(f"{prefix}active_units")
            return prefix, active
    return None, None


def expand_skill_logits(
    logits: Tensor,
    state: Mapping[str, Tensor],
    mastered_classes,
    output_dim: int,
) -> Tensor:
    """Pad a skill's local classifier logits into the current global head.

    Skill snapshots can have different classifier sizes. Metrics, however,
    expect one common output dimension. Known class rows are copied into that
    global space; unknown rows are set to ``-inf`` so a skill cannot win a
    routing decision by predicting a class it never mastered.
    """
    if output_dim < 1:
        raise ValueError("output_dim must be positive")

    expanded = logits.new_full((logits.shape[0], output_dim), float("-inf"))
    _, active_units = _classifier_prefix_and_active_units(state)
    if active_units is not None:
        class_ids = [int(v) for v in active_units.detach().cpu().tolist()]
    else:
        class_ids = list(range(logits.shape[1]))

    # Prefer the explicit registry mapping when the snapshot's active-unit
    # metadata is unavailable or inconsistent. This keeps skill indices and
    # class indices conceptually separate.
    if len(class_ids) != logits.shape[1]:
        class_ids = sorted(int(c) for c in mastered_classes)

    for row, class_id in enumerate(class_ids[: logits.shape[1]]):
        if 0 <= class_id < output_dim:
            expanded[:, class_id] = logits[:, row]
    return expanded

def route_probe_logits(logits_by_skill, states, mastered_classes=None):
    """Choose one skill per sample using label-free normalized margins.

    ``logits_by_skill`` contains one ``[batch, classes]`` tensor per stored
    skill. A one-class head has no entropy/margin against a runner-up, so its
    scalar evidence is its single logit. Multi-class heads use the top-two
    logit margin. Both are normalized by the stored classifier weight norm to
    reduce scale differences between independently trained skill snapshots.

    ``mastered_classes`` is bookkeeping only; it is used to distinguish a
    one-class skill from a genuinely multi-class head and is never populated
    from evaluation targets.

    This is intentionally a routing heuristic, not an accuracy oracle: no
    target labels are inspected.
    """
    if not logits_by_skill:
        raise ValueError("at least one skill is required")
    if len(logits_by_skill) != len(states):
        raise ValueError("logits_by_skill and states must have the same length")
    if mastered_classes is None:
        mastered_classes = [None] * len(states)
    if len(mastered_classes) != len(states):
        raise ValueError("mastered_classes and states must have the same length")

    scores = []
    for logits, state, classes in zip(logits_by_skill, states, mastered_classes):
        weight_keys = [key for key in state if key.endswith("classifier.weight")]
        if weight_keys:
            scale = state[weight_keys[0]].float().norm().clamp_min(1e-8)
        else:
            scale = logits.new_tensor(1.0)

        if logits.shape[1] == 1:
            score = logits[:, 0] / scale.to(device=logits.device)
        elif classes is not None and len(classes) == 1:
            score = logits[:, 0] / scale.to(device=logits.device)
        else:
            top2 = torch.topk(logits, k=2, dim=1).values
            score = (top2[:, 0] - top2[:, 1]) / scale.to(device=logits.device)
        scores.append(score)

    return torch.stack(scores, dim=0).argmax(dim=0)

def predictive_entropy(model, state_dict, experience, x) -> float:
    """Same reuse-one-scratch-model contract as `evaluate_state` above."""
    apply_skill_state(model, state_dict, experience)
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).mean()
    return float(entropy.item())


def predict_logits(model, state_dict: Mapping[str, Tensor], x) -> Tensor:
    """Return per-sample logits under one stored skill snapshot."""
    apply_skill_state_exact(model, state_dict)
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device)
    with torch.no_grad():
        return model(x)
