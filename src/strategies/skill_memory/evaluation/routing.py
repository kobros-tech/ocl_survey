"""Input-only evaluation and routing helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RoutingResult:
    """Input-only skill-routing result for one minibatch."""

    skill_indices: Tensor
    probabilities: Tensor
    best_probability: Tensor
    second_probability: Tensor
    confidence_gap: Tensor


def _routing_scores(
    raw_skill_logits: Sequence[Tensor],
    states: Sequence[Mapping[str, torch.Tensor]],
    skill_classes: Sequence[Sequence[int]],
) -> torch.Tensor:
    """Score each skill by probability mass on its owned classes.

    ``logits`` here are a skill's own *raw*, unpadded forward-pass output
    (see the caller in ``skill_memory_plugin.after_eval_forward``), not a
    globally-padded tensor. Avalanche's ``IncrementalClassifier`` indexes its
    output units directly by raw class label (verified against
    ``avalanche.models.IncrementalClassifier`` - a skill's classifier only
    ever grows to cover the classes it was actually trained on), so once a
    skill owns a class, that class's column must already exist in the
    skill's own raw logits. If it doesn't, class bookkeeping (``skill_classes``)
    and the model's own output space have silently drifted apart - most
    likely because the skill's classifier was queried with input from a
    class it was never trained on, or a benchmark relabels class ids from
    zero per experience, breaking the whole class_id convention. Either way,
    silently treating that class as "not present" would make the skill score
    zero and never win routing regardless of how well it actually matches
    the input, which looks exactly like a routing failure rather than a
    bookkeeping bug - so this raises instead of silently dropping the class.
    """
    del states

    scores = []
    for logits, owned_classes in zip(raw_skill_logits, skill_classes, strict=False):
        if not owned_classes:
            scores.append(torch.zeros(logits.shape[0], device=logits.device))
            continue

        if logits.shape[1] == 1:
            # A one-unit head is a binary "is this the owned class" score,
            # not a per-class-id column - it doesn't use the class_id-as-
            # column-index convention the strict check below assumes.
            scores.append(torch.sigmoid(logits[:, 0]))
            continue

        out_of_range = sorted(
            class_id
            for class_id in owned_classes
            if not 0 <= class_id < logits.shape[1]
        )
        if out_of_range:
            raise RuntimeError(
                f"skill owns classes {out_of_range} but its raw output only "
                f"has {logits.shape[1]} columns; class bookkeeping and the "
                "model's own output space have drifted apart (see "
                "_routing_scores docstring)"
            )
        valid_classes = sorted(owned_classes)

        probabilities = torch.softmax(logits, dim=1)
        scores.append(probabilities[:, valid_classes].sum(dim=1))

    if not scores:
        raise RuntimeError("No skills available for probe routing.")
    return torch.stack(scores, dim=0)


def _normalize_routing_scores(scores: Tensor, temperature: float) -> Tensor:
    """Normalize bounded routing scores across candidate skills."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    powered = scores.clamp_min(0).pow(1.0 / temperature)
    totals = powered.sum(dim=0, keepdim=True)
    eps = torch.finfo(powered.dtype).eps
    probabilities = powered / totals.clamp_min(eps)

    zero_total = totals.squeeze(0) <= 0
    if zero_total.any():
        probabilities = probabilities.clone()
        probabilities[:, zero_total] = 1.0 / probabilities.shape[0]
    return probabilities


def find_best_routing_skill(
    raw_skill_logits: list[Tensor],
    states: list[Mapping[str, torch.Tensor]],
    skill_classes: list[set[int]],
    temperature: float = 1.0,
) -> RoutingResult:
    """Select the best stored skill for every unlabeled probe sample."""
    scores = _routing_scores(raw_skill_logits, states, skill_classes)
    probabilities = _normalize_routing_scores(scores, temperature)
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
    states: list[Mapping[str, torch.Tensor]],
    skill_classes: list[set[int]],
) -> Tensor:
    """Compatibility wrapper returning only selected skill indices."""
    return find_best_routing_skill(
        raw_skill_logits, states, skill_classes
    ).skill_indices
