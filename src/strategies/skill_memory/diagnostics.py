"""Optional, intentionally expensive diagnostics for anonymous routing."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from torch import Tensor, nn

from .probing import incremental_active_units, incremental_out_features


def routing_rank_diagnostics(routes: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute rank and confusion diagnostics from retained route records.

    This helper is deliberately separate from evaluation so normal routing does
    not need to compute or retain diagnostic aggregates. Call it only when
    ``diagnose=True`` and route history is available.
    """
    if not routes:
        return {
            "samples": 0,
            "top1_accuracy": 0.0,
            "top2_accuracy": 0.0,
            "top3_accuracy": 0.0,
            "mean_reciprocal_rank": 0.0,
            "mean_correct_class_rank": 0.0,
        }

    ranks: list[int] = []
    confusion = Counter()
    for route in routes:
        true_class = route.get("evaluation_y")
        candidates = route.get("candidates", [])
        if true_class is None or not candidates:
            continue
        ordered = sorted(
            candidates,
            key=lambda item: item.get("score", float("-inf")),
            reverse=True,
        )
        rank = next(
            (
                index
                for index, candidate in enumerate(ordered, start=1)
                if int(candidate.get("class", -1)) == int(true_class)
            ),
            len(ordered) + 1,
        )
        ranks.append(rank)
        predicted = int(ordered[0].get("class", -1))
        if predicted != int(true_class):
            confusion[(int(true_class), predicted)] += 1

    if not ranks:
        return {"samples": 0}

    n = len(ranks)
    return {
        "samples": n,
        "top1_accuracy": sum(rank <= 1 for rank in ranks) / n,
        "top2_accuracy": sum(rank <= 2 for rank in ranks) / n,
        "top3_accuracy": sum(rank <= 3 for rank in ranks) / n,
        "mean_reciprocal_rank": sum(1.0 / rank for rank in ranks) / n,
        "mean_correct_class_rank": sum(ranks) / n,
        "confusion_pairs": [
            {"true_class": true_class, "wrong_class": wrong_class, "count": count}
            for (true_class, wrong_class), count in confusion.most_common()
        ],
    }


def class_index_alignment_report(
    model: nn.Module,
    states: Sequence[Mapping[str, Tensor]],
    slot_ids: Sequence[int],
    owned_classes_by_slot: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Report, per skill, whether its owned global class ids fit its own
    classifier's output space.

    This is the diagnostic the reverse-router and probe-routing correctness
    fixes both depend on: a skill's own classifier is only guaranteed to
    have a column for a class if that class was actually trained on that
    specific skill's classifier module (Avalanche's ``IncrementalClassifier``
    indexes output units directly by raw class label - see
    ``persistent_skill_memory_plugin._fit_reverse_router`` and
    ``evaluation._routing_scores`` for where this assumption is load-bearing).
    A benchmark that relabels classes to start from zero within each
    experience breaks that assumption; this report is the fastest way to
    tell which situation a given run is actually in, by inspecting each
    skill's *stored* classifier snapshot directly (no forward pass needed).

    Call this once, e.g. at the end of training, with ``states`` = the
    frozen snapshot for each slot in ``slot_ids`` (``memory.state(slot)``)
    and ``owned_classes_by_slot`` = the recorded classes for each slot
    (``class_map.classes_for_skill(slot)``), in the same order as
    ``slot_ids``.
    """
    per_skill: list[dict[str, Any]] = []
    any_misaligned = False

    for slot_id, state_dict, owned_classes in zip(
        slot_ids, states, owned_classes_by_slot, strict=True
    ):
        owned = sorted(int(c) for c in owned_classes)
        width = incremental_out_features(model, state_dict)
        active_units = incremental_active_units(model, state_dict)
        active_indices = (
            sorted(int(i) for i in active_units.nonzero().flatten().tolist())
            if active_units is not None
            else None
        )

        out_of_range = (
            [c for c in owned if width is None or not 0 <= c < width] if owned else []
        )
        misaligned = bool(out_of_range)
        any_misaligned = any_misaligned or misaligned

        per_skill.append(
            {
                "skill_id": int(slot_id),
                "owned_global_classes": owned,
                "logits_width": width,
                "active_units": active_indices,
                "misaligned": misaligned,
                "out_of_range_classes": out_of_range,
            }
        )

    return {
        "skills": per_skill,
        "any_misaligned": any_misaligned,
        "verdict": (
            "global-index alignment holds: every skill's own classifier "
            "already has a column for every class it owns"
            if not any_misaligned
            else "global-index alignment is BROKEN for at least one skill: "
            "see out_of_range_classes per skill below"
        ),
    }
