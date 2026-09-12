"""Per-class Skill Memory decisions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from torch import nn
from torch.utils.data import ConcatDataset

from .probing import (
    _sample_batches,
    class_subset,
    evaluate_state,
    incremental_out_features,
    probe_class,
)


def _probe_class_across(experiences, target_class, batch_size, n_batches, seed=None):
    """Pool only ``target_class`` samples from every matching experience."""
    subsets = []
    for experience in experiences:
        try:
            subsets.append(class_subset(experience, target_class))
        except RuntimeError:
            continue
    if not subsets:
        raise RuntimeError(f"class {target_class} not found in seen experiences")
    dataset = subsets[0] if len(subsets) == 1 else ConcatDataset(subsets)
    return _sample_batches(dataset, batch_size, n_batches, seed)


def _first_experience_with_class(experiences, target_class):
    for experience in experiences:
        try:
            class_subset(experience, target_class)
            return experience
        except RuntimeError:
            continue
    return None


def _strongest_candidates(results, key, floor):
    ranked = sorted(results, key=lambda result: result[key], reverse=True)
    if not ranked:
        return set()
    if len(ranked) == 1:
        return {ranked[0]["skill"]} if ranked[0][key] > floor else set()

    values = [result[key] for result in ranked]
    gaps = [values[i] - values[i + 1] for i in range(len(values) - 1)]
    split = max(range(len(gaps)), key=gaps.__getitem__)
    if gaps[split] <= 0:
        return set()

    return {
        result["skill"]
        for result in ranked[: split + 1]
        if result[key] > floor
    }


def find_best_skill(
    imagination_results: list[dict[str, Any]],
    forgetting_margin: float,
    score_floor: float | None = 0.9,
):
    """Select a new-class skill only when all its old classes remain safe."""
    if not imagination_results:
        return None

    safe_results = [
        result
        for result in imagination_results
        if result["old_accuracy"] > result["chance"] + forgetting_margin
    ]
    if not safe_results:
        return None

    floor_score = (
        max(result["chance"] for result in safe_results)
        if score_floor is None
        else score_floor
    )
    floor_accuracy = max(result["chance"] for result in safe_results)

    score_candidates = _strongest_candidates(safe_results, "new_score", floor_score)
    accuracy_candidates = _strongest_candidates(
        safe_results, "new_accuracy", floor_accuracy
    )
    intersection = score_candidates & accuracy_candidates
    if not intersection:
        return None

    candidates = [
        result for result in safe_results if result["skill"] in intersection
    ]
    return max(candidates, key=lambda r: (r["new_score"], r["new_accuracy"]))


def score_class_against_skills(
    strategy,
    experience,
    target_class: int,
    memory,
    class_map,
    probe_batch_size: int,
    probe_batches: int,
    probe_seed: int | None,
    seen_experiences: list,
) -> list[dict[str, Any]]:
    """Probe every skill against one *new* class.

    New-class evidence is sampled only from ``target_class`` in the current
    experience.  Old-skill safety is evaluated against *every class currently
    mastered by that skill*.  The aggregate ``old_accuracy`` is the worst
    old-class accuracy, so a mutable REUSE cannot hide forgetting of one of
    its existing classes behind good performance on another.
    """
    new_x, new_y = probe_class(
        experience, target_class, probe_batch_size, probe_batches, probe_seed
    )
    model_factory = lambda: deepcopy(strategy.model)
    results = []

    for slot in sorted(memory.slots()):
        state_dict = memory.state(slot)
        mastered_classes = sorted(class_map.classes_for_skill(slot))
        if not mastered_classes:
            continue

        old_metrics = []
        for old_class in mastered_classes:
            old_experience = _first_experience_with_class(
                seen_experiences, old_class
            )
            if old_experience is None:
                old_metrics = []
                break
            try:
                old_x, old_y = _probe_class_across(
                    seen_experiences,
                    old_class,
                    probe_batch_size,
                    probe_batches,
                    None if probe_seed is None else probe_seed + 100003 + old_class,
                )
            except RuntimeError:
                old_metrics = []
                break
            old_loss, old_score, old_accuracy = evaluate_state(
                model_factory,
                state_dict,
                old_x,
                old_y,
                nn.functional.cross_entropy,
                old_experience,
            )
            old_metrics.append(
                {
                    "class": old_class,
                    "loss": old_loss,
                    "score": old_score,
                    "accuracy": old_accuracy,
                }
            )

        if not old_metrics:
            continue

        new_loss, new_score, new_accuracy = evaluate_state(
            model_factory,
            state_dict,
            new_x,
            new_y,
            nn.functional.cross_entropy,
            experience,
        )
        out_features = incremental_out_features(strategy.model, state_dict)
        chance = 1.0 / out_features if out_features else 0.0

        results.append(
            {
                "skill": slot,
                "class": target_class,
                "old_classes": mastered_classes,
                "old_metrics": old_metrics,
                "old_accuracy": min(m["accuracy"] for m in old_metrics),
                "old_score": min(m["score"] for m in old_metrics),
                "new_loss": new_loss,
                "new_score": new_score,
                "new_accuracy": new_accuracy,
                "chance": chance,
            }
        )

    return results


def known_class_decision(target_class: int, skill: int) -> dict[str, Any]:
    """Return the deterministic decision for a previously mastered class."""
    return {
        "class": target_class,
        "decision": "reuse",
        "skill": skill,
        "new_score": 0.0,
        "old_accuracy": 0.0,
        "new_accuracy": 0.0,
        "results": [],
        "known_class": True,
    }


def decide_class(
    strategy,
    experience,
    target_class: int,
    memory,
    class_map,
    probe_batch_size: int,
    probe_batches: int,
    probe_seed: int | None,
    seen_experiences: list,
    forgetting_margin: float,
    score_floor: float | None,
    force_decision: str | None,
    logger_fn,
) -> dict[str, Any]:
    """Decide for one class, never for an entire multi-class experience.

    If the class was already mastered, its canonical class->skill mapping
    wins.  The generic imagination search is only for genuinely new classes.
    """
    known_skill = class_map.find_skill_for_class_anywhere(target_class)
    if known_skill is not None and force_decision != "scratch":
        decision = known_class_decision(target_class, known_skill)
        logger_fn(
            f"Class {target_class}: REUSE known skill {known_skill} "
            "(canonical class mapping)"
        )
        return decision

    results = score_class_against_skills(
        strategy,
        experience,
        target_class,
        memory,
        class_map,
        probe_batch_size,
        probe_batches,
        probe_seed,
        seen_experiences,
    )

    logger_fn(f"\nImagination for class {target_class}:")
    for result in results:
        logger_fn(
            f"  skill {result['skill']} (classes={result['old_classes']}): "
            f"old_acc={result['old_accuracy']:.3f}, "
            f"new_score={result['new_score']:.3f}, "
            f"new_acc={result['new_accuracy']:.3f}"
        )

    best = None
    if force_decision is None:
        best = find_best_skill(results, forgetting_margin, score_floor)
    elif force_decision == "reuse" and results:
        best = max(results, key=lambda r: (r["new_score"], r["new_accuracy"]))

    decision: dict[str, Any] = {
        "class": target_class,
        "decision": "scratch",
        "skill": None,
        "new_score": 0.0,
        "old_accuracy": 0.0,
        "new_accuracy": 0.0,
        "results": results,
        "known_class": False,
    }

    if best is not None:
        decision.update(
            {
                "decision": "reuse",
                "skill": best["skill"],
                "new_score": best["new_score"],
                "old_accuracy": best["old_accuracy"],
                "new_accuracy": best["new_accuracy"],
            }
        )
        logger_fn(
            f"Class {target_class}: REUSE skill {best['skill']} "
            f"(score={best['new_score']:.3f}, accuracy={best['new_accuracy']:.3f})"
        )
    else:
        logger_fn(f"Class {target_class}: no compatible skill -> SCRATCH")

    return decision
