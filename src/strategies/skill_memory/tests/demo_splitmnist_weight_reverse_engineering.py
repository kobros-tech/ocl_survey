"""SplitMNIST demo for anonymous routing from learned classifier weights."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.models.dynamic_modules import IncrementalClassifier
from avalanche.training.templates import SupervisedTemplate

from skill_memory import PersistentFingerprintSkillMemoryPlugin, SkillMemory


class Tee:
    """Write output to several streams at once."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


class SkillMemoryMLP(nn.Module):
    """Small MLP with an Avalanche growing classifier head."""

    def __init__(self, input_dim: int, hidden_size: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(inplace=True),
        )
        self.classifier = IncrementalClassifier(hidden_size, initial_out_features=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().view(x.size(0), -1)
        return self.classifier(self.features(x))


def evaluate_seen(strategy, test_stream, up_to_index: int) -> list[float]:
    """Evaluate seen experiences through the anonymous routing path."""
    results = strategy.eval([test_stream[i] for i in range(up_to_index + 1)])
    keys = sorted(key for key in results if key.startswith("Top1_Acc_Exp"))
    return [float(results[key]) for key in keys]


def _correct_candidate_rank(route: dict) -> int | None:
    """Return the 1-based rank of the true class among candidate scores."""
    true_class = route.get("evaluation_y")
    candidates = route.get("candidates", [])
    if true_class is None or not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda item: float(item.get("score", float("-inf"))),
        reverse=True,
    )
    for rank, candidate in enumerate(ranked, start=1):
        if candidate.get("class") == true_class:
            return rank
    return None


def flatten_route(route: dict, train_index: int) -> dict:
    """Convert one routing record into a compact analysis row."""
    candidates = route.get("candidates", [])
    ranked = sorted(
        candidates,
        key=lambda item: float(item.get("score", float("-inf"))),
        reverse=True,
    )
    top = ranked[0] if ranked else {}
    second = ranked[1] if len(ranked) > 1 else {}
    true_class = route.get("evaluation_y")
    true_candidate = next(
        (candidate for candidate in candidates if candidate.get("class") == true_class),
        {},
    )
    top_score = top.get("score")
    true_score = true_candidate.get("score")
    score_margin = (
        float(top_score) - float(true_score)
        if top_score is not None and true_score is not None
        else None
    )
    correct_rank = _correct_candidate_rank(route)
    return {
        "training_step": train_index,
        "evaluation_experience": route.get("evaluation_experience"),
        "batch_index": route.get("batch_index", -1),
        "sample_index": route.get("sample_index", -1),
        "evaluation_y": true_class,
        "status": route.get("status"),
        "inferred_class": route.get("class"),
        "selected_skill": route.get("skill"),
        "model_predicted_class": route.get("model_predicted_class"),
        "model_correct": route.get("model_correct"),
        "correct_class_rank": correct_rank,
        "top_candidate_class": top.get("class"),
        "top_candidate_skill": top.get("skill"),
        "top_candidate_correct": top.get("class") == true_class,
        "top_class_score": top_score,
        "top_probability": top.get("probability"),
        "true_class_score": true_score,
        "true_probability": true_candidate.get("probability"),
        "top_minus_true_score": score_margin,
        "second_candidate_class": second.get("class"),
        "second_candidate_skill": second.get("skill"),
        "second_candidate_correct": second.get("class") == true_class,
        "second_class_score": second.get("score"),
        "second_probability": second.get("probability"),
        "reference_accuracy": route.get("reference_accuracy"),
    }


def rank_summary(rows: list[dict]) -> dict:
    """Summarize true-class rank without feeding labels into routing."""
    ranks = [
        int(row["correct_class_rank"])
        for row in rows
        if row.get("status") == "IDENTIFIED"
        and row.get("correct_class_rank") is not None
    ]
    if not ranks:
        return {
            "samples": 0,
            "top1_accuracy": 0.0,
            "top2_accuracy": 0.0,
            "top3_accuracy": 0.0,
            "top5_accuracy": 0.0,
            "top10_accuracy": 0.0,
            "mean_reciprocal_rank": 0.0,
            "mean_correct_class_rank": 0.0,
            "rank_histogram": {},
        }

    return {
        "samples": len(ranks),
        "top1_accuracy": sum(rank <= 1 for rank in ranks) / len(ranks),
        "top2_accuracy": sum(rank <= 2 for rank in ranks) / len(ranks),
        "top3_accuracy": sum(rank <= 3 for rank in ranks) / len(ranks),
        "top5_accuracy": sum(rank <= 5 for rank in ranks) / len(ranks),
        "top10_accuracy": sum(rank <= 10 for rank in ranks) / len(ranks),
        "mean_reciprocal_rank": sum(1.0 / rank for rank in ranks) / len(ranks),
        "mean_correct_class_rank": float(np.mean(ranks)),
        "rank_histogram": {
            str(rank): ranks.count(rank) for rank in sorted(set(ranks))
        },
    }


def rank_summary_by_experience(rows: list[dict]) -> dict:
    """Return the same rank diagnostics separately for each eval experience."""
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        experience = row.get("evaluation_experience")
        if experience is not None:
            grouped.setdefault(str(experience), []).append(row)
    return {
        experience: rank_summary(experience_rows)
        for experience, experience_rows in sorted(grouped.items())
    }


def _score_margin_stats(rows: list[dict]) -> dict:
    """Summarize winner-vs-true score margins for Top-1 routing failures."""
    failures = [
        row
        for row in rows
        if row.get("status") == "IDENTIFIED"
        and row.get("correct_class_rank") is not None
        and int(row["correct_class_rank"]) > 1
        and row.get("top_minus_true_score") is not None
    ]
    margins = np.asarray(
        [float(row["top_minus_true_score"]) for row in failures],
        dtype=float,
    )
    if margins.size == 0:
        return {
            "samples": 0,
            "mean_margin": 0.0,
            "median_margin": 0.0,
            "min_margin": 0.0,
            "max_margin": 0.0,
        }
    return {
        "samples": int(margins.size),
        "mean_margin": float(np.mean(margins)),
        "median_margin": float(np.median(margins)),
        "min_margin": float(np.min(margins)),
        "max_margin": float(np.max(margins)),
    }


def score_margin_summary(rows: list[dict]) -> dict:
    """Diagnose how strongly wrong winners beat the true candidate."""
    failures = [
        row
        for row in rows
        if row.get("status") == "IDENTIFIED"
        and row.get("correct_class_rank") is not None
        and int(row["correct_class_rank"]) > 1
    ]
    confusion_counts: dict[str, int] = {}
    wrong_candidate_counts: dict[str, int] = {}
    for row in failures:
        true_class = row.get("evaluation_y")
        wrong_class = row.get("top_candidate_class")
        if true_class is not None and wrong_class is not None:
            pair = f"{true_class}->{wrong_class}"
            confusion_counts[pair] = confusion_counts.get(pair, 0) + 1
        if wrong_class is not None:
            key = str(wrong_class)
            wrong_candidate_counts[key] = wrong_candidate_counts.get(key, 0) + 1

    by_experience: dict[str, list[dict]] = {}
    for row in failures:
        experience = row.get("evaluation_experience")
        if experience is not None:
            by_experience.setdefault(str(experience), []).append(row)

    return {
        "top1_failures": len(failures),
        "overall": _score_margin_stats(rows),
        "by_evaluation_experience": {
            experience: _score_margin_stats(experience_rows)
            for experience, experience_rows in sorted(by_experience.items())
        },
        "most_frequent_confusion_pairs": [
            {"true_class_to_wrong_class": pair, "count": count}
            for pair, count in sorted(
                confusion_counts.items(), key=lambda item: (-item[1], item[0])
            )[:20]
        ],
        "most_frequent_wrong_candidates": [
            {"wrong_class": wrong_class, "count": count}
            for wrong_class, count in sorted(
                wrong_candidate_counts.items(),
                key=lambda item: (-item[1], int(item[0])),
            )[:20]
        ],
    }


def routing_summary(rows: list[dict]) -> dict:
    """Summarize routing and final-model correctness by eval experience."""
    by_experience: dict[str, dict] = {}
    for row in rows:
        experience = row.get("evaluation_experience")
        if experience is None:
            continue
        key = str(experience)
        summary = by_experience.setdefault(
            key,
            {
                "samples": 0,
                "identified": 0,
                "ambiguous": 0,
                "failed": 0,
                "identified_correct_class": 0,
                "identified_final_model_correct": 0,
                "final_model_correct": 0,
            },
        )
        summary["samples"] += 1
        summary[row["status"].lower()] += 1
        if row["status"] == "IDENTIFIED":
            summary["identified_correct_class"] += int(
                row["inferred_class"] == row["evaluation_y"]
            )
            summary["identified_final_model_correct"] += int(bool(row["model_correct"]))
        summary["final_model_correct"] += int(bool(row["model_correct"]))

    for summary in by_experience.values():
        identified = summary["identified"]
        samples = summary["samples"]
        summary["identified_class_accuracy"] = summary[
            "identified_correct_class"
        ] / max(identified, 1)
        summary["identified_final_model_accuracy"] = summary[
            "identified_final_model_correct"
        ] / max(identified, 1)
        summary["final_model_accuracy"] = summary["final_model_correct"] / max(
            samples, 1
        )
    return by_experience


def write_accuracy_matrix(
    log_dir: Path,
    run_id: str,
    accuracy_history: list[list[float]],
) -> Path:
    """Write train-step x evaluation-experience routed accuracy matrix."""
    path = log_dir / f"weight_reverse_engineering_accuracy_{run_id}.csv"
    max_experiences = max((len(row) for row in accuracy_history), default=0)
    fieldnames = ["training_step"] + [
        f"eval_exp_{index}" for index in range(max_experiences)
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for train_index, accuracies in enumerate(accuracy_history):
            row = {"training_step": train_index}
            row.update(
                {
                    f"eval_exp_{index}": accuracy
                    for index, accuracy in enumerate(accuracies)
                }
            )
            writer.writerow(row)
    return path


def write_analysis_files(
    log_dir: Path,
    run_id: str,
    rows: list[dict],
    accuracy_history: list[list[float]],
    accuracy_curve: np.ndarray,
    forgetting: np.ndarray,
) -> None:
    """Write artifacts that keep routing diagnostics separate from metrics."""
    csv_path = log_dir / f"weight_reverse_engineering_{run_id}.csv"
    json_path = log_dir / f"weight_reverse_engineering_{run_id}.json"
    matrix_path = write_accuracy_matrix(log_dir, run_id, accuracy_history)

    fieldnames = [
        "training_step",
        "evaluation_experience",
        "batch_index",
        "sample_index",
        "evaluation_y",
        "status",
        "inferred_class",
        "selected_skill",
        "model_predicted_class",
        "model_correct",
        "correct_class_rank",
        "top_candidate_class",
        "top_candidate_skill",
        "top_candidate_correct",
        "top_class_score",
        "top_probability",
        "true_class_score",
        "true_probability",
        "top_minus_true_score",
        "second_candidate_class",
        "second_candidate_skill",
        "second_candidate_correct",
        "second_class_score",
        "second_probability",
        "reference_accuracy",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    identified = sum(row["status"] == "IDENTIFIED" for row in rows)
    summary = {
        "rows": len(rows),
        "status_counts": {
            status: sum(row["status"] == status for row in rows)
            for status in ("IDENTIFIED", "AMBIGUOUS", "FAILED")
        },
        "identified_class_accuracy": (
            sum(
                row["status"] == "IDENTIFIED"
                and row["inferred_class"] == row["evaluation_y"]
                for row in rows
            )
            / max(identified, 1)
        ),
        "routed_accuracy_curve": accuracy_curve.tolist(),
        "routed_forgetting": forgetting.tolist(),
        "accuracy_matrix_csv": matrix_path.name,
        "routing_by_evaluation_experience": routing_summary(rows),
        "routing_rank_diagnostics": rank_summary(rows),
        "routing_rank_by_evaluation_experience": rank_summary_by_experience(rows),
        "routing_score_margin_diagnostics": score_margin_summary(rows),
        "analysis_csv": csv_path.name,
        "metric_scope": {
            "accuracy_curve": "anonymous routed model accuracy",
            "forgetting": "anonymous routed model forgetting",
            "identified_class_accuracy": (
                "class-identification accuracy conditional on IDENTIFIED"
            ),
            "rank_diagnostics": (
                "post-routing diagnostic: evaluation labels are used only to "
                "measure where the true class ranked among already-computed "
                "candidate scores; labels are never routing inputs"
            ),
            "score_margin_diagnostics": (
                "post-routing diagnostic: for Top-1 failures, top candidate "
                "score minus true candidate score is measured after routing; "
                "labels are never routing inputs"
            ),
            "warning": (
                "Routed accuracy and forgetting combine routing errors with "
                "classifier errors. They must not be interpreted as raw skill "
                "retention without a separate oracle-skill evaluation."
            ),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2))
    print("Analysis CSV saved to:", csv_path)
    print("Accuracy matrix CSV saved to:", matrix_path)
    print("Analysis JSON saved to:", json_path)
    print("Routing rank diagnostics:", json.dumps(summary["routing_rank_diagnostics"], indent=2))
    print(
        "Routing score-margin diagnostics:",
        json.dumps(summary["routing_score_margin_diagnostics"], indent=2),
    )


def main() -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"weight_reverse_engineering_{run_id}.txt"
    log_file = open(log_path, "w")
    real_stdout = sys.stdout
    sys.stdout = Tee(real_stdout, log_file)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        benchmark = SplitMNIST(n_experiences=10, seed=0)
        train_stream = benchmark.train_stream
        test_stream = benchmark.test_stream

        print("Device:", device)
        print(
            "Routing: persistent binary fingerprint -> class -> canonical skill"
        )
        print("No experience ID or target class is supplied to routing.")
        print(
            "Reported accuracy/forgetting are routed metrics; routing and "
            "classifier errors are not separated by these values."
        )

        model = SkillMemoryMLP(input_dim=784).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        plugin = PersistentFingerprintSkillMemoryPlugin(
            memory=SkillMemory(max_skills=10),
            forgetting_margin=0.05,
            probe_batch_size=10,
            probe_batches=5,
            probe_seed=0,
            class_train_epochs=1,
            class_train_batch_size=64,
            reuse_is_mutable=True,
            eval_routing="probe",
            verbose=True,
        )
        strategy = SupervisedTemplate(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=64,
            train_epochs=1,
            eval_mb_size=64,
            device=device,
            plugins=[plugin],
        )

        accuracy_history: list[list[float]] = []
        analysis_rows: list[dict] = []
        for train_index, train_exp in enumerate(train_stream):
            strategy.train(train_exp)
            accuracies = evaluate_seen(strategy, test_stream, train_index)
            accuracy_history.append(accuracies)
            analysis_rows.extend(
                flatten_route(route, train_index)
                for route in plugin.fingerprint_route_history
            )
            routes = plugin.last_fingerprint_routes
            identified = sum(r["status"] == "IDENTIFIED" for r in routes)
            ambiguous = sum(r["status"] == "AMBIGUOUS" for r in routes)
            failed = sum(r["status"] == "FAILED" for r in routes)
            print(
                f"Step {train_index}: classes="
                f"{sorted(train_exp.classes_in_this_experience)} "
                f"mean_seen_routed_accuracy={np.mean(accuracies):.3f} "
                f"last_batch_routes=(identified={identified}, "
                f"ambiguous={ambiguous}, failed={failed})"
            )
            for eval_index, accuracy in enumerate(accuracies):
                print(f"  routed_eval_exp={eval_index}: accuracy={accuracy:.3f}")

        n = len(accuracy_history)
        accuracy_curve = np.array([accuracy_history[i][i] for i in range(n)])
        forgetting = np.zeros(n)
        for class_index in range(n):
            seen = [row[class_index] for row in accuracy_history[class_index:]]
            if len(seen) > 1:
                forgetting[class_index] = max(seen[:-1]) - seen[-1]

        print("Routed accuracy:", np.round(accuracy_curve, 3))
        print("Routed forgetting:", np.round(forgetting, 3))
        print("Routed evaluation accuracy matrix:")
        for train_index, accuracies in enumerate(accuracy_history):
            print(
                f"  train_step={train_index}: "
                + ", ".join(
                    f"Exp{eval_index}={accuracy:.3f}"
                    for eval_index, accuracy in enumerate(accuracies)
                )
            )
        print(
            "Final fingerprint records:", len(plugin.behavior.state_dict()["records"])
        )
        write_analysis_files(
            log_dir,
            run_id,
            analysis_rows,
            accuracy_history,
            accuracy_curve,
            forgetting,
        )
        print("Log saved to:", log_path)
    finally:
        sys.stdout = real_stdout
        log_file.close()


if __name__ == "__main__":
    main()
