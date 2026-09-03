#!/usr/bin/env python3
"""Analyze Skill Memory probe diagnostics across replicate runs.

This script deliberately analyzes only pre-training compatibility diagnostics.
It does not tune thresholds against test accuracy. Its purpose is to answer:

1. Are score and probe accuracy mathematically consistent?
2. How often does each decision occur?
3. How variable are probe signals across seeds/tasks?

Usage:
    python -m experiments.analyze_skill_memory_audit ROOT_DIR OUTPUT.csv
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def max_score_for_accuracy(accuracy: float, num_classes: int = 100) -> float:
    return math.exp(-(1.0 - accuracy) * math.log(2.0) / math.log(num_classes))


def load_audits(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("**/skill_memory_audit.json")):
        seed = None
        for part in path.parts:
            if part.isdigit():
                seed = int(part)
                break
        with path.open("r", encoding="utf-8") as f:
            records = json.load(f)
        for record in records:
            row = dict(record)
            row["seed"] = seed
            row["audit_file"] = str(path)
            rows.append(row)
    if not rows:
        raise SystemExit(f"No skill_memory_audit.json files found below {root}")
    return pd.DataFrame(rows)


def bootstrap_mean(values: np.ndarray, seed: int = 0, samples: int = 2000):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m experiments.analyze_skill_memory_audit ROOT_DIR OUTPUT.csv")

    root = Path(sys.argv[1])
    output = Path(sys.argv[2])
    df = load_audits(root)

    df["max_score_allowed"] = df["compatibility_accuracy"].map(max_score_for_accuracy)
    df["score_accuracy_violation"] = df["compatibility_score"] > df["max_score_allowed"] + 1e-6
    df["zero_accuracy_high_score"] = (
        (df["compatibility_accuracy"] == 0.0)
        & (df["compatibility_score"] >= 0.90)
    )

    rows = []
    for decision, group in df.groupby("decision", sort=True):
        mean_score, score_lo, score_hi = bootstrap_mean(group["compatibility_score"].to_numpy())
        mean_acc, acc_lo, acc_hi = bootstrap_mean(group["compatibility_accuracy"].to_numpy())
        rows.append({
            "decision": decision,
            "n": len(group),
            "n_seeds": group["seed"].nunique(),
            "mean_probe_score": mean_score,
            "probe_score_ci95_low": score_lo,
            "probe_score_ci95_high": score_hi,
            "mean_probe_accuracy": mean_acc,
            "probe_accuracy_ci95_low": acc_lo,
            "probe_accuracy_ci95_high": acc_hi,
            "score_accuracy_violations": int(group["score_accuracy_violation"].sum()),
            "zero_accuracy_score_ge_0_90": int(group["zero_accuracy_high_score"].sum()),
        })

    summary = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)

    violations = int(df["score_accuracy_violation"].sum())
    zero_high = int(df["zero_accuracy_high_score"].sum())
    print(f"records={len(df)} seeds={df['seed'].nunique()}")
    print(f"score_accuracy_violations={violations}")
    print(f"zero_accuracy_score_ge_0_90={zero_high}")
    if violations or zero_high:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
