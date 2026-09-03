#!/usr/bin/env python3
"""Analyze Skill Memory probe diagnostics across replicate runs.

This script deliberately analyzes only pre-training compatibility diagnostics.
It does not tune thresholds against test accuracy. Its purpose is to answer:

1. Are score and probe accuracy mathematically consistent?
2. How often does each decision occur?
3. How variable are probe signals across independent seeds?

The statistical unit is the experimental seed. Multiple audit records from the
same seed are aggregated before bootstrap confidence intervals are computed.

Usage:
    python -m experiments.analyze_skill_memory_audit ROOT_DIR OUTPUT.csv
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_SEEDS = set(range(10))


def max_score_for_accuracy(accuracy: float, num_classes: int = 100) -> float:
    return math.exp(-(1.0 - accuracy) * math.log(2.0) / math.log(num_classes))


def load_audits(root: Path) -> pd.DataFrame:
    rows = []
    paths = sorted(root.glob("**/skill_memory_audit.json"))
    for path in paths:
        seed = None
        for part in path.parts:
            if part.isdigit():
                seed = int(part)
                break
        if seed is None:
            raise SystemExit(f"Could not determine seed from audit path: {path}")
        with path.open("r", encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list) or not records:
            raise SystemExit(f"Audit file is empty or malformed: {path}")
        for record in records:
            row = dict(record)
            row["seed"] = seed
            row["audit_file"] = str(path)
            rows.append(row)
    if not rows:
        raise SystemExit(f"No skill_memory_audit.json files found below {root}")
    return pd.DataFrame(rows)


def validate_schema(df: pd.DataFrame) -> None:
    required = {"seed", "decision", "compatibility_score", "compatibility_accuracy"}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"Missing required audit fields: {sorted(missing)}")

    seeds = set(df["seed"].dropna().astype(int).unique())
    missing_seeds = EXPECTED_SEEDS.difference(seeds)
    unexpected_seeds = seeds.difference(EXPECTED_SEEDS)
    if missing_seeds:
        raise SystemExit(f"Missing expected seeds: {sorted(missing_seeds)}")
    if unexpected_seeds:
        raise SystemExit(f"Unexpected seed values: {sorted(unexpected_seeds)}")

    for column in ("compatibility_score", "compatibility_accuracy"):
        values = pd.to_numeric(df[column], errors="coerce")
        if not np.isfinite(values.to_numpy()).all():
            raise SystemExit(f"Non-finite values found in {column}")

    accuracy = pd.to_numeric(df["compatibility_accuracy"], errors="coerce")
    if ((accuracy < 0.0) | (accuracy > 1.0)).any():
        raise SystemExit("compatibility_accuracy must be in [0, 1]")

    score = pd.to_numeric(df["compatibility_score"], errors="coerce")
    if ((score <= 0.0) | (score > 1.0)).any():
        raise SystemExit("compatibility_score must be in (0, 1]")


def bootstrap_mean(values: np.ndarray, seed: int = 0, samples: int = 2000):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def seed_level_summary(group: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated audit records to one estimate per seed/decision."""
    return (
        group.groupby(["seed", "decision"], as_index=False)
        .agg(
            probe_score=("compatibility_score", "mean"),
            probe_accuracy=("compatibility_accuracy", "mean"),
        )
    )


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m experiments.analyze_skill_memory_audit ROOT_DIR OUTPUT.csv")

    root = Path(sys.argv[1])
    output = Path(sys.argv[2])
    df = load_audits(root)
    validate_schema(df)

    df["max_score_allowed"] = df["compatibility_accuracy"].map(max_score_for_accuracy)
    df["score_accuracy_violation"] = df["compatibility_score"] > df["max_score_allowed"] + 1e-6
    df["zero_accuracy_high_score"] = (
        (df["compatibility_accuracy"] == 0.0)
        & (df["compatibility_score"] >= 0.90)
    )

    # These checks remain record-level because they validate every diagnostic.
    violations = int(df["score_accuracy_violation"].sum())
    zero_high = int(df["zero_accuracy_high_score"].sum())

    rows = []
    for decision, group in df.groupby("decision", sort=True):
        seed_df = seed_level_summary(group)
        mean_score, score_lo, score_hi = bootstrap_mean(seed_df["probe_score"].to_numpy())
        mean_acc, acc_lo, acc_hi = bootstrap_mean(seed_df["probe_accuracy"].to_numpy())
        rows.append({
            "decision": decision,
            "n_records": len(group),
            "n_seeds": len(seed_df),
            "mean_seed_probe_score": mean_score,
            "seed_probe_score_ci95_low": score_lo,
            "seed_probe_score_ci95_high": score_hi,
            "mean_seed_probe_accuracy": mean_acc,
            "seed_probe_accuracy_ci95_low": acc_lo,
            "seed_probe_accuracy_ci95_high": acc_hi,
            "score_accuracy_violations": int(group["score_accuracy_violation"].sum()),
            "zero_accuracy_score_ge_0_90": int(group["zero_accuracy_high_score"].sum()),
        })

    summary = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)

    print(f"records={len(df)} seeds={df['seed'].nunique()}")
    print(f"score_accuracy_violations={violations}")
    print(f"zero_accuracy_score_ge_0_90={zero_high}")
    if violations or zero_high:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
