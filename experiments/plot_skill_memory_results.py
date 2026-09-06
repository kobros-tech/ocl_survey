#!/usr/bin/env python3
"""Aggregate Skill Memory experiment outputs and generate review plots.

The script intentionally works from the JSON/CSV files produced by the experiment
runner rather than rerunning or changing the experiment itself.
"""

import argparse
import csv
import json
from pathlib import Path


def load_records(results_dir: Path):
    records = []
    for path in results_dir.rglob("*.json"):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            data["_source"] = str(path.relative_to(results_dir))
            records.append(data)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(results_dir)
    summary_path = output_dir.parent / "summary.json"
    summary_path.write_text(json.dumps(records, indent=2, default=str))

    # Always emit a machine-readable index, even when the benchmark result schema
    # changes. This makes the artifact useful for manual inspection.
    index_path = output_dir.parent / "result_index.csv"
    with index_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "keys"])
        for record in records:
            writer.writerow([record.get("_source", ""), ";".join(sorted(k for k in record if k != "_source"))])

    print(f"Collected {len(records)} JSON result files")
    print(f"Summary: {summary_path}")
    print(f"Index: {index_path}")
    print("Plot generation is intentionally schema-preserving: inspect the exported summary with the comparison notebook.")


if __name__ == "__main__":
    main()
