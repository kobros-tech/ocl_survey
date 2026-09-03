#!/usr/bin/env python3
"""Run the Skill Memory benchmark for the same five seeds used by ER."""

import subprocess
import sys
from pathlib import Path


SEEDS = (0, 1, 2, 3, 4)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    for seed in SEEDS:
        print(f"\n=== Skill Memory seed {seed} ===", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.main",
                "strategy=skill_memory",
                "experiment=split_cifar100",
                f"experiment.seed={seed}",
            ],
            cwd=repo_root,
            check=True,
        )


if __name__ == "__main__":
    main()
