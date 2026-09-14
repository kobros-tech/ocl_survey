"""Diagnostic SplitMNIST run for oracle-skill retention.

This intentionally uses ``class_oracle`` evaluation. Ground-truth labels are
used only to select the canonical skill for each sample, so this is not an
anonymous-routing result. Its purpose is to measure whether the stored skill
snapshots themselves retain old classes independently of fingerprint routing.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.models.dynamic_modules import IncrementalClassifier
from avalanche.training.templates import SupervisedTemplate

from skill_memory import PersistentFingerprintSkillMemoryPlugin, SkillMemory


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
    """Evaluate seen experiences with canonical class-to-skill routing."""
    results = strategy.eval([test_stream[i] for i in range(up_to_index + 1)])
    keys = sorted(key for key in results if key.startswith("Top1_Acc_Exp"))
    return [float(results[key]) for key in keys]


def main() -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    benchmark = SplitMNIST(n_experiences=10, seed=0)

    model = SkillMemoryMLP(input_dim=784).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    plugin = PersistentFingerprintSkillMemoryPlugin(
        memory=SkillMemory(max_skills=10),
        forgetting_margin=0.05,
        probe_batch_size=10,
        probe_batches=5,
        probe_seed=0,
        class_train_epochs=1,
        class_train_batch_size=64,
        reuse_is_mutable=True,
        eval_routing="class_oracle",
        verbose=True,
    )
    strategy = SupervisedTemplate(
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.CrossEntropyLoss(),
        train_mb_size=64,
        train_epochs=1,
        eval_mb_size=64,
        device=device,
        plugins=[plugin],
    )

    history: list[list[float]] = []
    for train_index, train_exp in enumerate(benchmark.train_stream):
        strategy.train(train_exp)
        accuracies = evaluate_seen(strategy, benchmark.test_stream, train_index)
        history.append(accuracies)
        assignments = plugin.class_map.class_skill_for_experience(train_index)
        decisions = plugin.last_class_decisions.get(train_index, {})
        decision_text = ", ".join(
            f"{class_id}: {decision['decision']}"
            for class_id, decision in sorted(decisions.items())
        )
        print(
            f"Step {train_index}: classes={sorted(train_exp.classes_in_this_experience)} "
            f"class->skill={assignments} decisions={{{decision_text}}}"
        )
        print(
            "  oracle_eval="
            + ", ".join(
                f"Exp{index}={accuracy:.3f}"
                for index, accuracy in enumerate(accuracies)
            )
        )

    n = len(history)
    curve = np.array([history[i][i] for i in range(n)])
    forgetting = np.zeros(n)
    for exp_index in range(n):
        seen = [row[exp_index] for row in history[exp_index:]]
        if len(seen) > 1:
            forgetting[exp_index] = max(seen[:-1]) - seen[-1]

    print("Oracle-skill accuracy:", np.round(curve, 3))
    print("Oracle-skill forgetting:", np.round(forgetting, 3))
    print("Oracle-skill evaluation accuracy matrix:")
    for train_index, accuracies in enumerate(history):
        print(
            f"  train_step={train_index}: "
            + ", ".join(
                f"Exp{index}={accuracy:.3f}"
                for index, accuracy in enumerate(accuracies)
            )
        )

    output = {
        "run_id": run_id,
        "routing": "class_oracle",
        "warning": "Diagnostic only: target labels select canonical skills.",
        "oracle_accuracy_curve": curve.tolist(),
        "oracle_forgetting": forgetting.tolist(),
        "accuracy_matrix": history,
        "class_skill_mapping": {
            str(index): plugin.class_map.class_skill_for_experience(index)
            for index in range(n)
        },
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
