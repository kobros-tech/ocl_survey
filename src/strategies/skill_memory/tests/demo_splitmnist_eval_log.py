# # Skill-memory continual learning demo (v0.1.4 package) -- eval log edition
#
# Deliberately named `demo_*.py`, not `test_*.py`: this is a runnable demo
# script (real SplitMNIST training, network access to download MNIST, no
# `test_` functions), not a pytest unit test.

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.models.dynamic_modules import IncrementalClassifier
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate

from skill_memory import SkillMemory, SkillMemoryPlugin


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


# Keep one ID for the complete run so the CSV and text log can be matched.
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = Path(__file__).resolve().parent / "logs"
log_dir.mkdir(exist_ok=True)
terminal_log_path = log_dir / f"run_{run_id}.txt"
csv_log_path = log_dir / f"run_{run_id}.csv"

_terminal_log_file = open(terminal_log_path, "w")
_real_stdout = sys.stdout
sys.stdout = Tee(_real_stdout, _terminal_log_file)
print(f"Logging full terminal output to {terminal_log_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


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
        x = self.features(x)
        return self.classifier(x)


class PredictionLogger(SupervisedPlugin):
    """Capture the strategy's final routed prediction for every eval sample."""

    def __init__(self):
        super().__init__()
        self.records: list[dict] = []
        self._train_step: int | None = None
        self._exp_id: int | None = None
        self._sample_index = 0

    def set_train_step(self, step: int) -> None:
        self._train_step = step
        self._sample_index = 0

    def before_eval_exp(self, strategy, **kwargs) -> None:
        # This is logging metadata only. It is never supplied to or used by
        # SkillMemoryPlugin routing; routing remains probe-based and
        # per-sample. Keeping the metadata lets us group errors by benchmark
        # experience after the fact.
        experience = strategy.experience
        self._exp_id = getattr(experience, "current_experience", None)

    def after_eval_iteration(self, strategy, **kwargs) -> None:
        """Log predictions after all eval-forward plugins have run."""
        y_true = strategy.mbatch[1].detach().cpu()
        logits = strategy.mb_output.detach().cpu()
        probs = torch.softmax(logits, dim=1)
        y_pred = logits.argmax(dim=1)
        confidence = probs.gather(1, y_pred.unsqueeze(1)).squeeze(1)

        for i in range(len(y_true)):
            self.records.append(
                {
                    "train_step": self._train_step,
                    "sample_index": self._sample_index,
                    "experience": self._exp_id,
                    "true_y": int(y_true[i]),
                    "pred_y": int(y_pred[i]),
                    "confidence": float(confidence[i]),
                    "correct": bool(y_true[i] == y_pred[i]),
                }
            )
            self._sample_index += 1

    def print_misclassifications(self, train_step: int) -> None:
        """Print every misclassification for one training step."""
        rows = [
            r
            for r in self.records
            if r["train_step"] == train_step and not r["correct"]
        ]
        total = sum(r["train_step"] == train_step for r in self.records)
        print(f"\n=== MISCLASSIFICATIONS AFTER TRAINING STEP {train_step} ===")
        print(f"Misclassified: {len(rows)} / {total}")
        if not rows:
            print("None")
            return

        by_exp: dict[int, list[dict]] = {}
        for row in rows:
            by_exp.setdefault(int(row["experience"]), []).append(row)

        for exp_id in sorted(by_exp):
            exp_rows = by_exp[exp_id]
            print(f"\nExperience {exp_id}: {len(exp_rows)} misclassified samples")
            print("sample  true_y  pred_y  confidence")
            print("-" * 38)
            for row in exp_rows:
                print(
                    f"{row['sample_index']:>6}  {row['true_y']:>6}  "
                    f"{row['pred_y']:>6}  {row['confidence']:.4f}"
                )

    def print_misclassification_summary(self, train_step: int) -> None:
        """Print per-experience error counts for one training step."""
        rows = [r for r in self.records if r["train_step"] == train_step]
        print(f"\n--- misclassification summary for step {train_step} ---")
        if not rows:
            print("(no evaluation rows)")
            return

        by_exp: dict[int, list[dict]] = {}
        for row in rows:
            by_exp.setdefault(int(row["experience"]), []).append(row)

        for exp_id in sorted(by_exp):
            exp_rows = by_exp[exp_id]
            errors = sum(not row["correct"] for row in exp_rows)
            accuracy = 1.0 - errors / len(exp_rows)
            print(
                f"Exp {exp_id:>2}: errors={errors:>4} / {len(exp_rows):>4}, "
                f"accuracy={accuracy:.3f}"
            )

    def to_csv(self, path: str) -> None:
        """Dump every evaluated sample to CSV."""
        fieldnames = [
            "train_step",
            "sample_index",
            "experience",
            "true_y",
            "pred_y",
            "confidence",
            "correct",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)
        print(f"Wrote {len(self.records)} eval rows to {path}")


def evaluate_seen_experiences(
    strategy,
    test_stream,
    up_to_index: int,
    pred_logger: PredictionLogger,
    train_step: int,
) -> list[float]:
    """Evaluate all seen experiences in one strategy evaluation call.

    The benchmark experience boundaries are retained only for evaluation
    metrics and post-hoc diagnostics. SkillMemoryPlugin receives no target
    experience or class selection signal and continues to route each sample
    with its configured probe-based evaluator.
    """
    pred_logger.set_train_step(train_step)
    seen_stream = [test_stream[i] for i in range(up_to_index + 1)]
    results = strategy.eval(seen_stream)

    acc_keys = sorted(k for k in results if k.startswith("Top1_Acc_Exp"))
    if len(acc_keys) != up_to_index + 1:
        raise RuntimeError(
            "Expected one Top1_Acc_Exp metric per seen experience, "
            f"found {len(acc_keys)} for {up_to_index + 1} experiences."
        )
    return [float(results[key]) for key in acc_keys]


def compute_cl_metrics(accuracy_history: list[list[float]]):
    """Build accuracy/forgetting curves from a ragged accuracy history."""
    n = len(accuracy_history)
    matrix = np.full((n, n), np.nan)
    for t, row in enumerate(accuracy_history):
        matrix[t, : len(row)] = row

    accuracy_curve = np.array([matrix[t, t] for t in range(n)])

    forgetting_curve = np.zeros(n)
    for i in range(n):
        seen = matrix[i:, i]
        seen = seen[~np.isnan(seen)]
        if len(seen) > 1:
            forgetting_curve[i] = np.max(seen[:-1]) - seen[-1]

    return accuracy_curve, forgetting_curve


benchmark = SplitMNIST(n_experiences=10, seed=0)
train_stream = benchmark.train_stream
test_stream = benchmark.test_stream

print("Number of experiences:", len(train_stream))
for i, exp in enumerate(train_stream):
    print(i, sorted(exp.classes_in_this_experience), len(exp.dataset))


model = SkillMemoryMLP(input_dim=784).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

skill_plugin = SkillMemoryPlugin(
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
pred_logger = PredictionLogger()

strategy = SupervisedTemplate(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_mb_size=64,
    train_epochs=1,
    eval_mb_size=64,
    device=device,
    plugins=[skill_plugin, pred_logger],
)

accuracy_history: list[list[float]] = []

try:
    for t, train_exp in enumerate(train_stream):
        strategy.train(train_exp)

        current_accuracies = evaluate_seen_experiences(
            strategy, test_stream, t, pred_logger, train_step=t
        )
        accuracy_history.append(current_accuracies)

        print(
            f"Experience {t}: "
            f"trained on {sorted(train_exp.classes_in_this_experience)}, "
            f"mean seen accuracy = {np.mean(current_accuracies):.3f}"
        )

        # The text artifact contains a complete error listing for every
        # training step, not just the final step. The CSV contains every
        # correct and incorrect prediction for offline analysis.
        pred_logger.print_misclassification_summary(train_step=t)
        pred_logger.print_misclassifications(train_step=t)

    accuracy_curve, forgetting_curve = compute_cl_metrics(accuracy_history)

    print("\nAccuracy:", np.round(accuracy_curve, 3))
    print("Forgetting:", np.round(forgetting_curve, 3))
finally:
    # Export partial results too if training/evaluation fails or is cancelled.
    pred_logger.to_csv(str(csv_log_path))
    sys.stdout = _real_stdout
    _terminal_log_file.close()
    print(f"Terminal log saved to {terminal_log_path}")
    print(f"Eval CSV saved to {csv_log_path}")
