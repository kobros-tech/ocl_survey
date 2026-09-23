"""Independent ML evaluator for measuring class retention.

This module provides the evaluation-memory and independent-classifier
components used by ``SkillMemoryStrategy``.

The evaluator answers a specific question:

    Given an anonymous input x, can a separately trained classifier recover y
    from the accumulated examples retained during continual learning?

The evaluator is therefore deliberately independent of Skill Memory's
weights, features, logits, routing decisions, and skill IDs.

The lifecycle is:

    Skill Memory training
        |
        +--> EvaluationMemoryPlugin retains raw x,y examples
        |
        +--> consolidate_evaluation_memory()
        |
        +--> train_evaluator()
        |
        +--> evaluate_model_by_class()
        |
        +--> accuracy / loss / forgetting

The experience index is used only for reporting and forgetting bookkeeping.
It is never an input to the evaluator.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from avalanche.training.plugins import SupervisedPlugin
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..cl.skill_memory_plugin import SkillMemoryPlugin


@dataclass
class EvaluationMemory:
    """Frozen raw examples retained for one class."""

    inputs: torch.Tensor
    targets: torch.Tensor
    class_id: int

    @property
    def size(self) -> int:
        """Return the number of retained samples."""
        return int(self.targets.numel())


class EvaluationMemoryPlugin(SkillMemoryPlugin):
    """Skill Memory plugin with independent evaluation-memory retention.

    Skill Memory remains responsible for:

    - REUSE/SCRATCH decisions
    - skill allocation
    - class-to-skill bookkeeping
    - training
    - storing skill states

    This subclass additionally retains a bounded set of raw examples for
    every class encountered by the training experience.

    The retained data contains only:

        x, y

    It contains no Skill Memory weights or derived representations.
    """

    def __init__(
        self,
        *args,
        eval_memory_per_class: int = 20,
        eval_memory_seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if eval_memory_per_class <= 0:
            raise ValueError("eval_memory_per_class must be positive")
        self.eval_memory_per_class = int(eval_memory_per_class)
        self.eval_memory_seed = int(eval_memory_seed)
        self.eval_memory: list[EvaluationMemory] = []

    def after_training_exp(self, strategy, **kwargs) -> None:
        """Run Skill Memory lifecycle, then retain evaluation examples.

        ``SkillMemoryPlugin.after_training_exp`` is called first so the
        Skill Memory bookkeeping is completed before evaluation memory is
        captured.
        """
        super().after_training_exp(
            strategy,
            **kwargs,
        )
        experience = strategy.experience
        memories = self._build_evaluation_memory(experience)
        self.eval_memory.extend(memories)
        if self.verbose:
            experience_index = int(
                getattr(
                    experience,
                    "current_experience",
                    0,
                )
            )

            print(
                f"Evaluation memory {experience_index}: "
                f"{sum(memory.size for memory in memories)} "
                f"samples, "
                f"classes={[memory.class_id for memory in memories]}"
            )

    def _build_evaluation_memory(
        self,
        experience,
    ) -> list[EvaluationMemory]:
        """Retain a deterministic bounded sample for each actual class.

        Classes are discovered from the actual samples in the experience
        dataset rather than from an assumed experience layout.

        This is important for generic Avalanche benchmarks where the number
        of classes per experience is not necessarily fixed.
        """
        dataset = experience.dataset
        samples_by_class: dict[int, list[int]] = {}

        for index in range(len(dataset)):
            sample = dataset[index]
            if len(sample) < 2:
                raise RuntimeError(
                    "Evaluation dataset samples must contain (input, target)."
                )
            target = int(sample[1])
            samples_by_class.setdefault(
                target,
                [],
            ).append(index)

        experience_index = int(
            getattr(
                experience,
                "current_experience",
                0,
            )
        )
        generator = torch.Generator()
        generator.manual_seed(self.eval_memory_seed + experience_index)
        memories: list[EvaluationMemory] = []

        for class_id in sorted(samples_by_class):
            indices = samples_by_class[class_id]

            if len(indices) > self.eval_memory_per_class:
                permutation = torch.randperm(
                    len(indices),
                    generator=generator,
                ).tolist()

                indices = [
                    indices[position]
                    for position in permutation[: self.eval_memory_per_class]
                ]

            inputs: list[torch.Tensor] = []
            targets: list[int] = []

            for index in indices:
                sample = dataset[index]
                input_tensor = sample[0]

                if not isinstance(
                    input_tensor,
                    torch.Tensor,
                ):
                    input_tensor = torch.as_tensor(input_tensor)

                inputs.append(input_tensor.detach().cpu())
                targets.append(int(sample[1]))

            if not inputs:
                raise RuntimeError(
                    f"Class {class_id} produced an empty evaluation memory."
                )

            memories.append(
                EvaluationMemory(
                    inputs=torch.stack(inputs),
                    targets=torch.tensor(
                        targets,
                        dtype=torch.long,
                    ),
                    class_id=class_id,
                )
            )

        return memories


def make_loader(
    memory: list[EvaluationMemory],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int = 0,
) -> DataLoader:
    """Build a loader from frozen evaluation memory."""
    if not memory:
        raise RuntimeError("Cannot build an evaluation loader from empty memory.")

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    inputs = torch.cat(
        [item.inputs for item in memory],
        dim=0,
    )
    targets = torch.cat(
        [item.targets for item in memory],
        dim=0,
    )
    dataset = TensorDataset(
        inputs,
        targets,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def consolidate_evaluation_memory(
    memory: list[EvaluationMemory],
) -> list[EvaluationMemory]:
    """Combine all retained examples belonging to the same class.

    Experience boundaries are deliberately discarded.

    The result therefore represents the accumulated class-level memory:

        class -> all retained x,y examples for that class
    """
    if not memory:
        return []

    by_class: dict[
        int,
        list[EvaluationMemory],
    ] = {}

    for item in memory:
        by_class.setdefault(
            int(item.class_id),
            [],
        ).append(item)

    consolidated: list[EvaluationMemory] = []

    for class_id in sorted(by_class):
        inputs = torch.cat(
            [item.inputs for item in by_class[class_id]],
            dim=0,
        )
        targets = torch.cat(
            [item.targets for item in by_class[class_id]],
            dim=0,
        )
        consolidated.append(
            EvaluationMemory(
                inputs=inputs,
                targets=targets,
                class_id=class_id,
            )
        )

    return consolidated


def train_evaluator(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    memory: list[EvaluationMemory],
    *,
    batch_size: int,
    epochs: int,
    device: torch.device,
    seed: int,
) -> None:
    """Train the anonymous x -> y evaluator on accumulated memory.

    The evaluator receives only raw inputs and targets.

    No Skill Memory state, skill ID, experience ID, task label, or routing
    information is supplied.
    """
    if not memory:
        raise RuntimeError("Cannot train evaluator with empty memory.")

    if epochs < 1:
        return

    loader = make_loader(
        memory,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    model.to(device)
    model.train()

    for _ in range(epochs):
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(inputs)

            if logits.ndim != 2:
                raise RuntimeError(
                    "The evaluator model must return [batch, num_classes] logits."
                )

            if logits.shape[0] != targets.shape[0]:
                raise RuntimeError(
                    "Evaluator output batch size does not match target batch size."
                )

            loss = criterion(
                logits,
                targets,
            )

            loss.backward()
            optimizer.step()


@torch.no_grad()
def evaluate_model_by_class(
    model: nn.Module,
    test_stream,
    up_to_index: int,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[int, dict[str, float]]:
    """Evaluate anonymous x -> predicted y for every seen class.

    No experience-level prediction mask is applied.

    The model sees only ``inputs``. Labels are used only afterward to
    calculate accuracy and loss.

    Therefore this is genuinely:

        anonymous x -> model(x) -> predicted y
    """
    if up_to_index < 0:
        return {}

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    model.to(device)
    model.eval()

    results: dict[
        int,
        dict[str, float],
    ] = {}

    for experience_index in range(up_to_index + 1):
        experience = test_stream[experience_index]

        loader = DataLoader(
            experience.dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        class_loss: dict[int, float] = {}
        class_correct: dict[int, int] = {}
        class_total: dict[int, int] = {}

        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)

            if logits.ndim != 2:
                raise RuntimeError(
                    "The evaluator model must return [batch, num_classes] logits."
                )

            if logits.shape[1] <= int(targets.max().item()):
                raise RuntimeError(
                    "Evaluator output space does not contain "
                    "one or more target class IDs."
                )

            per_sample_loss = nn.functional.cross_entropy(
                logits,
                targets,
                reduction="none",
            )

            predictions = logits.argmax(dim=1)

            for class_id in torch.unique(targets).tolist():
                class_id = int(class_id)

                mask = targets == class_id

                class_loss[class_id] = class_loss.get(
                    class_id,
                    0.0,
                ) + float(per_sample_loss[mask].sum().item())

                class_correct[class_id] = class_correct.get(
                    class_id,
                    0,
                ) + int((predictions[mask] == targets[mask]).sum().item())

                class_total[class_id] = class_total.get(
                    class_id,
                    0,
                ) + int(mask.sum().item())

        for class_id in class_loss:
            total = class_total[class_id]

            if total == 0:
                raise RuntimeError(f"Class {class_id} has no test samples.")

            results[class_id] = {
                "loss": (class_loss[class_id] / total),
                "accuracy": (class_correct[class_id] / total),
                "experience": float(experience_index),
            }

    return results


def aggregate_experience_metrics(
    class_results: dict[int, dict[str, float]],
    test_stream,
    up_to_index: int,
) -> tuple[list[float], list[float]]:
    """Aggregate class metrics by introducing experience.

    Classes are weighted equally within each experience.
    """
    losses: list[float] = []
    accuracies: list[float] = []

    for experience_index in range(up_to_index + 1):
        classes = sorted(
            int(class_id)
            for class_id in (test_stream[experience_index].classes_in_this_experience)
        )

        missing = [class_id for class_id in classes if class_id not in class_results]

        if missing:
            raise RuntimeError(
                f"Missing evaluation results for classes "
                f"{missing} in experience "
                f"{experience_index}."
            )

        losses.append(
            float(np.mean([class_results[class_id]["loss"] for class_id in classes]))
        )

        accuracies.append(
            float(
                np.mean([class_results[class_id]["accuracy"] for class_id in classes])
            )
        )

    return losses, accuracies


def compute_class_forgetting(
    accuracy_history: list[dict[int, float]],
    class_to_experience: dict[int, int],
    num_experiences: int,
) -> np.ndarray:
    """Compute acquisition-relative forgetting.

    Each class is compared against its accuracy immediately after its
    introduction.
    """
    forgetting_by_experience = [[] for _ in range(num_experiences)]

    all_classes = sorted(
        {class_id for history in accuracy_history for class_id in history}
    )

    for class_id in all_classes:
        introduction = class_to_experience[class_id]

        if introduction >= len(accuracy_history):
            continue

        acquisition_accuracy = accuracy_history[introduction][class_id]

        final_accuracy = accuracy_history[-1].get(class_id)

        if final_accuracy is None:
            continue

        forgetting = max(
            0.0,
            acquisition_accuracy - final_accuracy,
        )

        forgetting_by_experience[introduction].append(forgetting)

    result = np.zeros(
        num_experiences,
        dtype=np.float64,
    )

    for experience_index, values in enumerate(forgetting_by_experience):
        if values:
            result[experience_index] = float(np.mean(values))

    return result


def compute_peak_class_forgetting(
    accuracy_history: list[dict[int, float]],
    class_to_experience: dict[int, int],
    num_experiences: int,
) -> np.ndarray:
    """Compute peak-relative continual-learning forgetting.

    For every class:

        forgetting = peak_accuracy - final_accuracy

    with the peak restricted to evaluations after that class was introduced.
    """
    forgetting_by_experience = [[] for _ in range(num_experiences)]

    all_classes = sorted(
        {class_id for history in accuracy_history for class_id in history}
    )

    for class_id in all_classes:
        introduction = class_to_experience[class_id]

        if introduction >= len(accuracy_history):
            continue

        final_accuracy = accuracy_history[-1].get(class_id)

        if final_accuracy is None:
            continue

        observed = [
            history[class_id]
            for history in accuracy_history[introduction:]
            if class_id in history
        ]

        if not observed:
            continue

        peak_accuracy = max(observed)

        forgetting = max(
            0.0,
            peak_accuracy - final_accuracy,
        )

        forgetting_by_experience[introduction].append(forgetting)

    result = np.zeros(
        num_experiences,
        dtype=np.float64,
    )

    for experience_index, values in enumerate(forgetting_by_experience):
        if values:
            result[experience_index] = float(np.mean(values))

    return result


def build_evaluator(
    model_factory: Callable[[], nn.Module],
    *,
    device: torch.device,
    learning_rate: float,
) -> tuple[
    nn.Module,
    torch.optim.Optimizer,
    nn.Module,
]:
    """Create an independent evaluator.

    The factory must return a fresh model with the complete global class
    output space.
    """
    model = model_factory().to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
    )

    criterion = nn.CrossEntropyLoss()

    return (
        model,
        optimizer,
        criterion,
    )


class MLEvaluationPlugin(SupervisedPlugin):
    """Avalanche plugin for anonymous standalone ML evaluation.

    The evaluator is trained on the accumulated raw evaluation memory before
    each call to ``strategy.eval()``. During the normal Avalanche evaluation
    loop, the evaluator replaces ``strategy.mb_output`` so Avalanche's own
    evaluation metrics operate on:

        x -> standalone ML evaluator -> y

    No Skill Memory weights, skill IDs, task labels, or experience IDs are
    supplied to the evaluator.
    """

    def __init__(
        self,
        *,
        memory_plugin: EvaluationMemoryPlugin,
        model_factory: Callable[[], nn.Module],
        epochs: int = 1,
        batch_size: int = 64,
        learning_rate: float = 0.01,
        seed: int = 0,
        verbose: bool = True,
    ) -> None:
        super().__init__()

        if epochs < 1:
            raise ValueError("epochs must be at least 1")

        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        self.memory_plugin = memory_plugin
        self.model_factory = model_factory
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.verbose = verbose

        self.model: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.criterion = nn.CrossEntropyLoss()

        self._active = False

        self._class_to_experience: dict[int, int] = {}
        self._accuracy_history: list[dict[int, float]] = []
        self._loss_history: list[dict[int, float]] = []
        self._diagonal_accuracy_history: list[float] = []
        self._diagonal_loss_history: list[float] = []

        self._current_class_loss: dict[int, float] = {}
        self._current_class_correct: dict[int, int] = {}
        self._current_class_total: dict[int, int] = {}
        self._current_experience_classes: set[int] = set()

    # ------------------------------------------------------------------
    # Training-memory bookkeeping
    # ------------------------------------------------------------------

    def after_training_exp(self, strategy, **kwargs) -> None:
        """Record class introduction after the normal training hook."""
        experience = strategy.experience

        experience_index = int(
            getattr(
                experience,
                "current_experience",
                len(self._class_to_experience),
            )
        )

        for class_id in experience.classes_in_this_experience:
            class_id = int(class_id)
            self._class_to_experience.setdefault(
                class_id,
                experience_index,
            )

    # ------------------------------------------------------------------
    # Normal Avalanche evaluation lifecycle
    # ------------------------------------------------------------------

    def before_eval(self, strategy, **kwargs) -> None:
        """Train the independent evaluator before Avalanche eval starts."""
        memory = consolidate_evaluation_memory(self.memory_plugin.eval_memory)

        self._active = bool(memory)

        if not self._active:
            return

        self.model = self.model_factory().to(strategy.device)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        with torch.enable_grad():
            train_evaluator(
                self.model,
                self.optimizer,
                self.criterion,
                memory,
                batch_size=self.batch_size,
                epochs=self.epochs,
                device=strategy.device,
                seed=self.seed,
            )

        self._current_class_loss = {}
        self._current_class_correct = {}
        self._current_class_total = {}
        self._current_experience_classes = set()

        if self.verbose:
            print(
                "ML evaluation: trained standalone evaluator on "
                f"{sum(item.size for item in memory)} retained samples"
            )

    def before_eval_exp(self, strategy, **kwargs) -> None:
        """Reset per-experience accumulation."""
        if not self._active:
            return

        experience = strategy.experience

        self._current_experience_classes = {
            int(class_id) for class_id in experience.classes_in_this_experience
        }

    @torch.no_grad()
    def after_eval_forward(self, strategy, **kwargs) -> None:
        """Replace the main model output with standalone ML predictions."""
        if not self._active or self.model is None:
            return

        inputs = strategy.mbatch[0]

        self.model.eval()

        # The evaluator sees only x.
        strategy.mb_output = self.model(inputs)

    def after_eval_iteration(self, strategy, **kwargs) -> None:
        """Collect class-level metrics from the standalone evaluator."""
        if not self._active:
            return

        outputs = strategy.mb_output
        targets = strategy.mbatch[1]

        predictions = outputs.argmax(dim=1)

        per_sample_loss = nn.functional.cross_entropy(
            outputs,
            targets,
            reduction="none",
        )

        for class_id in torch.unique(targets).tolist():
            class_id = int(class_id)
            mask = targets == class_id

            self._current_class_loss[class_id] = self._current_class_loss.get(
                class_id, 0.0
            ) + float(per_sample_loss[mask].sum().item())

            self._current_class_correct[class_id] = self._current_class_correct.get(
                class_id, 0
            ) + int((predictions[mask] == targets[mask]).sum().item())

            self._current_class_total[class_id] = self._current_class_total.get(
                class_id, 0
            ) + int(mask.sum().item())

    def after_eval_exp(self, strategy, **kwargs) -> None:
        """Finalize metrics for this evaluation experience."""
        if not self._active:
            return

        # Nothing needs to be emitted here. The complete stream-level
        # result is finalized in after_eval().
        return

    def after_eval(self, strategy, **kwargs) -> None:
        """Finalize one normal Avalanche eval() call."""
        if not self._active:
            return

        current_accuracy: dict[int, float] = {}
        current_loss: dict[int, float] = {}

        for class_id in sorted(self._current_class_total):
            total = self._current_class_total[class_id]

            if total == 0:
                continue

            current_accuracy[class_id] = self._current_class_correct[class_id] / total

            current_loss[class_id] = self._current_class_loss[class_id] / total

        if not current_accuracy:
            self._active = False
            return

        self._accuracy_history.append(current_accuracy)
        self._loss_history.append(current_loss)

        diagonal_classes = [
            class_id
            for class_id, introduction in self._class_to_experience.items()
            if introduction == len(self._accuracy_history) - 1
            and class_id in current_accuracy
        ]

        if diagonal_classes:
            self._diagonal_accuracy_history.append(
                float(
                    np.mean(
                        [current_accuracy[class_id] for class_id in diagonal_classes]
                    )
                )
            )

            self._diagonal_loss_history.append(
                float(
                    np.mean([current_loss[class_id] for class_id in diagonal_classes])
                )
            )

        if self.verbose:
            print(
                "ML evaluation: "
                f"mean_accuracy="
                f"{np.mean(list(current_accuracy.values())):.4f}"
            )

        self._active = False

    # ------------------------------------------------------------------
    # Public results
    # ------------------------------------------------------------------

    @property
    def evaluator_model(self) -> nn.Module | None:
        return self.model

    @property
    def current_accuracy(self) -> dict[int, float]:
        """Return the latest per-class ML accuracy."""
        return self.ml_evaluation_plugin.current_accuracy

    @property
    def current_loss(self) -> dict[int, float]:
        """Return the latest per-class ML loss."""
        return self.ml_evaluation_plugin.current_loss

    def results(self) -> dict[str, Any]:
        if not self._accuracy_history:
            raise RuntimeError(
                "No ML evaluation results are available. "
                "Call strategy.eval() after training."
            )

        final_accuracy = self._accuracy_history[-1]
        final_loss = self._loss_history[-1]

        result: dict[str, Any] = {
            "final_class_accuracy": dict(final_accuracy),
            "final_class_loss": dict(final_loss),
            "mean_final_accuracy": float(np.mean(list(final_accuracy.values()))),
            "mean_final_loss": float(np.mean(list(final_loss.values()))),
            "diagonal_accuracy": np.asarray(
                self._diagonal_accuracy_history,
                dtype=np.float64,
            ),
            "diagonal_loss": np.asarray(
                self._diagonal_loss_history,
                dtype=np.float64,
            ),
        }

        if len(self._accuracy_history) > 0:
            result["peak_forgetting"] = compute_peak_class_forgetting(
                self._accuracy_history,
                self._class_to_experience,
                len(self._accuracy_history),
            )

        return result
