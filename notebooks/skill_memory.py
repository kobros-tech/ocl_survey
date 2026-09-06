"""
skill_memory.py

A task-free continual learning strategy based on dynamic "skill" (linear
expert head) allocation, driven by a probe-based compatibility test rather
than a task oracle.

Core idea
---------
For each new experience, decide whether to:
  (a) reuse an existing skill, only if a small probe shows it is BOTH
      - still safe on OLD data (forgetting guard), and
      - compatible with the NEW data (score/accuracy cluster test with an
        absolute performance floor, not just relative separation), or
  (b) allocate a fresh skill, otherwise.

This module is intentionally framework-agnostic: it expects any object
with a `.dataset` attribute (Avalanche experience-like) and any dataset
that yields (x, y, task_id)-style triples.

Fixes vs. the original notebook prototype
------------------------------------------
1. Forgetting guard: old_score / old_accuracy (already computed in
   `imagine`) are now actually used to filter out reuse candidates whose
   grip on old data is already weak, instead of being discarded.
2. Absolute floor: candidate clusters must also clear an absolute
   performance floor (better than chance), not just win on relative gap
   size. This fixes the n==2-skills degenerate case where relative
   clustering alone always collapses to a top-1-per-metric comparison.
3. Multi-batch probing: probes pull several batches instead of a single
   batch of 10, cutting decision variance; the number of batches is
   configurable and can be seeded for reproducible studies.
4. Optional replay-on-reuse: when a skill is reused, a small number of
   old-data batches can be interleaved into training so reuse doesn't
   silently overwrite what that head already knew.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

logger = logging.getLogger(__name__)


# ============================================================
# CONFIG
# ============================================================

@dataclass
class SkillMemoryConfig:
    # Model
    n_skills: int = 10
    input_dim: int = 784
    num_classes: int = 10

    # Training
    batch_size: int = 64
    epochs_per_experience: int = 1
    learning_rate: float = 0.01

    # Probing (used both for the reuse-vs-allocate decision and for
    # eval-time skill routing)
    probe_batch_size: int = 10
    probe_batches: int = 5  # batches concatenated per probe; more = lower variance

    # Allocation decision
    forgetting_margin: float = 0.05   # old_accuracy must exceed chance + margin to be reuse-safe
    score_floor: Optional[float] = None  # absolute floor on new_score candidates; auto-derived if None

    # Optional replay while training a REUSED skill
    replay_old_during_reuse: bool = False
    replay_batches_per_epoch: int = 1

    verbose: bool = True
    seed: Optional[int] = None
    device: str = "cpu"


# ============================================================
# MODEL
# ============================================================

class SkillClassifierBank(nn.Module):
    """A bank of linear "skill" classifiers; one is active at a time."""

    def __init__(self, config: SkillMemoryConfig):
        super().__init__()
        self.config = config
        self.skill_classifiers = nn.ModuleList(
            [
                nn.Linear(config.input_dim, config.num_classes)
                for _ in range(config.n_skills)
            ]
        )
        self.learned_skills: Set[int] = set()
        self.active_skill: Optional[int] = None

    def allocate_skill(self) -> int:
        for skill_index in range(len(self.skill_classifiers)):
            if skill_index not in self.learned_skills:
                self.learned_skills.add(skill_index)
                return skill_index
        raise RuntimeError("No free skill slots available.")

    def set_active_skill(self, skill_index: int) -> None:
        self.active_skill = skill_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        if self.active_skill is None:
            raise RuntimeError("No active skill selected.")
        return self.skill_classifiers[self.active_skill](x)


# ============================================================
# PROBING HELPERS
# ============================================================

def _probe(loader: DataLoader, n_batches: int):
    """
    Pull up to n_batches from loader and concatenate them into one
    lower-variance probe sample, instead of trusting a single small batch.
    """
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    it = iter(loader)
    for _ in range(n_batches):
        try:
            x, y, *_ = next(it)
        except StopIteration:
            break
        xs.append(x)
        ys.append(y)
    if not xs:
        raise RuntimeError("Probe loader produced no batches.")
    return torch.cat(xs), torch.cat(ys)


def score_from_loss(loss_value: float) -> float:
    """
    Confidence-like score in (0, 1], derived from a CrossEntropyLoss value.

    Uses exp(-loss), i.e. the geometric mean of the predicted probability
    of the true class. (Kept as a standalone function so the transform is
    easy to audit/replace independently of the routing logic.)
    """
    return float(np.exp(-loss_value))


def compute_cl_metrics(accuracy_history: List[List[float]]):
    """
    Standard accuracy / forgetting curves from a per-experience accuracy
    history, where accuracy_history[t][j] is the accuracy on experience j
    after training through experience t (j <= t).
    """
    n_experiences = len(accuracy_history)
    accuracy_curve = []
    forgetting_curve = []

    for t in range(n_experiences):
        seen = np.asarray(accuracy_history[t], dtype=float)
        accuracy_curve.append(float(seen.mean()))

        if t == 0:
            forgetting_curve.append(0.0)
            continue

        forgetting_values = []
        for j in range(t):
            past_accuracies = [accuracy_history[k][j] for k in range(j, t)]
            best_accuracy = max(past_accuracies)
            current_accuracy = accuracy_history[t][j]
            forgetting_values.append(best_accuracy - current_accuracy)

        forgetting_curve.append(float(np.mean(forgetting_values)))

    return accuracy_curve, forgetting_curve


# ============================================================
# STRATEGY
# ============================================================

class SkillMemoryStrategy:
    def __init__(
        self,
        model: SkillClassifierBank,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        config: SkillMemoryConfig,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config

        # Previously seen experiences (used only to source OLD probe samples)
        self.seen_datasets: List[Any] = []

        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)

    def _log(self, msg: str) -> None:
        if self.config.verbose:
            print(msg)
        else:
            logger.info(msg)

    def _chance_for(self, skill_index: int) -> float:
        return 1.0 / self.model.skill_classifiers[skill_index].out_features

    # ============================================================
    # IMAGINATION FOR TRAINING DECISION
    # ============================================================

    @torch.no_grad()
    def imagine(self, new_x, new_y, old_x, old_y) -> List[Dict[str, Any]]:
        """
        Compare every existing skill on:
          - old data: probe samples from previously seen experiences
          - new data: probe samples from the current experience

        No training or allocation happens here.
        """
        new_x = new_x.flatten(start_dim=1)
        old_x = old_x.flatten(start_dim=1)

        results = []

        for skill_index in self.model.learned_skills:
            classifier = self.model.skill_classifiers[skill_index]

            old_logits = classifier(old_x)
            old_loss = self.criterion(old_logits, old_y).item()
            old_score = score_from_loss(old_loss)
            old_predictions = old_logits.argmax(dim=1)
            old_accuracy = (old_predictions == old_y).float().mean().item()

            new_logits = classifier(new_x)
            new_loss = self.criterion(new_logits, new_y).item()
            new_score = score_from_loss(new_loss)
            new_predictions = new_logits.argmax(dim=1)
            new_accuracy = (new_predictions == new_y).float().mean().item()

            results.append(
                {
                    "skill": skill_index,
                    "old_loss": old_loss,
                    "old_score": old_score,
                    "old_accuracy": old_accuracy,
                    "new_loss": new_loss,
                    "new_score": new_score,
                    "new_accuracy": new_accuracy,
                }
            )

        return results

    # ============================================================
    # FIND BEST SKILL
    # ============================================================

    def find_best_skill(self, imagination_results: List[Dict[str, Any]]):
        """
        Select an existing skill only when there is evidence that it is
        BOTH safe to reuse (forgetting guard on old data) AND compatible
        with the new experience (relative clustering + absolute floor on
        new data).
        """
        if not imagination_results:
            return None

        # ---------------------------------------------------------
        # FORGETTING GUARD: drop any skill whose grip on old data is
        # already weak -- reusing it would trivially "forget" further.
        # ---------------------------------------------------------
        safe_results = [
            result
            for result in imagination_results
            if result["old_accuracy"]
            > self._chance_for(result["skill"]) + self.config.forgetting_margin
        ]

        if not safe_results:
            return None

        # ---------------------------------------------------------
        # Special case: only one safe skill remains.
        # ---------------------------------------------------------
        if len(safe_results) == 1:
            result = safe_results[0]
            if result["new_accuracy"] > self._chance_for(result["skill"]):
                return result
            return None

        # ---------------------------------------------------------
        # Find the strongest cluster on one metric, with BOTH a
        # relative-gap requirement and an absolute floor.
        # ---------------------------------------------------------
        def strongest_candidates(results, key, floor):
            ranked = sorted(results, key=lambda r: r[key], reverse=True)
            values = [r[key] for r in ranked]

            gaps = [values[i] - values[i + 1] for i in range(len(values) - 1)]
            split = max(range(len(gaps)), key=lambda i: gaps[i])

            if gaps[split] <= 0:
                return set()

            candidates = ranked[: split + 1]
            candidates = [r for r in candidates if r[key] > floor]
            return {r["skill"] for r in candidates}

        score_floor = self.config.score_floor
        if score_floor is None:
            # Default: require score to beat a maximally-uncertain
            # (uniform) prediction's score, i.e. exp(-log(num_classes)).
            score_floor = 1.0 / max(
                self.model.skill_classifiers[r["skill"]].out_features
                for r in safe_results
            )

        accuracy_floor = max(self._chance_for(r["skill"]) for r in safe_results)

        score_candidates = strongest_candidates(safe_results, "new_score", score_floor)
        accuracy_candidates = strongest_candidates(
            safe_results, "new_accuracy", accuracy_floor
        )

        self._log(f"Score candidates: {sorted(score_candidates)}")
        self._log(f"Accuracy candidates: {sorted(accuracy_candidates)}")

        intersection = score_candidates & accuracy_candidates
        self._log(f"Compatible candidates: {sorted(intersection)}")

        if not intersection:
            return None

        candidates = [r for r in safe_results if r["skill"] in intersection]

        return max(
            candidates,
            key=lambda result: (result["new_score"], result["new_accuracy"]),
        )

    # ============================================================
    # SIMPLE IMAGINATION FOR EVALUATION
    # ============================================================

    @torch.no_grad()
    def imagine_current(self, x, y) -> List[Dict[str, Any]]:
        """
        Evaluate all existing skills on the current experience.
        Used at eval time to pick which skill to route the experience to.
        """
        x = x.flatten(start_dim=1)
        results = []

        for skill_index in self.model.learned_skills:
            classifier = self.model.skill_classifiers[skill_index]
            logits = classifier(x)
            loss_value = self.criterion(logits, y).item()
            score = score_from_loss(loss_value)
            predictions = logits.argmax(dim=1)
            accuracy = (predictions == y).float().mean().item()

            results.append(
                {
                    "skill": skill_index,
                    "loss": loss_value,
                    "score": score,
                    "accuracy": accuracy,
                }
            )

        return results

    # ============================================================
    # TRAIN EXPERIENCE
    # ============================================================

    def train_experience(self, train_exp) -> None:
        cfg = self.config

        train_loader = DataLoader(
            train_exp.dataset, batch_size=cfg.batch_size, shuffle=True
        )
        probe_loader = DataLoader(
            train_exp.dataset, batch_size=cfg.probe_batch_size, shuffle=True
        )

        new_x, new_y = _probe(probe_loader, cfg.probe_batches)
        new_x = new_x.to(cfg.device)
        new_y = new_y.to(cfg.device)

        reused = False

        # ====================================================
        # NO EXISTING SKILLS
        # ====================================================
        if not self.model.learned_skills:
            skill_index = self.model.allocate_skill()
            self._log(f"No existing skills -> allocated skill {skill_index}")

        # ====================================================
        # EXISTING SKILLS
        # ====================================================
        else:
            if not self.seen_datasets:
                raise RuntimeError(
                    "Model has learned skills, but no previously seen "
                    "datasets are available."
                )

            old_dataset = ConcatDataset(self.seen_datasets)
            old_loader = DataLoader(
                old_dataset, batch_size=cfg.probe_batch_size, shuffle=True
            )
            old_x, old_y = _probe(old_loader, cfg.probe_batches)
            old_x = old_x.to(cfg.device)
            old_y = old_y.to(cfg.device)

            imagination = self.imagine(new_x, new_y, old_x, old_y)

            self._log("\nImagination:")
            for result in imagination:
                self._log(
                    f"  skill {result['skill']}: "
                    f"old_score={result['old_score']:.3f}, "
                    f"old_acc={result['old_accuracy']:.3f}, "
                    f"new_score={result['new_score']:.3f}, "
                    f"new_acc={result['new_accuracy']:.3f}"
                )

            best = self.find_best_skill(imagination)

            if best is None:
                skill_index = self.model.allocate_skill()
                self._log(f"\nNo compatible existing skill -> allocated skill {skill_index}")
            else:
                skill_index = best["skill"]
                reused = True
                self._log(
                    f"\nBest compatible skill: {skill_index} "
                    f"(new_score={best['new_score']:.3f}, "
                    f"new_accuracy={best['new_accuracy']:.3f})"
                )

        # ====================================================
        # REAL TRAINING
        # ====================================================
        self.model.set_active_skill(skill_index)
        self.model.train()

        replay_loader = None
        if reused and cfg.replay_old_during_reuse and self.seen_datasets:
            replay_loader = DataLoader(
                ConcatDataset(self.seen_datasets),
                batch_size=cfg.batch_size,
                shuffle=True,
            )

        for _epoch in range(cfg.epochs_per_experience):
            for x, y, *_ in train_loader:
                x = x.to(cfg.device)
                y = y.to(cfg.device)

                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()

            # Optional: interleave a few old-data batches so a reused
            # skill doesn't purely overwrite what it already knew.
            if replay_loader is not None:
                replay_iter = iter(replay_loader)
                for _ in range(cfg.replay_batches_per_epoch):
                    try:
                        rx, ry, *_ = next(replay_iter)
                    except StopIteration:
                        break
                    rx = rx.to(cfg.device)
                    ry = ry.to(cfg.device)

                    self.optimizer.zero_grad()
                    logits = self.model(rx)
                    loss = self.criterion(logits, ry)
                    loss.backward()
                    self.optimizer.step()

        self.seen_datasets.append(train_exp.dataset)

    # ============================================================
    # EVALUATE EXPERIENCE
    # ============================================================

    @torch.no_grad()
    def evaluate_experience(self, experience) -> float:
        cfg = self.config
        self.model.eval()

        loader = DataLoader(experience.dataset, batch_size=cfg.batch_size, shuffle=False)
        probe_loader = DataLoader(
            experience.dataset, batch_size=cfg.probe_batch_size, shuffle=False
        )

        probe_x, probe_y = _probe(probe_loader, cfg.probe_batches)
        probe_x = probe_x.to(cfg.device)
        probe_y = probe_y.to(cfg.device)

        imagination = self.imagine_current(probe_x, probe_y)
        if not imagination:
            return 0.0

        best = max(imagination, key=lambda result: result["score"])
        skill_index = best["skill"]
        self.model.set_active_skill(skill_index)

        self._log(
            f"Evaluation using skill {skill_index}: "
            f"loss={best['loss']:.4f}, score={best['score']:.3f}, "
            f"accuracy={best['accuracy']:.3f}"
        )

        correct = 0
        total = 0
        for x, y, *_ in loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            logits = self.model(x)
            predictions = logits.argmax(dim=1)
            correct += (predictions == y).sum().item()
            total += y.numel()

        return correct / total


# ============================================================
# BENCHMARK HELPERS
# ============================================================

def evaluate_seen_experiences(strategy: SkillMemoryStrategy, test_stream, current_experience: int):
    accuracies = []
    for experience_index in range(current_experience + 1):
        accuracy = strategy.evaluate_experience(test_stream[experience_index])
        accuracies.append(accuracy)
    return accuracies
