from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np
from torch import Tensor, nn
from torch.utils.data import DataLoader
from avalanche.models.dynamic_modules import IncrementalClassifier, avalanche_model_adaptation
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin


@dataclass
class SkillRecord:
    name: str
    state_dict: dict[str, Tensor]
    metadata: dict[str, Any] = field(default_factory=dict)


class SkillMemory:
    """Bounded registry of immutable, independently stored model states."""

    def __init__(self, max_skills: int = 20):
        if max_skills < 1:
            raise ValueError("max_skills must be positive")
        self.max_skills = max_skills
        self._records: dict[str, SkillRecord] = {}

    def register(self, name: str, state_dict: Mapping[str, Tensor], metadata=None):
        if name not in self._records and len(self._records) >= self.max_skills:
            raise RuntimeError(f"skill memory is at capacity ({self.max_skills})")
        self._records[name] = SkillRecord(
            name=name,
            state_dict={k: v.detach().cpu().clone() for k, v in state_dict.items()},
            metadata=dict(metadata or {}),
        )

    def names(self):
        return list(self._records)

    def __len__(self):
        return len(self._records)

    def records(self):
        return list(self._records.values())

    def get(self, name: str) -> SkillRecord:
        return self._records[name]

    def load_into(self, name: str, model: nn.Module):
        state_dict = self._records[name].state_dict
        _resize_incremental_classifiers_for_state(model, state_dict)
        model.load_state_dict(deepcopy(state_dict), strict=False)


def _resize_incremental_classifiers_for_state(model: nn.Module, state_dict: Mapping[str, Tensor]):
    for module_name, module in model.named_modules():
        if not isinstance(module, IncrementalClassifier):
            continue
        prefix = f"{module_name}." if module_name else ""
        target_weight = state_dict.get(f"{prefix}classifier.weight")
        if target_weight is None or target_weight.ndim != 2:
            continue
        if module.classifier.out_features == target_weight.shape[0]:
            continue
        device = module.classifier.weight.device
        dtype = module.classifier.weight.dtype
        module.classifier = nn.Linear(module.classifier.in_features, target_weight.shape[0]).to(device=device, dtype=dtype)
        active_key = f"{prefix}active_units"
        if active_key in state_dict:
            module.active_units = state_dict[active_key].to(device=device).clone()


def _restore_initial_state(model: nn.Module, initial_state: Mapping[str, Tensor]):
    current = model.state_dict()
    for name, initial in initial_state.items():
        if name not in current:
            continue
        target = current[name]
        if target.shape == initial.shape:
            target.copy_(initial.to(device=target.device, dtype=target.dtype))
        elif name.endswith("classifier.weight") and target.ndim == 2:
            rows = min(target.shape[0], initial.shape[0])
            target[:rows].copy_(initial[:rows].to(device=target.device, dtype=target.dtype))
        elif name.endswith("classifier.bias") and target.ndim == 1:
            rows = min(target.shape[0], initial.shape[0])
            target[:rows].copy_(initial[:rows].to(device=target.device, dtype=target.dtype))
        elif name.endswith("active_units"):
            continue
        else:
            raise RuntimeError(f"Cannot restore initial state for {name}")


def _origin_experience(experience):
    return getattr(experience, "origin_experience", experience)


def score_from_loss(loss_value: float) -> float:
    """Geometric-mean true-class probability induced by cross entropy."""
    return float(np.exp(-loss_value))


def _probe(experience, samples: int, batches: int, seed: int | None = None):
    if len(experience.dataset) == 0:
        raise RuntimeError("Cannot probe an empty experience")
    generator = None
    if seed is not None:
        generator = __import__("torch").Generator().manual_seed(seed)
    loader = DataLoader(
        experience.dataset,
        batch_size=min(samples, len(experience.dataset)),
        shuffle=True,
        generator=generator,
    )
    xs, ys = [], []
    for i, batch in enumerate(loader):
        if i >= max(1, batches):
            break
        xs.append(batch[0])
        ys.append(batch[1])
    if not xs:
        raise RuntimeError("Probe loader produced no batches")
    return __import__("torch").cat(xs), __import__("torch").cat(ys)


def _load_and_adapt(model_factory: Callable[[], nn.Module], record: SkillRecord, experience):
    model = model_factory()
    _resize_incremental_classifiers_for_state(model, record.state_dict)
    model.load_state_dict(record.state_dict, strict=False)
    avalanche_model_adaptation(model, _origin_experience(experience))
    model.eval()
    return model


def _evaluate_state(model_factory, record, experience, x, y, loss_fn):
    model = _load_and_adapt(model_factory, record, experience)
    device = next(model.parameters()).device
    x, y = x.to(device), y.to(device)
    with __import__("torch").no_grad():
        logits = model(x)
        loss = float(loss_fn(logits, y).item())
        accuracy = float((logits.argmax(dim=1) == y).float().mean().item())
    return loss, score_from_loss(loss), accuracy


class ProbeCompatibilityScorer:
    """Probe a stored skill on training data from a new experience."""

    def __init__(self, model_factory, loss_fn, probe_fn, reference_fn, probe_samples=64, probe_batches=5, seed=None):
        self.model_factory = model_factory
        self.loss_fn = loss_fn
        self.probe_fn = probe_fn
        self.reference_fn = reference_fn
        self.probe_samples = probe_samples
        self.probe_batches = probe_batches
        self.seed = seed

    def __call__(self, record, experience):
        x, y = self.probe_fn(experience)
        _, score, _ = _evaluate_state(self.model_factory, record, experience, x, y, self.loss_fn)
        return score


def make_probe(experience, samples=64, batches=5, seed=None):
    return _probe(experience, samples=samples, batches=batches, seed=seed)


def make_compatibility(model_factory, num_classes, probe_samples=64, probe_batches=5, seed=None):
    return ProbeCompatibilityScorer(
        model_factory=model_factory,
        loss_fn=nn.functional.cross_entropy,
        probe_fn=lambda exp: make_probe(exp, samples=probe_samples, batches=probe_batches, seed=seed),
        reference_fn=lambda _y: float(np.log(num_classes)),
        probe_samples=probe_samples,
        probe_batches=probe_batches,
        seed=seed,
    )


class SkillMemoryPlugin(SupervisedPlugin):
    """Probe-based Skill Memory retaining REUSE, CLONE and SCRATCH decisions."""

    REUSE, CLONE, SCRATCH = "reuse", "clone", "scratch"

    def __init__(self, memory=None, *, compatibility=None, skill_name=None, max_skills=20,
                 reuse_threshold=0.90, clone_threshold=0.30, forgetting_margin=0.05,
                 probe_samples=64, probe_batches=5, probe_seed=None, force_decision=None):
        super().__init__()
        if force_decision not in (None, self.REUSE, self.CLONE, self.SCRATCH):
            raise ValueError("invalid force_decision")
        if not 0 <= clone_threshold <= reuse_threshold <= 1:
            raise ValueError("require 0 <= clone_threshold <= reuse_threshold <= 1")
        self.memory = memory if memory is not None else SkillMemory(max_skills=max_skills)
        self.compatibility = compatibility
        self.skill_name = skill_name or (lambda exp: f"experience-{getattr(getattr(exp, 'origin_experience', None), 'current_experience', exp.current_experience)}")
        self.reuse_threshold = reuse_threshold
        self.clone_threshold = clone_threshold
        self.forgetting_margin = forgetting_margin
        self.probe_samples = probe_samples
        self.probe_batches = probe_batches
        self.probe_seed = probe_seed
        self.force_decision = force_decision
        self.last_decision = self.SCRATCH
        self.last_selected_skill = None
        self.last_compatibility_score = 0.0
        self.last_old_accuracy = 0.0
        self.last_new_accuracy = 0.0
        self._initial_state = None
        self._saved_train_epochs = None
        self._task_active = False
        self._seen_experiences: list[Any] = []

    def _reset_optimizer(self, strategy):
        optimizer = strategy.optimizer
        if optimizer is None:
            return
        params = list(strategy.model.parameters())
        if not optimizer.param_groups:
            optimizer.add_param_group({"params": params})
        else:
            optimizer.param_groups[0]["params"] = params
            for group in optimizer.param_groups[1:]:
                group["params"] = []
        optimizer.state.clear()

    def _scratch(self, strategy):
        _restore_initial_state(strategy.model, self._initial_state)
        self._reset_optimizer(strategy)

    @staticmethod
    def _is_first_subexp(experience):
        return getattr(experience, "is_first_subexp", True)

    @staticmethod
    def _is_last_subexp(experience):
        return getattr(experience, "is_last_subexp", True)

    def _adapt_to_original_task(self, strategy, experience):
        avalanche_model_adaptation(strategy.model, _origin_experience(experience))

    def _probe_current(self, experience, seed_offset=0):
        seed = None if self.probe_seed is None else self.probe_seed + seed_offset
        return make_probe(experience, samples=self.probe_samples, batches=self.probe_batches, seed=seed)

    def _score_records(self, strategy, experience):
        new_x, new_y = self._probe_current(experience, 0)
        model_factory = lambda: deepcopy(strategy.model)
        results = []
        for record in self.memory.records():
            new_loss, new_score, new_accuracy = _evaluate_state(model_factory, record, experience, new_x, new_y, nn.functional.cross_entropy)
            old_index = record.metadata.get("experience")
            if isinstance(old_index, int) and 0 <= old_index < len(self._seen_experiences):
                old_experience = self._seen_experiences[old_index]
            else:
                old_experience = self._seen_experiences[0]
            old_x, old_y = self._probe_current(old_experience, 100003 + len(results))
            old_loss, old_score, old_accuracy = _evaluate_state(model_factory, record, old_experience, old_x, old_y, nn.functional.cross_entropy)
            results.append({"record": record, "skill": record.name, "old_loss": old_loss,
                            "old_score": old_score, "old_accuracy": old_accuracy,
                            "new_loss": new_loss, "new_score": new_score, "new_accuracy": new_accuracy})
        return results

    def before_training_exp(self, strategy, **kwargs):
        experience = strategy.experience
        if self._task_active and not self._is_first_subexp(experience):
            return
        self._task_active = True
        if self._initial_state is None:
            self._initial_state = {k: v.detach().cpu().clone() for k, v in strategy.model.state_dict().items()}
        self.last_decision = self.SCRATCH
        self.last_selected_skill = None
        self.last_compatibility_score = 0.0
        self.last_old_accuracy = 0.0
        self.last_new_accuracy = 0.0
        self._saved_train_epochs = None

        if len(self.memory) == 0:
            self._scratch(strategy)
            return
        if not self._seen_experiences:
            raise RuntimeError("Skill Memory has skills but no previous experiences to probe")

        results = self._score_records(strategy, experience)
        classifier = getattr(strategy.model, "classifier", None)
        chance = 1.0 / max(1, getattr(classifier, "out_features", 10))
        safe = [r for r in results if r["old_accuracy"] > chance + self.forgetting_margin]
        best = max(safe or results, key=lambda r: (r["new_score"], r["new_accuracy"]))
        self.last_selected_skill = best["skill"]
        self.last_compatibility_score = best["new_score"]
        self.last_old_accuracy = best["old_accuracy"]
        self.last_new_accuracy = best["new_accuracy"]

        decision = self.force_decision
        if decision is None:
            if best in safe and best["new_score"] >= self.reuse_threshold:
                decision = self.REUSE
            elif best in safe and best["new_score"] >= self.clone_threshold:
                decision = self.CLONE
            else:
                decision = self.SCRATCH

        if decision in (self.REUSE, self.CLONE):
            self.memory.load_into(best["record"].name, strategy.model)
            self._adapt_to_original_task(strategy, experience)
            self._reset_optimizer(strategy)
            self.last_decision = decision
            if decision == self.REUSE:
                self._saved_train_epochs = strategy.train_epochs
                strategy.train_epochs = 0
        else:
            self._scratch(strategy)
            self.last_decision = self.SCRATCH

    def after_training_exp(self, strategy, **kwargs):
        experience = strategy.experience
        if not self._is_last_subexp(experience):
            return
        if self._saved_train_epochs is not None:
            strategy.train_epochs = self._saved_train_epochs
            self._saved_train_epochs = None
        if self.last_decision != self.REUSE:
            name = self.skill_name(experience)
            if name not in self.memory._records:
                self.memory.register(name, strategy.model.state_dict(), metadata={
                    "acquisition_decision": self.last_decision,
                    "selected_skill": self.last_selected_skill,
                    "compatibility_score": self.last_compatibility_score,
                    "old_accuracy": self.last_old_accuracy,
                    "new_accuracy": self.last_new_accuracy,
                    "probe_samples": self.probe_samples,
                    "probe_batches": self.probe_batches,
                    "probe_seed": self.probe_seed,
                    "experience": getattr(getattr(experience, "origin_experience", None), "current_experience", experience.current_experience),
                })
        self._seen_experiences.append(_origin_experience(experience))
        self._task_active = False
