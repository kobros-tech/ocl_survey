from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import torch
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

    def best_match(self, query, scorer: Callable[[SkillRecord, Any], float]):
        if not self._records or scorer is None:
            return None, 0.0
        scored = [(float(scorer(r, query)), r) for r in self._records.values()]
        score, record = max(scored, key=lambda x: x[0])
        return record, score

    def load_into(self, name: str, model: nn.Module):
        state_dict = self._records[name].state_dict
        _resize_incremental_classifiers_for_state(model, state_dict)
        model.load_state_dict(deepcopy(state_dict))


def _resize_incremental_classifiers_for_state(
    model: nn.Module, state_dict: Mapping[str, Tensor]
):
    """Recreate dynamic classifiers so their architecture matches a stored state."""
    for module_name, module in model.named_modules():
        if not isinstance(module, IncrementalClassifier):
            continue

        prefix = f"{module_name}." if module_name else ""
        weight_key = f"{prefix}classifier.weight"
        target_weight = state_dict.get(weight_key)
        if target_weight is None or target_weight.ndim != 2:
            continue

        target_units = target_weight.shape[0]
        if module.classifier.out_features == target_units:
            continue

        device = module.classifier.weight.device
        dtype = module.classifier.weight.dtype
        module.classifier = nn.Linear(
            module.classifier.in_features, target_units
        ).to(device=device, dtype=dtype)

        active_key = f"{prefix}active_units"
        if active_key in state_dict:
            module.active_units = state_dict[active_key].to(device=device).clone()


def _restore_initial_state(model: nn.Module, initial_state: Mapping[str, Tensor]):
    """Restore a scratch model without undoing Avalanche's current adaptation."""
    current = model.state_dict()
    for name, initial in initial_state.items():
        if name not in current:
            continue
        target = current[name]
        if target.shape == initial.shape:
            target.copy_(initial.to(device=target.device, dtype=target.dtype))
            continue

        if name.endswith("classifier.weight") and target.ndim == 2:
            rows = min(target.shape[0], initial.shape[0])
            target[:rows].copy_(initial[:rows].to(device=target.device, dtype=target.dtype))
        elif name.endswith("classifier.bias") and target.ndim == 1:
            rows = min(target.shape[0], initial.shape[0])
            target[:rows].copy_(initial[:rows].to(device=target.device, dtype=target.dtype))
        elif name.endswith("active_units"):
            continue
        else:
            raise RuntimeError(
                f"Cannot restore scratch state for {name}: "
                f"current shape {tuple(target.shape)}, "
                f"initial shape {tuple(initial.shape)}"
            )


def _origin_experience(experience):
    """Return the original task experience behind an online sub-experience."""
    return getattr(experience, "origin_experience", experience)


class ProbeCompatibilityScorer:
    """Zero-shot compatibility measured only on training data from the new experience."""

    def __init__(self, model_factory, loss_fn, probe_fn, reference_fn, probe_samples=64):
        self.model_factory = model_factory
        self.loss_fn = loss_fn
        self.probe_fn = probe_fn
        self.reference_fn = reference_fn
        self.probe_samples = probe_samples

    def __call__(self, record, experience):
        model = self.model_factory()
        _resize_incremental_classifiers_for_state(model, record.state_dict)
        model.load_state_dict(record.state_dict)

        adaptation_experience = _origin_experience(experience)
        model.train()
        avalanche_model_adaptation(model, adaptation_experience)
        model.eval()

        x, y = self.probe_fn(experience)
        with torch.no_grad():
            loss = float(self.loss_fn(model(x), y))
        reference = float(self.reference_fn(y))
        if reference <= 1e-8:
            return 1.0 if loss <= 1e-8 else 0.0
        return max(0.0, min(1.0, 1.0 - loss / reference))


def make_probe(experience, samples=64, seed=0):
    """Build a deterministic probe from the current training experience only."""
    n = min(samples, len(experience.dataset))
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(experience.dataset), generator=generator)[:n].tolist()
    subset = torch.utils.data.Subset(experience.dataset, indices)
    loader = DataLoader(subset, batch_size=n, shuffle=False)
    batch = next(iter(loader))
    return batch[0], batch[1]


def make_compatibility(model_factory, num_classes, probe_samples=64, probe_seed=0):
    return ProbeCompatibilityScorer(
        model_factory=model_factory,
        loss_fn=nn.functional.cross_entropy,
        probe_fn=lambda exp: make_probe(exp, probe_samples, seed=probe_seed),
        reference_fn=lambda _y: float(torch.log(torch.tensor(float(num_classes))).item()),
        probe_samples=probe_samples,
    )


class SkillMemoryPlugin(SupervisedPlugin):
    """Select REUSE, CLONE, or SCRATCH once per original task in an online stream."""

    REUSE, CLONE, SCRATCH = "reuse", "clone", "scratch"

    def __init__(self, memory=None, *, compatibility=None, skill_name=None,
                 max_skills=20, reuse_threshold=0.90, clone_threshold=0.30,
                 force_decision=None):
        super().__init__()
        if force_decision not in (None, self.REUSE, self.CLONE, self.SCRATCH):
            raise ValueError("invalid force_decision")
        if not 0 <= clone_threshold <= reuse_threshold <= 1:
            raise ValueError("require 0 <= clone_threshold <= reuse_threshold <= 1")
        self.memory = memory if memory is not None else SkillMemory(max_skills=max_skills)
        self.compatibility = compatibility
        self.skill_name = skill_name or (
            lambda exp: f"experience-{getattr(getattr(exp, 'origin_experience', None), 'current_experience', exp.current_experience)}"
        )
        self.reuse_threshold = reuse_threshold
        self.clone_threshold = clone_threshold
        self.force_decision = force_decision
        self.last_decision = self.SCRATCH
        self.last_selected_skill = None
        self.last_compatibility_score = 0.0
        self._initial_state = None
        self._saved_train_epochs = None
        self._task_active = False

    def _reset_optimizer(self, strategy):
        """Clear state and rebind optimizer groups after dynamic-module replacement."""
        optimizer = strategy.optimizer
        if optimizer is None:
            return

        current_params = list(strategy.model.parameters())
        old_count = sum(len(group["params"]) for group in optimizer.param_groups)
        if old_count != len(current_params):
            raise RuntimeError(
                "Skill Memory changed the model parameter count after optimizer "
                f"creation ({old_count} -> {len(current_params)}). Recreate the "
                "optimizer after model adaptation instead of rebinding it."
            )

        offset = 0
        for group in optimizer.param_groups:
            count = len(group["params"])
            group["params"] = current_params[offset : offset + count]
            offset += count
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
        """Expand the classifier to the complete task after loading a skill."""
        avalanche_model_adaptation(strategy.model, _origin_experience(experience))

    def before_training_exp(self, strategy, **kwargs):
        experience = strategy.experience

        if self._task_active and not self._is_first_subexp(experience):
            return

        self._task_active = True
        if self._initial_state is None:
            self._initial_state = {
                k: v.detach().cpu().clone()
                for k, v in strategy.model.state_dict().items()
            }
        self.last_decision = self.SCRATCH
        self.last_selected_skill = None
        self.last_compatibility_score = 0.0
        self._saved_train_epochs = None

        if len(self.memory) == 0:
            self._scratch(strategy)
            return

        record, score = self.memory.best_match(experience, self.compatibility)
        self.last_compatibility_score = score
        decision = self.force_decision
        if decision is None:
            if record is not None and score >= self.reuse_threshold:
                decision = self.REUSE
            elif record is not None and score >= self.clone_threshold:
                decision = self.CLONE
            else:
                decision = self.SCRATCH
        if record is None:
            decision = self.SCRATCH

        if decision == self.REUSE:
            self.memory.load_into(record.name, strategy.model)
            self._adapt_to_original_task(strategy, experience)
            self.last_decision = self.REUSE
            self.last_selected_skill = record.name
            self._saved_train_epochs = strategy.train_epochs
            strategy.train_epochs = 0
        elif decision == self.CLONE:
            self.memory.load_into(record.name, strategy.model)
            self._adapt_to_original_task(strategy, experience)
            self._reset_optimizer(strategy)
            self.last_decision = self.CLONE
            self.last_selected_skill = record.name
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
                    "experience": getattr(
                        getattr(experience, "origin_experience", None),
                        "current_experience",
                        experience.current_experience,
                    ),
                })

        self._task_active = False
