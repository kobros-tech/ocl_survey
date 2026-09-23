"""Class-level probe-based Skill Memory for Avalanche."""

from .cl.decision import find_best_skill
from .cl.skill_memory_plugin import SkillMemoryPlugin
from .cl.skill_registry import ClassRecord, ExperienceClassMap, SkillMemory
from .evaluation.behavior import (
    BehaviorFingerprintCache,
    ClassBehaviorRecord,
    compare_binary_behavior,
    identify_binary_behavior,
    reverse_engineer_scores_from_weights,
    reverse_engineer_y,
    reverse_engineer_y_from_weights,
)
from .evaluation.diagnostics import class_index_alignment_report
from .evaluation.fingerprint_routing import PersistentFingerprintSkillMemoryPlugin
from .evaluation.ml_cl_evaluator import (
    EvaluationMemory,
    EvaluationMemoryPlugin,
    MLEvaluationPlugin,
    aggregate_experience_metrics,
    build_evaluator,
    compute_class_forgetting,
    compute_peak_class_forgetting,
    consolidate_evaluation_memory,
    evaluate_model_by_class,
    make_loader,
    train_evaluator,
)
from .evaluation.reverse_engineering import CandidateParameters, NormalMLReverseEngineer
from .evaluation.routing import RoutingResult, find_best_routing_skill
from .strategy import SkillMemoryStrategy

__all__ = [
    "BehaviorFingerprintCache",
    "CandidateParameters",
    "ClassRecord",
    "ClassBehaviorRecord",
    "EvaluationMemory",
    "EvaluationMemoryPlugin",
    "ExperienceClassMap",
    "RoutingResult",
    "SkillMemory",
    "SkillMemoryPlugin",
    "PersistentFingerprintSkillMemoryPlugin",
    "NormalMLReverseEngineer",
    "aggregate_experience_metrics",
    "build_evaluator",
    "class_index_alignment_report",
    "compare_binary_behavior",
    "compute_class_forgetting",
    "compute_peak_class_forgetting",
    "consolidate_evaluation_memory",
    "evaluate_model_by_class",
    "MLEvaluationPlugin",
    "identify_binary_behavior",
    "make_loader",
    "reverse_engineer_scores_from_weights",
    "reverse_engineer_y",
    "reverse_engineer_y_from_weights",
    "find_best_routing_skill",
    "find_best_skill",
    "train_evaluator",
    "SkillMemoryStrategy",
]
