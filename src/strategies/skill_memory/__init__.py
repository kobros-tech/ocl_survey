"""Class-level probe-based Skill Memory for Avalanche."""

from .behavior import (
    BehaviorFingerprintCache,
    ClassBehaviorRecord,
    compare_binary_behavior,
    identify_binary_behavior,
    reverse_engineer_scores_from_weights,
    reverse_engineer_y,
    reverse_engineer_y_from_weights,
)
from .decision import find_best_skill
from .diagnostics import class_index_alignment_report
from .evaluation import RoutingResult, find_best_routing_skill
from .fingerprint_routing import PersistentFingerprintSkillMemoryPlugin
from .reverse_engineering import CandidateParameters, NormalMLReverseEngineer
from .skill_memory_plugin import SkillMemoryPlugin
from .skill_registry import ClassRecord, ExperienceClassMap, SkillMemory

__all__ = [
    "BehaviorFingerprintCache",
    "CandidateParameters",
    "ClassRecord",
    "ClassBehaviorRecord",
    "ExperienceClassMap",
    "RoutingResult",
    "SkillMemory",
    "SkillMemoryPlugin",
    "PersistentFingerprintSkillMemoryPlugin",
    "NormalMLReverseEngineer",
    "class_index_alignment_report",
    "compare_binary_behavior",
    "identify_binary_behavior",
    "reverse_engineer_scores_from_weights",
    "reverse_engineer_y",
    "reverse_engineer_y_from_weights",
    "find_best_routing_skill",
    "find_best_skill",
]
