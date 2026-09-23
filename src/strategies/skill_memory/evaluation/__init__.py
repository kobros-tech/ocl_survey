"""Machine-learning evaluation and anonymous routing components."""

from .reverse_engineering import (
    CandidateParameters as CandidateParameters,
)
from .reverse_engineering import (
    NormalMLReverseEngineer as NormalMLReverseEngineer,
)
from .routing import RoutingResult as RoutingResult
from .routing import find_best_routing_skill as find_best_routing_skill
from .routing import route_probe_logits as route_probe_logits

# `ml_cl_evaluator` is deliberately NOT re-exported here (only from the
# top-level `skill_memory` package, and always importable directly as
# `skill_memory.evaluation.ml_cl_evaluator`). It subclasses
# `skill_memory.cl.skill_memory_plugin.SkillMemoryPlugin`, and `cl/`'s own
# modules trigger this package's `__init__` (via `..evaluation.routing`
# compatibility re-exports in `utils/probing.py`) before `cl` itself has
# finished loading. Importing `ml_cl_evaluator` here would import `cl`
# back before it's ready - a real circular import, not just a lint
# warning. See `skill_memory/__init__.py`, where `cl` is already fully
# loaded by the time `ml_cl_evaluator` is imported.
