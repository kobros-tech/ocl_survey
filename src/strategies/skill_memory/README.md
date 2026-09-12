# Class-level Skill Memory

This package is an Avalanche plugin implementing probe-based Skill Memory at
**class level**, rather than treating an Avalanche experience as one semantic
unit.

## Core semantics

For an experience containing classes `{c1, c2, ...}`:

1. Extract the labels actually present in `experience.dataset`.
2. Process each class independently.
3. If the class has already been mastered, use its canonical `class -> skill`
   mapping. It is not re-assigned by the generic probe heuristic.
4. For a genuinely new class, probe each stored skill using only that class's
   samples.
5. A candidate skill must remain safe on **all** classes that skill already
   masters.
6. `REUSE` loads the same reserved slot and, when `reuse_is_mutable=True`,
   trains it on the new class and stores the updated state back to the same
   slot.
7. `SCRATCH` restores the pristine model, adapts it for the current
   experience, trains only the target class, and stores a new slot.

The bookkeeping is:

```text
experience -> [(skill, {classes})]
class      -> canonical skill
skill      -> all mastered classes
```

## Files

- `skill_registry.py` — skill snapshots and class/experience bookkeeping.
- `probing.py` — class filtering, probing, model-state helpers.
- `decision.py` — REUSE/SCRATCH decision logic.
- `training.py` — one-class-at-a-time training loop.
- `skill_memory_plugin.py` — Avalanche orchestration.

## Evaluation routing

`eval_routing="none"` is the recommended benchmark setting.

`oracle` and `probe` remain coarse diagnostic modes that swap one skill for an
entire evaluation experience. They are **not** class-level/sample-level
mixture-of-experts inference and should not be used as the headline result.

## Important invariant

A class is never silently remapped to another skill. If an attempt is made to
record a conflicting mapping, `ExperienceClassMap.record()` raises an error.
This prevents a later experience from accidentally changing the identity of a
class because of a noisy probe.
