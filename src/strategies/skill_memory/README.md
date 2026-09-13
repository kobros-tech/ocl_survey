# Class-level Skill Memory

This package is an Avalanche plugin implementing probe-based Skill Memory at
**class level**, rather than treating an Avalanche experience as one semantic
unit.

## Core semantics

For an experience containing classes `{c1, c2, ...}`:

1. Extract the labels actually present in `experience.dataset`.
2. Process each class independently, including classes found in later sub-experiences of the same logical experience.
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

`eval_routing="probe"` is the default and is the task-free evaluation mode.

Evaluation is still performed by Avalanche experience, matching the benchmark
protocol used by ordinary continual-learning baselines such as ER. Inside each
physical evaluation minibatch, however, samples may be routed to different
stored skills. The probe router does not inspect the target labels.

`class_oracle` is a diagnostic upper bound: it uses the true label to select the
canonical skill for each sample. `oracle` is retained for backwards
compatibility and swaps one skill for a whole evaluation experience; neither
should be reported as the task-free headline result.

The probe router does not use predictive entropy directly. Entropy is
problematic for snapshots with different classifier sizes, and a one-class
head has identically zero entropy for every input. Instead, routing uses a
classifier margin (or the single-class logit) normalized by the stored
classifier weight norm. This is an input-only routing heuristic; no target
label is used.

## Important invariant

A class is never silently remapped to another skill. If an attempt is made to
record a conflicting mapping, `ExperienceClassMap.record()` raises an error.
This prevents a later experience from accidentally changing the identity of a
class because of a noisy probe.
