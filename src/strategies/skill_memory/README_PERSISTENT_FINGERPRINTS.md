# Persistent binary class-behavior identification

This branch extends `v0.1.5-best-skill-routing` with one focused goal:

> Identify an anonymous class by reverse-engineering the learned classifier
> weights and testing the resulting binary `y` behavior for that candidate class.

The implementation does **not** use the benchmark experience ID or target label
for anonymous routing.

## Weight-based reverse engineering

For the default research path, a candidate class is evaluated from the stored
skill's learned parameters rather than by treating `model(x)` logits as the
reverse-engineering algorithm.

For an Avalanche `IncrementalClassifier`, the plugin captures the learned
feature representation `h` immediately before the classifier and explicitly
reconstructs:

```text
scores = h @ W.T + b
```

where `W` and `b` are the persisted classifier weights and bias. The candidate
binary behavior is then:

```text
y_hat(c) = argmax(scores) == c
```

This is the same affine classifier rule used by the learned head, but the
reverse-engineering implementation explicitly derives it from the learned
weights. PyTorch documents `nn.Linear` with the same `xA^T + b` formulation.

The package still accepts `reverse_engineer_y_fn` for experiments that need a
different research procedure. The injected function receives precomputed
scores/logits for backward compatibility; the default path is weight-based.

## Known-class validation

Reference samples for a known class have a known expected value:

```text
expected_y = True
```

The plugin persists the binary reference behavior and exposes
`reference_accuracy`. This validates whether the reverse-engineering procedure
can reproduce the expected class behavior before it is used for anonymous
routing.

## Anonymous identification

For every anonymous sample, the router evaluates every persistent class
fingerprint using its canonical skill's learned weights. The route is then:

1. reverse-engineer binary `y` for each candidate class;
2. identify the class that is compatible with the learned behavior;
3. resolve that class through the persistent canonical `class -> skill` map;
4. load the selected skill for the final prediction.

The routing result is intentionally discrete:

- `IDENTIFIED`: exactly one candidate class is compatible.
- `AMBIGUOUS`: more than one candidate class is compatible.
- `FAILED`: no candidate class is compatible.

There is no uniform-probability fallback to skill 0. A failed identification
remains failed instead of silently routing to the first slot.

No target label, task ID, or experience ID is consumed by anonymous routing.

## Persistent references

Each `ClassBehaviorRecord` stores:

- global `class_id`;
- canonical `skill_id`;
- skill generation/version;
- deterministic reference inputs;
- binary `reference_y` values;
- expected `y`.

If mutable `REUSE` changes a skill, every class mastered by that skill gets a
new generation. The original reference inputs are retained and reused when
refreshing the fingerprints. Unchanged skills are not refreshed. `SCRATCH`
creates the initial fingerprint for each newly mastered class.

## Diagnostics

Every anonymous routing record contains enough information to reconstruct the
decision:

- sample index;
- final status;
- selected class and skill, when identified;
- every candidate class and skill;
- predicted class and binary `y`;
- candidate class score from the learned weights;
- expected `y`;
- reference accuracy;
- candidate correctness.

This makes `IDENTIFIED`, `AMBIGUOUS`, and `FAILED` decisions directly
inspectable instead of reducing the result to a skill index.

## Tests and benchmark

The focused tests include an exact reconstruction check: the weight-based
reverse-engineering scores must match the classifier's own affine output, and
the resulting binary `y` must match the classifier argmax.

`skill_memory/tests/demo_splitmnist_weight_reverse_engineering.py` runs the
SplitMNIST benchmark through the persistent fingerprint plugin. The CI demo
uses this plugin directly; it does not pass experience IDs or labels to the
router.

The original `SkillMemoryPlugin` remains available separately for the older
probe-routing experiments.
