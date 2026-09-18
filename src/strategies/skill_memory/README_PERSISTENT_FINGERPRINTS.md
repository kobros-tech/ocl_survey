# Persistent class identification with normal ML reverse engineering

This branch extends `v0.1.5-best-skill-routing` with one focused goal:

> Identify an anonymous class with a standalone ML model trained from raw
> samples and responses produced by frozen Skill Memory snapshots.

The reverse-engineering model is deliberately separated from continual
learning. It never consumes the live CL feature representation and it is never
trained during evaluation.

## Normal ML reverse engineering

For each mastered class, the plugin keeps deterministic reference samples and
the complete frozen skill state for the corresponding skill generation. For a
reference sample and each candidate class/skill pair, the frozen skill produces
its full classifier response. The normal ML input is:

```text
input = [flattened raw sample x,
         padded full frozen logits,
         full frozen softmax response]
```

The supervised target is `1` only when the candidate class is the reference
sample's canonical class; otherwise it is `0`.

A small PyTorch MLP learns:

```text
P(candidate_class | raw sample, frozen skill response)
```

This is a learned compatibility model rather than a handcrafted argmax,
cosine similarity, fingerprint threshold, or binary decision rule.

## CL-safety boundary

The lifecycle is intentionally one-way:

```text
CL training
    |
    | complete frozen Skill Memory state
    v
reference data + frozen skill responses
    |
    | normal supervised ML training
    v
cached reverse model
    |
    | inference only
    v
anonymous class -> canonical skill -> frozen prediction
```

The reverse model is fitted **once after each completed logical training
experience** when the frozen candidate set changes. It is not fitted from
`after_eval_forward`. Therefore evaluation batch order, `torch.no_grad()`, and
the live CL representation cannot alter the reverse model.

For this plugin, `REUSE` is immutable by default. A stored skill snapshot is
therefore not changed by later training merely because a class is encountered
again. `CLONE` and `SCRATCH` produce new skill generations according to the
underlying Skill Memory policy.

## Known-class calibration

Reference samples provide the only labels used to train the reverse model.
For every reference sample, the canonical class is known while the model is
being calibrated. Candidate class/skill pairs from the stored memory create
both positive and negative examples.

Those labels are never supplied during anonymous evaluation. Evaluation uses
only the raw sample and the frozen candidate responses.

## Anonymous identification

For an anonymous sample, every persistent class candidate is evaluated by the
same cached reverse model:

```text
sample x + frozen candidate response
              |
              v
       normal ML reverse model
              |
              v
       candidate probability
              |
              v
       class -> canonical skill
```

The highest learned candidate probability selects the anonymous class. The
canonical `class -> skill` mapping then selects the stored skill used for the
final classifier prediction. No evaluation target label, task ID, or
experience ID is used for routing.

The router intentionally does not apply a universal `0.5` compatibility gate.
The output is a ranked learned probability over the currently known candidate
classes.

## Persistent references

Each `ClassBehaviorRecord` stores:

- global `class_id`;
- canonical `skill_id`;
- skill generation/version;
- deterministic reference inputs;
- binary reference diagnostics;
- expected `y`.

If a mutable Skill Memory configuration changes a skill generation, the
associated frozen behavior records are invalidated and rebuilt. With the
persistent plugin's default immutable `REUSE`, old generations remain stable.

## Diagnostics

Each anonymous routing record contains:

- sample index;
- selected class and skill;
- learned candidate probability;
- all candidate probabilities;
- evaluation experience index **only as a diagnostic field**;
- evaluation label **only as a diagnostic field**;
- final classifier prediction/correctness.

The evaluation label is recorded after routing for analysis; it is not an
input to the reverse model or the routing decision.

Routed accuracy and forgetting therefore combine two distinct effects:

1. reverse-routing correctness;
2. the prediction quality of the selected frozen skill.

An oracle-skill evaluation should be used to measure raw skill retention
separately from routing error.

## Tests and benchmark

The focused reverse-engineering tests verify that the standalone model can
learn candidate identity, does not mutate candidate parameters, and can be
saved/restored without changing predictions.

`skill_memory/tests/demo_splitmnist_weight_reverse_engineering.py` runs the
SplitMNIST benchmark through the persistent plugin. The reverse model is
trained after training experiences and only performs inference during
`eval`.

The original `SkillMemoryPlugin` remains available separately for older
probe-routing experiments.
