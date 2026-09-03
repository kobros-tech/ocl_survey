# OCL Survey

This repository provides a reproducible benchmark for online/continual learning (OCL) strategies.

## Experimental integrity

Benchmark results are generated artifacts and should not be treated as source-controlled inputs. Raw experiment outputs, TensorBoard logs, and machine-specific generated configurations are ignored by default. Curated tables or figures may be committed separately when they are intentionally part of the project record.

When comparing strategies, use the same benchmark, experience/task protocol, model, optimizer, training budget, evaluation protocol, and seed set. Report differences in additional resources explicitly. In particular, Skill Memory stores model states, while replay methods store examples; these are different resource types and should not be described as equal memory budgets without an explicit accounting.

## Skill Memory validation

Skill Memory is evaluated at two distinct levels:

1. **Mechanism validation:** immutable storage, independent REUSE/CLONE/SCRATCH behavior, compatibility scoring, and controlled transfer tests. Oracle routing may be used here only to verify that transfer is possible when the intended source skill is known.
2. **Benchmark comparison:** Skill Memory must make its own acquisition decision and must not receive oracle class-to-skill routing or other privileged test information. Compatibility probes use training data from the current experience only.

The compatibility score used by the current implementation is a normalized probe-loss score. It is **not a calibrated probability**. Thresholds therefore require methodological justification or independent calibration and must not be interpreted as probabilities by default.

## Reproducibility

The comparison should report the exact seed set, number of completed runs, and variability (for example, mean and standard deviation). Missing or inconsistent runs should be surfaced rather than silently mixed with stale artifacts.
