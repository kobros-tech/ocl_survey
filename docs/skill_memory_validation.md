# Skill Memory validation protocol

This document defines the validation boundary for Skill Memory. It is intentionally separate from the final OCL benchmark comparison.

## Mechanism validation

The following properties should be tested independently of benchmark ranking:

- A registered state is copied to CPU storage and is independent of the caller's tensors.
- Stored states are not mutated by later training.
- Memory capacity is enforced.
- Best-match selection chooses the highest observed compatibility score.
- REUSE does not train the selected state.
- CLONE starts from the selected state but remains an independently trainable model.
- SCRATCH starts from the defined initial model state after current-task adaptation.
- A controlled transfer experiment can demonstrate benefit when the source skill is known to be useful.

An oracle class-to-skill mapping may be used for this controlled mechanism experiment because its purpose is to establish an upper-bound/transfer property. It must not be used by the final benchmark policy.

## Benchmark comparison

For the benchmark comparison, Skill Memory must choose its source skill and action from information available before training on the current experience. The compatibility probe may use training samples from that experience, but must never use validation/test labels or test predictions.

The comparison must keep the benchmark, task/experience protocol, model, optimizer settings, training budget, evaluation protocol, and seed set common across methods. Additional resources must be reported separately. A model-state memory budget is not equivalent to an example-replay budget by default.

## Compatibility score

The implementation currently reports

`score = clip(1 - probe_loss / log(K), 0, 1)`

where `K` is the configured number of classes. This is a normalized loss-derived score. It is not a probability and should not be described as an estimated probability of transfer.

The default thresholds (`clone_threshold=0.30`, `reuse_threshold=0.90`) are therefore operating points on this score. They must be fixed before final evaluation or selected using a documented independent calibration/sensitivity procedure that does not use final benchmark test results.

## Replication reporting

Every benchmark table should expose:

- strategy and configuration identity,
- benchmark and number of experiences,
- seed set and completed-run count,
- mean and standard deviation (or another pre-specified uncertainty summary),
- memory/resource accounting,
- whether any run was incomplete or excluded and why.

Missing runs should not silently disappear from a comparison.
