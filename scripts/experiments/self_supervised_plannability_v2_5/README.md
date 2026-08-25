# Self-supervised plannability v2.5

This package implements
[`SELF_SUPERVISED_PLANNABILITY_V2_5_SPEC.md`](../../../SELF_SUPERVISED_PLANNABILITY_V2_5_SPEC.md).

The formal launch order is:

1. clean committed source and repository tests;
2. create-only formal preparation for TwoRoom, PushT, Cube, and Reacher;
3. real-checkpoint local-response and progressive-overfit diagnostics;
4. three formal training replicates per task;
5. frozen validation promotion, held-out profile, then conditional CLEAR.

Preparation, smoke, queued, partial, and failed artifacts are not formal
evidence.

## Unit correction

SSP-v2.5 applies the four registered epsilon values to a strict float32
mean-MSE verifier. The corrected mapping is:

- PushT: mean `1.5`, sum-SSE equivalent `288`;
- Cube: mean `1.0`, sum-SSE equivalent `192`;
- TwoRoom: mean `1.5`, sum-SSE equivalent `288`;
- Reacher: mean `0.7`, sum-SSE equivalent `134.4`.

The generic Slurm entrypoints cover all four tasks: prepare/smoke arrays
`0-3`, train/profile arrays `0-11`, and conditional CLEAR array `0-39`.
Every output root and artifact protocol ID is distinct from SSP-v2.
