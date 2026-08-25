# Self-supervised plannability v2

This package implements
[`SELF_SUPERVISED_PLANNABILITY_V2_SPEC.md`](../../../SELF_SUPERVISED_PLANNABILITY_V2_SPEC.md).

The formal launch order is:

1. clean committed source and repository tests;
2. create-only formal preparation for TwoRoom, PushT, and Reacher;
3. real-checkpoint local-response and progressive-overfit diagnostics;
4. three formal training replicates per task;
5. frozen validation promotion, held-out profile, then conditional CLEAR.

Cube is diagnostic-only under v2 and is excluded from the formal launch array.
Preparation, smoke, queued, partial, and failed artifacts are not formal
evidence.

## Reacher extension

Reacher is a formal SSP-v2 task with the preregistered strict float32 sum-SSE
verifier threshold `epsilon_task=0.7`. Its config pins the project-trained
epoch-10 checkpoint, the immutable Reacher HDF5 revision, both corrected
CLEAR v0.5 manifests, and the same SSP-v2 optimizer/CEM contract used by the
other tasks.

The `slurm_reacher_*.sh` entrypoints isolate the new Reacher cells from the
already-run TwoRoom/PushT roots:

- prepare and smoke are single jobs;
- train and profile are arrays `0-2`, mapping to seeds
  `260822/260823/260824`;
- CLEAR indices `0-3` are identity and repeated-identity controls; learned
  indices `4-9` map to the same three seeds, two protocols each. Submit only
  the learned indices whose held-out internal gates satisfy the frozen CLEAR
  promotion rule.
