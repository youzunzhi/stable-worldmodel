# Experiment T: goal-threshold data-sensitivity study

## Question

How sensitive is the frozen-encoder pointwise goal threshold to the sampled
calibration data when the task contract, dataset revision, preprocessing,
checkpoint, residual, dtype, and selection constraints remain fixed?

This study adds four data seeds to the existing formal seed and reports five
paired estimates per task under two pair-family conditions:

1. current method: 100,000,000 uniform ordered pairs plus 20,000,000 50/50
   task-stratified pairs;
2. stratified only: no uniform pairs and the same 20,000,000 50/50
   task-stratified pairs.

Cube retains the registered no-feasible-result semantics. For every Cube
replicate, the report additionally gives the descriptive best operating point:
the smallest epsilon attaining the maximum anchor-group macro TPR among finite
candidates with macro FPR at most 0.10. It is not promoted as a feasible
threshold when macro TPR is below 0.90.

## Frozen identity

For each task, the dataset path, dataset full-file SHA-256, label variant,
checkpoint path and SHA-256, encoder/projector parameter hash, preprocessing,
float32 `mean_D((z_i-z_j)^2)` residual with `D=192`, split fractions, and
selection constraints are inherited from the immutable
`formal-142ffa7-20260820` baseline. Predictor, action encoder, planner,
environment construction, and environment stepping remain outside scope.

The existing frozen embedding array may be reused only after its manifest,
row-ID mapping, checkpoint identity, dataset identity, latent dimension, dtype,
and file SHA-256 are verified. Reuse is exact computation reuse, not a change
in the observations or checkpoint.

## Paired seeds

The preregistered seeds are stored in
`scripts/experiments/observation_goal_threshold/data_sensitivity_config.json`.
`seed-0` is the existing formal run (`threshold_split_seed=260820`,
`pair_sampling_seed=90210`). `seed-1` through `seed-4` jointly change the
group-first split seed and pair-sampling seed. The same task-stratified sample
is used for the two conditions within a task/seed pair.

This is a paired design. It estimates joint sensitivity to group allocation
and pair sampling; it does not separately identify their individual effects.

## Pair counts and access boundary

The current method still materializes exactly 100M uniform and 20M
task-stratified pairs per new task/seed replicate and allocates them 60/20/20
by the existing group-first fit/validation/audit contract. Therefore threshold
selection sees exactly 60M uniform and 12M task-stratified fit pairs. The
stratified-only estimate uses exactly those same 12M fit task-stratified pairs,
which are the fit allocation of its preregistered 20M total design.

Only `threshold_fit` scores are opened in this sensitivity study. Validation
and audit pair manifests remain closed because the estimand is variability of
the threshold estimator, not promotion of nine new downstream thresholds.
These outputs are formal evidence for fit-estimator sensitivity only and must
not be described as newly validated, locked, audited, planner, execution, or
CLEAR evidence.

## Why the comparison is paired

Under the frozen selection rule, `min_population_precision` is null. Macro TPR
and macro FPR are computed exclusively from the task-stratified sample;
uniform pairs only estimate population prevalence, precision, and secondary
metrics. Consequently, removing uniform pairs is expected to produce an
exactly equal epsilon for a paired task/seed. The implementation must compute
both paths and fail if their reported epsilon or best operating point differs.

Changing the stratified sample or group split may still change epsilon. That
cross-seed variation is the primary data-sensitivity result.

## Outputs and acceptance checks

Each task/seed artifact records both seeds, split hash, pair-manifest hash,
fit score-manifest hashes, baseline embedding and checkpoint hashes, exact pair
counts, both selection payloads, and their paired epsilon delta. The aggregate
JSON and Markdown report include all per-seed values and, per task, the mean,
sample standard deviation, minimum, maximum, range, coefficient of variation,
and maximum absolute deviation from `seed-0`.

Before launch, targeted tests, formatting/lint checks, and the full repository
test suite must pass on the immutable execution commit. The final report must
record the execution commit and preserve the boundary that Cube's descriptive
best point is not a feasible selected threshold.
