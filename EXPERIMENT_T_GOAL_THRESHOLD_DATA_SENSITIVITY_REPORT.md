# Experiment T: goal-threshold data sensitivity

## Result

The fitted threshold is only mildly sensitive to the joint group-split and
pair-sampling seed in these five replicates. PushT spans `1.630940914` to
`1.642531395` (range `0.011590481`, CV `0.3015%`), and TwoRoom spans
`1.386452556` to `1.401974201` (range `0.015521646`, CV `0.4436%`). Cube's
descriptive best operating point is even tighter, but Cube is infeasible in
all five replicates and therefore still has no promoted threshold.

Removing all 100M Uniform pairs changes the reported epsilon by exactly `0.0`
for every task and every paired seed. This is a structural consequence of the
frozen selector, not evidence that 100M random draws happened to agree: with
`min_population_precision=null`, Uniform pairs do not enter the macro-TPR or
macro-FPR threshold decision. They only support population prevalence,
precision, and secondary reporting.

## Per-seed thresholds

The existing formal run is `seed-0`; `seed-1` through `seed-4` are the four
new runs. Each row's current-method estimate used the existing 100M Uniform +
20M 50/50 task-stratified design. The paired stratified-only estimate used the
exact same task-stratified fit sample and is numerically identical.

| Task | Seed | Split seed | Pair seed | Current method | Stratified only | Fit macro TPR | Fit macro FPR | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PushT | seed-0 | 260820 | 90210 | 1.641965866 | 1.641965866 | 0.943112 | 0.099999 | feasible fit threshold |
| PushT | seed-1 | 260821 | 90211 | 1.635059595 | 1.635059595 | 0.933142 | 0.100000 | feasible fit threshold |
| PushT | seed-2 | 260822 | 90212 | 1.630940914 | 1.630940914 | 0.928853 | 0.100000 | feasible fit threshold |
| PushT | seed-3 | 260823 | 90213 | 1.639685392 | 1.639685392 | 0.933739 | 0.099999 | feasible fit threshold |
| PushT | seed-4 | 260824 | 90214 | 1.642531395 | 1.642531395 | 0.944619 | 0.100000 | feasible fit threshold |
| Cube | seed-0 | 260820 | 90210 | 1.679481864 | 1.679481864 | 0.707882 | 0.100000 | best point; not feasible |
| Cube | seed-1 | 260821 | 90211 | 1.679521322 | 1.679521322 | 0.709355 | 0.099999 | best point; not feasible |
| Cube | seed-2 | 260822 | 90212 | 1.680142164 | 1.680142164 | 0.709308 | 0.099999 | best point; not feasible |
| Cube | seed-3 | 260823 | 90213 | 1.679500580 | 1.679500580 | 0.709070 | 0.100000 | best point; not feasible |
| Cube | seed-4 | 260824 | 90214 | 1.679564834 | 1.679564834 | 0.708652 | 0.099997 | best point; not feasible |
| TwoRoom | seed-0 | 260820 | 90210 | 1.392462969 | 1.392462969 | 1.000000 | 0.029390 | feasible fit threshold |
| TwoRoom | seed-1 | 260821 | 90211 | 1.390118122 | 1.390118122 | 1.000000 | 0.027546 | feasible fit threshold |
| TwoRoom | seed-2 | 260822 | 90212 | 1.387668371 | 1.387668371 | 1.000000 | 0.027842 | feasible fit threshold |
| TwoRoom | seed-3 | 260823 | 90213 | 1.386452556 | 1.386452556 | 1.000000 | 0.028328 | feasible fit threshold |
| TwoRoom | seed-4 | 260824 | 90214 | 1.401974201 | 1.401974201 | 1.000000 | 0.029399 | feasible fit threshold |

## Across-seed sensitivity

| Task | Reported value | Mean | Sample SD | Min | Max | Range | CV | Max absolute change from seed-0 | Feasible |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PushT | selected fit threshold | 1.638036633 | 0.004938931 | 1.630940914 | 1.642531395 | 0.011590481 | 0.3015% | 0.011024952 | 5/5 |
| Cube | descriptive best point | 1.679642153 | 0.000281214 | 1.679481864 | 1.680142164 | 0.000660300 | 0.0167% | 0.000660300 | 0/5 |
| TwoRoom | selected fit threshold | 1.391735244 | 0.006173070 | 1.386452556 | 1.401974201 | 0.015521646 | 0.4436% | 0.009511232 | 5/5 |

Five seeds are enough to show that there is no large seed-driven movement in
this run, but they do not establish a precise sampling distribution. The seed
change jointly perturbs group allocation and pair sampling, so this study also
does not attribute the variation to one of those two sources separately.

## Contract and provenance

- Execution commit: `2b537e5337dbfaa306f576df2436d556addd388a`
- Remote artifact root: `/public/home/xsy0001/workspace/data/stable-worldmodel/experiments/observation_goal_threshold/data-sensitivity-2b537e5-20260821`
- Aggregate results SHA-256: `2393f85211098fd70f9e2d0296e4d88fdc9be4a4561bed22f3c2d95019a32300`
- Generated remote report SHA-256: `8664880e21bea7d6d75067842ef6cac58bf406b34a94617a02dae6f66d0fc994`
- Artifacts: 15/15 `fit_selection.json`, 15/15 `status.json`, and 15/15 effective task configs; total retained size approximately 45 GB.
- Each task has five distinct split hashes and five distinct pair-manifest hashes, but exactly one checkpoint hash and one frozen-embedding hash across seeds.
- Every new current-method replicate materialized exactly 100M Uniform + 20M task-stratified pairs. Under the unchanged 60/20/20 boundary, threshold selection opened 60M Uniform + 12M task-stratified fit pairs; validation and audit stayed closed.
- Live full-file dataset hashes and checkpoint hashes matched the existing formal baseline before launch. Frozen embedding files and row-ID maps were also re-hashed before reuse.
- Remote targeted tests: `38 passed`. Remote full suite: `1104 passed, 11 skipped, 1 xfailed`. Targeted Ruff passed; repo-wide Ruff remains blocked by 400 pre-existing violations outside this change.

The first background wrapper attempt exited before creating any run directory
because of launcher shell quoting (`bash: -m: command not found`). Its log is
retained. The corrected launcher then ran all preregistered task/seed pairs
once; no experimental seed was retried.

## Interpretation boundary

These are formal results for sensitivity of the `threshold_fit` estimator.
They are not newly validated, threshold-locked, or audit-opened calibrations.
Cube's values are explicitly descriptive best points under macro FPR `<=0.10`
and must not be treated as selected thresholds because their macro TPR remains
well below `0.90`.

The results concern frozen pointwise encoder geometry only. They are not
predictor, reachability, planner, execution, or official CLEAR evidence, and
the thresholds remain specific to the task, label contract, dataset revision,
checkpoint, preprocessing, residual, and dtype recorded in the artifacts.
