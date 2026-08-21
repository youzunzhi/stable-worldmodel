# find-goal-threshold implementation

This package implements
`FIND_GOAL_THRESHOLD_SPEC.md` for the three primary
pointwise variants: PushT joint XY, Cube block position, and TwoRoom agent XY.

The full driver constructs and hashes group-first splits and pair manifests,
encodes every eligible observation once with a frozen encoder/projector, scores
fit pairs, selects the threshold, applies it unchanged to validation, locks the
complete threshold tuple, and only then permits one-time audit scoring. A smoke
run uses fewer groups/pairs/bootstrap replicates and is explicitly marked as
non-formal evidence.

```bash
PYTHONPATH=. python -m \
  scripts.experiments.observation_goal_threshold.run \
  --config scripts/experiments/observation_goal_threshold/configs/pusht.json \
  --run-dir /path/to/immutable/pusht-run
```

Add `--smoke` for preflight. Never reuse a smoke output as a formal result.
Every formal task needs a new output directory on an immutable clean commit.

After the three task directories are complete:

```bash
PYTHONPATH=. python -m \
  scripts.experiments.observation_goal_threshold.summarize_three \
  --root /path/containing/pusht-cube-tworoom
```

The run writes `epsilon_tpr_fpr_curve.png`, with epsilon on the x-axis and
fit-split anchor-group macro TPR/FPR on the y-axis. To derive the same plot
from an immutable existing run without modifying it:

```bash
PYTHONPATH=. python -m \
  scripts.experiments.observation_goal_threshold.curve_plot \
  --run-dir /path/to/immutable/task-run \
  --output-dir /path/to/new/derived-curve-directory
```

## CLEAR endpoint self-eval

Self-eval is a downstream test performed only after epsilon is locked. The
full matrix is three tasks by CLEAR Moderate/Strict by 100 fixed pairs (600
trajectories). Each cell uses the same checkpoint as that task's threshold.
The normal CLEAR planner and evaluator produce the actual S/F label; after the
pair ends, the frozen encoder/projector computes endpoint-to-goal mean-D MSE
and predicts success exactly when the distance is at most epsilon.

Pass the locked artifact to the normal evaluator:

```bash
PYTHONPATH=. python scripts/plan/eval_wm.py --config-name=pusht \
  policy=/absolute/path/to/weights_epoch_10.pt \
  eval.dataset_name=/absolute/path/to/pusht_expert_train.h5 \
  eval.manifest=/absolute/path/to/pusht/moderate-seed42-n100.json \
  eval.find_goal_threshold=/absolute/path/to/pusht/selected_threshold.json \
  eval.video=false output.dir=/new/immutable/output
```

Repeat with each task config and both protocols, then aggregate the six
`results.txt.json` files with `summarize_self_eval`. The aggregator rejects
non-100-pair cells, solver/CPU-contract mismatches, duplicate pair IDs, and
checkpoint/threshold hash mismatches. It reports Wilson uncertainty for pair
accuracy and a deterministic paired bootstrap interval for predicted-minus-
actual success-rate error.

The current Cube calibration has no promoted epsilon. Cube self-eval must
therefore remain unavailable unless a new threshold-selection contract is
pre-registered and calibrated; do not substitute its best failed fit point.

## Threshold data sensitivity

The paired data-sensitivity study is preregistered in
`data_sensitivity_config.json` and run with the separate `data_sensitivity`
module. It reuses verified frozen embeddings, materializes the unchanged full
pair design for four new seeds, opens fit scores only, and derives a paired
stratified-only estimate from the same task-stratified shards. See
`EXPERIMENT_T_GOAL_THRESHOLD_DATA_SENSITIVITY_SPEC.md` for the access and
interpretation boundaries.
