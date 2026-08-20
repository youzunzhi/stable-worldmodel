# Experiment T implementation

This package implements
`EXPERIMENT_T_DEMO_CALIBRATED_GOAL_THRESHOLD_SPEC.md` for the three primary
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
