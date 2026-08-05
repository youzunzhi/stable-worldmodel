# CLEAR-LeWM evaluation

This checkout evaluates our `stable_worldmodel` planner and environments on
the fixed CLEAR-LeWM v0.5 manifests. The protocol source is pinned in
`clear_eval.json`; do not use a moving CLEAR-LeWM `main` checkout for reported
results.

The adapter keeps our model loading, CEM implementation, action clipping, and
physical rollout loop. It changes only the evaluation contract:

- fixed episode-balanced, initially unsolved start-goal pairs;
- Moderate PushT scores pusher plus T position within 20 px and T angle within
  20 degrees; the first hit succeeds;
- Strict PushT scores only the T within 10 px and 10 degrees for 3 steps;
- Moderate Cube scores cube-center position within 4 cm; the first hit
  succeeds;
- Strict Cube scores cube-center position within 3 cm and orientation within
  15 degrees modulo 24 proper cube rotations for 3 steps;
- Moderate TwoRoom uses continuous swept-disk collision and succeeds within
  16 px on clean cross-room pairs;
- Strict TwoRoom additionally requires a legal door crossing, the goal side,
  a valid complete route, and endpoint distance below 8 px;
- 100 pairs, goal offset 25, control budget 50; canonical manifests use
  policy seeds 0, 1, and 42, with seed 42 used for community submissions;
- CEM 300 samples, 30 elites, 30 iterations, solver batch size 1.
- Python, NumPy, Torch, CUDA, and policy seed 42; Torch CPU threads 1.

Upstream v0.5 also defines Reacher. This adapter supports PushT, Cube, and
TwoRoom, the three tasks in our trained-checkpoint registry. Use the upstream
evaluator for Reacher rather than labeling an unsupported local run as
CLEAR-LeWM v0.5.

This is intentionally a reproduction with our runtime, not a claim that our
newer package and numerical stack are byte-identical to CLEAR-LeWM's published
`stable-worldmodel==0.1.0`, PyTorch 2.6.0, CUDA 12.4 environment.

## Source checkout

```bash
git clone --recurse-submodules \
  https://github.com/DavidSunok/CLEAR-LeWM.git /path/to/CLEAR-LeWM
git -C /path/to/CLEAR-LeWM checkout \
  df026185a36bd9997c69d94753854db0b1a46f54
```

Verify the selected manifest hash against `clear_eval.json` before evaluation.

## One matched evaluation

```bash
scripts/experiments/eval.sh \
  --model lewm --task pusht \
  --run-name clear-pusht-strict-random-seed42 \
  --random --seed 42 --no-video \
  --manifest /path/to/CLEAR-LeWM/manifests/v0.5/pusht/strict-seed42-n100.json

scripts/experiments/eval.sh \
  --model lewm --task pusht \
  --run-name clear-pusht-strict-official-seed42 \
  --checkpoint /path/to/pusht/weights.pt --seed 42 --no-video \
  --manifest /path/to/CLEAR-LeWM/manifests/v0.5/pusht/strict-seed42-n100.json
```

Run the same pair for `moderate`, and for `cube` or `tworoom`. The pinned
official TwoRoom checkpoint is `quentinll/lewm-tworooms` revision
`77adaae0bc31deab21c93740d1f8bb947cd0bdec`; its source `weights.pt` SHA-256 is
`566f223624ea4bfb39dbfe6ae731198dd6ea73b7b8919fed6b1ecafca810f7dd`.
Each result JSON records the
manifest SHA-256, embedded criterion, exact pair rows, resolved config,
checkpoint/config SHA-256, optional checkpoint provenance, dataset path,
per-episode outcomes, runtime duration, and TwoRoom route diagnostics.

Omit `--no-video` when rollout videos are needed. The flag affects only video
collection and encoding; metrics and structured result files are still saved.

## Matched solver ablation

Use the same manifest, checkpoint, seed, horizon, action block, success
criterion, and control budget while replacing CEM with gradient-based planning:

```bash
scripts/experiments/eval.sh \
  --model lewm --task pusht \
  --run-name solver-ablation-pusht-strict-gd-seed42 \
  --checkpoint /path/to/pusht/weights.pt --seed 42 --no-video \
  --solver gd \
  --manifest /path/to/CLEAR-LeWM/manifests/v0.5/pusht/strict-seed42-n100.json
```

The GD config uses 100 AdamW restarts, 30 updates, learning rate 0.1, and the
same seeded Gaussian initialization scale as CEM. Override fields such as
`solver.num_samples` or `solver.optimizer_kwargs.lr` only as an explicitly
reported tuning ablation.

Because the solver is part of the CLEAR contract, this is a matched-pair
solver ablation rather than a protocol-conformant CLEAR result. Its structured
JSON records `solver_contract_matched: false` and the complete resolved solver
configuration. It is also not compute-matched to CEM: every GD update includes
backpropagation, while every CEM iteration evaluates sampled candidates
without backward.

## Canonical results

The current seed-42 v0.5 result matrix, bundled manifests, checkpoint hashes,
and all 16 structured result JSONs are tracked under
`results/clear_eval/v0.5/`. The version README is the headline report and
`summary.json` is the machine-readable index. Treat the JSON files under
`raw/` as the primary evidence.
