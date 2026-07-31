# CLEAR-LeWM evaluation

This checkout evaluates our `stable_worldmodel` planner and environments on
the fixed CLEAR-LeWM v0.3 manifests. The protocol source is pinned in
`clear_eval.json`; do not use a moving CLEAR-LeWM `main` checkout for reported
results.

The adapter keeps our model loading, CEM implementation, action clipping, and
physical rollout loop. It changes only the evaluation contract:

- fixed episode-balanced, initially unsolved start-goal pairs;
- PushT success from T-block position and angle, held for 3 or 5 steps;
- Cube success from position and orientation modulo 24 proper cube rotations,
  held for 3 or 5 steps;
- 100 pairs, goal offset 25, control budget 50, seed 42;
- CEM 300 samples, 30 elites, 30 iterations, solver batch size 1.
- Python, NumPy, Torch, CUDA, and policy seed 42; Torch CPU threads 1.

This is intentionally a reproduction with our runtime, not a claim that our
newer package and numerical stack are byte-identical to CLEAR-LeWM's published
`stable-worldmodel==0.1.0`, PyTorch 2.6.0, CUDA 12.4 environment.

## Source checkout

```bash
git clone --recurse-submodules \
  https://github.com/DavidSunok/CLEAR-LeWM.git /path/to/CLEAR-LeWM
git -C /path/to/CLEAR-LeWM checkout \
  f06b66b358f5e42aa582e4a5599d3356c29edcf4
```

Verify the four manifest hashes against `clear_eval.json` before evaluation.

## One matched evaluation

```bash
scripts/experiments/eval.sh \
  --model lewm --task pusht \
  --run-name clear-pusht-strict-random-seed42 \
  --random --seed 42 --no-video \
  --manifest /path/to/CLEAR-LeWM/manifests/v0.3/pusht/strict-seed42-n100.json

scripts/experiments/eval.sh \
  --model lewm --task pusht \
  --run-name clear-pusht-strict-official-seed42 \
  --checkpoint /path/to/pusht/weights.pt --seed 42 --no-video \
  --manifest /path/to/CLEAR-LeWM/manifests/v0.3/pusht/strict-seed42-n100.json
```

Run the same pair for `moderate` and for `cube`. Each result JSON records the
manifest SHA-256, embedded criterion, exact pair rows, resolved config,
checkpoint and dataset paths, per-episode outcomes, and runtime duration.

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
  --manifest /path/to/CLEAR-LeWM/manifests/v0.3/pusht/strict-seed42-n100.json
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
