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
  --random --seed 42 \
  --manifest /path/to/CLEAR-LeWM/manifests/v0.3/pusht/strict-seed42-n100.json

scripts/experiments/eval.sh \
  --model lewm --task pusht \
  --run-name clear-pusht-strict-official-seed42 \
  --checkpoint /path/to/pusht/weights.pt --seed 42 \
  --manifest /path/to/CLEAR-LeWM/manifests/v0.3/pusht/strict-seed42-n100.json
```

Run the same pair for `moderate` and for `cube`. Each result JSON records the
manifest SHA-256, embedded criterion, exact pair rows, resolved config,
checkpoint and dataset paths, per-episode outcomes, and runtime duration.
