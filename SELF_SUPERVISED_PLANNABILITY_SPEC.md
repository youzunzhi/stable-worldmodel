# Self-Supervised Plannability (SSP) experiment specification

Status: implementation-ready specification, not an experiment result

Protocol ID: `self-supervised-plannability-v1`

Short name: `SSP`

Source branch: `codex/self-supervised-plannability`

Source worktree: `outputs/worktrees/self-supervised-plannability`

Main base revision: `83f229b267f1c6be229546cb2bec93cbb253d5cf`

Tasks: `pusht`, `cube`, `tworoom`

## 1. Research question

SSP asks one narrowly scoped question:

> With a trained LeWM frozen, can a black-box outer optimizer learn a latent
> geometry that lets the unchanged finite-budget hard CEM search enter a
> pre-registered goal set earlier or more often?

For each task, SSP learns a task-specific map `G_psi`. It does not learn the
world-model encoder, predictor, action representation, policy, threshold, or
environment success predicate.

The causal chain under test is:

```text
psi -> G_psi search cost -> CEM elite ranking -> sampled action sequences
    -> first threshold-hit iteration T -> AUC reward -> outer ES update
```

The experiment is self-supervised in the limited sense that its training
signal is generated entirely inside the frozen world model. It uses no task
success labels, reward model, expert action target, simulator execution, or
CLEAR outcome during `G_psi` training.

## 2. Direct SSP scope

SSP is a direct learning experiment. It has no dependency on a preceding
random-geometry scan and must not consume artifacts from one.

The following are deliberately out of scope:

- scanning random metric families before training;
- using a scan to choose the threshold, ES perturbation scale, reward, or
  parameterization;
- training separate fixed-budget and AUC reward arms;
- learning or fine-tuning `E_0`, `F_0`, the action encoder, or the checkpoint;
- differentiating through CEM or replacing hard CEM with a differentiable
  planner;
- using the magnitude of a failed original distance as an outer reward;
- changing SSP hyperparameters after inspecting a formal task result;
- presenting smoke, preparation, submitted, partial, or failed jobs as formal
  evidence.

One fixed SSP configuration is pre-registered below. If it produces no
learning signal, that is an SSP outcome, not permission to start an unrecorded
metric or sigma sweep.

## 3. Frozen model and two-distance contract

Let the fixed trained LeWM be

```text
M_0 = (E_0, F_0).
```

For a start observation `o_t`, goal observation `o_g`, and action sequence
`A`, define

```text
z_t        = E_0(o_t)
z_g        = E_0(o_g)
z_hat_H(A) = F_0^(H)(z_t, A)
r(A)       = z_hat_H(A) - z_g.
```

`E_0` and `F_0` are in evaluation/inference mode and frozen for every SSP
stage. Their weights and configuration hashes must be identical before and
after a run.

SSP keeps two quantities separate:

1. Search cost, used by CEM for ranking:

   ```text
   c_psi(A) = ||G_psi(z_hat_H(A)) - G_psi(z_g)||_2^2.
   ```

2. Original hit distance, used only as a yes/no query:

   ```text
   d_0(A) = sum_{d=1}^{192} r_d(A)^2.
   ```

The hit bit is

```text
hit(A) = 1[d_0(A) < epsilon_task].
```

The comparison is strictly `<`, not `<=`.

### 3.1 Locked task thresholds

| Task | `epsilon_task` | Exact predicate |
|---|---:|---|
| PushT | 1.5 | `sum_D((z_hat_H - z_g)^2) < 1.5` |
| Cube | 1.0 | `sum_D((z_hat_H - z_g)^2) < 1.0` |
| TwoRoom | 1.5 | `sum_D((z_hat_H - z_g)^2) < 1.5` |

These are user-specified pre-registered constants. SSP does not calibrate,
validate, or tune them.

The values apply to a float32 **sum over the 192 latent dimensions**. They do
not apply to `mean_D((z_i-z_j)^2)`. Dividing by 192, changing preprocessing,
changing the checkpoint, or changing latent dimension creates a different
threshold contract and therefore a different protocol ID.

### 3.2 Information firewall

- `c_psi` is the only tensor given to `torch.topk` for elite selection.
- `d_0` is detached and converted to a boolean threshold result before the
  outer reward sees it.
- Failed values `d_0=1.6` and `d_0=100` are identical to the outer optimizer.
- Formal training artifacts store hit bits and `T`, not failed-distance
  magnitudes.
- No term proportional to `d_0`, `epsilon-d_0`, a sigmoid of either quantity,
  or conditional mean `d_0` may enter the outer objective.
- No learned success classifier or task-state label may replace the predicate.

## 4. Immutable input identities

The default is the project-trained seed-3072 M0 checkpoint for each task. An
official Hugging Face checkpoint is not an SSP substitute. Every path below is
a launch-time candidate that must exist and hash-match; the recorded hash, not
the path string, is authoritative.

| Task | Dataset identity | Project-trained checkpoint | Checkpoint SHA-256 | Config SHA-256 |
|---|---|---|---|---|
| PushT | `quentinll/lewm-pusht@655cd446b9929369d7d406001da85c15d1457850`, `pusht_expert_train.h5` | `.../lewm-dagger-pusht-v3/runs/checkpoints/lewm-dagger-pusht-m0-seed3072/weights_epoch_10.pt` | `e4f14a2276918bcb34876fb8d86d16dbd8683ae6077a13a2275dee008a68c775` | `c4a21dab4e6ab5803e9ae159bc7345104e4fe083116580fa5719357aea0af5b8` |
| Cube | `quentinll/lewm-cube@02a19a67a0dc8c9d6215f89c19e0a597691e152a`, `cube_single_expert.h5` | `.../lewm-dagger-r1-cube-tworoom-v2-73e9902d/cube/seed-3072/runtime/checkpoints/lewm-dagger-cube-m0-seed3072/weights_epoch_10.pt` | `eece65ce87e451d8ee953d83da0c566f77ad2d8f8f6ee5e77f7ddbae5bedf883` | `86f2ed24c61b48354416c23af51aa51279ae28a33cb36b7ebc3d057eec2b8c0d` |
| TwoRoom | `quentinll/lewm-tworooms@6903a2de048b13819d812da0b4dd661290bc01e4`, `tworoom.h5` | `.../lewm-dagger-r1-cube-tworoom-v2-73e9902d/tworoom/seed-3072/runtime/checkpoints/lewm-dagger-tworoom-m0-seed3072/weights_epoch_10.pt` | `68e6ca32ec5f7bdfb728ef89164ab62f566147e7724b74e5cf8e5858db746a65` | `c4a21dab4e6ab5803e9ae159bc7345104e4fe083116580fa5719357aea0af5b8` |

The expected current remote data roots are:

```text
/public/home/xsy0001/workspace/data/stable-worldmodel/hf/datasets/
/public/home/xsy0001/workspace/data/stable-worldmodel/tworoom/
```

The expected project-checkpoint roots are under the existing remote DAgger
worktrees. A formal launcher must resolve the full live paths, record them,
hash the files, and refuse a mismatch. Absence of an older `/ssd/sxu` path is
not evidence that a checkpoint is missing.

Every formal protocol records at least:

- source Git commit and dirty-state flag;
- dataset absolute path, byte size, upstream repository/revision, metadata
  fingerprint, and full-file SHA-256 when feasible;
- checkpoint/config absolute paths and SHA-256 values;
- preprocessing identity, dtype, latent dimension, and model class;
- Python, PyTorch, CUDA, GPU, and container/environment identities;
- the exact SSP config and hashes of every pair manifest and basis file.

## 5. Observation and pair construction

Each SSP planning instance is

```text
x = (o_t, o_{t+25}).
```

The goal offset is exactly 25 environment steps. The existing planner contract
represents this as five model steps with five-action blocks.

### 5.1 Eligibility

A pair is eligible only when:

- both rows are in the same source episode;
- the start and goal pixels exist and match the checkpoint rendering contract;
- `step_idx(goal) = step_idx(start) + 25`;
- neither row is corrupt or padding;
- the start row is not one of the fixed CLEAR Moderate/Strict start rows;
- the pair is not initially inside the SSP goal set:
  `sum_D((E_0(o_t)-E_0(o_g))^2) >= epsilon_task`.

Initial-hit filtering may inspect the boolean predicate only. Counts before
and after every filter are recorded.

### 5.2 Leakage groups and deterministic split

Split before sampling pairs:

- PushT group: `episode_idx // 101`, preserving all 101 recorded variants of a
  source demonstration in one split;
- Cube group: `ep_idx`;
- TwoRoom group: `ep_idx`.

Use split seed `260822` and group proportions `70/15/15` for
train/validation/test. The group lists are written before any formal planning
run and may not change afterward.

Sample without replacement and episode-balance within each split:

| Split | Pairs per task | Use |
|---|---:|---|
| train | 800 | 50 disjoint outer batches of 16 |
| validation | 256 | checkpoint selection only |
| test | 512 | held-out plannability profile |

If a task cannot supply these counts, preparation fails with
`SSP_INSUFFICIENT_ELIGIBLE_PAIRS`; it must not silently shrink, duplicate, or
move pairs between splits.

Formal CLEAR pairs are never used in ES reward, validation selection, or the
SSP internal test manifest. Episode overlap between CLEAR and SSP data must be
reported because the released manifests may span the available episode set;
start-row exclusion alone must not be described as episode-level isolation.

## 6. Learnable geometry family

Each task learns its own geometry. There is no cross-task sharing because
latent coordinates, checkpoints, datasets, and thresholds are different
contracts.

The first SSP version uses a 16-parameter positive diagonal metric in a fixed
task-specific subspace. Let `D=192`, `p=16`, and
`B_task in R^(D x p)` have orthonormal columns and be orthogonal to the
all-ones vector. Define

```text
ell_psi = B_task psi
w_psi   = exp(ell_psi)
G_psi(z) = diag(exp(ell_psi / 2)) z
c_psi(A) = sum_d w_psi,d * r_d(A)^2.
```

This gives:

- `psi=0` exactly equals the current identity/`GoalMSE` search cost;
- positive weights, so the cost is a valid diagonal squared metric;
- no unidentifiable global-scale direction, because `B_task^T 1 = 0`;
- only 16 outer parameters, keeping black-box ES feasible.

### 6.1 Basis construction

Construct `B_task` once on CPU in float64:

1. use NumPy `PCG64DXSM` and task basis seed;
2. draw a `192 x 16` standard-normal matrix;
3. project every column off the normalized all-ones vector;
4. perform reduced QR;
5. canonicalize each column sign so its largest-absolute entry is positive;
6. verify orthogonality and cast to float32;
7. save the array and its SHA-256.

| Task | Basis seed |
|---|---:|
| PushT | `26082201` |
| Cube | `26082202` |
| TwoRoom | `26082203` |

The basis is fixed across the three training replicates for a task. A basis
seed is not an outer-training seed.

### 6.2 Metric bound

After every optimizer update, project the center parameter by a single radial
rescale when necessary so that

```text
max_d |(B_task psi)_d| <= log(4).
```

Thus every coordinate weight remains in `[0.25, 4]`. Apply the same projection
operator to both antithetic evaluation points. Record how often projection is
active. Per-coordinate clipping, normalization by observed reward, or a
learned global scale is not allowed.

## 7. Inner hard-CEM contract

The planner must preserve the current `CEMSolver` semantics and the CLEAR
solver budget:

| Field | Value |
|---|---:|
| `batch_size` | 1 |
| candidates per CEM iteration | 300 |
| CEM iterations `K` | 30 |
| elites | 30 |
| initial variance scale | 1.0 |
| model horizon | 5 |
| receding horizon | 5 |
| action block | 5 |
| environment-step planning horizon | 25 |

Candidate 0 remains the current CEM mean. SSP must not introduce action
clipping or smoothing that the identity control does not receive. Any future
action-bound correction is a new shared planner contract and must be evaluated
from a fresh identity baseline.

At CEM iteration `k`, after rolling out all 300 candidates and before updating
the elite distribution, compute

```text
H_k = 1[there exists i with d_0(A_k,i) < epsilon_task].
```

The first-hit iteration is

```text
T = min{k in 1..30 : H_k = 1},
```

and `T=infinity` if no hit occurs. The full 30 iterations are executed in the
formal protocol even after a hit, keeping compute and random-number schedules
matched. Only the first hit enters the reward.

## 8. Common random numbers

Every `+eta/-eta` comparison uses common random numbers (CRN):

- the same ordered batch of 16 start-goal pairs;
- the same CEM initial means and variances;
- the same standard-normal candidate-noise tensor at every CEM iteration;
- the same candidate-0 mean insertion;
- the same model, dtype, preprocessing, and device.

Different geometry costs may select different elites after the first ranking,
so their means, variances, and realized action candidates naturally diverge.
That divergence is the causal effect SSP wants to measure.

The implementation uses a keyed, deterministic noise schedule. For every
`(task, replicate, outer_step, direction, pair_slot)` it records:

- seed/key;
- generator algorithm and runtime version;
- pre/post generator-state hashes;
- a streaming SHA-256 of the generated standard-normal blocks.

The two signs of a direction must reference the same noise-schedule ID and
content digest. A mismatch fails the step; it must not be averaged into a
gradient.

## 9. Outer ES training

For one pair, convert `T` to the locked AUC reward

```text
r_auc(T) = (30 - T + 1) / 30,  if T <= 30
           0,                  otherwise.
```

Equivalently,

```text
r_auc(T) = (1/30) * sum_{k=1}^{30} 1[T <= k].
```

This is the discrete area under the solve-vs-budget curve for a uniformly
distributed budget in `{1,...,30}`. It uses only threshold-hit timing, never a
failed `d_0` magnitude.

For an outer batch of `N=16` pairs,

```text
R(psi) = mean_n r_auc(T_psi,n).
```

At current center `mu`, sample `P=8` directions
`eta_j ~ Normal(0, I_16)` and evaluate

```text
psi_j+ = Project(mu + sigma eta_j)
psi_j- = Project(mu - sigma eta_j).
```

The raw antithetic estimator is

```text
g_hat = (1 / (2 P sigma)) * sum_j (R_j+ - R_j-) eta_j.
```

Use gradient ascent through Adam on `mu` only. Do not rank-transform rewards,
standardize sign-pair deltas, add a `d_0` baseline, or backpropagate through
the planner.

### 9.1 Locked outer hyperparameters

| Field | Value |
|---|---:|
| parameter dimension `p` | 16 |
| antithetic directions `P` | 8 |
| pairs per outer step `N` | 16 |
| perturbation scale `sigma` | 0.25 |
| optimizer | Adam ascent |
| learning rate | 0.05 |
| Adam betas | `(0.9, 0.999)` |
| Adam epsilon | `1e-8` |
| outer steps | 50 |
| checkpoint interval | 5 steps |
| validation interval | 5 steps |
| center initialization | `mu=0` (identity) |
| training replicates | 3 per task |
| replicate seeds | `260822`, `260823`, `260824` |

Each training replicate consumes the same fixed 800-pair train manifest but a
different deterministic permutation, ES-direction stream, and CRN schedule.
Every train pair appears exactly once in each replicate.

### 9.2 Checkpoint selection

Evaluate the center checkpoint at steps `0,5,...,50` on the fixed 256-pair
validation manifest with validation planner seed `26082290`. Select the largest
validation AUC. Exact ties choose the earlier outer step. `S(30)`, CLEAR
success, and test results are not selection criteria.

Step 0 is identity and is eligible only as a transparent null outcome. If it
wins, the result is “SSP did not learn a validation-improving geometry,” not a
learned improvement.

## 10. Preparation, smoke, and no-signal handling

Preparation and smoke exist to prove code paths, not to choose scientific
hyperparameters.

Required order per task:

1. `prepare-only`: resolve and hash source, data, checkpoint, manifests,
   preprocessing, basis, and split manifests without GPU planning;
2. unit/invariant tests;
3. one real-checkpoint, one-pair, three-CEM-iteration smoke for identity and
   one antithetic direction;
4. one complete 30-iteration paired-direction smoke on four pairs;
5. a fresh immutable formal root for all three replicates;
6. held-out profile evaluation;
7. secondary CLEAR evaluation.

Smoke roots contain `formal_evidence=false` and must never be renamed or
promoted into formal roots.

If every one of the eight direction deltas is exactly zero for ten consecutive
formal outer steps, stop that replicate and write
`SSP_NO_LEARNING_SIGNAL`. Record whether observed hit bits were all-zero,
all-one, or mixed. Do not change `epsilon`, `sigma`, reward, or metric family
inside that formal root. Other replicates continue independently.

A root exception during preparation must be reproduced and captured before
submitting a task matrix. A wrapper-level subprocess error alone is not enough
diagnostic evidence.

## 11. Primary held-out evaluation: plannability profile

The primary scientific object is the full held-out profile

```text
S_G(k) = Pr(T_G <= k),  k=1,...,30.
```

For the fixed 512-pair test manifest, evaluate:

- identity `G_0`;
- the selected checkpoint from each of the three SSP replicates;
- five planner seeds: `42,43,44,45,46`;
- exactly matched CRN between identity and every learned geometry.

Report for each task and geometry:

- all 30 values of `S_G(k)`;
- AUC `(1/30) sum_k S_G(k)`;
- endpoint solve rate `S_G(30)`;
- the empirical distribution of `T`, including censored failures;
- `B_50 = min{k:S_G(k)>=0.5}` when it exists;
- paired deltas from identity for AUC and every `S(k)`;
- all three training replicates separately and their fixed-replicate mean.

Do not report only the scalar training reward. Do not pool the three tasks as
the primary result.

### 11.1 Uncertainty

The unit of resampling is the start-goal pair. A paired bootstrap resamples
pairs and keeps all five matched planner seeds for a sampled pair together.
Use 10,000 deterministic bootstrap replicates with analysis seed `20260822`.

Report percentile 95% paired intervals for AUC and `S(30)`. For the 30-point
profile delta, report a simultaneous 95% band across `k=1..30` using the
bootstrap maximum absolute centered deviation. Pointwise intervals alone are
not enough for a stochastic-dominance statement.

### 11.2 Outcome language

- **Broad plannability improvement:** the simultaneous lower band for
  `S_learned(k)-S_identity(k)` is nonnegative for every `k` and strictly
  positive for at least one `k`.
- **Compute-efficiency trade-off:** AUC improves, but the curves cross or
  `S(30)` decreases. This is not broad dominance.
- **Endpoint-only trade-off:** `S(30)` improves while early budgets regress.
- **No clear evidence:** the paired uncertainty includes no-effect for the
  relevant estimand.
- **Regression:** the learned profile is credibly worse for the stated
  budgets or endpoint.

The report must not convert curve crossing into a single universal ordering.

## 12. Secondary evaluation: executed CLEAR v0.5

After geometry selection is locked, test whether internal plannability
transfers to task execution. This is secondary because SSP trains on a latent
hit event, while CLEAR scores real environment task completion.

Use the existing codebase contract:

- CLEAR-LeWM v0.5 revision
  `df026185a36bd9997c69d94753854db0b1a46f54`;
- PushT, Cube, and TwoRoom;
- Moderate and Strict fixed manifests, 100 pairs per cell;
- `batch_size=1`, `num_samples=300`, `n_steps=30`, `topk=30`;
- `cpu_threads=1`, policy seed 42, no video for formal scoring;
- task configs with `horizon/receding_horizon/action_block=5/5/5`;
- the same project-trained checkpoint as SSP;
- identity and all three selected SSP replicate geometries;
- a fresh identity rerun from the same source revision, not a historical score
  copied from another commit.

The learned `c_psi` replaces only the terminal `GoalMSE` ranking cost. The
original CLEAR task predicates remain the source of executed success. The SSP
`d_0` threshold does not replace, modify, or relabel CLEAR success.

Report task/protocol success counts, paired success delta, 95% paired bootstrap
interval, exact McNemar result, completed-pair count, and runtime. Treat CLEAR
transfer as supporting evidence, not proof that the encoder geometry is
globally correct.

## 13. Required codebase integration

Implementation should use the current seams rather than create a second world
model or policy stack.

### 13.1 Existing integration points

- `stable_worldmodel/planning/evaluator.py`
  - `ShootingCostEvaluator` already owns goal encoding and one-shot rollout.
- `stable_worldmodel/planning/objective.py`
  - `GoalMSE` is the exact identity baseline; despite its name, it sums the
    final latent squared residual over dimensions.
- `stable_worldmodel/planning/solver/cem.py`
  - `CEMSolver` owns candidate sampling, ranking, and elite updates.
- `stable_worldmodel/policy.py`
  - `WorldModelPolicy` and `PlanConfig` own MPC execution and warm start.
- `scripts/plan/eval_wm.py`
  - existing trained-checkpoint loading and CLEAR v0.5 execution path.
- `scripts/plan/clear_protocol.py`
  - fixed manifests, task success adapters, solver validation, and provenance.

### 13.2 Minimal new interfaces

Add behavior-preserving optional seams:

1. one evaluator call rolls out candidates once and returns both the search
   cost and detached SSP hit diagnostics;
2. CEM accepts an explicit deterministic candidate-noise source;
3. an iteration observer receives read-only hit bits/`T` diagnostics but
   cannot replace the ranking cost or candidates;
4. the default path with no SSP objective, observer, or noise source remains
   bitwise-compatible with current `GoalMSE + CEMSolver` behavior.

Do not run a second predictor rollout merely to compute `d_0`; `c_psi` and
`d_0` must be derived from the same predicted terminal embedding.

### 13.3 Proposed SSP package

```text
scripts/experiments/self_supervised_plannability/
  README.md
  contracts.py
  geometry.py
  pairs.py
  noise.py
  planner.py
  es.py
  run.py
  evaluate_profile.py
  summarize.py
  configs/
    pusht.json
    cube.json
    tworoom.json

tests/experiments/test_self_supervised_plannability.py
```

All newly written machine IDs, JSON schemas, paths, module names, and report
labels use `self-supervised-plannability` or `ssp` only.

## 14. Mandatory invariant tests

At minimum, tests must prove:

1. `psi=0` reproduces current `GoalMSE` candidate costs on real-shaped tensors;
2. `G_psi` weights are positive, bounded, and have no global-scale component;
3. `d_0` is float32 sum-SSE over exactly 192 dimensions;
4. task thresholds are exactly PushT 1.5, Cube 1.0, TwoRoom 1.5, with strict
   `<` behavior at the boundary;
5. the hit check observes all 300 candidates before the elite update;
6. changing failed `d_0` magnitudes while holding `c_psi` and hit bits fixed
   cannot change `T`, reward, or the ES update;
7. changing `c_psi` can change elite selection without changing the hit
   predicate implementation;
8. antithetic signs reference identical pair order and noise digests;
9. different elite choices may diverge means while consuming the same standard
   noise schedule;
10. reward equals the discrete solve-curve AUC identity;
11. failures are right-censored at 30 and contribute zero reward;
12. validation selection cannot read test or CLEAR outcomes;
13. the frozen checkpoint has no gradients, optimizer membership, or weight
   changes;
14. environment constructors and `step()` calls are zero during SSP training;
15. formal output roots are create-only and refuse overwrite/resume from an
   incompatible protocol;
16. the ordinary non-SSP planner path retains its existing tests and parity.

## 15. Artifact contract

Every formal root is immutable and create-only:

```text
outputs/experiments/self-supervised-plannability-v1/<task>/<replicate>/
```

Required artifacts include:

```text
protocol.json
source.json
environment.json
input_hashes.json
pre_registered_config.json
pair_manifests/
  train.jsonl
  validation.jsonl
  test.jsonl
geometry_basis.npy
geometry_basis.json
outer_steps.jsonl
noise_schedule.jsonl
checkpoints/
  step-000.pt
  ...
selected_geometry.pt
selected_geometry.json
training.completed.json
profile/
  per_pair_seed.jsonl
  solve_curve.csv
  summary.json
clear/
  moderate/results.txt.json
  strict/results.txt.json
audit.json
```

`outer_steps.jsonl` stores for every direction:

- center, projected `psi+/-`, `eta`, and projection incidence;
- ordered pair IDs and noise-schedule IDs;
- `T+/-`, hit bits, per-pair AUC rewards, `R+/-`, and raw delta;
- raw ES gradient, Adam state summary, center before/after;
- checkpoint and basis hashes.

It must not store failed `d_0` magnitudes as an optimizer-facing field.

`training.completed.json` is written only after all required outer steps or a
pre-registered no-signal stop, artifact validation, and final hash inventory.
A submitted job, checkpoint file, or partial JSONL is not completion evidence.

## 16. Validation and launch gates

Before an expensive formal launch:

1. `git diff --check`;
2. Ruff lint and format checks on changed Python files;
3. targeted SSP, evaluator, CEM, policy, and CLEAR protocol tests;
4. the full repository test suite;
5. clean committed SSP source revision;
6. prepare-only success with exact source hashes;
7. real-checkpoint smoke success;
8. one-cell full-budget local/interactive preparation success;
9. remote allocation, free disk, CUDA, dataset, checkpoint, and manifest
   preflight;
10. launch from a commit-pinned Git worktree or verified bundle;
11. bounded post-launch check of allocation, CUDA visibility, protocol hashes,
    and the first completed stage.

Only after these gates may a three-task matrix be submitted. Formal task roots
must be separate, and failed roots remain immutable provenance rather than
being overwritten by a retry.

## 17. Failure codes

At minimum, the implementation uses structured terminal codes:

| Code | Meaning |
|---|---|
| `SSP_INPUT_HASH_MISMATCH` | dataset/checkpoint/config/manifest identity differs |
| `SSP_LATENT_CONTRACT_MISMATCH` | preprocessing, dtype, or latent dimension differs |
| `SSP_INSUFFICIENT_ELIGIBLE_PAIRS` | locked split counts cannot be materialized |
| `SSP_INITIAL_HIT_LEAKAGE` | an initially solved pair reached a formal manifest |
| `SSP_IDENTITY_PARITY_FAILURE` | `psi=0` differs from current `GoalMSE` |
| `SSP_CRN_MISMATCH` | an antithetic pair does not share pair/noise identity |
| `SSP_FROZEN_MODEL_MUTATION` | M0 weights/config changed or received an update |
| `SSP_ENVIRONMENT_CALL_DURING_TRAINING` | training constructed or stepped a simulator |
| `SSP_NO_LEARNING_SIGNAL` | ten consecutive formal steps had all-zero direction deltas |
| `SSP_FORMAL_ROOT_EXISTS` | create-only output target already exists |
| `SSP_INCOMPLETE` | required terminal artifacts or evaluations are absent |

## 18. Interpretation boundary

A positive SSP result supports only this statement:

> Within the fixed checkpoint, data, metric family, thresholds, and finite CEM
> budget, black-box self-supervision learned a terminal latent geometry whose
> search reached the registered latent goal set with a better held-out
> solve-vs-budget profile.

It does not by itself prove:

- improved predictor accuracy;
- a more semantic or globally better representation;
- reachability or feasibility of every latent hit;
- improved open-loop action quality;
- task completion without the secondary executed evaluation;
- transfer to another checkpoint, task, dataset, threshold, planner, or
  planning budget.

The report must keep predictor error, search geometry, latent threshold hits,
and executed task success as separate evidence layers.
