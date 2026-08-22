# Self-Supervised Plannability v2 experiment specification

Status: frozen implementation and launch contract; not an experiment result

Protocol ID: `self-supervised-plannability-v2`

Short name: `SSP-v2`

Source branch: `codex/self-supervised-plannability`

Tasks promoted to formal training: `tworoom`, `pusht`

Task held out from the v2 go/no-go claim: `cube`

## 1. Research question and boundary

With a project-trained LeWM frozen, can binary feedback from its registered
latent goal predicate learn a search geometry that increases the probability
that fixed-budget hard CEM returns a model-verified plan?

SSP-v2 does **not** test whether an arbitrary geometry exists in a continuous
space. It contains no Monte-Carlo `existence scan`. Negative evidence is
scoped to the registered rotated family, optimizer, data, and compute budget.

The primary internal estimand is the paired difference in returned verified
hit probability after 30 CEM iterations. CLEAR execution is a later secondary
transfer test and is not part of training or checkpoint selection.

Cube is not a formal go/no-go task because its pointwise latent verifier has no
feasible operating point under the preceding threshold study. It may be run
later as a declared mechanistic diagnostic, but cannot decide SSP-v2.

## 2. Frozen components and information firewall

For start observation `o_t`, goal observation `o_g`, and normalized action
plan `A`, the checkpoint supplies

```text
z_t = E_0(o_t)
z_g = E_0(o_g)
z_hat_1:5(A) = F_0(z_t, A).
```

`E_0`, `F_0`, the action encoder, dataset, observation preprocessing, task
threshold, and hard-CEM budget remain frozen. The immutable hit predicate is

```text
y(A) = 1[min_{h=1..5} sum_D((z_hat_h(A) - z_g)^2) < epsilon_task].
```

The comparison is strict `<`; accumulation is float32 sum-SSE over all 192
latent coordinates. Thresholds remain PushT `1.5`, Cube `1.0`, and TwoRoom
`1.5`.

Only `y(A)` may enter success gating, archive state, outer reward, or formal
internal evaluation. For failed candidates, `d0=1.6` and `d0=100` remain
indistinguishable to the outer learner. No failed-distance magnitude,
simulator reward, expert action target, CLEAR result, or task-state label may
enter learning.

## 3. Shared verified-hit planner contract

Identity and learned arms use the exact same solver implementation and
candidate-noise tapes.

### 3.1 Clip-consistent candidates

The LeWM was trained on z-score normalized dataset actions, while executed
actions are clipped to each task's raw `[-1,1]` action box. Before every model
rollout and elite update, each proposed normalized candidate is transformed by

```text
A_raw     = A_norm * action_std + action_mean
A_clipped = clip(A_raw, -1, 1)
A_model   = (A_clipped - action_mean) / action_std.
```

Thus the model evaluates the normalized form of the action that would be sent
to the environment. The full-dataset action mean/std, raw bounds, and their
hashes are preparation artifacts. This transform is shared by identity and
learned geometry.

### 3.2 Lexicographic elite selection

At every CEM iteration candidates are ordered by

```text
(not y(A), c_theta(A), candidate_index).
```

Every verified hit precedes every failure; `c_theta` orders candidates only
within the same binary class. This does not rank failures by original `d0`.

### 3.3 Verified-hit archive and return value

Every evaluated hit is eligible for an absorbing archive. The archive keeps
the lowest search-cost hit, with deterministic iteration/index tie-breaking.
After all 30 iterations:

```text
if archive non-empty: return archived verified candidate
else:                 return lowest-cost evaluated candidate
```

The unevaluated final elite mean is retained only as a diagnostic and is never
the v2 controller output. Candidate, elite, archive, and returned-hit fields
are recorded at every solve.

### 3.4 Inner budget

The locked CEM contract remains batch size 1, 300 candidates, 30 iterations,
30 elites, initial standard-deviation scale 1.0, five model steps, five raw
actions per model step, and 25 environment steps per plan.

## 4. Learned geometry

SSP-v2 replaces the single random diagonal basis with an action-effect aligned
rotated metric. Preparation takes the 800 training pairs and their recorded
25-step expert action sequences. It z-score normalizes and clip-projects those
actions, then applies a deterministic Gaussian perturbation of scale `0.5`.
For each pair it computes

```text
delta_z = terminal(F_0(z_t, A + delta_A))
        - terminal(F_0(z_t, A)).
```

The top 32 right singular vectors of the uncentered action-effect matrix form
orthonormal columns `Q in R^(192 x 32)`. Validation/test pairs and their
actions never contribute to `Q`. The singular spectrum, explained second
moment, orthogonality error, perturbation seed, clip incidence, and array hash
are saved.

For terminal residual `r=z_hat_5-z_g`, define

```text
lambda(theta) = log(4) * tanh(theta)
c_theta(r) = ||r||^2
           + sum_j (exp(lambda_j)-1) * (q_j^T r)^2.
```

Consequently `theta=0` is bitwise-parity identity GoalMSE, eigenvalues are in
`[0.25,4]` in the action-effect subspace and exactly one outside it, and the
metric contains cross-coordinate terms in the original latent axes. The only
learned values are 32 task-specific parameters; the LeWM and `Q` are buffers.

## 5. Training objective and optimizer

For iteration `k`, let `m_k` be the fraction of its 300 candidates satisfying
the binary trajectory-aware predicate. Let `a` indicate a non-empty archive
after iteration 30. The fixed-budget reward is

```text
R(theta) = a + 0.25 * mean(m_26, ..., m_30).
```

The archive term is primary; late hit mass provides binary, goal-aligned dense
credit and cannot make a failed plan equivalent to a verified return.
First-hit AUC is diagnostic only.

Training uses antithetic evolution strategies on `theta`:

- 32 parameters;
- 16 mutually orthogonal directions per outer step, scaled to norm `sqrt(32)`;
- perturbation scale `sigma=0.5`;
- 32 start-goal pairs per step;
- two independent planner tapes per pair/sign;
- common pairs, directions, and tapes for every plus/minus comparison;
- all directions in a step share the same pair/tape set;
- four deterministic epochs over the 800 train pairs, 100 outer steps total;
- Adam ascent, learning rate `0.03`, betas `(0.9,0.999)`, epsilon `1e-8`;
- checkpoints and validation every 10 steps;
- three optimizer replicates `260822`, `260823`, `260824`.

No random-geometry screening or test-dependent sigma selection is allowed.
The preformal diagnostic evaluates local response at sigma `0.25/0.5/1.0`,
but `0.5` remains locked regardless of its result; the diagnostic estimates
learnability and catches implementation failures, not hyperparameter choice.

## 6. Validation and identity fallback

Every checkpoint is evaluated on the fixed 256-pair validation manifest with
five planner tapes `42..46`. The same tapes are used for identity and every
checkpoint.

Two records are produced:

1. `best_candidate`: earliest checkpoint within one standard error of the
   largest mean fixed-budget reward;
2. `promoted_geometry`: `best_candidate` only if its paired returned-hit gain
   over identity exceeds one standard error; otherwise exact step-zero
   identity is promoted.

Both records are retained. Only the promoted geometry is eligible for the
one-shot held-out profile and CLEAR. This prevents noisy best-of-11 selection
from silently becoming a learned-improvement claim.

## 7. Preformal diagnostics and progressive overfit

Before formal training, each promoted task must complete a real-checkpoint
diagnostic root, explicitly marked `formal_evidence=false`, containing:

- exact identity parity of rotated cost and GoalMSE;
- candidate clip bounds and clip incidence;
- a synthetic end-to-end archive/lexicographic/return test;
- expert-action model hit rate under the trajectory-aware verifier;
- identity any-hit, elite-hit, archive-hit, returned-hit, and late-hit-mass
  funnel on fixed pairs;
- antithetic CRN equality;
- local response at sigma `0.25/0.5/1.0`;
- single-pair then eight-pair short overfit traces.

Failure to overfit is a structured diagnostic outcome, not evidence that no
geometry exists. Missing artifacts, hash mismatch, model mutation, CRN
mismatch, non-finite values, or planner invariant failure blocks formal jobs.

## 8. Formal held-out evaluation

The 512-pair test split stays unopened until checkpoint promotion is frozen.
Identity and promoted geometry use five paired planner tapes.

Primary:

```text
Delta returned_verified_hit_rate at K=30
```

Secondary: late hit mass, archive hit, first-hit AUC over budgets 2..30,
candidate-to-elite retention, return modes, and clip incidence. Budget 1 is a
registered structural equality and is excluded from simultaneous dominance
testing. Uncertainty resamples leakage groups (PushT source group or task
episode), preserving all pairs and planner tapes within each group.

An internal task is positive only if the paired 95% confidence interval for
returned-hit difference excludes zero in the positive direction. AUC or a
single replicate cannot override this primary gate.

## 9. CLEAR promotion and causal controls

CLEAR is launched only after a task passes its internal gate in at least two
optimizer replicates. Identity and learned cells must both use the v2 rotated
cost code path: identity is represented by the same `Q` and exact zero
`theta`. They must use the same manifest, pair order, seed, candidate-noise
schedule, GPU class, deterministic settings, solver contract, and action
clipping. A repeated zero-theta cell is a mandatory numerical control.

CLEAR remains secondary execution transfer evidence. A positive internal
result with null CLEAR transfer diagnoses verifier/model/execution mismatch;
it does not erase the internal search result.

## 10. Formal launch order and terminal codes

For TwoRoom and PushT:

1. repository tests, formatting, clean committed revision;
2. create-only formal preparation with full dataset/checkpoint/config hashes;
3. nonformal real-checkpoint diagnostic and progressive overfit;
4. six create-only formal training replicates;
5. frozen validation selection;
6. one-shot paired held-out profile;
7. conditional CLEAR promotion.

Submitted, queued, smoke, diagnostic, partial, or failed work is never formal
evidence. Stable failures include `SSP_V2_INPUT_HASH_MISMATCH`,
`SSP_V2_CRN_MISMATCH`, `SSP_V2_FROZEN_MODEL_MUTATION`,
`SSP_V2_DIAGNOSTIC_FAILED`, `SSP_V2_INCOMPLETE`, and
`SSP_V2_FORMAL_ROOT_EXISTS`.
