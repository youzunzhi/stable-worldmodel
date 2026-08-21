# find-goal-threshold spec: demo-calibrated observation-observation latent goal threshold

## 0. Scope and decision

This specification defines one standalone experiment whose only purpose is to
select and audit a binary threshold on the geometry of a frozen encoder:

\[
S_\epsilon(o_i,o_j)=\mathbf 1\{d_E(o_i,o_j)\le\epsilon\},
\]

\[
d_E(o_i,o_j)=\frac1D\left\|E_0(o_i)-E_0(o_j)\right\|_2^2.
\]

Both observations come directly from demonstration-data rows. The experiment
deliberately excludes predictor error: it does not compute a predicted endpoint
and it does not call the LeWM predictor or rollout path.

Positive and negative labels are computed from the privileged simulator state
stored at the two rows under a pre-registered, single-time-step task-space
predicate with an explicit tolerance gap:

```text
positive: task_error < positive_threshold
negative: task_error > negative_threshold
ignored:  positive_threshold <= task_error <= negative_threshold
```

where `negative_threshold > positive_threshold`. The ignored region is an
intentional abstention band and is not silently assigned to either class.

The selected latent threshold is locked into a provenance-complete artifact
before any downstream predictor, planning, geometry-metric, or environment
experiment reads it.

This experiment does **not** implement or evaluate:

- the LeWM predictor \(F_0\), predicted endpoints, or autoregressive rollout;
- action-conditioned reachability or any fixed-horizon dynamics question;
- alternative planning metrics, Mahalanobis geometry, or metric selection;
- CEM, candidate ranking, or planner comparison;
- policy or world-model training;
- simulator execution;
- path legality, collision history, route completion, or sustained success;
- official CLEAR Moderate/Strict success unless a pointwise predicate is
  separately proven equivalent;
- threshold tuning from downstream results.

The unit of a threshold is the complete tuple

```text
(task, pointwise label variant, encoder checkpoint, dataset version,
 observation/preprocessing semantics, residual definition, dtype)
```

Changing any member of the tuple, either side of the task-space tolerance gap,
or whether the latent residual is a sum or mean requires a new calibration
artifact.

---

## 1. Research question

Freeze the encoder \(E_0\) from one pretrained LeWM checkpoint. For each
eligible ordered pair of demonstration observations and simulator states,

\[
(o_i,s_i),\qquad(o_j,s_j),
\]

compute

\[
z_i=E_0(o_i),\qquad z_j=E_0(o_j),
\]

\[
d_{ij}=\frac1D\|z_i-z_j\|_2^2.
\]

Let the task-specific pointwise error be

\[
\Delta_{ij}=\Delta_v(s_i,s_j),
\]

where \(v\) identifies the exact label variant. Given pre-registered
task-space thresholds \(\tau_T<\tau_F\), assign

\[
y_{ij}=\begin{cases}
T,&\Delta_{ij}<\tau_T,\\
F,&\Delta_{ij}>\tau_F,\\
U,&\tau_T\le\Delta_{ij}\le\tau_F.
\end{cases}
\]

`T` means task-space equivalent under the pointwise predicate, `F` means
task-space non-equivalent, and `U` means deliberately unresolved.

The experiment asks:

> Can one pre-registered global latent radius \(\epsilon^*\), selected only
> from dedicated demonstration calibration groups, include a high fraction of
> pointwise task-space-equivalent observation pairs while excluding most
> task-space-non-equivalent pairs, and does that behavior generalize to held-out
> demonstration groups?

This is an encoder-geometry binary-classification calibration question. It is
not a prediction, reachability, ranking, or planning-performance question.

---

## 2. Why observation-observation pairs are primary

Using

\[
\|\hat z_H-E_0(o_j)\|^2
\]

would combine encoder geometry with multi-step predictor error. find-goal-threshold
intentionally isolates encoder geometry. Every latent in this experiment is a
direct encoding of a recorded observation. Predictor error requires a separate
experiment and must not influence \(\epsilon^*\).

Exact self-pairs `(i, i)` are excluded from primary analysis because they are
guaranteed zero-distance positives. They may be retained only as a separate
encoder-determinism sanity check.

Primary positives must include non-identical rows and, where compatible data
exist, cross-trajectory pairs whose simulator states satisfy the positive
predicate. The experiment therefore tests a task-space equivalence
neighborhood, not identity reconstruction.

---

## 3. Pointwise label contract and tolerance gap

### 3.1 Single-row semantics

Each side contributes exactly one recorded observation and its single-row
simulator state. Labels must not depend on:

- actions or reachability between the observations;
- collisions, route history, or room crossing;
- previous or following frames;
- success holding for multiple steps;
- latent or pixel distance.

If an official task rule contains such requirements, this experiment uses a
separately named pointwise variant.

### 3.2 PushT default variant

The pre-registered PushT variant is:

```yaml
task_label:
  variant: pusht_joint_xy_pointwise_gap20_30
  state_fields: [pusher_x, pusher_y, block_x, block_y]
  metric: joint_l2
  positive_if_lt: 20.0
  negative_if_gt: 30.0
  unit: pixel
  ignored_fields: [block_angle, pusher_velocity_x, pusher_velocity_y]
  exact_self_pairs: exclude
```

For rows \(i,j\),

\[
\Delta_{ij}=\sqrt{
(p^x_i-p^x_j)^2+(p^y_i-p^y_j)^2+
(b^x_i-b^x_j)^2+(b^y_i-b^y_j)^2}.
\]

The strict inequalities are part of the contract: exactly `20.0` and `30.0`
belong to the ignored band. This variant must not be called PushT Moderate or
Strict because it ignores block angle and held success.

### 3.3 Cube default position variant

The primary pre-registered Cube variant is position-only:

```yaml
task_label:
  variant: cube_block_xyz_pointwise_gap03_04
  state_fields: [privileged_block_0_pos]
  metric: l2
  positive_if_lt: 0.03
  negative_if_gt: 0.04
  unit: metre
  ignored_fields:
    - privileged_block_0_quat
    - privileged_block_0_yaw
    - qvel
    - robot_qpos
    - effector_state
    - gripper_contact
  exact_self_pairs: exclude
```

For rows \(i,j\),

\[
\Delta^{\mathrm{pos}}_{ij}=
\left\|p^{\mathrm{block}}_i-p^{\mathrm{block}}_j\right\|_2.
\]

Thus `<0.03 m` is `T`, `0.03..0.04 m` inclusive is `U`, and `>0.04 m`
is `F`. The inner and outer radii reuse the existing strict and moderate Cube
position tolerances only as pointwise task-space anchors. This variant is not
Cube Moderate or Strict: it ignores orientation and never infers held success
from neighboring rows.

Cube compatibility must fix the target object identity, object inventory,
scene/layout, camera, and rendering semantics. If a target marker is visible in
the encoded observation, its rendering metadata must also match across a pair;
otherwise the observation contract must declare that the marker is absent or
removed by pre-registered preprocessing.

### 3.4 Cube secondary symmetry-aware pose variant

Any orientation-aware calibration is a separate experiment and produces a
separate threshold artifact. It must not pool pairs, distances, or thresholds
with the position-only variant. The pre-registered secondary variant is:

```yaml
task_label:
  variant: cube_block_pose_sym24_pointwise_gap03cm15deg_04cm30deg
  state_fields:
    - privileged_block_0_pos
    - privileged_block_0_quat
  metric: max_normalized_pose_margin
  quaternion_convention: wxyz
  cube_rotational_symmetries: 24
  positive_position_lt_m: 0.03
  positive_orientation_lt_deg: 15.0
  negative_position_gt_m: 0.04
  negative_orientation_gt_deg: 30.0
  positive_if_lt: 0.0
  negative_if_gt: 1.0
  unit: dimensionless
  ignored_fields:
    - qvel
    - robot_qpos
    - effector_state
    - gripper_contact
  exact_self_pairs: exclude
```

Let

\[
r_{ij}=\left\|p^{\mathrm{block}}_i-p^{\mathrm{block}}_j\right\|_2
\]

and let \(\theta_{ij}\) be the geodesic orientation error in degrees after
minimizing over the cube's 24 proper rotational symmetries. Define the scalar
task error

\[
\Delta^{\mathrm{pose}}_{ij}=
\max\left(
\frac{r_{ij}-0.03}{0.04-0.03},
\frac{\theta_{ij}-15}{30-15}
\right).
\]

The resulting label is equivalently:

```text
T: position < 0.03 m AND symmetry-aware orientation < 15 degrees
F: position > 0.04 m OR  symmetry-aware orientation > 30 degrees
U: every remaining combination, including every exact boundary
```

This normalization prevents metres and degrees from being added implicitly.
Quaternion sign aliases must give identical angles. Missing or ambiguous
quaternion state is blocking; a yaw-only dataset requires a separately named
yaw-only variant and cannot be reported as full 3D pose calibration.

### 3.5 TwoRoom default variant

The pre-registered TwoRoom variant is endpoint position only:

```yaml
task_label:
  variant: tworoom_agent_xy_pointwise_gap8_16
  state_fields: [pos_agent]
  metric: l2
  positive_if_lt: 8.0
  negative_if_gt: 16.0
  unit: pixel
  ignored_fields:
    - goal_pos_agent
    - action
    - collision_history
    - route_valid
    - room_crossing_history
    - goal_side_history
    - success_hold
  exact_self_pairs: exclude
```

For rows \(i,j\),

\[
\Delta^{\mathrm{xy}}_{ij}=\left\|a_i-a_j\right\|_2.
\]

Thus `<8 px` is `T`, `8..16 px` inclusive is `U`, and `>16 px` is `F`.
The two radii reuse the existing strict and moderate TwoRoom endpoint
tolerances only as pointwise anchors. The variant must not be called TwoRoom
Moderate or Strict because it excludes swept collision, route legality, room
crossing, goal-side arrival, and all history.

Eligible TwoRoom pairs must share a navigation-geometry signature containing
at least wall axis, wall center, wall thickness, door count, door positions,
door sizes, agent radius, image size, and camera/rendering semantics. A field
may be a dataset-global constant, but it must still be materialized in the
compatibility metadata and hashed. The default contract requires an unrendered
target. Rendering a target, changing geometry, or omitting the geometry
signature requires a new variant or is blocking.

### 3.6 Variant isolation

Each formal run selects exactly one task and one pointwise label variant. In
particular, Cube position, Cube pose, and TwoRoom XY runs have independent pair
manifests, selected thresholds, validation decisions, and audit artifacts. No
threshold is shared across them.

### 3.7 Ignored-region reporting

`U` pairs are intentional abstentions. They are sampled and counted but do not
enter `T/F` threshold selection. Every partition must report:

- uniform-sample prevalence of `T`, `F`, and `U`;
- \(P(d_{ij}\le\epsilon^*\mid y=U)\);
- the task-error distribution of ignored pairs inside and outside the ball.

No correctness claim is allowed for `U` pairs.

---

## 4. Absolute invariants

### 4.1 Frozen encoder only

- Load \(E_0\) from the exact checkpoint declared by the experiment.
- Set the checkpoint to `eval()` and every parameter to
  `requires_grad=False`.
- Run under `torch.inference_mode()` or equivalent.
- Do not update, adapt, or copy-and-modify the encoder or projector.
- Do not call the predictor, action encoder, `predict()`, `rollout()`, or a
  planner-facing cost evaluator.
- Record encoder/projector parameter hashes before and after scoring; they must
  match.

### 4.2 Exact observation and residual semantics

Observation encoding must match the intended frozen encoder path, including:

- pixel source and channel order;
- resize and interpolation;
- uint8-to-float scaling;
- image normalization;
- encoder and projector symbols;
- evaluation mode and selected output tensor;
- dtype and numerical tolerance.

The residual is exactly

\[
d_{ij}=\operatorname{mean}_D((z_i-z_j)^2).
\]

The repository's existing `GoalMSE` sums terminal latent squared errors and is
not the residual defined here. Its thresholds are incompatible without an
explicit conversion and a different artifact contract.

### 4.3 No latent-dependent labels or sampling

- Compatibility, `T/F/U` labels, splits, and task-space strata use only
  provenance and simulator state.
- Pair samplers may not import the encoder or read latent tables.
- Latent distance, pixel similarity, or manual image inspection may not select,
  relabel, or exclude pairs.
- Changing latent scores in a fixture must not change pair IDs or labels.

### 4.4 No predictor, planner, or simulator

- Predictor-call count is zero.
- Planner-call count is zero.
- Environment-constructor and `step()` counts are zero.
- Labels are pure functions of stored single-row state.
- Missing state is a pre-scoring exclusion or failure, never a reason to restore
  and step a simulator.

### 4.5 No downstream feedback

Selection code may not read predictor analyses, metric scans, CEM results,
planner rates, environment outcomes, or downstream validation/test artifacts.
Changing \(\epsilon^*\), the pointwise predicate, or the tolerance gap after a
downstream experiment starts requires a new immutable experiment version.

---

## 5. Repository and data mapping before scoring

Record the actual symbols and paths for:

1. checkpoint loading;
2. encoder and projector forward paths;
3. observation preprocessing;
4. latent dimension \(D\) and dtype;
5. dataset reader;
6. row, episode, trajectory, and source-group IDs;
7. task-state columns;
8. pointwise task-error function and boundaries;
9. compatibility metadata;
10. group-aware split construction;
11. downstream validation of `selected_threshold.json`.

Missing state fields, source-group identity, compatibility metadata, or a pure
pointwise predicate is a blocking condition. Cube pose additionally requires a
verified `wxyz` block quaternion and symmetry implementation. TwoRoom requires
the complete navigation-geometry signature even when its values are constant
throughout the dataset.

### 5.1 Reference dataset inventory

Verified on `shou_node09` on 2026-08-20:

| Task | Rows | Episodes | Pixel shape | HDF5 size | Relevant state columns |
|---|---:|---:|---|---:|---|
| PushT | 2,336,736 | 18,685 | 224x224 RGB | 46.3 GB | `state`, `proprio` |
| Cube | 2,010,000 | 10,000 | 224x224 RGB | 101.9 GB | `privileged_block_0_pos`, `privileged_block_0_quat`, `qpos`, `qvel`, effector state |
| TwoRoom | 920,809 | 10,000 | 224x224 RGB | 12.8 GB | `pos_agent`, `proprio`, navigation-geometry metadata or constants |

PushT's 18,685 episodes are 185 source groups times 101 variants. Splits and
uncertainty must use the source group, not treat the variants as independent.

Every formal run must repeat the inventory and hash the dataset or immutable
index. A mismatch creates a new dataset version; this table is not a substitute
for live preflight.

---

## 6. Immutable data schemas

### 6.1 Observation record

```python
{
    "observation_id": stable_unique_id,
    "partition": "threshold_fit" | "threshold_validation" | "threshold_audit",
    "raw_group_id": source_group_or_episode_id,
    "episode_id": trajectory_or_episode_id,
    "row_id": immutable_dataset_row_id,
    "step_id": step_within_episode,
    "observation_ref": immutable_pixel_and_input_reference,
    "simulator_state": declared_single_row_state,
    "compatibility_metadata": task_scene_object_or_variation_metadata,
    "checkpoint_hash": sha256,
    "dataset_hash": sha256_or_versioned_index_hash,
}
```

Pixels may remain in the source dataset. The resolved ordered observation list
must be hashed.

### 6.2 Pair record

```python
{
    "pair_id": stable_unique_id,
    "partition": "threshold_fit" | "threshold_validation" | "threshold_audit",
    "sample_family": "uniform" | "task_stratified",
    "anchor_observation_id": observation_i,
    "goal_observation_id": observation_j,
    "anchor_group_id": raw_group_i,
    "goal_group_id": raw_group_j,
    "task_error": delta_ij,
    "label": "T" | "F" | "U",
    "task_error_stratum": configured_stratum,
    "sampling_probability": known_probability_or_design_weight,
    "analysis_weight": inverse_probability_or_one,
    "latent_distance": d_ij,  # scorer only
    "latent_dim": D,
    "checkpoint_hash": sha256,
    "dataset_hash": sha256,
}
```

Pair IDs and labels are materialized and hashed before scoring. The pair builder
must not import model code.

---

## 7. Eligible pair population

The population consists of compatible ordered pairs of distinct observations
within one partition:

```text
(i, j), i != j
```

A metadata-only rule defines compatibility, including task, scene, object, or
variation semantics. Impossible pairings are outside the population, not easy
negatives.

Requirements:

- exclude exact self-pairs from primary metrics;
- forbid duplicate pair IDs within a sample family;
- report same-trajectory and cross-trajectory pairs separately;
- retain PushT source-group and clean/noisy variant identity;
- never form cross-partition pairs;
- record every compatibility rejection reason;
- do not let long episodes silently dominate group-macro results.

---

## 8. Split and leakage control

Split raw independent groups before pair sampling:

```text
threshold_fit         60% of groups
threshold_validation  20% of groups
threshold_audit       20% of groups
```

Use only calibration groups reserved before downstream experiments.

Requirements:

- PushT splits by 185 source groups, never row, clip, replay variant, or the
  18,685 episode IDs alone;
- Cube and TwoRoom split by whole episode or a stronger source group;
- use stable hashing and a fixed `threshold_split_seed`;
- build pairs only after group assignment;
- both rows of each pair are in the same partition;
- save all IDs and split hashes;
- assert no raw-group overlap;
- never move hard examples after scoring;
- audit distances and metrics are unreadable until validation passes and the
  threshold is locked.

Audit IDs and sampling seeds may be precommitted, but selection code cannot
resolve audit latent, distance, or metric paths.

---

## 9. Encode every eligible observation once

Full configuration encodes 100% of eligible observations. Observation
subsampling is not the runtime-control mechanism.

Eligibility depends on the declared observation and simulator-state fields, not
on the existence of an outgoing action. Initial rows and terminal rows with a
NaN or missing outgoing-action sentinel remain eligible when their observation
and required state are finite, because find-goal-threshold never consumes actions.

For every row:

\[
z_i=E_0(o_i).
\]

Store one finite vector per row in sharded or memory-mapped form. The scorer
must preserve deterministic order, exact preprocessing, declared dtype, and
row-level provenance; fail on non-finite values; hash every shard and the shard
manifest; and verify parameter hashes before and after encoding.

Reference float32 storage at \(D=192\):

| Task | Full latent table |
|---|---:|
| PushT | about 1.79 GB |
| Cube | about 1.54 GB |
| TwoRoom | about 0.71 GB |

Raw latents and pair distances remain inside the threshold experiment.
Downstream code reads only the locked threshold artifact.

---

## 10. Pair sampling protocol

Materializing the complete \(N^2\) matrix is prohibited.

| Task | Ordered pairs before exclusions | One float32 distance per pair |
|---|---:|---:|
| PushT | 5,460,335,133,696 | 21.84 TB |
| Cube | 4,040,100,000,000 | 16.16 TB |
| TwoRoom | 847,889,214,481 | 3.39 TB |

Use two complementary sample families.

### 10.1 Uniform population sample

Draw `100,000,000` compatible ordered pairs uniformly, without replacement
within each partition and sample family:

```text
threshold_fit         60,000,000
threshold_validation  20,000,000
threshold_audit       20,000,000
```

The uniform sample retains `T/F/U`, estimates their prevalence, is the primary
source for population precision, and records its sampling frame, probability,
accepted count, and rejected proposals.

Approximate rates relative to the reference pair universes:

| Task | 100M / all pairs |
|---|---:|
| PushT | 0.00183% |
| Cube | 0.00248% |
| TwoRoom | 0.01179% |

### 10.2 Task-space-stratified sample

Draw another `20,000,000` pairs using simulator state only:

```text
threshold_fit         12,000,000
threshold_validation   4,000,000
threshold_audit        4,000,000
```

Target equal accepted counts of `T` and `F`. Split `F` into pre-registered
task-space strata such as `boundary_outside`, `medium`, and `far`. Exact edges
must be fixed in task units before latent scoring.

The sampler may use a task-space grid, tree, or deterministic reservoir, but
never latent distance. Save weights for unequal inclusion probabilities. This
sample stabilizes conditional TPR/FPR; its 50/50 class balance may not be used
as population precision.

### 10.3 Full target and shortfall

```text
100% observations encoded
+ 100M uniform pairs
+ 20M task-stratified pairs
= 120M scored pairs per task
```

If a stratum lacks enough unique pairs, do not duplicate. Include every
available pair, report group coverage and the shortfall, and require a new
pre-scoring config version to change the target or task thresholds.

---

## 11. Metrics and deterministic selection

### 11.1 Recall and false-positive rate

Ignoring `U` pairs:

\[
\operatorname{TPR}(e)=\Pr(d_{ij}\le e\mid y=T),
\]

\[
\operatorname{FPR}(e)=\Pr(d_{ij}\le e\mid y=F).
\]

Report FPR by every negative task-error stratum. Primary TPR/FPR are
anchor-group macro averages; pair-weighted results are secondary.

The primary TPR/FPR estimator uses the task-space-stratified sample with its
saved design weights. The uniform sample supplies a separately reported
cross-check. Records that happen to occur in both sample families must not be
pooled or counted twice in either estimator.

### 11.2 Population precision

On the eligible `T/F` population:

\[
\operatorname{Precision}(e)=
\frac{\Pr(y=T,d\le e)}
{\Pr(y=T,d\le e)+\Pr(y=F,d\le e)}.
\]

Estimate it directly from the uniform sample with design weights. Also report

\[
\operatorname{Precision}(e)=
\frac{\pi\operatorname{TPR}(e)}
{\pi\operatorname{TPR}(e)+(1-\pi)\operatorname{FPR}(e)},
\]

where \(\pi=\Pr(y=T\mid y\in\{T,F\})\). Direct and reconstructed precision
should agree within uncertainty. Balanced-sample raw precision is invalid.

For this reconstruction, \(\pi\), TPR, and FPR must all be the same
uniform-sample, design-weighted pair-population estimands. Substituting the
anchor-group-macro TPR/FPR into this identity is invalid and need not agree with
direct population precision.

### 11.3 Operating contract

Declare before fit scoring:

```yaml
selection:
  min_positive_recall: 0.90
  max_negative_fpr: 0.10
  min_population_precision: null  # report-only unless pre-registered
  tie_break: smallest_epsilon
```

If `min_population_precision` is non-null, it is an additional uniform-sample
feasibility constraint. It cannot be filled after viewing scores.

### 11.4 Selector

Candidates are the finite sorted unique union of fit `T/F` distances from both
sample families plus no-hit/all-hit sentinels. Candidate metrics are still
computed separately by estimator: weighted stratified TPR/FPR and uniform
population precision. Among candidates satisfying every configured FPR and
optional precision constraint:

1. maximize anchor-group-macro TPR;
2. break equal-TPR ties with the smallest \(e\);
3. require `min_positive_recall`.

If none is feasible, stop with:

```text
THRESHOLD_CALIBRATION_NO_FEASIBLE_OPERATING_POINT
```

Do not relax the latent operating contract or task-space gap after scoring.

An existing threshold is comparable only if its complete tuple and mean-MSE
residual match. Summed `GoalMSE` and predicted-endpoint thresholds are not
direct comparators.

---

## 12. Validation, lock, and audit

### 12.1 Validation

Apply fit-selected \(\epsilon^*\) unchanged to validation. Report:

- macro and pair-weighted TPR;
- overall and per-stratum FPR;
- uniform population precision and reconstructed precision;
- weighted `T/F/U` prevalence;
- ignored-band ball occupancy;
- group-clustered 95% CIs;
- score, precision-recall, and ROC curves;
- same- versus cross-trajectory strata.

Point estimates must meet all pre-registered constraints. If they fail, stop
with:

```text
THRESHOLD_CALIBRATION_FAILED_VALIDATION
```

Do not adjust epsilon, state fields, gap, pair weights, or sampling composition
from validation outcomes.

### 12.2 Threshold lock

After validation passes and before audit scoring, write immutable
`selected_threshold.json` containing at least:

```text
epsilon
task
pointwise_label_variant
task_state_fields
task_error_definition_and_unit
positive_and_negative_thresholds_with_boundary_semantics
ignored_region_definition
residual_definition
D
encoder_checkpoint_path_or_id_and_sha256
encoder_and_projector_parameter_hashes
dataset_version_and_hash
observation_manifest_hash
fit_and_validation_split_hashes
uniform_and_stratified_pair_sample_hashes
selection_rule_and_constraints
observation_preprocessing
encoder_projector_symbol_mapping
git_commit_and_dirty_status
software_versions
timestamp
artifact_schema_version
```

### 12.3 One-time audit

After lock, score and evaluate the precommitted audit observations and pairs
once. Never change epsilon or the label contract from audit results.

Bug reruns retain the original artifact, identify the affected code, use a new
run ID, and state whether audit embeddings, distances, or metrics were viewed.

---

## 13. Statistical analysis

Pairs are dependent because they share observations and groups. Primary
uncertainty is conditional on the fixed goal-row population and clustered over
independent anchor raw groups. Every raw group must appear as an anchor so the
macro estimand is symmetric. Where feasible, also report a two-sided bootstrap
over anchor and goal groups.

For at least 10,000 bootstrap replicates:

1. resample anchor groups with replacement;
2. retain their pair records and design weights;
3. compute TPR, FPR, population precision, `U` prevalence, and ignored-band
   occupancy at the locked threshold;
4. retain matched pairs for comparator deltas.

Use a fixed `analysis_seed` and percentile 95% CIs. Pre-aggregate per-group
threshold counts or histograms so bootstrap does not rescan 120M rows 10,000
times.

Report the bootstrap distribution of fit-selected epsilon as stability, but do
not replace deterministic \(\epsilon^*\) with a bootstrap optimum.

---

## 14. Outputs

Equivalent repository structure:

```text
experiments/observation_goal_threshold/
  observations.py
  labels.py
  split.py
  encode.py
  sample_pairs.py
  score_pairs.py
  select.py
  validate.py
  audit.py
  analyze.py
  configs/
  tests/
```

Each full run produces:

```text
run_dir/
  manifest.yaml
  repository_mapping.json
  dataset_inventory.json
  split_ids.json
  observation_records.parquet
  embedding_shards/
  embedding_manifest.json
  pair_manifests/
  pair_scores/                  # sharded, experiment-local
  task_label_prevalence.json
  threshold_candidates.parquet
  selected_threshold.json       # only after validation passes
  validation_metrics.json
  audit_metrics.json            # only after threshold lock
  timing.json
  score_distributions.png
  precision_recall_curve.png
  roc_curve.png
  threshold_stability.png
  report.md
```

The manifest records commit/dirty status; checkpoint, parameter, dataset,
observation, split, sample, and output hashes; exact label/gap contract;
preprocessing and residual; target/realized counts and weights; seeds;
selection constraints; timing, device, dtype, batch size, peak memory, software
versions; exclusions and rejections; and zero forbidden-call counters.

---

## 15. Required tests

### 15.1 Encoding and frozen-model tests

- preprocessing, encoder output, projected latent, \(D\), and dtype match the
  intended path;
- fixed inputs reproduce identical latents;
- gradients, optimizer, and backward paths are absent;
- parameter hashes match before and after;
- predictor and action-encoder calls remain zero.

### 15.2 Label and gap tests

- labels use only declared single-row state fields;
- latent, pixels, actions, path, collision, route, and hold metadata cannot
  change labels;
- known below-gap, above-gap, and ignored fixtures produce `T`, `F`, and `U`;
- for PushT: `<20` is `T`, `20..30` inclusive is `U`, and `>30` is `F`;
- for Cube position: `<0.03 m` is `T`, `0.03..0.04 m` inclusive is `U`,
  and `>0.04 m` is `F`;
- for Cube pose: `T` requires both inner tolerances, `F` requires either outer
  tolerance to be exceeded, and all remaining combinations are `U`;
- Cube pose treats all 24 proper cube rotations and quaternion sign aliases
  correctly and rejects non-finite or non-`wxyz` quaternion inputs;
- for TwoRoom: `<8 px` is `T`, `8..16 px` inclusive is `U`, and `>16 px`
  is `F`.

### 15.3 Observation, split, and sampling tests

- every eligible row appears exactly once; exclusions have reason codes;
- shard counts and hashes match the observation manifest;
- self-pairs and duplicate pair IDs are absent from primary tables;
- raw groups never cross partitions;
- all PushT variants from one source group remain together;
- incompatible Cube object/scene/rendering metadata cannot form pairs;
- incompatible TwoRoom navigation-geometry signatures cannot form pairs;
- uniform sampling retains `T/F/U` without class rejection and has correct
  probabilities;
- stratified sampling imports no model/latent code, reproduces IDs from its
  seed, matches requested strata, and reports rather than duplicates shortages.

### 15.4 Precision and selector tests

Synthetic imbalanced fixtures verify:

- direct uniform `TP/(TP+FP)` precision;
- prevalence reconstruction;
- rejection of raw 50/50-sample precision as population precision;
- exclusion of `U` from the `T/F` denominator;
- monotonic TPR/FPR with epsilon;
- rejection of infeasible candidates;
- optional precision constraints only when pre-registered;
- maximum feasible macro TPR and smallest-epsilon tie break;
- fit-only selector access and documented failure codes.

### 15.5 Isolation and forbidden-call tests

- validation cannot change epsilon or the task gap;
- audit requires a locked artifact and cannot rewrite it;
- selection cannot resolve audit/downstream paths;
- predictor, planner, world/environment constructors, and environment steps are
  all mocked and called zero times;
- downstream consumers fail fast on any tuple/schema mismatch.

---

## 16. Smoke, full execution, and five-hour budget

Smoke may reduce groups, pair counts, batch size, and bootstrap replicates. It
may not change preprocessing, mean-MSE residual, label fields, gap boundaries,
split isolation, sample-family meaning, or selector logic.

Minimum config:

```yaml
checkpoint: ...
dataset: ...
task: ...

task_label:
  variant: ...
  state_fields: [...]
  metric: ...
  positive_if_lt: ...
  negative_if_gt: ...
  unit: ...
  exact_self_pairs: exclude

data:
  threshold_split_seed: ...
  pair_sampling_seed: ...
  fit_fraction: 0.60
  validation_fraction: 0.20
  audit_fraction: 0.20

pair_sampling:
  uniform_pairs: 100000000
  stratified_pairs: 20000000
  stratified_positive_fraction: 0.50
  negative_strata: {...}

selection:
  min_positive_recall: 0.90
  max_negative_fpr: 0.10
  min_population_precision: null
  tie_break: smallest_epsilon

analysis:
  bootstrap_replicates: 10000
  analysis_seed: ...
```

Equivalent CLI stages must separately inventory/split/label, encode all
observations, sample pairs, score latent distances, select on fit, validate and
lock, audit once, and analyze/report.

### 16.1 Reference throughput probe

A bounded non-artifact probe on 2026-08-20 used the current ViT-Tiny LeWM
encoder, float32, batch 256, one RTX 4090, real HDF5 reads, and configured image
normalization:

| Dataset | End-to-end observations/s | Full-pass extrapolation |
|---|---:|---:|
| PushT | 2,863 | 13.6 minutes |
| Cube | 2,221 | 15.1 minutes |
| TwoRoom | 2,697 | 5.7 minutes |

A resident-GPU 10M-pair probe covering 192-D mean squared distance, four-D
task error, `T/F` masks, and 2,048-bin histograms observed about 97.6M pairs/s.
This is a kernel upper bound, not an end-to-end guarantee.

Formal runs repeat a bounded probe with the exact task checkpoint, data,
device, dtype, and batch size. Reference numbers cannot replace measured timing.

### 16.2 Five-hour contract

Target one task within five wall-clock hours on one RTX 4090-class GPU:

| Stage | Budget |
|---|---:|
| provenance/schema preflight | 15 minutes |
| encode 100% observations | 30 minutes |
| task-space index and 120M pair construction | 60 minutes |
| pair scoring and threshold selection | 30 minutes |
| validation, lock, one-time audit, bootstrap, plots, report | 90 minutes |
| contingency | 75 minutes |
| total | 300 minutes |

Audit consumes its reserved portion only after lock. If preflight projects over
five hours, do not reduce observation coverage silently. First optimize pair
streaming, task indexing, histogram aggregation, and group summaries. Any
reduced pair target requires a new config hash and explicit precision/power
justification.

---

## 17. Interpretation rules

| Observation | Allowed conclusion |
|---|---|
| Held-out recall is high and FPR/precision satisfy the contract | The frozen encoder supports this exact pointwise observation-neighborhood contract |
| Recall is high only at high FPR | Task-equivalent and task-non-equivalent observations are not sharply separated by this latent distance |
| Balanced precision looks strong but uniform precision is weak | Artificial class balance created misleading purity |
| Fit passes but validation fails | Threshold overfit or group shift; do not promote |
| Validation passes but audit degrades with wide CI | Generalization is uncertain |
| Many `U` pairs fall inside the ball | The ball covers much of the deliberately unresolved task-space band; no correctness claim applies there |
| Encoder geometry succeeds | No conclusion about predictor accuracy, reachability, CEM, or execution |
| PushT joint-XY succeeds | No conclusion about block angle or official Moderate/Strict success |
| Cube position succeeds | No conclusion about orientation, held success, or official Moderate/Strict success |
| Cube pose succeeds | No conclusion about held success, manipulation reachability, or execution |
| TwoRoom XY succeeds | No conclusion about collision-free reachability, legal crossing, route validity, or official Moderate/Strict success |

---

## 18. Final report requirements

`report.md` must answer:

1. What exact encoder checkpoint, dataset, preprocessing, residual, and dtype
   were calibrated?
2. What state fields, task metric, units, and `T/F/U` boundaries defined labels?
3. Which official task semantics were deliberately excluded?
4. Were all observations and independent source groups accounted for and hashed?
5. Were fit, validation, and audit groups disjoint?
6. What target/realized uniform and stratified counts, weights, and shortfalls
   occurred?
7. What were uniform `T/F/U` prevalence and effective sampling rates?
8. What recall, FPR, and optional precision constraints were pre-registered?
9. Why was \(\epsilon^*\) the deterministic fit winner?
10. Did unchanged epsilon pass validation?
11. What were audit recall, FPR, precision, ignored-band occupancy, and 95% CIs?
12. Did direct and reconstructed population precision agree?
13. How stable was epsilon under group bootstrap?
14. Did any post-score exclusion, rerun, or data issue occur?
15. What were throughput, peak memory, stage timings, and total wall time?
16. Is the result promotable, uncertain, or failed for this exact contract?

---

## 19. Definition of Done

- [ ] Repository mapping and all provenance hashes are recorded.
- [ ] Labels use only declared single-row state fields.
- [ ] Positive and negative task thresholds have an explicit ignored gap.
- [ ] Exact boundary semantics are tested.
- [ ] The pointwise variant is not misnamed as a path/hold-aware official rule.
- [ ] Cube position, Cube pose, and TwoRoom XY use separate pair, threshold,
      validation, and audit artifacts.
- [ ] Cube pose uses verified `wxyz` quaternions and all 24 proper rotational
      symmetries; metre/degree scaling is explicit.
- [ ] TwoRoom compatibility includes the complete hashed navigation-geometry
      signature and an unrendered-target observation contract.
- [ ] 100% of eligible observations are encoded exactly once.
- [ ] Preprocessing and projected latents match the frozen encoder path.
- [ ] Encoder/projector hashes are unchanged.
- [ ] Predictor and action-encoder calls are zero.
- [ ] Planner, world/environment constructors, and `step()` calls are zero.
- [ ] Self-pairs are excluded from primary metrics.
- [ ] Raw groups are disjoint across fit/validation/audit.
- [ ] PushT variants remain grouped by source demonstration.
- [ ] Pair construction and labels cannot access latent distances.
- [ ] Uniform pairs retain `T/F/U` and support population precision.
- [ ] Stratified pairs stabilize TPR/FPR and are not used for raw precision.
- [ ] Pair counts, probabilities, weights, and shortfalls are recorded.
- [ ] Selection reads fit only and validation passes before lock.
- [ ] Audit runs only after lock and cannot change epsilon or the gap.
- [ ] Group-clustered uncertainty is reported.
- [ ] Raw latents/distances remain experiment-local.
- [ ] Downstream consumers verify the complete compatibility tuple.
- [ ] No predictor, metric scan, CEM, or environment result influences selection.
- [ ] One-task runtime is at most five hours or explicitly fails that contract.

The central review question is:

> Was \(\epsilon^*\) selected solely from frozen-encoder distances between
> pre-partitioned demonstration observations whose `T/F/U` labels came only
> from a pre-registered single-time-step simulator-state gap predicate, with
> population precision estimated from uniform pairs and the threshold locked
> before any downstream predictor or planner result was observed?

---

## 20. Downstream CLEAR endpoint self-eval

This is a post-lock validation experiment, not part of threshold selection.
No CLEAR result may alter epsilon, labels, splits, or the operating contract.

- Matrix: PushT, Cube, and TwoRoom x CLEAR v0.5 Moderate and Strict x 100
  fixed seed-42 manifest pairs, reported as six separate cells.
- Planner contract: CEM `batch_size=1`, `num_samples=300`, `n_steps=30`,
  `topk=30`, `cpu_threads=1`, goal offset 25, execution budget 50.
- Model: each task uses the exact checkpoint SHA-256 recorded in its locked
  `selected_threshold.json`; no official or cross-task checkpoint fallback.
- Actual label: the installed CLEAR evaluator's per-pair S/F result.
- Self label: after the pair terminates or exhausts its budget, freshly render
  the final observation, encode it and the fixed goal using only the frozen
  encoder/projector, compute float32 `mean_D((z_final-z_goal)^2)`, and predict
  success iff that distance is `<= epsilon`.
- Artifacts: preserve every pair ID, endpoint distance, prediction, evaluator
  label, correctness bit, confusion counts, accuracy with Wilson 95% interval,
  actual SR, predicted SR, signed/absolute SR error, and a paired bootstrap
  95% interval for predicted-minus-actual SR error.
- Reporting: never pool tasks or Moderate/Strict into the primary result;
  pointwise geometry omits angle, pose, sustained hold, collision, route, and
  reachability semantics present in some CLEAR cells.
- Missing-threshold rule: a task without a promoted epsilon is
  `THRESHOLD_UNAVAILABLE`. It is forbidden to substitute a fit diagnostic,
  tune on CLEAR outcomes, or call the three-task matrix complete.

### 20.1 Post-lock epsilon-pair-accuracy curve

The diagnostic curve uses the same fixed 3-task x Moderate/Strict x 100-pair
CLEAR endpoints. Within each cell, sweep every non-negative epsilon at the
exact observed distance breakpoints and report the empirical paired accuracy
of `endpoint_distance <= epsilon` against the evaluator S/F vector. Keep all
six task/protocol curves separate in one 3x2 figure; do not pool pairs.

PushT and TwoRoom mark their already locked epsilon and its paired accuracy.
Because Cube has no promoted epsilon, its two cells may run in score-only mode:
the exact formal calibration config/status and checkpoint identity are locked,
but the run records only endpoint distances and evaluator labels. Cube draws
no epsilon marker, produces no fixed-threshold S/F prediction, and remains
`THRESHOLD_UNAVAILABLE` for self-eval.

This sweep is post-lock descriptive analysis. It must not select, promote, or
retune epsilon, and a high point on any curve is not validation evidence
because the same 100 CLEAR labels define the displayed accuracy.

### 20.2 Evaluator-relative endpoint TPR/FPR curve

Using the identical six endpoint result artifacts, also report TPR and FPR as
functions of epsilon, with the CLEAR evaluator S/F vector defining the positive
and negative classes. The predicate remains `endpoint_distance <= epsilon`;
TPR is the fraction of evaluator successes predicted successful, and FPR is
the fraction of evaluator failures predicted successful.

The six-panel display fixes Cube's epsilon axis to `[0, 4]` and TwoRoom's to
`[0, 3]`, extending each curve horizontally at its terminal all-positive rate
when the observed maximum distance is smaller. PushT retains a data-driven
range containing all breakpoints and its locked epsilon. Locked epsilon markers
remain available only for PushT and TwoRoom; Cube remains score-only. This is
still post-lock descriptive analysis and must not be used to select epsilon.
