# CLEAR-LeWM v0.5 four-task LeWM baseline

This report consolidates the PushT, Cube, TwoRoom, and Reacher LeWM baselines.
Every four-task baseline and official-reference score is a completed 100-pair,
policy-seed-42, no-video evaluation against CLEAR-LeWM v0.5.0 at revision
`df026185a36bd9997c69d94753854db0b1a46f54`. CEM uses batch size 1, 300
samples, 30 iterations, top-k 30, and one Torch CPU thread. Planning uses
`horizon=5`, `receding_horizon=5`, `action_block=5`, and `eval_budget=50`.

## Project-trained 10-epoch baseline

All four project-trained checkpoints use training seed 3072 and the complete
epoch-10 weights.

| Task | Moderate | Strict |
|---|---:|---:|
| PushT | [96/100](raw/clear-v05-61c41d0-pusht-moderate-trained-seed3072-policyseed42/results.txt.json) | [85/100](raw/clear-v05-61c41d0-pusht-strict-trained-seed3072-policyseed42/results.txt.json) |
| Cube | [47/100](raw/clear-v05-61c41d0-cube-moderate-trained-seed3072-policyseed42/results.txt.json) | [28/100](raw/clear-v05-61c41d0-cube-strict-trained-seed3072-policyseed42/results.txt.json) |
| TwoRoom | [88/100](raw/clear-v05-53ca790-tworoom-moderate-trained-epoch10-seed3072-policyseed42/results.txt.json) | [81/100](raw/clear-v05-53ca790-tworoom-strict-trained-epoch10-seed3072-policyseed42/results.txt.json) |
| Reacher (corrected B runtime) | [80/100](raw/trained-b-suppressed-fix-moderate-seed3072-policyseed42/results.txt.json) | [81/100](raw/trained-b-suppressed-fix-strict-seed3072-policyseed42/results.txt.json) |

These are task-specific success rates, not a single interchangeable metric.
Moderate and Strict use the success predicates encoded by their respective
fixed manifests.

## Official-checkpoint reference

| Task | Moderate | Strict |
|---|---:|---:|
| PushT | [88/100](raw/clear-v05-61c41d0-pusht-moderate-official-policyseed42/results.txt.json) | [71/100](raw/clear-v05-61c41d0-pusht-strict-official-policyseed42/results.txt.json) |
| Cube | [52/100](raw/clear-v05-61c41d0-cube-moderate-official-policyseed42/results.txt.json) | [22/100](raw/clear-v05-61c41d0-cube-strict-official-policyseed42/results.txt.json) |
| TwoRoom | [80/100](raw/clear-v05-53ca790-tworoom-moderate-official-policyseed42/results.txt.json) | [57/100](raw/clear-v05-53ca790-tworoom-strict-official-policyseed42/results.txt.json) |
| Reacher (corrected B runtime) | [77/100](raw/official-b-suppressed-fix-moderate-seed42/results.txt.json) | [79/100](raw/official-b-suppressed-fix-strict-seed42/results.txt.json) |

PushT official exactly reproduces the upstream seed-42 reference. Official
Cube is a runtime drift: local `52/22` versus upstream `51/25`; both local runs
satisfy the exact manifest, solver, and CPU-thread contract. Official TwoRoom
Strict matches upstream; Moderate differs by one pair while retaining valid
topology audits.

Reacher is reported only under the corrected B runtime supported by this
codebase. B suppresses the underlying dm-control qpos termination during the
outer CLEAR action-repeat step and records
`reacher_internal_termination_mode=suppressed-local-fix` with
`reference_compatible=false`. It must not be presented as an exact reproduction
of the released upstream-v0.5 A runtime.

## Partial controls and solver ablation

Random-policy controls are currently complete only for PushT and Cube:

| Policy | PushT Moderate | PushT Strict | Cube Moderate | Cube Strict |
|---|---:|---:|---:|---:|
| Random | [3/100](raw/clear-v05-61c41d0-pusht-moderate-random-policyseed42/results.txt.json) | [7/100](raw/clear-v05-61c41d0-pusht-strict-random-policyseed42/results.txt.json) | [15/100](raw/clear-v05-61c41d0-cube-moderate-random-policyseed42/results.txt.json) | [8/100](raw/clear-v05-61c41d0-cube-strict-random-policyseed42/results.txt.json) |

All four available random controls exactly reproduce the upstream seed-42
references.

The project-trained GD ablation is also limited to PushT and Cube:

| Solver | PushT Moderate | PushT Strict | Cube Moderate | Cube Strict |
|---|---:|---:|---:|---:|
| CEM | 96/100 | 85/100 | 47/100 | 28/100 |
| GD | [90/100](raw/solver-ablation-gd-v05-61c41d0-pusht-moderate-trained-seed3072-policyseed42/results.txt.json) | [82/100](raw/solver-ablation-gd-v05-61c41d0-pusht-strict-trained-seed3072-policyseed42/results.txt.json) | [46/100](raw/solver-ablation-gd-v05-61c41d0-cube-moderate-trained-seed3072-policyseed42/results.txt.json) | [20/100](raw/solver-ablation-gd-v05-61c41d0-cube-strict-trained-seed3072-policyseed42/results.txt.json) |

GD uses 100 AdamW restarts, 30 updates, and learning rate 0.1. It is a
manifest-matched solver ablation, not a protocol-conformant CLEAR baseline;
the raw JSONs record `solver_contract_matched=false` and
`solver_ablation_opt_in=true`.

## Provenance

Evaluation implementation commits differ because task support was added in
validated stages:

- PushT and Cube: `fba769e6444d29b9790b68413d6f447e0a3aac05`.
- TwoRoom: `53ca79025322f6a8b598ea144d07bacc17eb73b3`.
- Reacher B evaluation: `665e01d532392148ff2683664205ece389da8abd`;
  the B-only implementation was subsequently promoted to `main` in
  `211ea6b9d8cf2bd118a5a15226f520f37129a2dc`.

Project-trained checkpoint SHA-256 values:

- PushT: `1cf72a6616b9625e056595d98794b61595bddce923ea87bb1d527fc6f43d98b7`.
- Cube: `6ed18d2aacbd4b6c51fd2ba79fe57f8b0b456691e6cf69455723269099151d6f`.
- TwoRoom: `b42b2a215185989ef19cd2c72619f9f3116967b55cf114f33ae1ae4a9c34b31b`.
- Reacher: `e23682a16e772469d7dc76ed2c6f303e0fd31d6f6ccff4cb751566bf5d3cfec5`.

Official checkpoint SHA-256 values:

- PushT: `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607`.
- Cube: `2839a907362f403f9136383016e91774373a295d958ae75121791f22a9fddf89`.
- TwoRoom: `566f223624ea4bfb39dbfe6ae731198dd6ea73b7b8919fed6b1ecafca810f7dd`.
- Reacher: `1c8670e75d1a30550ff1d0a1127c56099bd24d8c6967f6cf43fc5237502af8ab`.

Manifest SHA-256 values:

| Task | Moderate | Strict |
|---|---|---|
| PushT | `dcdce1f5e90c29246b70670d6a61171bd8ba17814bc88726ac65994714c14afa` | `2042018fc346927af5da75cce2565db8476121d1db67a99d0fc50f218eddcbc9` |
| Cube | `03f9c3a375707bdcd40d32aa1be26ef15ab2529fe513e67eea59789b36c70d63` | `fccf9d6de3336431a09f146b4fadbf1a1150aff4b2d699e75ca32835b3242ecd` |
| TwoRoom | `216250de76495a17a58bb16538b678408063798c3e4f55381943b79214ac83f6` | `b1833f5e500db94157b5f091bef8c77380a8d49a4862701eba7c2cad8b87fc35` |
| Reacher | `1ab745023ffc673aeef2f073aa7b7d1ba2ab00cd0caf39cc392a6a6ff54d387c` | `01c4d061dfbd50a431f4bcb14cda6d00a11dc0841d3ebfeaae4fe22b9361cc65` |

The compact machine-readable index is [`summary.json`](summary.json).
[`manifests/`](manifests/) contains the exact inputs, and [`raw/`](raw/)
contains the terminal structured results, including per-pair success vectors,
resolved configurations, timings, checkpoint paths, and task-specific runtime
audits.

## Interpretation boundary

The manifests use `split=all` and `heldout_fraction=0`. These evaluations are
fixed task-transfer tests, but the report does not claim trajectory-held-out
generalization relative to every local training clip. Scores should be compared
within the same task and protocol, with the recorded runtime contract intact.
