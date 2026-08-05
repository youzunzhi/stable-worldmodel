# CLEAR-LeWM v0.5 results

These are 100-pair, seed-42, no-video evaluations against CLEAR-LeWM v0.5.0 at
revision `df026185a36bd9997c69d94753854db0b1a46f54`. CEM uses batch size 1,
300 samples, 30 iterations, top-k 30, and one Torch CPU thread.

## CEM results

| Policy | PushT Moderate | PushT Strict | Cube Moderate | Cube Strict |
|---|---:|---:|---:|---:|
| Project-trained | 96/100 | 85/100 | 47/100 | 28/100 |
| Official LeWM | 88/100 | 71/100 | 52/100 | 22/100 |
| Random | 3/100 | 7/100 | 15/100 | 8/100 |

PushT official and all four random results exactly reproduce the upstream
seed-42 references. Official Cube is a runtime drift: local `52/22` versus the
upstream `51/25`; both runs satisfy the exact manifest, solver, and CPU-thread
contract.

## GD solver ablation

| Policy and solver | PushT Moderate | PushT Strict | Cube Moderate | Cube Strict |
|---|---:|---:|---:|---:|
| Project-trained CEM | 96/100 | 85/100 | 47/100 | 28/100 |
| Project-trained GD | 90/100 | 82/100 | 46/100 | 20/100 |

GD uses 100 AdamW restarts, 30 updates, and learning rate 0.1. It is a
manifest-matched solver ablation, not a protocol-conformant CLEAR baseline;
the raw JSONs record `solver_contract_matched: false` and
`solver_ablation_opt_in: true`.

## Provenance

- Evaluation source: `fba769e6444d29b9790b68413d6f447e0a3aac05`.
- Policy seed: 42; all runs completed 100/100 pairs.
- PushT manifest SHA-256: Moderate `dcdce1f5e90c29246b70670d6a61171bd8ba17814bc88726ac65994714c14afa`, Strict `2042018fc346927af5da75cce2565db8476121d1db67a99d0fc50f218eddcbc9`.
- Cube manifest SHA-256: Moderate `03f9c3a375707bdcd40d32aa1be26ef15ab2529fe513e67eea59789b36c70d63`, Strict `fccf9d6de3336431a09f146b4fadbf1a1150aff4b2d699e75ca32835b3242ecd`.
- Project-trained PushT checkpoint SHA-256: `1cf72a6616b9625e056595d98794b61595bddce923ea87bb1d527fc6f43d98b7`.
- Project-trained Cube checkpoint SHA-256: `6ed18d2aacbd4b6c51fd2ba79fe57f8b0b456691e6cf69455723269099151d6f`.
- Official PushT checkpoint SHA-256: `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607`.
- Official Cube checkpoint SHA-256: `2839a907362f403f9136383016e91774373a295d958ae75121791f22a9fddf89`.

`summary.json` is the compact machine-readable index. `manifests/` contains
the exact inputs, while `raw/` contains the full resolved configs, per-pair
success vectors, timing, and checkpoint paths.
