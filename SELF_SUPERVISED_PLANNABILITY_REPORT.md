# Self-supervised plannability（SSP）正式實驗報告

- **Protocol:** `self-supervised-plannability-v1`
- **Status:** 完成
- **Tasks:** PushT、Cube、TwoRoom
- **Formal source:** `7e4c97c80b908ebc67474d1f0185a14802f056b2`
- **Report date:** 2026-08-22

## 1. 摘要

SSP 要回答的是：在完全固定 project-trained LeWM、latent hit verifier 與 CEM
budget 的條件下，只學一個 16-parameter、positive diagonal terminal cost geometry，
是否能讓 hard CEM 更快、更常抵達預先註冊的 latent goal set，並進一步改善實際
CLEAR task execution。

正式矩陣已完整結束：9/9 training replicates、9/9 held-out profiles 與 24/24
CLEAR cells 都以 Slurm exit `0:0` 完成。所有 CLEAR cells 都完成 100/100 pairs、
符合既定 solver contract，且使用 `cpu_threads=1`。Frozen model 的訓練與 profile
前後 hashes 相同；SSP training 的 environment constructor 與 `step()` calls 都是
0。

主要結果是負向但有一個局部訊號：

- **沒有任何 replicate 達到 broad plannability improvement。** 9/9 simultaneous
  95% solve-curve bands 都未符合「所有 budget 不劣且至少一處嚴格改善」。
- **8/9 validation selections 是非 identity geometry，但大多沒有泛化到 held-out
  pairs。** 唯一 AUC paired 95% CI 排除 0 的結果是 Cube seed `260824`：
  `+0.005833 [0.001211, 0.010391]`，但 budgets 5–10 變差、11–30 變好，屬於
  curve-crossing compute-efficiency trade-off，不是 broad dominance。
- **CLEAR 沒有可靠 transfer。** 18 個 learned-vs-identity paired comparisons 的
  bootstrap 95% CI 全部包含 0，McNemar exact `p >= 0.1797`。TwoRoom Strict 的
  descriptive replicate mean 是 `+3 pp`，Cube Strict 是 `-3 pp`，但都不能宣稱
  improvement 或 regression。

因此，正式結論是：SSP v1 能在 validation 上選出非 identity geometry，但沒有
證明該 geometry 能穩健改善 held-out latent plannability，也沒有證明能提升實際
CLEAR success。這個結論只適用於本次固定 checkpoint、dataset、threshold、
16-parameter basis、bounds、ES 與 CEM budget。

## 2. 固定實驗合約

SSP 凍結 `M_0=(E_0,F_0)`。學到的 geometry `G_psi` 只用於 CEM candidate
ranking；原始 latent distance `d_0` 只負責回答是否 hit：

```text
psi in R^16
ell = B psi
w = exp(ell), with w_d in [0.25, 4]
c_psi = sum_d w_d (z_hat_terminal,d - z_goal,d)^2
```

`psi=0` 精確對應 identity `GoalMSE`。Optimizer 只看到 first-hit iteration
`T` 所衍生的 30-step AUC reward，不會看到 failed `d_0` magnitude。

| Contract item | Fixed value |
|---|---:|
| Latent / parameter dimension | 192 / 16 |
| Task hit thresholds | PushT `d_0 < 1.5`; Cube `< 1.0`; TwoRoom `< 1.5` |
| Train / validation / test pairs | 800 / 256 / 512 per task |
| Replicate seeds | `260822`, `260823`, `260824` |
| Profile planner seeds | `42`, `43`, `44`, `45`, `46` |
| CEM batch / samples / iterations / elites | 1 / 300 / 30 / 30 |
| Planning / receding horizon / action block | 5 / 5 / 5 |
| ES directions / pair batch / sigma | 8 / 16 / 0.25 |
| Optimizer / learning rate | Adam ascent / 0.05 |
| Outer steps / validation interval | 50 / 5 |
| Paired bootstrap | 10,000 pair-resamples; seed `20260822` |

Training 的三個 splits 使用固定 manifests。Held-out profile 對每個 test pair、
五個 planner seeds 比較 identity 與 selected geometry，並以 CRN 保留一對一
planner-noise identity。Primary profile estimands 是完整 `S_G(k)`、AUC 與
`S_G(30)`；CLEAR v0.5 是 geometry selection 鎖定後的 secondary execution
evidence。

## 3. 完成與 selection audit

所有 9 個 replicates 都執行完整 50 outer steps，terminal code 都是
`SSP_COMPLETED`；沒有 replicate 觸發 `SSP_NO_LEARNING_SIGNAL`。

| Task | Seed | Selected step | Validation AUC | Selection |
|---|---:|---:|---:|---|
| PushT | 260822 | 0 | 0.232031 | identity / null outcome |
| PushT | 260823 | 15 | 0.234635 | learned |
| PushT | 260824 | 5 | 0.228906 | learned |
| Cube | 260822 | 35 | 0.101562 | learned |
| Cube | 260823 | 50 | 0.108594 | learned |
| Cube | 260824 | 40 | 0.101823 | learned |
| TwoRoom | 260822 | 5 | 0.146875 | learned |
| TwoRoom | 260823 | 50 | 0.128516 | learned |
| TwoRoom | 260824 | 10 | 0.135156 | learned |

非零 selection 本身只代表在固定 validation manifest 上勝過 step 0；它不構成
held-out 或 execution improvement。PushT seed `260822` 由 step 0 勝出，依預註冊
語言是 transparent null outcome，不把 identity 稱為 learned improvement。

## 4. Primary held-out plannability profile

### 4.1 Fixed-replicate mean

下表先對每個 replicate 的 512 pairs x 5 planner seeds profile 計算曲線，再對三個
固定 replicates 取平均。`pp` 是 absolute percentage points。這是 descriptive
fixed-replicate mean；預註冊 protocol 沒有為三 replicate mean 定義另一個 pooled
CI，因此不把它當成 task-level significance test。

| Task | Identity AUC | SSP AUC | Delta | Identity `S(30)` | SSP `S(30)` | Delta | Broad improvements |
|---|---:|---:|---:|---:|---:|---:|---:|
| PushT | 24.220% | 24.300% | +0.080 pp | 52.344% | 52.344% | 0.000 pp | 0/3 |
| Cube | 11.686% | 11.984% | +0.298 pp | 32.383% | 32.630% | +0.247 pp | 0/3 |
| TwoRoom | 13.543% | 13.527% | -0.016 pp | 22.500% | 22.982% | +0.482 pp | 0/3 |

### 4.2 Replicate-level paired uncertainty

Intervals resample start-goal pairs while keeping all five matched planner seeds together。
數值保持 probability scale；例如 `0.005833` 等於 `+0.5833 pp`。

| Task | Seed | AUC delta [paired 95% CI] | `S(30)` delta [paired 95% CI] | Interpretation |
|---|---:|---:|---:|---|
| PushT | 260822 | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] | selected identity; exact profile parity |
| PushT | 260823 | +0.000221 [-0.003607, +0.004089] | -0.001953 [-0.011328, +0.007422] | no clear evidence |
| PushT | 260824 | +0.002174 [-0.001367, +0.005547] | +0.001953 [-0.007031, +0.010547] | no clear evidence |
| Cube | 260822 | +0.001732 [-0.003398, +0.006732] | -0.003516 [-0.015625, +0.008203] | no clear evidence |
| Cube | 260823 | +0.001380 [-0.003255, +0.006003] | +0.001562 [-0.010156, +0.013281] | no clear evidence |
| Cube | 260824 | **+0.005833 [+0.001211, +0.010391]** | +0.009375 [-0.001563, +0.020313] | **AUC-positive curve crossing; not broad dominance** |
| TwoRoom | 260822 | -0.000065 [-0.004193, +0.004076] | +0.000781 [-0.007812, +0.008994] | no clear evidence |
| TwoRoom | 260823 | +0.000013 [-0.006003, +0.006185] | +0.004687 [-0.005078, +0.014844] | no clear evidence |
| TwoRoom | 260824 | -0.000417 [-0.005456, +0.004649] | +0.008984 [0.000000, +0.017969] | endpoint boundary touches zero; no broad evidence |

Cube seed `260824` 的 point-estimate delta 在 budgets 5–10 為負、11–30 為正。
所以它支持的是在這個 replicate 上，較晚 budgets 的累積 solve-rate 增益足以抵銷
早期退步並得到較高總 AUC；它不支持每個 compute budget 都更好。其 simultaneous
band 仍在部分 budgets 低於 0，endpoint interval 也包含 0。

### 4.3 Task-specific primary interpretation

- **PushT:** 一個 replicate 選回 identity，其餘兩個的 AUC 與 endpoint intervals
  都包含 no-effect；fixed-replicate mean 幾乎不變。
- **Cube:** 三個 point-estimate AUC deltas 都為正，但只有 seed `260824` 的 paired
  AUC interval 排除 0，而且曲線交叉。這是局部 compute-efficiency signal，不是
  replicated broad improvement。
- **TwoRoom:** AUC 基本不變；endpoint point estimates 偏正，但 simultaneous
  curves 與 AUC uncertainty 都不支持 broad improvement。

## 5. Secondary CLEAR v0.5 execution

### 5.1 Descriptive fixed-replicate mean

每個 identity 與 selected geometry cell 使用相同的 100-pair manifest/order。
下表的 SSP 是三個 selected geometries 的 success-rate mean；同一 task/protocol 的
identity 是共同基準。這個 mean 沒有取代逐 replicate paired analysis。

| Task | Protocol | Identity | SSP replicate mean | Mean delta |
|---|---|---:|---:|---:|
| PushT | Moderate | 90.0% | 90.3% | +0.3 pp |
| PushT | Strict | 68.0% | 67.0% | -1.0 pp |
| Cube | Moderate | 52.0% | 50.3% | -1.7 pp |
| Cube | Strict | 25.0% | 22.0% | -3.0 pp |
| TwoRoom | Moderate | 92.0% | 92.0% | 0.0 pp |
| TwoRoom | Strict | 82.0% | 85.0% | +3.0 pp |

### 5.2 Per-geometry paired results

Bootstrap intervals 的單位是 success-rate percentage points。McNemar 是同一
100-pair order 上的 two-sided exact test，未做 multiple-comparison adjustment。

| Task | Protocol | Seed | Identity -> SSP successes | Delta [paired bootstrap 95% CI] | McNemar p |
|---|---|---:|---:|---:|---:|
| PushT | Moderate | 260822 | 90 -> 90 | 0 pp [0, 0] | 1.0000 |
| PushT | Moderate | 260823 | 90 -> 90 | 0 pp [-3, +3] | 1.0000 |
| PushT | Moderate | 260824 | 90 -> 91 | +1 pp [-2, +5] | 1.0000 |
| PushT | Strict | 260822 | 68 -> 69 | +1 pp [-2, +4] | 1.0000 |
| PushT | Strict | 260823 | 68 -> 64 | -4 pp [-11, +3] | 0.4240 |
| PushT | Strict | 260824 | 68 -> 68 | 0 pp [-7, +7] | 1.0000 |
| Cube | Moderate | 260822 | 52 -> 46 | -6 pp [-14, +2] | 0.2379 |
| Cube | Moderate | 260823 | 52 -> 55 | +3 pp [-4, +10] | 0.5811 |
| Cube | Moderate | 260824 | 52 -> 50 | -2 pp [-10, +6] | 0.8145 |
| Cube | Strict | 260822 | 25 -> 24 | -1 pp [-9, +7] | 1.0000 |
| Cube | Strict | 260823 | 25 -> 21 | -4 pp [-12, +4] | 0.4545 |
| Cube | Strict | 260824 | 25 -> 21 | -4 pp [-13, +5] | 0.5235 |
| TwoRoom | Moderate | 260822 | 92 -> 92 | 0 pp [0, 0] | 1.0000 |
| TwoRoom | Moderate | 260823 | 92 -> 92 | 0 pp [0, 0] | 1.0000 |
| TwoRoom | Moderate | 260824 | 92 -> 92 | 0 pp [0, 0] | 1.0000 |
| TwoRoom | Strict | 260822 | 82 -> 83 | +1 pp [-5, +8] | 1.0000 |
| TwoRoom | Strict | 260823 | 82 -> 87 | +5 pp [-1, +11] | 0.1797 |
| TwoRoom | Strict | 260824 | 82 -> 85 | +3 pp [-3, +10] | 0.5488 |

所有 18 個 paired CIs 都包含 0。TwoRoom Strict 的正向 point estimates 與 Cube
Strict 的負向 point estimates 可以保留為後續假說，但目前分別不是可靠 transfer
或 regression。CLEAR 使用 actual environment task predicate；它不能與 latent
hit profile 合併成單一 success definition。

## 6. Provenance 與品質稽核

### 6.1 Source、jobs 與 formal roots

| Item | Evidence |
|---|---|
| Source branch / commit | `codex/self-supervised-plannability` / `7e4c97c80b908ebc67474d1f0185a14802f056b2` |
| Base revision | `83f229b267f1c6be229546cb2bec93cbb253d5cf` |
| Formal root | `/public/home/xsy0001/workspace/code/stable-worldmodel/outputs/worktrees/self-supervised-plannability/outputs/experiments/self-supervised-plannability-v1` |
| Launch manifest | `outputs/launches/self-supervised-plannability-v1-7e4c97c/formal-launch.json` within the remote SSP worktree |
| Prepare / smoke | jobs `15735` / `15738`; 3/3 cells each, exit `0:0` |
| Training / profile / CLEAR | arrays `15741` / `15742` / `15743`; 9 / 9 / 24 cells, all exit `0:0` |
| Formal completion | 2026-08-22 18:26 CST / 20:26 AEST |

Launch gates recorded local full suite `1081 passed, 15 skipped, 1 xpassed`、remote
targeted suite `108 passed`、full-file dataset hashes、real-checkpoint smoke 與 clean
commit-pinned worktree。

### 6.2 Input identity

| Task | Dataset revision / SHA-256 | Project-trained checkpoint SHA-256 | Moderate / Strict manifest SHA-256 |
|---|---|---|---|
| PushT | `655cd446...` / `b6ebd9ac94bbe9e383f6e7a9cd92d74e9aa665ea57b758ed3717b0ee7df8d4fb` | `e4f14a2276918bcb34876fb8d86d16dbd8683ae6077a13a2275dee008a68c775` | `dcdce1f5...` / `2042018f...` |
| Cube | `02a19a67...` / `0664d507c4ff12009010644c9ae950836f954e700c172ccf22e7423af1a55625` | `eece65ce87e451d8ee953d83da0c566f77ad2d8f8f6ee5e77f7ddbae5bedf883` | `03f9c3a3...` / `fccf9d6d...` |
| TwoRoom | `6903a2de...` / `129a36aa93ea0de488d2bcc876e396de9e3907bf66c6aae6394e542ef6a6d623` | `68e6ca32ec5f7bdfb728ef89164ab62f566147e7724b74e5cf8e5858db746a65` | `216250de...` / `b1833f5e...` |

完整 hashes、逐 replicate profile/CLEAR results 與 paired intervals 另保存在
`SELF_SUPERVISED_PLANNABILITY_RESULTS.json`。正式 root 內有 9 個 training audits、
9 個 profile audits 與 24 個 terminal `results.txt.json`；每個 profile 包含
2,560 pair-seed rows。

### 6.3 Conclusion boundary

本實驗能回答的只有：在固定 LeWM、資料、16-parameter diagonal family、task
threshold 與有限 CEM budget 下，black-box SSP 是否學到更好的 registered latent
goal-set search geometry。結果沒有支持 task-independent broad improvement。

它不能推出：

- predictor accuracy、semantic representation 或 reachability 已改善；
- 所有 latent hits 都可執行或代表 task success；
- SSP 在其他 checkpoint、dataset、basis、threshold、planner 或 budget 上無效；
- 更高容量 geometry（full diagonal、low-rank 或 MLP）一定無法工作。

若要繼續，應將本次 v1 保留為 immutable negative/mixed evidence，另開明確的新
protocol 檢驗 capacity 或 basis sensitivity，而不是在這個 formal root 上依結果
回調 threshold、sigma、reward 或 metric family。
