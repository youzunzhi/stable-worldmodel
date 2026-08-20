# Experiment T：demo-calibrated goal threshold 正式報告

## 結論

正式實驗使用 immutable commit
`142ffa7156ad79eaefd7f8d757b7ed0e4a6d54ea`，結果如下。

| Task | Pointwise label variant | 找到的 threshold ε | Formal 終態 | Audit macro TPR / FPR | Audit uniform TPR / FPR | Audit population precision |
|---|---|---:|---|---:|---:|---:|
| PushT | `pusht_joint_xy_pointwise_gap20_30` | `1.6419658660888672` | threshold 已鎖定；fit/validation 合約通過 | `0.952936 / 0.100224` | `0.950860 / 0.143495` | `0.008638` |
| Cube | `cube_block_xyz_pointwise_gap03_04` | **沒有可行 threshold** | `THRESHOLD_CALIBRATION_NO_FEASIBLE_OPERATING_POINT` | audit 未開啟 | audit 未開啟 | audit 未開啟 |
| TwoRoom | `tworoom_agent_xy_pointwise_gap8_16` | `1.392462968826294` | threshold 已鎖定；fit/validation 合約通過 | `1.000000 / 0.028177` | `1.000000 / 0.065225` | `0.228997` |

Cube 的正式結論不是缺少實驗，而是在要求
`macro-TPR >= 0.90` 且 `macro-FPR <= 0.10` 的預註冊 operating
contract 下不存在可行 ε；在 FPR 約束內可達到的最佳 macro-TPR 為
`0.7078820482`。規格中的 label gap 與 operating constraints 沒有被放寬，
也沒有為了產生數字而開啟 audit 或重新選 threshold。

## PushT 解讀

PushT 在 fit 選出的 ε 為 `1.6419658660888672`：

| Split | Macro TPR | Macro FPR | Uniform TPR | Uniform FPR |
|---|---:|---:|---:|---:|
| Fit | `0.943112` | `0.099999` | `0.946735` | `0.143844` |
| Validation | `0.954280` | `0.098155` | `0.953750` | `0.150354` |
| Audit | `0.952936` | `0.100224` | `0.950860` | `0.143495` |

一次性 audit 的 macro-FPR 點估計為 `0.1002238`，比 0.10 高
`0.0002238`；10,000 次 source-group clustered bootstrap 的 95% CI 是
`[0.091592, 0.108988]`。因此這個 ε 確實通過預註冊的 fit/validation
選擇與 promotion gate，但 audit 顯示它在 10% FPR 邊界上，不能宣稱在
所有 split 都穩健低於 10%。audit 開啟後沒有重調 ε。

PushT 的 uniform population 中 T prevalence 僅約 `0.00131`，所以即使
TPR 約 0.95，audit population precision 仍只有 `0.00864`。這是 base-rate
結果，不應用 stratified-pair precision 取代。

## TwoRoom 解讀

TwoRoom 在 fit 選出的 ε 為 `1.392462968826294`：

| Split | Macro TPR | Macro FPR | Uniform TPR | Uniform FPR |
|---|---:|---:|---:|---:|
| Fit | `1.000000` | `0.029390` | `1.000000` | `0.066920` |
| Validation | `1.000000` | `0.027474` | `1.000000` | `0.065057` |
| Audit | `1.000000` | `0.028177` | `1.000000` | `0.065225` |

Audit macro-TPR 的 clustered 95% CI 為 `[1.000000, 1.000000]`，
macro-FPR CI 為 `[0.027207, 0.029153]`。這個 threshold 在三個 split
都保留很大的預註冊 operating margin。

## 固定合約與 provenance

- 每個 task 都實現 `100,000,000` 個 latent-blind uniform pairs 加
  `20,000,000` 個 task-space stratified pairs；fit/validation/audit
  分別為 60%/20%/20%，target 與 realized counts 完全相同。
- residual 固定為 `mean_D((z_i-z_j)^2)`，`D=192`、`float32`；每個
  threshold 僅適用於其 task、pointwise label variant、dataset、checkpoint、
  preprocessing 與 compatibility signature。
- encoder/projector 在正式執行前後的 parameter hash 完全相同；
  `planner`、`predictor`、`action_encoder`、environment construction/step
  的 forbidden-call counts 全為 0。
- PushT threshold file SHA-256：
  `5dc633561a94928eb37dcf2506423ba58c8cf35994cf3e8d7a74867880ae98fb`。
- TwoRoom threshold file SHA-256：
  `e2b00f0b38f42b37373e8d4576d915d0140943f5f8b0d0569d61f5b3b41ece4d`。
- 正式 root：
  `/public/home/xsy0001/workspace/data/stable-worldmodel/experiments/observation_goal_threshold/formal-142ffa7-20260820`。
- 完整 remote artifacts（pair manifests、all score rows、fit/validation/audit
  metrics、bootstrap arrays、plots、reports、locks）都保留在上述 root。

### Dataset 與 checkpoint SHA-256

| Task | Dataset SHA-256 | Project-trained checkpoint SHA-256 |
|---|---|---|
| PushT | `b6ebd9ac94bbe9e383f6e7a9cd92d74e9aa665ea57b758ed3717b0ee7df8d4fb` | `e4f14a2276918bcb34876fb8d86d16dbd8683ae6077a13a2275dee008a68c775` |
| Cube | `0664d507c4ff12009010644c9ae950836f954e700c172ccf22e7423af1a55625` | `eece65ce87e451d8ee953d83da0c566f77ad2d8f8f6ee5e77f7ddbae5bedf883` |
| TwoRoom | `129a36aa93ea0de488d2bcc876e396de9e3907bf66c6aae6394e542ef6a6d623` | `68e6ca32ec5f7bdfb728ef89164ab62f566147e7724b74e5cf8e5858db746a65` |

這三個 checkpoint 都是目前可用的 project-trained、seed-3072、從固定 D0
offline split 由零訓練的 canonical M0。舊規格指向的 `/ssd` full-demo
checkpoint 已不存在，因此這是報告中明示的 compatibility fallback，不是
官方 HF checkpoint。

## 測試與失敗保留

- 修正版 remote 全 repository suite：`1102 passed, 11 skipped, 1 xfailed`。
- 實驗專屬測試包含 state contracts、group-first split、latent-blind sampling、
  no-replacement refill、exact preprocessing parity、selection/validation、
  clustered bootstrap 與 audit-lock guard。
- 首次 formal root
  `/public/home/xsy0001/workspace/data/stable-worldmodel/experiments/observation_goal_threshold/formal-2d4f13d-20260820`
  保留為失敗 provenance。該版 PushT 在 6,000,000 個 unique pairs 目標下只
  得到 5,996,879；修正版沿用同一預註冊 RNG stream 增量補抽並重新去重，
  而不是換 seed 或複製 pairs。所有 task 隨後在新 commit、新 root 從零重跑，
  沒有沿用失敗 root 的 partial outputs。

## 能支持與不能支持的主張

結果支持的是 frozen project-trained encoder/projector 的 pointwise latent
geometry threshold calibration。它不是 predictor、reachability、planner，
也不是官方 CLEAR Moderate/Strict 成功率結果；不能由此推論 closed-loop
planning performance。
