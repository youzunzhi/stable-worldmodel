# find-goal-threshold：正式實驗報告

## 1. 摘要

本實驗要回答的是：對固定、project-trained 的 encoder/projector，是否能從
demo observation pairs 校準出一個 latent distance threshold
\(\epsilon\)，使

\[
d_0(o_i,o_j)=\operatorname{mean}_{D}\left[(z_i-z_j)^2\right]
\le \epsilon
\]

可作為「兩個單幀 observation 屬於同一 pointwise goal set」的二元判定。
正式校準使用 immutable commit
`142ffa7156ad79eaefd7f8d757b7ed0e4a6d54ea`，三個 task 的結果是：

| Task | Pointwise label variant | 正式 ε | Formal 終態 | Audit macro TPR / FPR |
|---|---|---:|---|---:|
| PushT | `pusht_joint_xy_pointwise_gap20_30` | `1.6419658660888672` | 已通過 validation 並鎖定 | `0.952936 / 0.100224` |
| Cube | `cube_block_xyz_pointwise_gap03_04` | — | `THRESHOLD_CALIBRATION_NO_FEASIBLE_OPERATING_POINT` | 未開啟 |
| TwoRoom | `tworoom_agent_xy_pointwise_gap8_16` | `1.392462968826294` | 已通過 validation 並鎖定 | `1.000000 / 0.028177` |

因此，PushT 與 TwoRoom 有 task-contract-specific threshold；Cube 在預先固定的
`macro-TPR >= 0.90`、`macro-FPR <= 0.10` 條件下沒有可行 threshold。後續
CLEAR endpoint self-eval 進一步顯示：這兩個 ε 都不能可靠取代 CLEAR evaluator
來預測 execution success rate。這不是矛盾；前者校準的是 demo 上的單幀 latent
geometry，後者還包含角度、持續成功、碰撞、路徑與可達性等語義。

## 2. 實驗合約

### 2.1 Ground-truth pointwise labels

每一對 observation 先只用同一列儲存的 simulator state 標成 `T/F/U`，再計算
latent distance。`U` 是預先定義的 tolerance gap：保留、統計，但不進入
threshold 選擇。

| Task | 使用的 state 與 task-space metric | T | U | F | 明確忽略 |
|---|---|---:|---:|---:|---|
| PushT | pusher XY + block XY 的 joint L2，pixel | `<20` | `[20,30]` | `>30` | block angle、velocity、held success |
| Cube | `privileged_block_0_pos` L2，metre | `<0.03` | `[0.03,0.04]` | `>0.04` | orientation、robot/gripper state、held success |
| TwoRoom | `pos_agent` L2，pixel | `<8` | `[8,16]` | `>16` | collision/route/room history、goal side、held success |

這些 variant 不等同於 CLEAR Moderate/Strict。特別是 Cube 本次只執行
position-only variant；規格中的 symmetry-aware pose variant 沒有被拿來產生本次
結果。

### 2.2 固定項目與選擇規則

- 每個 task 固定 dataset、checkpoint、observation preprocessing、label variant、
  residual、`float32` 與 latent dimension `D=192`；任何一項改變都需要新的
  calibration artifact，ε 不跨合約轉用。
- encoder/projector 全程 frozen；不呼叫 predictor、planner、action encoder，
  也不建構或 step environment。
- fit selector 的 primary estimand 是 task-stratified sample 的 anchor-group
  macro TPR/FPR。Uniform sample用來估計 population prevalence、precision 與
  secondary rates，不與 stratified sample 混算。
- selector 在 fit `T/F` distances 的有限候選集中，先限制
  `macro-FPR <= 0.10`，再最大化 macro-TPR，同 TPR 時取最小 ε，最後要求
  `macro-TPR >= 0.90`。
- validation 必須原封不動套用 fit ε；通過後才寫
  `selected_threshold.json`，再一次性開啟 audit。Audit 結果不得反向調整 ε。

## 3. 正式實驗操作

每個 task 都在新的 immutable output directory 從零執行以下流程：

| 階段 | 實際操作 | 防止的混淆或洩漏 |
|---|---|---|
| 1. Preflight | 驗證 clean commit、dataset/checkpoint SHA-256、schema、preprocessing、軟體與裝置資訊 | 錯用資料、模型或 compatibility fallback |
| 2. Group-first split | 先按獨立 source group/episode 做固定 seed 的 `60/20/20` fit/validation/audit split，再建 pairs | 同 episode、同 source 的跨 split leakage |
| 3. Frozen encoding | 對 100% eligible observations 各 encode 一次，保存 row ID、latent shards 與 hashes；前後重算 parameter hash | 模型更新、重複 encode 漂移、row 對錯 |
| 4. Latent-blind pair design | 每 task 無放回抽 `100M` Uniform + `20M` task-stratified pairs；pair IDs 與 `T/F/U` labels 在看 latent 前 materialize 並 hash | 依 latent 挑 easy/hard pairs |
| 5. Distance scoring | 只計算 `float32 mean_D((z_i-z_j)^2)`，Uniform 與 stratified shards 分開保存 | residual/dtype 偷換、重複計數 |
| 6. Fit selection | 只開 fit：`60M + 12M` pairs，依預註冊 selector 找 ε；無可行點立即停止 | 用 validation/audit 調 threshold |
| 7. Validation and lock | 對 `20M + 4M` validation pairs套用不變 ε；通過後 hash-lock 完整 threshold tuple | validation 後調參或合約漂移 |
| 8. One-time audit | 鎖定後才對 `20M + 4M` audit pairs評估，做 10,000 次 source-group clustered bootstrap | audit feedback tuning |

三個 task 各實現 `120M` pairs，target 與 realized counts 完全相同。正式入口為
`scripts/experiments/observation_goal_threshold/run.py`；衍生曲線由
`curve_plot.py` 從 immutable task artifacts 產生。首個 formal root 曾因
no-replacement collision 只得到 `5,996,879/6,000,000` unique pairs 而作廢；
修正版保留同一 RNG stream 增量補抽、重新去重，並在新 commit、新 root 將三個
task 全部從零重跑，未沿用 partial scores。

## 4. 正式校準結果

### 4.1 PushT 與 TwoRoom

| Task | Split | Macro TPR | Macro FPR | Uniform TPR | Uniform FPR | Population precision |
|---|---|---:|---:|---:|---:|---:|
| PushT | Fit | `0.943112` | `0.099999` | `0.946735` | `0.143844` | — |
| PushT | Validation | `0.954280` | `0.098155` | `0.953750` | `0.150354` | — |
| PushT | Audit | `0.952936` | `0.100224` | `0.950860` | `0.143495` | `0.008638` |
| TwoRoom | Fit | `1.000000` | `0.029390` | `1.000000` | `0.066920` | — |
| TwoRoom | Validation | `1.000000` | `0.027474` | `1.000000` | `0.065057` | — |
| TwoRoom | Audit | `1.000000` | `0.028177` | `1.000000` | `0.065225` | `0.228997` |

PushT 的 audit macro-FPR 點估計 `0.1002238` 比 0.10 高 `0.0002238`；
clustered bootstrap 95% CI 為 `[0.091592, 0.108988]`，macro-TPR CI 為
`[0.941702, 0.962902]`。所以它確實通過預註冊的 fit 與 validation promotion
gate，但 audit 顯示它落在 10% FPR 邊界，不能宣稱在所有 split 都穩健低於
10%。Audit 開啟後沒有重調 ε。

PushT uniform population 的 `T` prevalence 約為 `0.00131`，因此即使 TPR 約
0.95，audit population precision 仍只有 `0.00864`；這是 base-rate 結果，
不能用 50/50 stratified-pair precision 取代。

TwoRoom 的 audit macro-TPR 95% CI 為 `[1.000000, 1.000000]`，macro-FPR CI
為 `[0.027207, 0.029153]`，在三個 split 都保留明顯 operating margin。

### 4.2 Cube 的負結果

Cube 在 `macro-FPR <= 0.10` 內可達到的最佳 fit macro-TPR 只有
`0.7078820482`，低於要求的 `0.90`。正式終態因此是
`THRESHOLD_CALIBRATION_NO_FEASIBLE_OPERATING_POINT`：沒有 selected ε、沒有
threshold lock，也沒有開啟 validation/audit。Label gap 與 operating contract
沒有為了得到一個數字而放寬。

### 4.3 Fit ε–TPR/FPR curves

以下曲線以 fit split 的 anchor-group macro TPR/FPR 為 y 軸；虛線是
`TPR=0.90`、`FPR=0.10`，黑色垂直線只標示已鎖定的 PushT/TwoRoom ε。
每張圖的 `curve_manifest.json` 都 hash-lock config、status、fit score shards、
threshold（若存在）與 PNG。

![PushT epsilon–TPR/FPR curve](results/find-goal-threshold/curves/formal-142ffa7-20260820/pusht/epsilon_tpr_fpr_curve.png)

![Cube epsilon–TPR/FPR curve](results/find-goal-threshold/curves/formal-142ffa7-20260820/cube/epsilon_tpr_fpr_curve.png)

![TwoRoom epsilon–TPR/FPR curve](results/find-goal-threshold/curves/formal-142ffa7-20260820/tworoom/epsilon_tpr_fpr_curve.png)

## 5. Fit data sensitivity（補充實驗）

為檢查 split 與 pair-sampling seed 敏感度，正式 seed-0 加上四個新 seed，總共
五個 paired replicates。每個新 current-method replicate 都重新 materialize
`100M Uniform + 20M stratified` pairs，但只開 fit；validation/audit 保持關閉，
所以這一節不會產生或取代正式 locked thresholds。

| Task | 五 seed 報告值範圍 | Range | CV | 達到 fit 合約 |
|---|---:|---:|---:|---:|
| PushT | `1.630940914–1.642531395` | `0.011590481` | `0.3015%` | `5/5` |
| Cube | `1.679481864–1.680142164`（descriptive best point） | `0.000660300` | `0.0167%` | `0/5` |
| TwoRoom | `1.386452556–1.401974201` | `0.015521646` | `0.4436%` | `5/5` |

PushT/TwoRoom 在這五個 seeds 下只有輕微變動；Cube 則在 5/5 replicates 都不可行，
因此負結果不是 seed-0 特例。配對移除全部 Uniform pairs 時，三個 task、五個
seeds 的報告值變化都恰為 `0.0`。原因是目前
`min_population_precision=null`，Uniform pairs 本來就不進入 threshold selector；
它們仍是估計 population prevalence、precision 與 secondary rates 所必需，不能
因此解讀為「100M Uniform pairs 對整個分析沒有用途」。完整逐 seed 結果見
`EXPERIMENT_T_GOAL_THRESHOLD_DATA_SENSITIVITY_REPORT.md`。

## 6. Post-lock CLEAR endpoint self-eval

### 6.1 操作與固定合約

這是 threshold 鎖定後的 downstream agreement test，不是 threshold selection。
每個可用 task 以同一 checkpoint 執行 CLEAR v0.5 Moderate/Strict、固定 seed-42、
每 cell 100 pairs；CEM 固定 `batch_size=1`、`num_samples=300`、`n_steps=30`、
`topk=30`、`cpu_threads=1`、goal offset 25、execution budget 50。

CLEAR evaluator 回傳每個 pair 的 actual S/F。每個 pair 結束後，再用 frozen
encoder/projector 計算 final observation 到 fixed goal 的 distance，並以
`distance <= locked epsilon` 預測 S/F。Primary 指標是 predicted SR 與 actual
SR 的 paired difference、其 bootstrap 95% CI，以及 pair confusion；task 與
protocol 不 pooling。

Cube 沒有 promoted ε，所以 fixed-threshold matrix 只能執行 4/6 cells；Cube
Moderate/Strict 被明確標為 `THRESHOLD_UNAVAILABLE`，沒有拿 failed-fit best
point 代替。

### 6.2 Fixed-threshold 結果

| Task | CLEAR rule | Actual SR | ε-predicted SR | Predicted−actual SR [paired bootstrap 95% CI] | Pair accuracy [Wilson 95% CI] | TP/TN/FP/FN |
|---|---|---:|---:|---:|---:|---:|
| PushT | Moderate | `90%` | `99%` | `+9 pp [4,15]` | `0.91 [0.838,0.952]` | `90/1/9/0` |
| PushT | Strict | `67%` | `98%` | `+31 pp [22,40]` | `0.69 [0.594,0.772]` | `67/2/31/0` |
| TwoRoom | Moderate | `92%` | `41%` | `−51 pp [−61,−41]` | `0.49 [0.394,0.587]` | `41/8/0/51` |
| TwoRoom | Strict | `82%` | `96%` | `+14 pp [8,21]` | `0.86 [0.779,0.915]` | `82/4/14/0` |

四個 cells 的 SR-error CI 都不含 0，因此目前 ε 不能可靠預測 CLEAR SR。
PushT 忽略 block angle 與 sustained success，因而以 false positive 為主；
TwoRoom Moderate evaluator 使用 `<16 px`，但 calibrated positive gap 是
`<8 px`，因而出現 51 個 false negative。TwoRoom Strict 雖同為 8 px endpoint
距離，仍要求 goal side、合法 route 與 collision semantics，所以有 14 個
false positive。

### 6.3 Endpoint ε–TPR/FPR diagnostic

另以同一組 3 tasks × 2 rules × 100 endpoints 掃描所有 observed distance
breakpoints。這裡的 positive/negative class 是 CLEAR evaluator S/F，與第 4 節
calibration fit macro curve 是不同 estimand；Cube 只做 score-only sweep，不畫
threshold marker，也不產生 fixed-ε prediction。

![3x2 epsilon–endpoint TPR/FPR curves](results/find-goal-threshold/self-eval/epsilon-accuracy-fb37755-20260821-cube-score-only/curve-tpr-fpr-v1/epsilon_endpoint_tpr_fpr_3x2.png)

| Task | CLEAR rule | Actual SR | Locked ε | TPR / FPR at locked ε |
|---|---|---:|---:|---:|
| PushT | Moderate | `90%` | `1.641965866` | `1.000 / 0.900` |
| PushT | Strict | `67%` | `1.641965866` | `1.000 / 0.939` |
| Cube | Moderate | `52%` | — | — |
| Cube | Strict | `25%` | — | — |
| TwoRoom | Moderate | `92%` | `1.392462969` | `0.446 / 0.000` |
| TwoRoom | Strict | `82%` | `1.392462969` | `1.000 / 0.778` |

這張圖是 post-lock descriptive diagnostic，不能從曲線的最佳點反選或重調 ε。
完整 3×2 score curves 也不會把 fixed-threshold self-eval 從 4/6 變成完整 6/6。

## 7. Provenance、品質檢查與結論邊界

### 7.1 主要 provenance

| 產物 | Revision / root |
|---|---|
| 三 task 正式 calibration | commit `142ffa7156ad79eaefd7f8d757b7ed0e4a6d54ea`; `/public/home/xsy0001/workspace/data/stable-worldmodel/experiments/observation_goal_threshold/formal-142ffa7-20260820` |
| Fit data sensitivity | commit `2b537e5337dbfaa306f576df2436d556addd388a`; `/public/home/xsy0001/workspace/data/stable-worldmodel/experiments/observation_goal_threshold/data-sensitivity-2b537e5-20260821` |
| Primary fixed-ε self-eval | revision `d0c466ba333d98f4b8aaae2b50cd0b86eddaf8d4`; `results/find-goal-threshold/self-eval/self-eval-d0c466b-20260821-available-thresholds/` |
| Complete 3×2 endpoint scores / curves | score revision `fb37755c352698aa6ab7faf8260383850c4d7685`; TPR/FPR renderer revision `c507b0598d0e7ae92502a1088fdad263b537a1a5` |

Primary machine-readable summary 是 `FIND_GOAL_THRESHOLD_RESULTS.json`。PushT
與 TwoRoom 的 `selected_threshold.json` SHA-256 分別為
`5dc633561a94928eb37dcf2506423ba58c8cf35994cf3e8d7a74867880ae98fb` 與
`e2b00f0b38f42b37373e8d4576d915d0140943f5f8b0d0569d61f5b3b41ece4d`。

| Task | Dataset SHA-256 | Project-trained checkpoint SHA-256 |
|---|---|---|
| PushT | `b6ebd9ac94bbe9e383f6e7a9cd92d74e9aa665ea57b758ed3717b0ee7df8d4fb` | `e4f14a2276918bcb34876fb8d86d16dbd8683ae6077a13a2275dee008a68c775` |
| Cube | `0664d507c4ff12009010644c9ae950836f954e700c172ccf22e7423af1a55625` | `eece65ce87e451d8ee953d83da0c566f77ad2d8f8f6ee5e77f7ddbae5bedf883` |
| TwoRoom | `129a36aa93ea0de488d2bcc876e396de9e3907bf66c6aae6394e542ef6a6d623` | `68e6ca32ec5f7bdfb728ef89164ab62f566147e7724b74e5cf8e5858db746a65` |

三個 checkpoint 都是 project-trained、seed-3072、從固定 D0 offline split 由零
訓練的 canonical M0。舊規格中的 `/ssd` full-demo checkpoint 已不存在，因此這是
明示的 compatibility fallback，不是官方 HF checkpoint。

### 7.2 實作與測試證據

- 正式 calibration 修正版的 remote full suite：`1102 passed, 11 skipped, 1 xfailed`。
- Data-sensitivity execution commit：targeted `38 passed`；full suite
  `1104 passed, 11 skipped, 1 xfailed`。
- Primary self-eval revision：targeted CLEAR/self-eval `55 passed`；其 parent
  remote full suite `1106 passed, 11 skipped, 1 xfailed`。
- `self-eval-91aea84-20260821-available-thresholds` 保留為 historical
  provenance；加入 paired-bootstrap uncertainty 後由 `d0c466b` 取代為 primary。
  兩版只有 PushT Strict 的 2 個 evaluator S/F labels flip，四個已執行 cells 的
  ε predictions 都沒有 flip；這是 runtime drift，不是 threshold retuning。
- Endpoint renderer revision：targeted `62 passed`；3×2 PNG 已做原始解析度視覺檢查。
- 正式執行前後 encoder/projector parameter hashes 相同；planner、predictor、
  action encoder、environment construction/step 的 forbidden-call counts 都為 0。
- 作廢的 `formal-2d4f13d-20260820` 保留為 failure provenance，不進入任何正式統計。

### 7.3 能支持與不能支持的主張

主校準能支持的是：在精確固定的 task/label/dataset/checkpoint/preprocessing/
residual/dtype identity 下，frozen encoder 的 pointwise latent geometry 是否有
符合預註冊 TPR/FPR 合約的 ε。它不能直接支持 predictor、reachability、planner、
execution 或 CLEAR performance。

Post-lock self-eval 能支持的是 endpoint predicate 與 CLEAR evaluator 的 paired
agreement 結果；它不能把高 pair accuracy 解讀為 planner quality，也不能用
CLEAR endpoints 回頭調 ε。Cube 的 score-only curves 不能補造 promoted threshold，
而 PushT audit 在 FPR 邊界上的結果也必須與其 fit/validation 通過事實同時保留。
