# PLANNING_HORIZON_260808

這是 CLEAR-LeWM v0.5 的 planning-horizon 與 execution-budget ablation。比較 canonical baseline `(planning_horizon=5, receding_horizon=5, eval_budget=50 primitive steps)`，以及 `(10,5,50)`、`(10,10,50)`、`(5,5,25)`，涵蓋 PushT、Cube、TwoRoom 的 Moderate/Strict，共 18 個 ablation evaluation cells；每格 100 個固定 pairs。

## 結果

括號內是相對於 5/5 baseline 的 success-count 變化。

| Task | Criterion | 5/5, budget 50 baseline | 5/5, budget 25 | 10/5, budget 50 | 10/10, budget 50 |
|---|---:|---:|---:|---:|---:|
| PushT | Moderate | 96 | 92 (-4) | 21 (-75) | 56 (-40) |
| PushT | Strict | 85 | 56 (-29) | 43 (-42) | 47 (-38) |
| Cube | Moderate | 47 | 45 (-2) | 40 (-7) | 38 (-9) |
| Cube | Strict | 28 | 24 (-4) | 12 (-16) | 14 (-14) |
| TwoRoom | Moderate | 88 | 72 (-16) | 37 (-51) | 56 (-32) |
| TwoRoom | Strict | 81 | 26 (-55) | 9 (-72) | 18 (-63) |
| **6-cell mean** |  | **70.83** | **52.50 (-18.33)** | **27.00 (-43.83)** | **38.17 (-32.67)** |

在 `planning_horizon=10` 內，`receding_horizon=10` 比 `receding_horizon=5` 平均高 11.17 successes，六格中五格較好；唯一例外是 Cube Moderate（38 vs 40）。不過兩個 horizon=10 設定都低於 5/5 baseline。

將 5/5 的 execution budget 從 50 降到 25 primitive steps 後，六格平均少 18.33 successes。影響集中在需要更多執行時間或 sustained success 的 cells：TwoRoom Strict 少 55、PushT Strict 少 29；Cube Moderate/Strict 只少 2/4。

逐 pair 檢查中，六格的 budget-25 `gained` 都是 0，成功集合完全是 budget-50 baseline 的子集。因此這些下降不是 pairs 或 optimizer 隨機差異；它們正好對應在後半 25 primitive steps 才達成成功的 pairs。這項比較同時移除了 baseline 在第 25 step 後的第二次 5-block planning，所以效果代表「少一半 execution budget，以及沒有第二次 replanning」的合併影響。

逐 pair 配對也顯示這不是單純 aggregate 波動：例如 PushT Moderate 從 10/5 換到 10/10 有 38 個 gained、3 個 lost；TwoRoom Moderate 有 26 個 gained、7 個 lost。完整 gained/lost 數字在 `summary.json`。

## 執行與解讀

固定條件為 project-trained epoch-10 checkpoints、Arm A、policy seed 42、CLEAR-LeWM v0.5 fixed manifests、CEM `300 samples × 30 iterations / top-k 30`、`cpu_threads=1`、`action_block=5`、不錄影。只有各 ablation 指定的 horizon/receding horizon 或 execution budget 不同。

平均 evaluation time 為 5/5 budget-50 baseline 92.25 秒、5/5 budget-25 67.09 秒、10/5 budget-50 206.56 秒、10/10 budget-50 113.45 秒。5/5 budget-25 只執行一次 5-block plan；baseline 會在 25 primitive steps 後再規劃一次。

較長 horizon 同時把 CEM 的 action-block 搜尋維度由 5 加到 10，也讓 predictor 的 rollout 更長。固定 samples/iterations 下，搜尋變稀疏及長 rollout model error 累積，是 performance 普遍下降的合理機制解釋；這是由設定與結果推導的診斷，不是本次 evaluation 直接隔離出的因果證明。

## Audit 與 artifacts

- 18/18 ablation 結果與 6/6 baseline 都完成 100 pairs；success vector 長度均為 100。
- 三組 sampled episode/flat/start index 陣列逐格與 baseline 完全相同。
- checkpoint、checkpoint config、manifest、solver contract、seed、CPU threads 逐檔相符。
- TwoRoom 每個 run 都有 100 筆 topology records，沒有 invalid route、unclear start 或 unclear goal。
- logs 未發現 traceback、OOM、ERROR 或 exception。
- Budget-25 JSON 明確記錄實際 `eval_budget=25`、canonical protocol budget 50、`eval_budget_ablation_opt_in=true`、`eval_budget_contract_matched=false`。臨時 evaluator 已恢復至原始 SHA-256。
- 原始新結果在 `raw/`，baseline JSON 在 `baseline_raw/`，執行 logs 在 `logs/`；所有 SHA-256 與配對統計在 `summary.json`。

5/5 budget-50 仍是 canonical planner baseline。10/5、10/10 與 5/5 budget-25 都是刻意改變 planner/execution contract 的 ablation，不應標成 canonical CLEAR-LeWM v0.5 baseline submission。
