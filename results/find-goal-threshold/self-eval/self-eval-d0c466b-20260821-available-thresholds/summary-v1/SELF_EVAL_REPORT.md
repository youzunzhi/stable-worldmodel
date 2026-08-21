# find-goal-threshold CLEAR endpoint self-eval

- Matrix status: **INCOMPLETE**
- Contract: 3 tasks x Moderate/Strict x 100 fixed CLEAR pairs.
- Prediction: final endpoint latent distance `<= epsilon`.

| Task | CLEAR rule | Epsilon | Actual SR | Predicted SR | SR error [paired 95% CI] | Accuracy [Wilson 95% CI] | TP/TN/FP/FN |
|---|---|---:|---:|---:|---:|---:|---:|
| pusht | moderate | 1.641965866 | 90.0% | 99.0% | +9.0 [+4.0, +15.0] pp | 0.910 [0.838, 0.952] | 90/1/9/0 |
| pusht | strict | 1.641965866 | 67.0% | 98.0% | +31.0 [+22.0, +40.0] pp | 0.690 [0.594, 0.772] | 67/2/31/0 |
| tworoom | moderate | 1.392462969 | 92.0% | 41.0% | -51.0 [-61.0, -41.0] pp | 0.490 [0.394, 0.587] | 41/8/0/51 |
| tworoom | strict | 1.392462969 | 82.0% | 96.0% | +14.0 [+8.0, +21.0] pp | 0.860 [0.779, 0.915] | 82/4/14/0 |

## Missing cells

- `cube/moderate`
- `cube/strict`

Accuracy is paired endpoint agreement with the installed CLEAR evaluator. It does not turn the pointwise predicate into a reachability, route, pose, or sustained-success rule.
