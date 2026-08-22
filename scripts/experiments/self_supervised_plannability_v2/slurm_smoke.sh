#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --qos=user_xsy0001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

: "${SSP_V2_REPO_ROOT:?}"
: "${SSP_V2_PYTHON:?}"
: "${SSP_V2_PREPARATION_ROOT:?}"
: "${SSP_V2_SMOKE_ROOT:?}"
: "${SSP_V2_COMMIT:?}"

tasks=(tworoom pusht)
task="${tasks[${SLURM_ARRAY_TASK_ID:?}]}"

cd "${SSP_V2_REPO_ROOT}"
test "$(git rev-parse HEAD)" = "${SSP_V2_COMMIT}"
test -z "$(git status --porcelain)"
export PYTHONPATH="${SSP_V2_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1

"${SSP_V2_PYTHON}" -m \
  scripts.experiments.self_supervised_plannability_v2.run smoke \
  --config "scripts/experiments/self_supervised_plannability_v2/configs/${task}.json" \
  --preparation-dir "${SSP_V2_PREPARATION_ROOT}/${task}" \
  --output-dir "${SSP_V2_SMOKE_ROOT}/${task}" \
  --device cuda
