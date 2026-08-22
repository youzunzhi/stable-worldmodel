#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --qos=user_xsy0001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00

set -euo pipefail

: "${SSP_V2_REPO_ROOT:?}"
: "${SSP_V2_PYTHON:?}"
: "${SSP_V2_PREPARATION_ROOT:?}"
: "${SSP_V2_FORMAL_ROOT:?}"
: "${SSP_V2_COMMIT:?}"

tasks=(tworoom tworoom tworoom pusht pusht pusht)
seeds=(260822 260823 260824 260822 260823 260824)
index="${SLURM_ARRAY_TASK_ID:?}"
task="${tasks[${index}]}"
seed="${seeds[${index}]}"

cd "${SSP_V2_REPO_ROOT}"
test "$(git rev-parse HEAD)" = "${SSP_V2_COMMIT}"
test -z "$(git status --porcelain)"
export PYTHONPATH="${SSP_V2_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1

"${SSP_V2_PYTHON}" -m \
  scripts.experiments.self_supervised_plannability_v2.run profile \
  --config "scripts/experiments/self_supervised_plannability_v2/configs/${task}.json" \
  --preparation-dir "${SSP_V2_PREPARATION_ROOT}/${task}" \
  --replicate-dir "${SSP_V2_FORMAL_ROOT}/${task}/${seed}" \
  --device cuda
