#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --qos=user_xsy0001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-11

set -euo pipefail

: "${SSP_V2_5_REPO_ROOT:?}"
: "${SSP_V2_5_PYTHON:?}"
: "${SSP_V2_5_PREPARATION_ROOT:?}"
: "${SSP_V2_5_FORMAL_ROOT:?}"
: "${SSP_V2_5_COMMIT:?}"

tasks=(tworoom tworoom tworoom pusht pusht pusht cube cube cube reacher reacher reacher)
seeds=(260822 260823 260824 260822 260823 260824 260822 260823 260824 260822 260823 260824)
index="${SLURM_ARRAY_TASK_ID:?}"
task="${tasks[${index}]}"
seed="${seeds[${index}]}"

cd "${SSP_V2_5_REPO_ROOT}"
test "$(git rev-parse HEAD)" = "${SSP_V2_5_COMMIT}"
test -z "$(git status --porcelain)"
export PYTHONPATH="${SSP_V2_5_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1

"${SSP_V2_5_PYTHON}" -m \
  scripts.experiments.self_supervised_plannability_v2_5.run train \
  --config "scripts/experiments/self_supervised_plannability_v2_5/configs/${task}.json" \
  --preparation-dir "${SSP_V2_5_PREPARATION_ROOT}/${task}" \
  --output-dir "${SSP_V2_5_FORMAL_ROOT}/${task}/${seed}" \
  --replicate-seed "${seed}" \
  --device cuda
