#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --qos=user_xsy0001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail

: "${SSP_REPO_ROOT:?}"
: "${SSP_PYTHON:?}"
: "${SSP_PREPARATION_ROOT:?}"
: "${SSP_COMMIT:?}"

tasks=(pusht cube tworoom)
task="${tasks[${SLURM_ARRAY_TASK_ID:?}]}"

cd "${SSP_REPO_ROOT}"
test "$(git rev-parse HEAD)" = "${SSP_COMMIT}"
test -z "$(git status --porcelain)"
export PYTHONPATH="${SSP_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1

"${SSP_PYTHON}" -m \
  scripts.experiments.self_supervised_plannability.run prepare \
  --config "scripts/experiments/self_supervised_plannability/configs/${task}.json" \
  --output-dir "${SSP_PREPARATION_ROOT}/${task}" \
  --repo-root "${SSP_REPO_ROOT}" \
  --device cuda \
  --formal \
  --full-dataset-hash
