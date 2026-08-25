#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --qos=user_xsy0001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8

: "${SSP_V2_REPO_ROOT:?}"
: "${SSP_V2_PYTHON:?}"
: "${SSP_V2_PREPARATION_ROOT:?}"
: "${SSP_V2_FORMAL_ROOT:?}"
: "${SSP_V2_COMMIT:?}"

geometries=(identity identity identity-repeat identity-repeat 260822 260822 260823 260823 260824 260824)
protocols=(moderate strict moderate strict moderate strict moderate strict moderate strict)
index="${SLURM_ARRAY_TASK_ID:?}"
geometry="${geometries[${index}]}"
protocol="${protocols[${index}]}"

cd "${SSP_V2_REPO_ROOT}"
test "$(git rev-parse HEAD)" = "${SSP_V2_COMMIT}"
test -z "$(git status --porcelain)"
export PYTHONPATH="${SSP_V2_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1

"${SSP_V2_PYTHON}" - "${geometry}" "${protocol}" <<'PY'
import json
import os
import sys
from pathlib import Path

from scripts.experiments.self_supervised_plannability_v2.clear import run_clear_cell
from scripts.experiments.self_supervised_plannability_v2.contracts import load_config

repo = Path.cwd()
task = 'reacher'
geometry, protocol = sys.argv[1:]
_, config = load_config(
    repo
    / 'scripts'
    / 'experiments'
    / 'self_supervised_plannability_v2'
    / 'configs'
    / f'{task}.json'
)
result = run_clear_cell(
    config=config,
    repo_root=repo,
    preparation_dir=Path(os.environ['SSP_V2_PREPARATION_ROOT']) / task,
    formal_root=os.environ['SSP_V2_FORMAL_ROOT'],
    geometry=geometry,
    protocol=protocol,
)
print(json.dumps({'completed': result['completed_trajectories']}, sort_keys=True))
PY
