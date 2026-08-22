#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --qos=user_xsy0001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

set -euo pipefail

: "${SSP_REPO_ROOT:?}"
: "${SSP_PYTHON:?}"
: "${SSP_FORMAL_ROOT:?}"
: "${SSP_COMMIT:?}"

tasks=(pusht pusht pusht pusht pusht pusht pusht pusht cube cube cube cube cube cube cube cube tworoom tworoom tworoom tworoom tworoom tworoom tworoom tworoom)
geometries=(identity identity 260822 260822 260823 260823 260824 260824 identity identity 260822 260822 260823 260823 260824 260824 identity identity 260822 260822 260823 260823 260824 260824)
protocols=(moderate strict moderate strict moderate strict moderate strict moderate strict moderate strict moderate strict moderate strict moderate strict moderate strict moderate strict moderate strict)
index="${SLURM_ARRAY_TASK_ID:?}"
task="${tasks[${index}]}"
geometry="${geometries[${index}]}"
protocol="${protocols[${index}]}"

cd "${SSP_REPO_ROOT}"
test "$(git rev-parse HEAD)" = "${SSP_COMMIT}"
test -z "$(git status --porcelain)"
export PYTHONPATH="${SSP_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1

"${SSP_PYTHON}" - "${task}" "${geometry}" "${protocol}" <<'PY'
import json
import sys
from pathlib import Path

from scripts.experiments.self_supervised_plannability.clear import run_clear_cell

repo = Path.cwd()
task, geometry, protocol = sys.argv[1:]
config = json.loads(
    (repo / 'scripts' / 'experiments' / 'self_supervised_plannability' / 'configs' / f'{task}.json').read_text()
)
result = run_clear_cell(
    config=config,
    repo_root=repo,
    formal_root=Path(__import__('os').environ['SSP_FORMAL_ROOT']),
    geometry=geometry,
    protocol=protocol,
)
print(json.dumps({'completed': result['completed_trajectories']}, sort_keys=True))
PY
