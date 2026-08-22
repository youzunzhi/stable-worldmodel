"""Run and audit one secondary SSP CLEAR v0.5 cell."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .contracts import PROTOCOL_ID, SSPFailure, resolve_repo_path


def run_clear_cell(
    *,
    config: dict,
    repo_root: Path,
    formal_root: str | Path,
    geometry: str,
    protocol: str,
) -> dict:
    if protocol not in {'moderate', 'strict'}:
        raise ValueError('CLEAR protocol must be moderate or strict')
    task = config['task']
    manifest = next(
        resolve_repo_path(repo_root, path)
        for path in config['clear_manifests']
        if protocol in Path(path).name
    )
    root = Path(formal_root).expanduser().resolve()
    overrides = [
        f'policy={config["checkpoint"]["path"]}',
        f'eval.dataset_name={config["dataset"]["path"]}',
        f'eval.manifest={manifest}',
        'eval.video=false',
        'seed=42',
    ]
    if geometry == 'identity':
        output = root / task / 'identity' / 'clear' / protocol
    else:
        seed = int(geometry)
        if seed not in config['replicate_seeds']:
            raise ValueError(f'unknown SSP geometry replicate {geometry}')
        replicate = root / task / str(seed)
        completion = replicate / 'training.completed.json'
        if not completion.is_file():
            raise SSPFailure(
                'SSP_INCOMPLETE', f'training is incomplete: {replicate}'
            )
        geometry_path = replicate / 'selected_geometry.pt'
        basis_path = replicate / 'geometry_basis.npy'
        overrides.extend(
            [
                f'ssp.geometry={geometry_path}',
                f'ssp.basis={basis_path}',
            ]
        )
        output = replicate / 'clear' / protocol
    command = [
        sys.executable,
        str(repo_root / 'scripts' / 'plan' / 'eval_wm.py'),
        '--config-name',
        task,
        *overrides,
        f'output.dir={output}',
        'output.filename=results.txt',
    ]
    subprocess.run(command, cwd=repo_root, check=True)
    result_path = output / 'results.txt.json'
    result = json.loads(result_path.read_text())
    clear = result.get('clear_lewm') or {}
    if (
        result.get('completed_trajectories') != 100
        or not clear.get('solver_contract_matched')
        or clear.get('cpu_threads') != 1
    ):
        raise SSPFailure(
            'SSP_INCOMPLETE', 'CLEAR result failed completion/contract audit'
        )
    ssp = result.get('self_supervised_plannability')
    if geometry == 'identity' and ssp is not None:
        raise SSPFailure(
            'SSP_INPUT_HASH_MISMATCH', 'identity CLEAR cell used SSP geometry'
        )
    if geometry != 'identity' and (
        ssp is None or ssp.get('protocol_id') != PROTOCOL_ID
    ):
        raise SSPFailure(
            'SSP_INPUT_HASH_MISMATCH',
            'learned CLEAR cell omitted SSP geometry',
        )
    return result
