"""Run one matched-code-path secondary SSP-v2 CLEAR cell."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .contracts import PROTOCOL_ID, SSPV2Failure, resolve_repo_path


def run_clear_cell(
    *,
    config: dict,
    repo_root: Path,
    preparation_dir: str | Path,
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
    preparation = Path(preparation_dir).expanduser().resolve()
    basis_path = preparation / 'action_effect_basis.npy'
    action_stats_path = preparation / 'action_stats.json'
    for path in (basis_path, action_stats_path):
        if not path.is_file():
            raise SSPV2Failure(
                'SSP_V2_INCOMPLETE', f'missing preparation artifact {path}'
            )
    root = Path(formal_root).expanduser().resolve()
    overrides = [
        f'policy={config["checkpoint"]["path"]}',
        f'eval.dataset_name={config["dataset"]["path"]}',
        f'eval.manifest={manifest}',
        'eval.video=false',
        'seed=42',
        'ssp.version=2',
        f'ssp.basis={basis_path}',
        f'ssp.action_stats={action_stats_path}',
    ]
    if geometry in {'identity', 'identity-repeat'}:
        overrides.append('ssp.identity=true')
        output = root / task / geometry / 'clear' / protocol
    else:
        seed = int(geometry)
        if seed not in config['replicate_seeds']:
            raise ValueError(f'unknown v2 geometry replicate {geometry}')
        replicate = root / task / str(seed)
        completion = replicate / 'training.completed.json'
        profile = replicate / 'profile' / 'summary.json'
        if not completion.is_file() or not profile.is_file():
            raise SSPV2Failure(
                'SSP_V2_INCOMPLETE',
                f'training/profile is incomplete: {replicate}',
            )
        profile_result = json.loads(profile.read_text())
        if not profile_result['primary']['positive_gate']:
            raise SSPV2Failure(
                'SSP_V2_INCOMPLETE',
                'CLEAR promotion requires a positive internal gate',
            )
        geometry_path = replicate / 'selected_geometry.pt'
        overrides.extend(
            [
                'ssp.identity=false',
                f'ssp.geometry={geometry_path}',
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
    result = json.loads((output / 'results.txt.json').read_text())
    clear = result.get('clear_lewm') or {}
    ssp = result.get('self_supervised_plannability') or {}
    if (
        result.get('completed_trajectories') != 100
        or not clear.get('solver_contract_matched')
        or clear.get('cpu_threads') != 1
        or ssp.get('protocol_id') != PROTOCOL_ID
    ):
        raise SSPV2Failure(
            'SSP_V2_INCOMPLETE', 'CLEAR completion/protocol audit failed'
        )
    expected_mode = (
        'identity-zero-theta'
        if geometry in {'identity', 'identity-repeat'}
        else 'promoted-learned'
    )
    if ssp.get('geometry_mode') != expected_mode:
        raise SSPV2Failure(
            'SSP_V2_INPUT_HASH_MISMATCH', 'CLEAR geometry mode mismatch'
        )
    return result
