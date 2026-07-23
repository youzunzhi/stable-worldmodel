"""CLEAR-LeWM v0.3 manifest and task-success adapter.

The protocol definitions are derived from DavidSunok/CLEAR-LeWM at revision
f06b66b358f5e42aa582e4a5599d3356c29edcf4.  CLEAR-LeWM is MIT licensed.
This module keeps stable-worldmodel's planner and rollout loop, while applying
the fixed start-goal manifest and external task-completion predicates.
"""

from __future__ import annotations

import json
from pathlib import Path


CLEAR_LEWM_REVISION = 'f06b66b358f5e42aa582e4a5599d3356c29edcf4'
CLEAR_MANIFEST_SCHEMA = 'clear-lewm-manifest-v1'
CLEAR_TASKS = {'pusht', 'cube'}
CLEAR_PROTOCOLS = {'moderate', 'strict'}
CLEAR_SOLVER = {
    'batch_size': 1,
    'num_samples': 300,
    'n_steps': 30,
    'topk': 30,
}
CLEAR_CPU_THREADS = 1


def load_manifest(path: str | Path) -> dict:
    """Load and validate the supported subset of a CLEAR-LeWM manifest."""
    manifest_path = Path(path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema_version') != CLEAR_MANIFEST_SCHEMA:
        raise ValueError(
            'Unsupported CLEAR manifest schema: '
            f"{manifest.get('schema_version')!r}"
        )
    if manifest.get('task') not in CLEAR_TASKS:
        raise ValueError(
            f"CLEAR task must be one of {sorted(CLEAR_TASKS)}, "
            f"got {manifest.get('task')!r}"
        )
    protocol = manifest.get('protocol', {})
    if protocol.get('name') not in CLEAR_PROTOCOLS:
        raise ValueError(
            f"CLEAR protocol must be one of {sorted(CLEAR_PROTOCOLS)}, "
            f"got {protocol.get('name')!r}"
        )
    pairs = manifest.get('pairs')
    if not isinstance(pairs, list) or not pairs:
        raise ValueError('CLEAR manifest must contain at least one pair')
    if len({pair['pair_id'] for pair in pairs}) != len(pairs):
        raise ValueError('CLEAR manifest pair_id values must be unique')
    if any(pair.get('initial_success') for pair in pairs):
        raise ValueError('CLEAR robust manifests cannot contain pre-solved pairs')
    return manifest
