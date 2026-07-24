"""CLEAR-LeWM v0.3 manifest and task-success adapter.

The protocol definitions are derived from DavidSunok/CLEAR-LeWM at revision
f06b66b358f5e42aa582e4a5599d3356c29edcf4.  CLEAR-LeWM is MIT licensed.
This module keeps stable-worldmodel's planner and rollout loop, while applying
the fixed start-goal manifest and external task-completion predicates.
"""

from __future__ import annotations

import hashlib
import json
import random
from itertools import permutations, product
from pathlib import Path

import numpy as np


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


def manifest_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).expanduser().resolve().read_bytes()).hexdigest()


def metadata_fingerprint(path: str | Path) -> str:
    """Match CLEAR-LeWM's metadata-only HDF5 fingerprint."""
    import h5py

    dataset_path = Path(path)
    with h5py.File(dataset_path, 'r') as dataset:
        schema = []
        for key in sorted(dataset.keys()):
            value = dataset[key]
            if isinstance(value, h5py.Dataset):
                schema.append(
                    {
                        'key': key,
                        'shape': list(value.shape),
                        'dtype': str(value.dtype),
                    }
                )
    payload = {'size_bytes': dataset_path.stat().st_size, 'schema': schema}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(',', ':')
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_dataset(dataset, manifest: dict) -> Path:
    """Verify the HDF5 identity recorded by the manifest."""
    dataset_path = Path(dataset.h5_path).resolve()
    expected = manifest['dataset']['fingerprint']
    if expected['kind'] != 'metadata-sha256':
        raise ValueError(
            f"Unsupported CLEAR dataset fingerprint kind: {expected['kind']}"
        )
    actual = metadata_fingerprint(dataset_path)
    if actual != expected['value']:
        raise ValueError(
            'Evaluation dataset does not match the CLEAR manifest: '
            f"{actual} != {expected['value']}"
        )
    return dataset_path


def resolve_manifest_pairs(
    dataset, manifest: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve fixed rows and verify their episode/step identity."""
    rows = np.asarray(
        [pair['start_row'] for pair in manifest['pairs']], dtype=np.int64
    )
    episode_key = manifest['dataset']['episode_column']
    episodes = np.asarray(dataset.get_col_data(episode_key))[rows]
    steps = np.asarray(dataset.get_col_data('step_idx'))[rows]
    expected_episodes = np.asarray(
        [pair['episode_id'] for pair in manifest['pairs']]
    )
    expected_steps = np.asarray(
        [pair['start_step'] for pair in manifest['pairs']]
    )
    if not np.array_equal(episodes, expected_episodes):
        raise ValueError('CLEAR manifest episode IDs do not match dataset rows')
    if not np.array_equal(steps, expected_steps):
        raise ValueError('CLEAR manifest start steps do not match dataset rows')
    return rows, episodes, steps


def validate_solver_config(solver) -> None:
    """Reject CLEAR-labelled runs with a different CEM budget."""
    mismatches = {
        name: (int(solver[name]), expected)
        for name, expected in CLEAR_SOLVER.items()
        if int(solver[name]) != expected
    }
    if mismatches:
        details = ', '.join(
            f'{name}={actual} (expected {expected})'
            for name, (actual, expected) in mismatches.items()
        )
        raise ValueError(f'CLEAR-LeWM solver contract mismatch: {details}')


def seed_runtime(seed: int) -> None:
    """Apply CLEAR's deterministic Python, NumPy, Torch, and CUDA seeds."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(CLEAR_CPU_THREADS)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cube_symmetry_matrices() -> np.ndarray:
    matrices = []
    identity = np.eye(3, dtype=np.float64)
    for permutation in permutations(range(3)):
        base = identity[:, permutation]
        for signs in product((-1.0, 1.0), repeat=3):
            matrix = base * np.asarray(signs, dtype=np.float64)[None, :]
            if np.linalg.det(matrix) > 0.5:
                matrices.append(matrix)
    result = np.asarray(matrices, dtype=np.float64)
    if result.shape != (24, 3, 3):
        raise RuntimeError(f'Expected 24 cube symmetries, got {result.shape}')
    return result


CUBE_SYMMETRY_MATRICES = _cube_symmetry_matrices()


def _quaternion_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion = quaternion / np.clip(
        np.linalg.norm(quaternion, axis=-1, keepdims=True), 1e-12, None
    )
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape((*quaternion.shape[:-1], 3, 3))
