"""CLEAR-LeWM v0.5 manifest and task-success adapter.

The protocol definitions are derived from DavidSunok/CLEAR-LeWM at revision
df026185a36bd9997c69d94753854db0b1a46f54.  CLEAR-LeWM is MIT licensed.
This module keeps stable-worldmodel's planner and rollout loop, while applying
the fixed start-goal manifest and external task-completion predicates.
"""

from __future__ import annotations

import hashlib
import json
import random
from itertools import permutations, product
from pathlib import Path
from types import MethodType

import numpy as np

try:
    from .clear_tworoom import (
        install_tworoom_success,
        topology_audit_records as _topology_audit_records,
    )
except ImportError:  # Direct execution via scripts/plan/eval_wm.py.
    from clear_tworoom import (
        install_tworoom_success,
        topology_audit_records as _topology_audit_records,
    )


CLEAR_LEWM_VERSION = '0.5.0'
CLEAR_LEWM_REVISION = 'df026185a36bd9997c69d94753854db0b1a46f54'
CLEAR_MANIFEST_SCHEMA = 'clear-lewm-manifest-v1'
CLEAR_TASKS = {'pusht', 'cube', 'tworoom'}
CLEAR_PROTOCOLS = {'moderate', 'strict'}
CLEAR_SOLVER = {
    'batch_size': 1,
    'num_samples': 300,
    'n_steps': 30,
    'topk': 30,
}
CLEAR_CPU_THREADS = 1


def topology_audit_records() -> list[dict]:
    """Expose any TwoRoom route diagnostics collected by the adapter."""
    return _topology_audit_records()


_V05_SHARED_PROTOCOL = {
    'goal_offset': 25,
    'eval_budget': 50,
    'sampling': 'episode-balanced',
    'split': 'all',
    'heldout_fraction': 0,
    'exclude_initial_success': True,
    'success_mode': 'task-sustained',
}
_V05_TASK_PROTOCOLS = {
    ('moderate', 'pusht'): {
        'pusht_block_only': False,
        'pusht_position_threshold': 20,
        'pusht_angle_threshold_deg': 20,
        'pusht_sustained_steps': None,
        'sustained_steps': 1,
    },
    ('strict', 'pusht'): {
        'pusht_block_only': True,
        'pusht_position_threshold': 10,
        'pusht_angle_threshold_deg': 10,
        'pusht_sustained_steps': 3,
        'sustained_steps': 1,
    },
    ('moderate', 'cube'): {
        'cube_position_threshold_m': 0.04,
        'cube_orientation_threshold_deg': None,
        'cube_symmetry_aware': False,
        'cube_sustained_steps': None,
        'sustained_steps': 1,
    },
    ('strict', 'cube'): {
        'cube_position_threshold_m': 0.03,
        'cube_orientation_threshold_deg': 15,
        'cube_symmetry_aware': True,
        'cube_sustained_steps': 3,
        'sustained_steps': 1,
    },
    ('moderate', 'tworoom'): {
        'tworoom_collision_mode': 'swept',
        'tworoom_crossroom_only': True,
        'tworoom_distance_threshold': 16,
        'tworoom_route_required': False,
        'tworoom_source_window_clean': True,
        'tworoom_sustained_steps': None,
        'sustained_steps': 1,
    },
    ('strict', 'tworoom'): {
        'tworoom_collision_mode': 'swept',
        'tworoom_crossroom_only': True,
        'tworoom_distance_threshold': 8,
        'tworoom_goal_side_required': True,
        'tworoom_route_required': True,
        'tworoom_source_window_clean': True,
        'tworoom_sustained_steps': None,
        'sustained_steps': 1,
    },
}


def _validate_v05_protocol(manifest: dict) -> None:
    """Reject older same-schema manifests before they can be mislabeled."""
    protocol = manifest['protocol']
    task = manifest['task']
    expected = {
        **_V05_SHARED_PROTOCOL,
        **_V05_TASK_PROTOCOLS[(protocol['name'], task)],
    }
    mismatches = {
        name: (protocol.get(name), value)
        for name, value in expected.items()
        if protocol.get(name) != value
    }
    if mismatches:
        details = ', '.join(
            f'{name}={actual!r} (expected {expected!r})'
            for name, (actual, expected) in mismatches.items()
        )
        raise ValueError(
            f'CLEAR-LeWM v{CLEAR_LEWM_VERSION} protocol mismatch: {details}'
        )


def _hold_steps(protocol: dict, task: str) -> int:
    task_steps = protocol.get(f'{task}_sustained_steps')
    return int(
        task_steps
        if task_steps is not None
        else protocol.get('sustained_steps', 1)
    )


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
    _validate_v05_protocol(manifest)
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
    mismatches = {}
    for name, expected in CLEAR_SOLVER.items():
        actual = solver.get(name)
        if actual is None:
            mismatches[name] = (None, expected)
            continue
        actual = int(actual)
        if actual != expected:
            mismatches[name] = (actual, expected)
    if mismatches:
        details = ', '.join(
            f'{name}={actual if actual is not None else "missing"} '
            f'(expected {expected})'
            for name, (actual, expected) in mismatches.items()
        )
        raise ValueError(f'CLEAR-LeWM solver contract mismatch: {details}')


def validate_policy_seed(manifest: dict, seed: int) -> None:
    """Require the policy RNG seed embedded in the selected manifest."""
    expected = int(manifest['policy_seed'])
    if int(seed) != expected:
        raise ValueError(
            'CLEAR-LeWM policy seed mismatch: '
            f'{int(seed)} != manifest seed {expected}'
        )


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


def cube_symmetry_angle_deg(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """Geodesic orientation error modulo the cube's 24 proper rotations."""
    current = _quaternion_matrix_wxyz(q0)
    target = _quaternion_matrix_wxyz(q1)
    relative = np.swapaxes(current, -1, -2) @ target
    equivalent = relative[..., None, :, :] @ CUBE_SYMMETRY_MATRICES
    traces = np.trace(equivalent, axis1=-2, axis2=-1)
    angles = np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0))
    return np.degrees(np.min(angles, axis=-1))


def _install_pusht_success(world, protocol: dict) -> None:
    def patch_environment(env) -> None:
        original_step = env.step
        original_set_goal = env._set_goal_state
        env._clear_lewm_hold_count = 0

        def set_goal_state(self, goal_state):
            result = original_set_goal(goal_state)
            self._clear_lewm_hold_count = 0
            return result

        def step(self, action):
            observation, reward, _, truncated, info = original_step(action)
            state = np.asarray(observation['state'])
            goal = np.asarray(self.goal_state)
            position_slice = (
                slice(2, 4)
                if protocol['pusht_block_only']
                else slice(0, 4)
            )
            position_error = float(
                np.linalg.norm(goal[position_slice] - state[position_slice])
            )
            angle_error = abs(float(goal[4] - state[4]))
            angle_error = min(angle_error, 2.0 * np.pi - angle_error)
            success = (
                position_error < protocol['pusht_position_threshold']
                and np.degrees(angle_error)
                < protocol['pusht_angle_threshold_deg']
            )
            self._clear_lewm_hold_count = (
                self._clear_lewm_hold_count + 1 if success else 0
            )
            terminated = (
                self._clear_lewm_hold_count
                >= _hold_steps(protocol, 'pusht')
            )
            info['clear_lewm_hold_count'] = self._clear_lewm_hold_count
            return observation, reward, terminated, truncated, info

        env._set_goal_state = MethodType(set_goal_state, env)
        env.step = MethodType(step, env)

    for wrapped in world.envs.envs:
        patch_environment(wrapped.unwrapped)


def _install_cube_success(world, protocol: dict) -> None:
    def patch_environment(env) -> None:
        original_post_step = env.post_step
        original_set_target = env.set_target_pos
        env._clear_lewm_hold_count = 0

        def set_target_pos(self, cube_id, target_pos, target_quat=None):
            result = original_set_target(cube_id, target_pos, target_quat)
            self._clear_lewm_hold_count = 0
            self._success = False
            return result

        def post_step(self):
            original_post_step()
            qpos = np.asarray(self._data.joint('object_joint_0').qpos)
            target_id = self._cube_target_mocap_ids[0]
            target_pos = np.asarray(self._data.mocap_pos[target_id])
            position_ok = (
                np.linalg.norm(qpos[:3] - target_pos)
                <= protocol['cube_position_threshold_m']
            )
            pose_ok = bool(position_ok)
            orientation_threshold = protocol[
                'cube_orientation_threshold_deg'
            ]
            if orientation_threshold is not None:
                if not protocol['cube_symmetry_aware']:
                    raise ValueError(
                        'This adapter only supports symmetry-aware Cube '
                        'orientation scoring'
                    )
                target_quat = np.asarray(self._data.mocap_quat[target_id])
                angle_deg = float(
                    cube_symmetry_angle_deg(
                        qpos[None, 3:7], target_quat[None]
                    )[0]
                )
                pose_ok = bool(
                    pose_ok and angle_deg <= orientation_threshold
                )
            self._clear_lewm_hold_count = (
                self._clear_lewm_hold_count + 1 if pose_ok else 0
            )
            self._success = (
                self._clear_lewm_hold_count >= _hold_steps(protocol, 'cube')
            )

        env.set_target_pos = MethodType(set_target_pos, env)
        env.post_step = MethodType(post_step, env)

    for wrapped in world.envs.envs:
        patch_environment(wrapped.unwrapped)


def install_success_criterion(world, manifest: dict) -> None:
    """Install the selected CLEAR v0.5 success rule on every raw env."""
    protocol = manifest['protocol']
    if manifest['task'] == 'pusht':
        _install_pusht_success(world, protocol)
    elif manifest['task'] == 'cube':
        _install_cube_success(world, protocol)
    elif manifest['task'] == 'tworoom':
        install_tworoom_success(world, protocol)
    else:
        raise ValueError(f"Unsupported CLEAR task: {manifest['task']}")
