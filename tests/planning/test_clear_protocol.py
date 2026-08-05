import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.plan.clear_protocol import (
    CLEAR_LEWM_REVISION,
    CLEAR_LEWM_VERSION,
    cube_symmetry_angle_deg,
    install_success_criterion,
    load_manifest,
    resolve_manifest_pairs,
    validate_policy_seed,
    validate_solver_config,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _manifest(task='pusht', protocol='moderate'):
    criteria = {
        ('moderate', 'pusht'): {
            'pusht_block_only': False,
            'pusht_position_threshold': 20,
            'pusht_angle_threshold_deg': 20,
            'pusht_sustained_steps': None,
        },
        ('strict', 'pusht'): {
            'pusht_block_only': True,
            'pusht_position_threshold': 10,
            'pusht_angle_threshold_deg': 10,
            'pusht_sustained_steps': 3,
        },
        ('moderate', 'cube'): {
            'cube_symmetry_aware': False,
            'cube_position_threshold_m': 0.04,
            'cube_orientation_threshold_deg': None,
            'cube_sustained_steps': None,
        },
        ('strict', 'cube'): {
            'cube_symmetry_aware': True,
            'cube_position_threshold_m': 0.03,
            'cube_orientation_threshold_deg': 15,
            'cube_sustained_steps': 3,
        },
    }
    return {
        'schema_version': 'clear-lewm-manifest-v1',
        'task': task,
        'seed': 42,
        'policy_seed': 42,
        'dataset': {'episode_column': 'episode_idx'},
        'protocol': {
            'name': protocol,
            'goal_offset': 25,
            'eval_budget': 50,
            'sampling': 'episode-balanced',
            'split': 'all',
            'heldout_fraction': 0,
            'exclude_initial_success': True,
            'success_mode': 'task-sustained',
            'sustained_steps': 1,
            **criteria[(protocol, task)],
        },
        'pairs': [
            {
                'pair_id': 0,
                'episode_id': 7,
                'start_step': 4,
                'start_row': 1,
                'initial_success': False,
            }
        ],
    }


class _FakePushT:
    def __init__(self):
        self.goal_state = np.zeros(7)
        self.state = np.zeros(7)

    def _set_goal_state(self, goal_state):
        self.goal_state = np.asarray(goal_state)

    def step(self, action):
        return {'state': self.state.copy()}, 0.0, True, False, {}


class _FakeJoint:
    def __init__(self, qpos):
        self.qpos = qpos


class _FakeData:
    def __init__(self):
        self.object_qpos = np.array(
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        )
        self.mocap_pos = np.zeros((1, 3))
        self.mocap_quat = np.array([[1.0, 0.0, 0.0, 0.0]])

    def joint(self, name):
        assert name == 'object_joint_0'
        return _FakeJoint(self.object_qpos)


class _FakeCube:
    def __init__(self):
        self._data = _FakeData()
        self._cube_target_mocap_ids = [0]
        self._success = False

    def set_target_pos(self, cube_id, target_pos, target_quat=None):
        self._data.mocap_pos[0] = target_pos
        if target_quat is not None:
            self._data.mocap_quat[0] = target_quat

    def post_step(self):
        return None


def _world(env):
    wrapped = SimpleNamespace(unwrapped=env)
    return SimpleNamespace(envs=SimpleNamespace(envs=[wrapped]))


def test_pusht_moderate_scores_pusher_and_block_on_first_hit():
    env = _FakePushT()
    env.goal_state = np.array([100.0, 100.0, 10.0, 20.0, 0.0, 0.0, 0.0])
    env.state = np.array([0.0, 0.0, 10.0, 20.0, 0.0, 0.0, 0.0])
    install_success_criterion(_world(env), _manifest())
    assert not env.step(np.zeros(2))[2]

    env.state = env.goal_state.copy()
    assert env.step(np.zeros(2))[2]


def test_pusht_strict_scores_block_only_and_requires_three_steps():
    env = _FakePushT()
    env.goal_state = np.array([100.0, 100.0, 10.0, 20.0, 0.0, 0.0, 0.0])
    env.state = np.array([0.0, 0.0, 10.0, 20.0, 0.0, 0.0, 0.0])
    install_success_criterion(_world(env), _manifest(protocol='strict'))
    outcomes = [env.step(np.zeros(2))[2] for _ in range(3)]
    assert outcomes == [False, False, True]


def test_cube_moderate_ignores_orientation_and_succeeds_on_first_hit():
    env = _FakeCube()
    env._data.object_qpos[3:7] = np.array(
        [np.cos(np.pi / 8), 0.0, 0.0, np.sin(np.pi / 8)]
    )
    install_success_criterion(_world(env), _manifest(task='cube'))
    env.post_step()
    assert env._success


def test_cube_strict_scores_symmetry_aware_orientation_and_holds_three_steps():
    env = _FakeCube()
    env._data.object_qpos[3:7] = np.array(
        [np.cos(np.pi / 8), 0.0, 0.0, np.sin(np.pi / 8)]
    )
    install_success_criterion(
        _world(env), _manifest(task='cube', protocol='strict')
    )
    env.post_step()
    assert not env._success

    env._data.object_qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    outcomes = []
    for _ in range(3):
        env.post_step()
        outcomes.append(env._success)
    assert outcomes == [False, False, True]


def test_cube_symmetry_accepts_equivalent_quarter_turn():
    identity = np.array([[1.0, 0.0, 0.0, 0.0]])
    quarter_turn_z = np.array(
        [[np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]]
    )
    assert cube_symmetry_angle_deg(identity, quarter_turn_z)[0] < 1e-6


def test_manifest_rows_must_match_episode_and_step_identity():
    dataset = SimpleNamespace(
        get_col_data=lambda key: {
            'episode_idx': np.array([3, 7]),
            'step_idx': np.array([1, 4]),
        }[key]
    )
    rows, episodes, steps = resolve_manifest_pairs(dataset, _manifest())
    assert rows.tolist() == [1]
    assert episodes.tolist() == [7]
    assert steps.tolist() == [4]


def test_manifest_rejects_pre_solved_pairs(tmp_path):
    manifest = _manifest()
    manifest['pairs'][0]['initial_success'] = True
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='pre-solved'):
        load_manifest(path)


def test_manifest_rejects_pre_v05_protocol_with_same_schema(tmp_path):
    manifest = _manifest()
    manifest['protocol']['heldout_fraction'] = 0.2
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='v0.5.0 protocol mismatch'):
        load_manifest(path)


def test_adapter_revision_matches_clear_eval_registry():
    registry = json.loads(
        (REPO_ROOT / 'scripts/experiments/clear_eval.json').read_text()
    )
    assert registry['clear_lewm']['version'] == CLEAR_LEWM_VERSION
    assert registry['clear_lewm']['revision'] == CLEAR_LEWM_REVISION


def test_policy_seed_must_match_manifest():
    validate_policy_seed(_manifest(), 42)
    with pytest.raises(ValueError, match='41 != manifest seed 42'):
        validate_policy_seed(_manifest(), 41)


def test_clear_solver_contract_rejects_short_cube_budget():
    solver = {
        'batch_size': 1,
        'num_samples': 300,
        'n_steps': 10,
        'topk': 30,
    }
    with pytest.raises(ValueError, match='n_steps=10'):
        validate_solver_config(solver)


def test_clear_solver_contract_reports_missing_cem_fields():
    solver = {
        'batch_size': 1,
        'num_samples': 100,
        'n_steps': 30,
    }
    with pytest.raises(ValueError, match='topk=missing'):
        validate_solver_config(solver)
