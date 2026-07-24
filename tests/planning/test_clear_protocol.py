import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.plan.clear_protocol import (
    cube_symmetry_angle_deg,
    install_success_criterion,
    load_manifest,
    resolve_manifest_pairs,
    validate_solver_config,
)


def _manifest(task='pusht', protocol='moderate'):
    hold = 3 if protocol == 'moderate' else 5
    return {
        'schema_version': 'clear-lewm-manifest-v1',
        'task': task,
        'dataset': {'episode_column': 'episode_idx'},
        'protocol': {
            'name': protocol,
            'goal_offset': 25,
            'eval_budget': 50,
            'sustained_steps': hold,
            'pusht_block_only': True,
            'pusht_position_threshold': 20.0 if hold == 3 else 15.0,
            'pusht_angle_threshold_deg': 20.0 if hold == 3 else 15.0,
            'cube_symmetry_aware': True,
            'cube_position_threshold_m': 0.04 if hold == 3 else 0.03,
            'cube_orientation_threshold_deg': 30.0 if hold == 3 else 15.0,
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


def _world(env):
    wrapped = SimpleNamespace(unwrapped=env)
    return SimpleNamespace(envs=SimpleNamespace(envs=[wrapped]))


def test_pusht_scores_block_pose_and_requires_hold():
    env = _FakePushT()
    env.goal_state = np.array([100.0, 100.0, 10.0, 20.0, 0.0, 0.0, 0.0])
    env.state = np.array([0.0, 0.0, 10.0, 20.0, 0.0, 0.0, 0.0])
    install_success_criterion(_world(env), _manifest())
    outcomes = [env.step(np.zeros(2))[2] for _ in range(3)]
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


def test_clear_solver_contract_rejects_short_cube_budget():
    solver = {
        'batch_size': 1,
        'num_samples': 300,
        'n_steps': 10,
        'topk': 30,
    }
    with pytest.raises(ValueError, match='n_steps=10'):
        validate_solver_config(solver)
