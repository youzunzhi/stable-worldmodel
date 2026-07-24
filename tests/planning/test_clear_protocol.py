import json

import pytest

from scripts.plan.clear_protocol import (
    load_manifest,
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
