"""One-shot held-out SSP-v2 returned-hit profile with clustered uncertainty."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from .contracts import (
    LOCKED,
    PROFILE_SEEDS,
    PROTOCOL_ID,
    SSPV2Failure,
    append_jsonl,
    hash_inventory,
    sha256_file,
    write_json,
)
from .es import (
    _load_latents,
    _load_preparation,
    _pair_lookup,
    _planner,
    _validate_crn,
)
from .pairs import _load_model
from .planner import model_state_sha256


def _solve_curve(first_hits: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            np.mean((first_hits > 0) & (first_hits <= budget))
            for budget in range(1, LOCKED['n_steps'] + 1)
        ],
        dtype=np.float64,
    )


def _group_ids(manifest: Path, pair_ids: list[str]) -> np.ndarray:
    records = {}
    with manifest.open() as stream:
        for line in stream:
            row = json.loads(line)
            records[row['pair_id']] = int(row['group_id'])
    return np.asarray(
        [records[pair_id] for pair_id in pair_ids], dtype=np.int64
    )


def _bootstrap(
    *,
    group_ids: np.ndarray,
    identity_returned: np.ndarray,
    learned_returned: np.ndarray,
    identity_hits: np.ndarray,
    learned_hits: np.ndarray,
) -> dict:
    """Resample leakage groups, preserving their pairs and planner tapes."""
    groups = sorted(set(group_ids.tolist()))
    members = {group: np.flatnonzero(group_ids == group) for group in groups}
    rng = np.random.Generator(np.random.PCG64DXSM(LOCKED['analysis_seed']))
    observed_curve = _solve_curve(learned_hits.reshape(-1)) - _solve_curve(
        identity_hits.reshape(-1)
    )
    endpoint = np.empty(LOCKED['bootstrap_replicates'])
    auc = np.empty_like(endpoint)
    curve = np.empty((LOCKED['bootstrap_replicates'], LOCKED['n_steps']))
    for index in range(LOCKED['bootstrap_replicates']):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        selected = np.concatenate(
            [members[int(group)] for group in sampled_groups]
        )
        endpoint[index] = (
            learned_returned[selected].mean()
            - identity_returned[selected].mean()
        )
        delta_curve = _solve_curve(
            learned_hits[selected].reshape(-1)
        ) - _solve_curve(identity_hits[selected].reshape(-1))
        curve[index] = delta_curve
        auc[index] = delta_curve[1:].mean()
    deviation = np.abs(curve[:, 1:] - observed_curve[None, 1:]).max(axis=1)
    critical = float(np.quantile(deviation, 0.95))
    return {
        'analysis_seed': LOCKED['analysis_seed'],
        'bootstrap_replicates': LOCKED['bootstrap_replicates'],
        'resampling_unit': 'leakage group with all pairs and planner tapes',
        'groups': len(groups),
        'returned_hit_delta_ci95': np.quantile(
            endpoint, [0.025, 0.975]
        ).tolist(),
        'auc_budgets_2_30_delta_ci95': np.quantile(
            auc, [0.025, 0.975]
        ).tolist(),
        'simultaneous_curve_delta_band95_budgets_2_30': {
            'critical_max_abs_centered_deviation': critical,
            'lower': (observed_curve[1:] - critical).tolist(),
            'upper': (observed_curve[1:] + critical).tolist(),
        },
        'budget_1_registered_structural_equality': True,
    }


def evaluate_profile(
    *,
    config: dict,
    preparation_dir: str | Path,
    replicate_dir: str | Path,
    device: str,
) -> dict:
    preparation = _load_preparation(preparation_dir, config)
    root = Path(replicate_dir).expanduser().resolve()
    completion_path = root / 'training.completed.json'
    selected_path = root / 'selected_geometry.pt'
    if not completion_path.is_file() or not selected_path.is_file():
        raise SSPV2Failure('SSP_V2_INCOMPLETE', 'training is not complete')
    profile = root / 'profile'
    if not profile.is_dir() or any(profile.iterdir()):
        raise SSPV2Failure(
            'SSP_V2_FORMAL_ROOT_EXISTS',
            f'profile target is absent or non-empty: {profile}',
        )
    checkpoint = torch.load(
        selected_path, map_location='cpu', weights_only=True
    )
    if checkpoint.get('protocol_id') != PROTOCOL_ID:
        raise SSPV2Failure(
            'SSP_V2_INPUT_HASH_MISMATCH', 'selected geometry protocol mismatch'
        )
    learned = checkpoint['center'].float()
    identity = torch.zeros_like(learned)
    latents = _load_latents(preparation['root'] / 'pair_latents.pt')
    lookup = _pair_lookup(latents, 'test')
    pair_ids = list(latents['test']['pair_ids'])
    if len(pair_ids) != LOCKED['test_pairs']:
        raise SSPV2Failure(
            'SSP_V2_INPUT_HASH_MISMATCH', 'test pair count is not locked'
        )
    groups = _group_ids(
        preparation['root'] / 'pair_manifests' / 'test.jsonl', pair_ids
    )
    model = _load_model(Path(config['checkpoint']['path']), device)
    before = model_state_sha256(model)
    planner = _planner(
        config=config,
        model=model,
        preparation_root=preparation['root'],
        device=device,
    )
    shape = (len(pair_ids), len(PROFILE_SEEDS))
    identity_returned = np.zeros(shape, dtype=bool)
    learned_returned = np.zeros(shape, dtype=bool)
    identity_hits = np.full(shape, -1, dtype=np.int16)
    learned_hits = np.full(shape, -1, dtype=np.int16)
    identity_late = np.zeros(shape, dtype=np.float32)
    learned_late = np.zeros(shape, dtype=np.float32)
    noise_path = profile / 'noise_schedule.jsonl'
    rows_path = profile / 'per_pair_seed.jsonl'
    for pair_index, pair_id in enumerate(pair_ids):
        start, goal = lookup[pair_id]
        for seed_index, planner_seed in enumerate(PROFILE_SEEDS):
            key = {
                'protocol_id': PROTOCOL_ID,
                'task': config['task'],
                'phase': 'held-out-profile',
                'planner_seed': planner_seed,
                'pair_slot': pair_index,
                'pair_id': pair_id,
            }
            identity_result = planner.run(
                pair_id=pair_id,
                start_embedding=start,
                goal_embedding=goal,
                theta=identity,
                noise_key=key,
            )
            learned_result = planner.run(
                pair_id=pair_id,
                start_embedding=start,
                goal_embedding=goal,
                theta=learned,
                noise_key=key,
            )
            _validate_crn([identity_result], [learned_result])
            append_jsonl(
                noise_path, {'geometry': 'identity', **identity_result.noise}
            )
            append_jsonl(
                noise_path, {'geometry': 'learned', **learned_result.noise}
            )
            identity_returned[pair_index, seed_index] = (
                identity_result.returned_verified_hit
            )
            learned_returned[pair_index, seed_index] = (
                learned_result.returned_verified_hit
            )
            identity_hits[pair_index, seed_index] = (
                identity_result.first_hit_iteration or -1
            )
            learned_hits[pair_index, seed_index] = (
                learned_result.first_hit_iteration or -1
            )
            identity_late[pair_index, seed_index] = np.mean(
                identity_result.population_hit_fractions[-5:]
            )
            learned_late[pair_index, seed_index] = np.mean(
                learned_result.population_hit_fractions[-5:]
            )
            append_jsonl(
                rows_path,
                {
                    'pair_id': pair_id,
                    'group_id': int(groups[pair_index]),
                    'planner_seed': planner_seed,
                    'identity_returned_hit': bool(
                        identity_result.returned_verified_hit
                    ),
                    'learned_returned_hit': bool(
                        learned_result.returned_verified_hit
                    ),
                    'identity_first_hit': identity_result.first_hit_iteration,
                    'learned_first_hit': learned_result.first_hit_iteration,
                    'identity_late_hit_mass': float(
                        identity_late[pair_index, seed_index]
                    ),
                    'learned_late_hit_mass': float(
                        learned_late[pair_index, seed_index]
                    ),
                },
            )
    identity_curve = _solve_curve(identity_hits.reshape(-1))
    learned_curve = _solve_curve(learned_hits.reshape(-1))
    delta_curve = learned_curve - identity_curve
    with (profile / 'solve_curve.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(
            stream, fieldnames=('budget', 'identity', 'learned', 'delta')
        )
        writer.writeheader()
        for budget in range(1, LOCKED['n_steps'] + 1):
            writer.writerow(
                {
                    'budget': budget,
                    'identity': identity_curve[budget - 1],
                    'learned': learned_curve[budget - 1],
                    'delta': delta_curve[budget - 1],
                }
            )
    uncertainty = _bootstrap(
        group_ids=groups,
        identity_returned=identity_returned,
        learned_returned=learned_returned,
        identity_hits=identity_hits,
        learned_hits=learned_hits,
    )
    returned_delta = float(learned_returned.mean() - identity_returned.mean())
    positive = uncertainty['returned_hit_delta_ci95'][0] > 0
    after = model_state_sha256(model)
    if before != after:
        raise SSPV2Failure(
            'SSP_V2_FROZEN_MODEL_MUTATION', 'model changed during profile'
        )
    summary = {
        'protocol_id': PROTOCOL_ID,
        'task': config['task'],
        'replicate_seed': checkpoint['replicate_seed'],
        'selected_step': checkpoint['step'],
        'selected_geometry_sha256': sha256_file(selected_path),
        'pairs': len(pair_ids),
        'leakage_groups': len(set(groups.tolist())),
        'planner_seeds': list(PROFILE_SEEDS),
        'primary': {
            'estimand': 'returned verified hit rate at K=30',
            'identity': float(identity_returned.mean()),
            'learned': float(learned_returned.mean()),
            'paired_delta': returned_delta,
            'positive_gate': bool(positive),
        },
        'secondary': {
            'identity_late_hit_mass': float(identity_late.mean()),
            'learned_late_hit_mass': float(learned_late.mean()),
            'late_hit_mass_delta': float(
                learned_late.mean() - identity_late.mean()
            ),
            'identity_auc_budgets_2_30': float(identity_curve[1:].mean()),
            'learned_auc_budgets_2_30': float(learned_curve[1:].mean()),
            'solve_curve_delta': delta_curve.tolist(),
        },
        'uncertainty': uncertainty,
        'model_state_sha256_before': before,
        'model_state_sha256_after': after,
    }
    write_json(profile / 'summary.json', summary)
    write_json(profile / 'audit.json', {'sha256': hash_inventory(profile)})
    return summary
