"""Held-out SSP solve-vs-budget profile with paired uncertainty."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from .contracts import (
    LOCKED,
    PROFILE_SEEDS,
    PROTOCOL_ID,
    SSPFailure,
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
            for budget in range(1, 31)
        ],
        dtype=np.float64,
    )


def _b50(curve: np.ndarray) -> int | None:
    indices = np.flatnonzero(curve >= 0.5)
    return int(indices[0] + 1) if len(indices) else None


def _bootstrap(identity_hits: np.ndarray, learned_hits: np.ndarray) -> dict:
    """Resample pair rows, preserving all five planner seeds per pair."""
    pair_count = identity_hits.shape[0]
    rng = np.random.Generator(np.random.PCG64DXSM(LOCKED['analysis_seed']))
    observed_identity = _solve_curve(identity_hits.reshape(-1))
    observed_learned = _solve_curve(learned_hits.reshape(-1))
    observed_delta = observed_learned - observed_identity
    auc_deltas = np.empty(LOCKED['bootstrap_replicates'])
    endpoint_deltas = np.empty(LOCKED['bootstrap_replicates'])
    curve_deltas = np.empty((LOCKED['bootstrap_replicates'], 30))
    for index in range(LOCKED['bootstrap_replicates']):
        selected = rng.integers(0, pair_count, size=pair_count)
        identity = identity_hits[selected].reshape(-1)
        learned = learned_hits[selected].reshape(-1)
        delta = _solve_curve(learned) - _solve_curve(identity)
        curve_deltas[index] = delta
        auc_deltas[index] = delta.mean()
        endpoint_deltas[index] = delta[-1]
    maximum_deviation = np.abs(curve_deltas - observed_delta[None, :]).max(
        axis=1
    )
    critical = float(np.quantile(maximum_deviation, 0.95))
    lower = observed_delta - critical
    upper = observed_delta + critical
    return {
        'analysis_seed': LOCKED['analysis_seed'],
        'bootstrap_replicates': LOCKED['bootstrap_replicates'],
        'resampling_unit': 'start-goal pair with five planner seeds',
        'auc_delta_ci95': np.quantile(auc_deltas, [0.025, 0.975]).tolist(),
        'endpoint_delta_ci95': np.quantile(
            endpoint_deltas, [0.025, 0.975]
        ).tolist(),
        'simultaneous_delta_band95': {
            'critical_max_abs_centered_deviation': critical,
            'lower': lower.tolist(),
            'upper': upper.tolist(),
        },
        'broad_plannability_improvement': bool(
            np.all(lower >= 0) and np.any(lower > 0)
        ),
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
        raise SSPFailure('SSP_INCOMPLETE', 'training is not complete')
    profile = root / 'profile'
    if not profile.is_dir() or any(profile.iterdir()):
        raise SSPFailure(
            'SSP_FORMAL_ROOT_EXISTS',
            f'profile target is absent or non-empty: {profile}',
        )
    checkpoint = torch.load(
        selected_path, map_location='cpu', weights_only=True
    )
    if checkpoint.get('protocol_id') != PROTOCOL_ID:
        raise SSPFailure(
            'SSP_INPUT_HASH_MISMATCH', 'selected geometry protocol mismatch'
        )
    learned_psi = checkpoint['center'].float()
    identity_psi = torch.zeros_like(learned_psi)
    basis = torch.from_numpy(np.load(root / 'geometry_basis.npy'))
    latents = _load_latents(preparation['root'] / 'pair_latents.pt')
    lookup = _pair_lookup(latents, 'test')
    pair_ids = list(latents['test']['pair_ids'])
    if len(pair_ids) != 512:
        raise SSPFailure(
            'SSP_INPUT_HASH_MISMATCH', 'test manifest is not 512 pairs'
        )
    model = _load_model(Path(config['checkpoint']['path']), device)
    before = model_state_sha256(model)
    planner = _planner(
        config=config,
        model=model,
        basis=basis,
        device=device,
        num_samples=LOCKED['num_samples'],
        n_steps=LOCKED['n_steps'],
    )
    identity_hits = np.full(
        (len(pair_ids), len(PROFILE_SEEDS)), -1, dtype=np.int16
    )
    learned_hits = np.full_like(identity_hits, -1)
    noise_path = profile / 'noise_schedule.jsonl'
    rows_path = profile / 'per_pair_seed.jsonl'
    for pair_index, pair_id in enumerate(pair_ids):
        for seed_index, planner_seed in enumerate(PROFILE_SEEDS):
            key = {
                'protocol_id': PROTOCOL_ID,
                'task': config['task'],
                'phase': 'held-out-profile',
                'planner_seed': planner_seed,
                'pair_slot': pair_index,
                'pair_id': pair_id,
            }
            start, goal = lookup[pair_id]
            identity = planner.run(
                pair_id=pair_id,
                start_embedding=start,
                goal_embedding=goal,
                psi=identity_psi,
                noise_key=key,
            )
            learned = planner.run(
                pair_id=pair_id,
                start_embedding=start,
                goal_embedding=goal,
                psi=learned_psi,
                noise_key=key,
            )
            _validate_crn([identity], [learned])
            append_jsonl(
                noise_path,
                {'geometry': 'identity', **identity.noise},
            )
            append_jsonl(
                noise_path,
                {'geometry': 'learned', **learned.noise},
            )
            identity_value = identity.first_hit_iteration or -1
            learned_value = learned.first_hit_iteration or -1
            identity_hits[pair_index, seed_index] = identity_value
            learned_hits[pair_index, seed_index] = learned_value
            append_jsonl(
                rows_path,
                {
                    'pair_id': pair_id,
                    'planner_seed': planner_seed,
                    'identity_T': identity.first_hit_iteration,
                    'learned_T': learned.first_hit_iteration,
                    'identity_auc': identity.auc_reward,
                    'learned_auc': learned.auc_reward,
                    'auc_delta': learned.auc_reward - identity.auc_reward,
                },
            )
    identity_curve = _solve_curve(identity_hits.reshape(-1))
    learned_curve = _solve_curve(learned_hits.reshape(-1))
    delta_curve = learned_curve - identity_curve
    with (profile / 'solve_curve.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=('budget', 'identity', 'learned', 'delta'),
        )
        writer.writeheader()
        for budget in range(1, 31):
            writer.writerow(
                {
                    'budget': budget,
                    'identity': identity_curve[budget - 1],
                    'learned': learned_curve[budget - 1],
                    'delta': delta_curve[budget - 1],
                }
            )
    uncertainty = _bootstrap(identity_hits, learned_hits)
    after = model_state_sha256(model)
    if before != after:
        raise SSPFailure(
            'SSP_FROZEN_MODEL_MUTATION', 'model changed during profile'
        )
    summary = {
        'protocol_id': PROTOCOL_ID,
        'task': config['task'],
        'replicate_seed': checkpoint['replicate_seed'],
        'selected_step': checkpoint['step'],
        'selected_geometry_sha256': sha256_file(selected_path),
        'pairs': len(pair_ids),
        'planner_seeds': list(PROFILE_SEEDS),
        'identity': {
            'solve_curve': identity_curve.tolist(),
            'auc': float(identity_curve.mean()),
            'endpoint_solve_rate': float(identity_curve[-1]),
            'B_50': _b50(identity_curve),
        },
        'learned': {
            'solve_curve': learned_curve.tolist(),
            'auc': float(learned_curve.mean()),
            'endpoint_solve_rate': float(learned_curve[-1]),
            'B_50': _b50(learned_curve),
        },
        'paired_delta': {
            'solve_curve': delta_curve.tolist(),
            'auc': float(delta_curve.mean()),
            'endpoint_solve_rate': float(delta_curve[-1]),
        },
        'first_hit_histogram': {
            'identity': np.bincount(
                np.where(identity_hits < 0, 31, identity_hits).reshape(-1),
                minlength=32,
            )[1:].tolist(),
            'learned': np.bincount(
                np.where(learned_hits < 0, 31, learned_hits).reshape(-1),
                minlength=32,
            )[1:].tolist(),
            'bin_31_is_right_censored_failure': True,
        },
        'uncertainty': uncertainty,
        'model_state_sha256_before': before,
        'model_state_sha256_after': after,
    }
    write_json(profile / 'summary.json', summary)
    write_json(profile / 'audit.json', {'sha256': hash_inventory(profile)})
    return summary
