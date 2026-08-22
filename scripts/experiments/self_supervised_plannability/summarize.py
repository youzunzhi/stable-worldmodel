"""Combine three fixed SSP replicates without pooling tasks."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .contracts import REPLICATE_SEEDS, write_json


def summarize_task(
    *, task: str, replicate_dirs: list[str | Path], output: str | Path
) -> dict:
    if len(replicate_dirs) != 3:
        raise ValueError('exactly three replicate directories are required')
    profiles = []
    for value in replicate_dirs:
        path = Path(value).expanduser().resolve() / 'profile' / 'summary.json'
        profiles.append(json.loads(path.read_text()))
    seeds = [int(profile['replicate_seed']) for profile in profiles]
    if sorted(seeds) != sorted(REPLICATE_SEEDS):
        raise ValueError(f'replicate seeds are not locked: {seeds}')
    if any(profile['task'] != task for profile in profiles):
        raise ValueError('profile task mismatch')
    identity_curves = np.asarray(
        [profile['identity']['solve_curve'] for profile in profiles]
    )
    learned_curves = np.asarray(
        [profile['learned']['solve_curve'] for profile in profiles]
    )
    deltas = learned_curves - identity_curves
    summary = {
        'task': task,
        'replicate_seeds': seeds,
        'replicates': profiles,
        'fixed_replicate_mean': {
            'identity_solve_curve': identity_curves.mean(axis=0).tolist(),
            'learned_solve_curve': learned_curves.mean(axis=0).tolist(),
            'paired_delta_solve_curve': deltas.mean(axis=0).tolist(),
            'identity_auc': float(identity_curves.mean()),
            'learned_auc': float(learned_curves.mean()),
            'paired_auc_delta': float(deltas.mean()),
            'identity_endpoint': float(identity_curves[:, -1].mean()),
            'learned_endpoint': float(learned_curves[:, -1].mean()),
            'paired_endpoint_delta': float(deltas[:, -1].mean()),
        },
        'tasks_pooled': False,
    }
    write_json(output, summary)
    return summary


def _episode_successes(result: dict) -> np.ndarray:
    values = result['metrics']['episode_successes']
    array = np.asarray(values, dtype=bool)
    if array.shape != (100,):
        raise ValueError(f'CLEAR result has success shape {array.shape}')
    return array


def _mcnemar_exact(identity: np.ndarray, learned: np.ndarray) -> dict:
    identity_only = int(np.sum(identity & ~learned))
    learned_only = int(np.sum(~identity & learned))
    discordant = identity_only + learned_only
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(identity_only, learned_only)
        tail = sum(math.comb(discordant, k) for k in range(lower + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        'identity_only_success': identity_only,
        'learned_only_success': learned_only,
        'discordant_pairs': discordant,
        'two_sided_exact_p_value': p_value,
    }


def summarize_clear(
    *, task: str, formal_root: str | Path, output: str | Path
) -> dict:
    """Compare fresh identity with every selected SSP geometry on CLEAR."""
    root = Path(formal_root).expanduser().resolve() / task
    rng = np.random.Generator(np.random.PCG64DXSM(20260822))
    comparisons = []
    for protocol in ('moderate', 'strict'):
        identity_result = json.loads(
            (
                root / 'identity' / 'clear' / protocol / 'results.txt.json'
            ).read_text()
        )
        identity = _episode_successes(identity_result)
        identity_rows = identity_result['sampled_flat_indices']
        for seed in REPLICATE_SEEDS:
            learned_result = json.loads(
                (
                    root / str(seed) / 'clear' / protocol / 'results.txt.json'
                ).read_text()
            )
            if learned_result['sampled_flat_indices'] != identity_rows:
                raise ValueError('CLEAR identity/learned pair order mismatch')
            learned = _episode_successes(learned_result)
            pair_delta = learned.astype(np.float64) - identity.astype(
                np.float64
            )
            boot = np.empty(10000)
            for index in range(len(boot)):
                selected = rng.integers(0, 100, size=100)
                boot[index] = pair_delta[selected].mean()
            comparisons.append(
                {
                    'task': task,
                    'protocol': protocol,
                    'replicate_seed': seed,
                    'completed_pairs': 100,
                    'identity_success_count': int(identity.sum()),
                    'learned_success_count': int(learned.sum()),
                    'paired_success_delta': float(pair_delta.mean()),
                    'paired_success_delta_ci95': np.quantile(
                        boot, [0.025, 0.975]
                    ).tolist(),
                    'mcnemar': _mcnemar_exact(identity, learned),
                    'identity_runtime_seconds': identity_result[
                        'evaluation_time_seconds'
                    ],
                    'learned_runtime_seconds': learned_result[
                        'evaluation_time_seconds'
                    ],
                }
            )
    summary = {
        'task': task,
        'secondary_evidence_only': True,
        'comparisons': comparisons,
    }
    write_json(output, summary)
    return summary
