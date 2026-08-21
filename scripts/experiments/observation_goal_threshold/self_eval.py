"""Endpoint scoring and self-evaluation for find-goal-threshold."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .encode import encode_projected, preprocess_pixels
from .io_utils import read_json, sha256_file

SELF_EVAL_SCHEMA = 'find-goal-threshold-self-eval-v1'
ENDPOINT_SCORE_SCHEMA = 'find-goal-threshold-endpoint-scores-v1'
RESIDUAL = 'mean_D((z_i-z_j)^2)'


def load_and_validate_threshold(
    path: str | Path,
    *,
    task: str,
    checkpoint_sha256: str,
    checkpoint_config_sha256: str,
) -> tuple[dict, dict]:
    """Load a promoted threshold and verify its evaluation identity."""
    threshold_path = Path(path).expanduser().resolve()
    if not threshold_path.is_file():
        raise FileNotFoundError(
            f'No locked find-goal-threshold artifact for {task}: '
            f'{threshold_path}'
        )
    artifact = read_json(threshold_path)
    mismatches = {}
    source_config_path = threshold_path.parent / 'pre_registered_config.json'
    source_config = (
        read_json(source_config_path) if source_config_path.is_file() else None
    )
    recorded_config_sha256 = artifact.get('encoder_checkpoint_config_sha256')
    if recorded_config_sha256 is None and source_config is not None:
        recorded_config_sha256 = source_config.get('checkpoint', {}).get(
            'config_sha256'
        )
    expected = {
        'task': task,
        'encoder_checkpoint_sha256': checkpoint_sha256,
        'encoder_checkpoint_config_sha256': checkpoint_config_sha256,
        'residual_definition': RESIDUAL,
        'D': 192,
        'dtype': 'float32',
    }
    for key, value in expected.items():
        actual = (
            recorded_config_sha256
            if key == 'encoder_checkpoint_config_sha256'
            else artifact.get(key)
        )
        if actual != value:
            mismatches[key] = {
                'actual': actual,
                'expected': value,
            }
    epsilon = artifact.get('epsilon')
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not math.isfinite(float(epsilon))
        or float(epsilon) < 0
    ):
        mismatches['epsilon'] = {
            'actual': epsilon,
            'expected': 'finite non-negative locked epsilon',
        }
    preprocessing = artifact.get('observation_preprocessing', {})
    preprocessing_expected = {
        'residual': RESIDUAL,
        'dtype': 'float32',
        'latent_dim': 192,
        'resize': [224, 224],
        'normalization': 'ImageNet mean/std from stable_pretraining',
    }
    for key, value in preprocessing_expected.items():
        if preprocessing.get(key) != value:
            mismatches[f'observation_preprocessing.{key}'] = {
                'actual': preprocessing.get(key),
                'expected': value,
            }
    if mismatches:
        raise ValueError(
            f'find-goal-threshold identity mismatch: {mismatches}'
        )
    provenance = {
        'selected_threshold_path': str(threshold_path),
        'selected_threshold_sha256': sha256_file(threshold_path),
        'epsilon': float(epsilon),
        'pointwise_label_variant': artifact['pointwise_label_variant'],
        'encoder_checkpoint_sha256': checkpoint_sha256,
        'encoder_checkpoint_config_sha256': checkpoint_config_sha256,
        'encoder_checkpoint_provenance': artifact.get(
            'encoder_checkpoint_provenance'
        ),
        'residual_definition': RESIDUAL,
        'latent_dim': 192,
        'dtype': 'float32',
        'preprocessing': preprocessing,
        'source_pre_registered_config_path': (
            str(source_config_path) if source_config is not None else None
        ),
        'source_pre_registered_config_sha256': (
            sha256_file(source_config_path)
            if source_config is not None
            else None
        ),
    }
    return artifact, provenance


def load_and_validate_score_contract(
    path: str | Path,
    *,
    task: str,
    checkpoint_sha256: str,
    checkpoint_config_sha256: str,
) -> dict:
    """Validate a formal calibration identity without requiring epsilon.

    This is the score-only path for diagnostic epsilon sweeps. It records
    endpoint distances and evaluator labels but cannot promote or apply a
    threshold.
    """
    source = Path(path).expanduser().resolve()
    config_path = (
        source / 'pre_registered_config.json' if source.is_dir() else source
    )
    if not config_path.is_file():
        raise FileNotFoundError(
            f'No find-goal-threshold scoring contract: {config_path}'
        )
    status_path = config_path.parent / 'status.json'
    if not status_path.is_file():
        raise FileNotFoundError(
            f'No formal find-goal-threshold status artifact: {status_path}'
        )
    config = read_json(config_path)
    status = read_json(status_path)
    preprocessing = config.get('preprocessing', {})
    mismatches = {}
    expected = {
        'task': (config.get('task'), task),
        'checkpoint.sha256': (
            config.get('checkpoint', {}).get('sha256'),
            checkpoint_sha256,
        ),
        'checkpoint.config_sha256': (
            config.get('checkpoint', {}).get('config_sha256'),
            checkpoint_config_sha256,
        ),
        'preprocessing.residual': (
            preprocessing.get('residual'),
            RESIDUAL,
        ),
        'preprocessing.dtype': (preprocessing.get('dtype'), 'float32'),
        'preprocessing.latent_dim': (preprocessing.get('latent_dim'), 192),
        'preprocessing.resize': (preprocessing.get('resize'), [224, 224]),
        'preprocessing.normalization': (
            preprocessing.get('normalization'),
            'ImageNet mean/std from stable_pretraining',
        ),
        'status.formal_evidence': (
            status.get('formal_evidence'),
            True,
        ),
    }
    for key, (actual, wanted) in expected.items():
        if actual != wanted:
            mismatches[key] = {'actual': actual, 'expected': wanted}
    if mismatches:
        raise ValueError(
            f'find-goal-threshold score-contract identity mismatch: '
            f'{mismatches}'
        )
    threshold_path = config_path.parent / 'selected_threshold.json'
    return {
        'mode': 'score-only diagnostic epsilon sweep',
        'threshold_applied': False,
        'threshold_selection_permitted': False,
        'calibration_run_dir': str(config_path.parent),
        'pre_registered_config_path': str(config_path),
        'pre_registered_config_sha256': sha256_file(config_path),
        'status_path': str(status_path),
        'status_sha256': sha256_file(status_path),
        'calibration_status': status.get('status'),
        'calibration_stage': status.get('stage'),
        'selected_threshold_exists': threshold_path.is_file(),
        'task': task,
        'pointwise_label_variant': config.get('task_label', {}).get('variant'),
        'encoder_checkpoint_sha256': checkpoint_sha256,
        'encoder_checkpoint_config_sha256': checkpoint_config_sha256,
        'encoder_checkpoint_provenance': config.get('checkpoint', {}).get(
            'provenance'
        ),
        'residual_definition': RESIDUAL,
        'latent_dim': 192,
        'dtype': 'float32',
        'preprocessing': preprocessing,
    }


def _nhwc_uint8(value: Any, name: str) -> np.ndarray:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a project dependency.
        torch = None
    if torch is not None and torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 5 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(
            f'{name} must have shape (N,H,W,3) or (N,1,H,W,3), '
            f'got {array.shape}'
        )
    if array.dtype != np.uint8:
        raise ValueError(f'{name} must be uint8 RGB, got {array.dtype}')
    return array


def endpoint_distances(
    model,
    endpoint_pixels: Any,
    goal_pixels: Any,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    """Encode only terminal/goal pixels and return float32 mean-D MSE."""
    import torch

    endpoint = _nhwc_uint8(endpoint_pixels, 'endpoint_pixels')
    goal = _nhwc_uint8(goal_pixels, 'goal_pixels')
    if endpoint.shape != goal.shape:
        raise ValueError(
            f'endpoint/goal pixel shapes differ: {endpoint.shape} != '
            f'{goal.shape}'
        )
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device('cpu')
    distances = np.empty(len(endpoint), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(endpoint), batch_size):
            stop = min(start + batch_size, len(endpoint))
            pixels = np.concatenate(
                (endpoint[start:stop], goal[start:stop]), axis=0
            )
            encoded = encode_projected(
                model,
                preprocess_pixels(pixels).to(device, non_blocking=True),
            ).float()
            count = stop - start
            current, target = encoded[:count], encoded[count:]
            distances[start:stop] = (
                (current - target)
                .square()
                .mean(dim=1)
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
    if not np.isfinite(distances).all():
        raise ValueError('self-eval produced non-finite endpoint distances')
    return distances


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _wilson_interval(successes: int, total: int) -> dict[str, float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / denominator
    )
    return {'low': center - radius, 'high': center + radius}


def _paired_bootstrap_mean_interval(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {
            'low': float('nan'),
            'high': float('nan'),
            'replicates': replicates,
            'seed': seed,
        }
    rng = np.random.default_rng(seed)
    distribution = np.empty(replicates, dtype=np.float64)
    chunk = 512
    for start in range(0, replicates, chunk):
        count = min(chunk, replicates - start)
        sampled = rng.integers(0, len(values), size=(count, len(values)))
        distribution[start : start + count] = values[sampled].mean(axis=1)
    low, high = np.quantile(distribution, [0.025, 0.975])
    return {
        'low': float(low),
        'high': float(high),
        'replicates': replicates,
        'seed': seed,
    }


def compare_predictions(
    distances: np.ndarray,
    actual_successes: np.ndarray,
    *,
    epsilon: float,
    pair_ids: list[str],
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20260821,
) -> dict:
    """Build paired records and a binary confusion summary."""
    distance = np.asarray(distances, dtype=np.float32)
    actual = np.asarray(actual_successes, dtype=bool)
    if distance.shape != actual.shape or distance.ndim != 1:
        raise ValueError(
            'distance and actual success vectors must have the same 1-D shape'
        )
    if len(pair_ids) != len(actual):
        raise ValueError('pair_ids length does not match evaluation vectors')
    predicted = distance <= np.float32(epsilon)
    tp = int(np.sum(predicted & actual))
    tn = int(np.sum(~predicted & ~actual))
    fp = int(np.sum(predicted & ~actual))
    fn = int(np.sum(~predicted & actual))
    correct = tp + tn
    total = len(actual)
    actual_sr = float(actual.mean()) if total else float('nan')
    predicted_sr = float(predicted.mean()) if total else float('nan')
    paired_sr_difference = predicted.astype(np.int8) - actual.astype(np.int8)
    paired_sr_interval = _paired_bootstrap_mean_interval(
        paired_sr_difference,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    paired_sr_interval['low'] *= 100
    paired_sr_interval['high'] *= 100
    paired_sr_interval['unit'] = 'percentage_points'
    tpr = _safe_ratio(tp, tp + fn)
    tnr = _safe_ratio(tn, tn + fp)
    balanced_accuracy = (
        None if tpr is None or tnr is None else float((tpr + tnr) / 2)
    )
    summary = {
        'pairs': total,
        'actual_successes': int(actual.sum()),
        'actual_failures': int((~actual).sum()),
        'predicted_successes': int(predicted.sum()),
        'predicted_failures': int((~predicted).sum()),
        'confusion': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn},
        'accuracy': _safe_ratio(correct, total),
        'accuracy_wilson_95ci': _wilson_interval(correct, total),
        'sensitivity_tpr': tpr,
        'specificity_tnr': tnr,
        'false_positive_rate': _safe_ratio(fp, fp + tn),
        'false_negative_rate': _safe_ratio(fn, fn + tp),
        'precision': _safe_ratio(tp, tp + fp),
        'negative_predictive_value': _safe_ratio(tn, tn + fn),
        'balanced_accuracy': balanced_accuracy,
        'actual_success_rate_percent': actual_sr * 100,
        'predicted_success_rate_percent': predicted_sr * 100,
        'success_rate_error_percentage_points': (predicted_sr - actual_sr)
        * 100,
        'success_rate_error_paired_bootstrap_95ci': paired_sr_interval,
        'absolute_success_rate_error_percentage_points': abs(
            predicted_sr - actual_sr
        )
        * 100,
    }
    records = [
        {
            'pair_id': pair_id,
            'endpoint_latent_distance': float(d),
            'epsilon': float(epsilon),
            'predicted_success': bool(prediction),
            'evaluator_success': bool(truth),
            'correct': bool(prediction == truth),
        }
        for pair_id, d, prediction, truth in zip(
            pair_ids, distance, predicted, actual
        )
    ]
    return {
        'artifact_schema_version': SELF_EVAL_SCHEMA,
        'summary': summary,
        'pairs': records,
    }


def build_endpoint_score_records(
    distances: np.ndarray,
    actual_successes: np.ndarray,
    *,
    pair_ids: list[str],
) -> dict:
    """Record endpoint distances and CLEAR labels without choosing epsilon."""
    distance = np.asarray(distances, dtype=np.float32)
    actual = np.asarray(actual_successes, dtype=bool)
    if distance.shape != actual.shape or distance.ndim != 1:
        raise ValueError(
            'distance and actual success vectors must have the same 1-D shape'
        )
    if len(pair_ids) != len(actual):
        raise ValueError('pair_ids length does not match evaluation vectors')
    if not np.isfinite(distance).all():
        raise ValueError('endpoint scores contain non-finite distances')
    total = len(actual)
    return {
        'artifact_schema_version': ENDPOINT_SCORE_SCHEMA,
        'summary': {
            'pairs': total,
            'actual_successes': int(actual.sum()),
            'actual_failures': int((~actual).sum()),
            'actual_success_rate_percent': (
                float(actual.mean()) * 100 if total else float('nan')
            ),
            'epsilon_applied': False,
        },
        'pairs': [
            {
                'pair_id': pair_id,
                'endpoint_latent_distance': float(value),
                'evaluator_success': bool(success),
            }
            for pair_id, value, success in zip(pair_ids, distance, actual)
        ],
    }


def evaluate_endpoints(
    model,
    endpoint_pixels: Any,
    goal_pixels: Any,
    actual_successes: np.ndarray,
    *,
    epsilon: float,
    pair_ids: list[str],
    batch_size: int = 256,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20260821,
) -> dict:
    distances = endpoint_distances(
        model,
        endpoint_pixels,
        goal_pixels,
        batch_size=batch_size,
    )
    return compare_predictions(
        distances,
        actual_successes,
        epsilon=epsilon,
        pair_ids=pair_ids,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )


def evaluate_endpoint_scores(
    model,
    endpoint_pixels: Any,
    goal_pixels: Any,
    actual_successes: np.ndarray,
    *,
    pair_ids: list[str],
    batch_size: int = 256,
) -> dict:
    """Encode endpoints for a diagnostic sweep without selecting epsilon."""
    distances = endpoint_distances(
        model,
        endpoint_pixels,
        goal_pixels,
        batch_size=batch_size,
    )
    return build_endpoint_score_records(
        distances,
        actual_successes,
        pair_ids=pair_ids,
    )
