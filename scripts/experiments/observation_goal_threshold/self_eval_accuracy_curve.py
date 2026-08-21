"""Plot epsilon against CLEAR endpoint metrics for all 3x2 cells."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from .io_utils import read_json, sha256_file, write_json

TASKS = ('pusht', 'cube', 'tworoom')
PROTOCOLS = ('moderate', 'strict')
TASK_LABELS = {'pusht': 'PushT', 'cube': 'Cube', 'tworoom': 'TwoRoom'}
TPR_FPR_X_LIMITS = {'cube': 4.0, 'tworoom': 3.0}


def epsilon_accuracy_curve(
    distances: np.ndarray, actual_successes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact non-negative epsilon breakpoints and pair accuracies."""
    distance = np.asarray(distances, dtype=np.float32)
    actual = np.asarray(actual_successes, dtype=bool)
    if distance.ndim != 1 or distance.shape != actual.shape or not len(actual):
        raise ValueError('distance and success vectors must be non-empty 1-D')
    if not np.isfinite(distance).all() or np.any(distance < 0):
        raise ValueError('endpoint distances must be finite and non-negative')
    epsilon = np.unique(
        np.concatenate((np.array([0], dtype=np.float32), distance))
    )
    predicted = distance[:, None] <= epsilon[None, :]
    accuracy = np.mean(predicted == actual[:, None], axis=0)
    return epsilon, accuracy


def epsilon_tpr_fpr_curve(
    distances: np.ndarray, actual_successes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact epsilon breakpoints and evaluator-relative TPR/FPR."""
    distance = np.asarray(distances, dtype=np.float32)
    actual = np.asarray(actual_successes, dtype=bool)
    if distance.ndim != 1 or distance.shape != actual.shape or not len(actual):
        raise ValueError('distance and success vectors must be non-empty 1-D')
    if not np.isfinite(distance).all() or np.any(distance < 0):
        raise ValueError('endpoint distances must be finite and non-negative')
    positives = int(actual.sum())
    negatives = int((~actual).sum())
    if positives == 0 or negatives == 0:
        raise ValueError(
            'TPR/FPR requires both evaluator classes in each cell'
        )
    epsilon = np.unique(
        np.concatenate((np.array([0], dtype=np.float32), distance))
    )
    predicted = distance[:, None] <= epsilon[None, :]
    tpr = np.sum(predicted & actual[:, None], axis=0) / positives
    fpr = np.sum(predicted & ~actual[:, None], axis=0) / negatives
    return epsilon, tpr, fpr


def _load_cell(path: Path) -> dict:
    result = read_json(path)
    clear = result.get('clear_lewm')
    if clear is None:
        raise ValueError(f'{path} is not a CLEAR result')
    if result.get('requested_trajectories') != 100:
        raise ValueError(f'{path} did not request 100 trajectories')
    if result.get('completed_trajectories') != 100:
        raise ValueError(f'{path} did not complete 100 trajectories')
    if clear.get('cpu_threads') != 1:
        raise ValueError(f'{path} did not use one Torch CPU thread')
    if not clear.get('solver_contract_matched'):
        raise ValueError(f'{path} did not match the CLEAR CEM contract')
    solver = result.get('resolved_config', {}).get('solver', {})
    solver_contract = {
        'batch_size': 1,
        'num_samples': 300,
        'n_steps': 30,
        'topk': 30,
    }
    if any(solver.get(key) != value for key, value in solver_contract.items()):
        raise ValueError(f'{path} has a mismatched resolved CEM contract')

    self_eval = result.get('find_goal_threshold_self_eval')
    endpoint_scores = result.get('find_goal_threshold_endpoint_scores')
    if (self_eval is None) == (endpoint_scores is None):
        raise ValueError(
            f'{path} must contain exactly one endpoint-scoring artifact'
        )
    artifact = self_eval if self_eval is not None else endpoint_scores
    source_kind = 'locked-epsilon self-eval' if self_eval else 'score-only'
    task = clear['task']
    protocol = clear['protocol']['name']
    if task != artifact['task'] or protocol != artifact['clear_protocol']:
        raise ValueError(f'{path} has inconsistent task/protocol identity')
    records = artifact['pairs']
    pair_ids = [row['pair_id'] for row in records]
    if len(records) != 100 or len(set(pair_ids)) != 100:
        raise ValueError(f'{path} lacks 100 unique endpoint records')
    actual = np.asarray(
        [row['evaluator_success'] for row in records], dtype=bool
    )
    if actual.tolist() != result['metrics']['episode_successes']:
        raise ValueError(
            f'{path} endpoint labels do not match evaluator output'
        )
    distance = np.asarray(
        [row['endpoint_latent_distance'] for row in records],
        dtype=np.float32,
    )
    epsilon, accuracy = epsilon_accuracy_curve(distance, actual)
    operating_epsilon, tpr, fpr = epsilon_tpr_fpr_curve(distance, actual)
    if not np.array_equal(epsilon, operating_epsilon):
        raise ValueError(
            f'{path} metric sweeps produced different breakpoints'
        )

    provenance = (
        artifact['threshold'] if self_eval else artifact['score_contract']
    )
    if provenance['encoder_checkpoint_sha256'] != result['checkpoint_sha256']:
        raise ValueError(f'{path} checkpoint/scoring-contract hashes differ')
    if (
        provenance['encoder_projector_parameter_hash_before_evaluation']
        != provenance['encoder_projector_parameter_hash_after_evaluation']
    ):
        raise ValueError(f'{path} encoder/projector hash changed')
    locked_epsilon = (
        float(artifact['threshold']['epsilon']) if self_eval else None
    )
    locked_accuracy = (
        float(np.mean((distance <= locked_epsilon) == actual))
        if locked_epsilon is not None
        else None
    )
    if locked_epsilon is not None:
        locked_prediction = distance <= locked_epsilon
        locked_tpr = float(np.sum(locked_prediction & actual) / np.sum(actual))
        locked_fpr = float(
            np.sum(locked_prediction & ~actual) / np.sum(~actual)
        )
    else:
        locked_tpr = None
        locked_fpr = None
    return {
        'task': task,
        'protocol': protocol,
        'result_path': str(path),
        'result_sha256': sha256_file(path),
        'manifest_sha256': clear['manifest_sha256'],
        'checkpoint_sha256': result['checkpoint_sha256'],
        'source_kind': source_kind,
        'actual_success_rate_percent': float(actual.mean() * 100),
        'locked_epsilon': locked_epsilon,
        'locked_epsilon_accuracy': locked_accuracy,
        'locked_epsilon_tpr': locked_tpr,
        'locked_epsilon_fpr': locked_fpr,
        'curve_points': [
            {
                'epsilon': float(value),
                'pair_accuracy': float(rate),
                'tpr': float(true_positive_rate),
                'fpr': float(false_positive_rate),
            }
            for value, rate, true_positive_rate, false_positive_rate in zip(
                epsilon, accuracy, tpr, fpr
            )
        ],
    }


def _load_matrix(paths: list[Path], curve_name: str) -> dict:
    cells = {}
    for path in paths:
        cell = _load_cell(path.expanduser().resolve())
        key = (cell['task'], cell['protocol'])
        if key in cells:
            raise ValueError(f'duplicate {curve_name} cell: {key}')
        cells[key] = cell
    expected = {(task, protocol) for task in TASKS for protocol in PROTOCOLS}
    missing = sorted(expected - set(cells))
    unexpected = sorted(set(cells) - expected)
    if missing or unexpected:
        raise ValueError(
            f'{curve_name} requires exact 3x2 matrix; '
            f'missing={missing}, unexpected={unexpected}'
        )
    return cells


def plot_matrix(paths: list[Path], output_dir: Path) -> dict:
    """Validate a complete 3x2 matrix and render one six-panel figure."""
    cells = _load_matrix(paths, 'epsilon-accuracy curve')

    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=False)
    plot_path = output_dir / 'epsilon_pair_accuracy_3x2.png'
    figure, axes = plt.subplots(
        len(TASKS),
        len(PROTOCOLS),
        figsize=(12, 11),
        sharey=True,
        constrained_layout=True,
    )
    for row, task in enumerate(TASKS):
        for column, protocol in enumerate(PROTOCOLS):
            axis = axes[row, column]
            cell = cells[(task, protocol)]
            epsilon = np.asarray(
                [point['epsilon'] for point in cell['curve_points']]
            )
            accuracy = np.asarray(
                [point['pair_accuracy'] for point in cell['curve_points']]
            )
            locked = cell['locked_epsilon']
            xmax = max(float(epsilon[-1]), float(locked or 0))
            xmax = xmax * 1.05 if xmax > 0 else 1.0
            display_epsilon = np.append(epsilon, xmax)
            display_accuracy = np.append(accuracy, accuracy[-1])
            axis.step(
                display_epsilon,
                display_accuracy,
                where='post',
                linewidth=1.8,
                color='tab:blue',
                label='pair accuracy',
            )
            if locked is not None:
                locked_accuracy = cell['locked_epsilon_accuracy']
                axis.axvline(
                    locked,
                    color='black',
                    linestyle='--',
                    linewidth=1.2,
                    label=f'locked epsilon = {locked:.4g}',
                )
                axis.scatter(
                    [locked],
                    [locked_accuracy],
                    color='black',
                    s=24,
                    zorder=3,
                )
                status = f'locked-epsilon accuracy = {locked_accuracy:.2f}'
            else:
                status = 'score-only; no promoted epsilon'
            axis.set(
                xlim=(0, xmax),
                ylim=(-0.02, 1.02),
                xlabel='epsilon (mean-D latent MSE)',
                ylabel='pair accuracy',
                title=(
                    f'{TASK_LABELS[task]} / {protocol.title()} '
                    f'(actual SR {cell["actual_success_rate_percent"]:.0f}%)\n'
                    f'{status}'
                ),
            )
            axis.grid(alpha=0.2)
            axis.legend(loc='best', fontsize=8)
    figure.suptitle(
        'find-goal-threshold: epsilon vs CLEAR endpoint pair accuracy\n'
        'Post-lock diagnostic sweep only; no epsilon is selected from this figure',
        fontsize=14,
    )
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    ordered_cells = [
        cells[(task, protocol)] for task in TASKS for protocol in PROTOCOLS
    ]
    payload = {
        'artifact_schema_version': (
            'find-goal-threshold-epsilon-pair-accuracy-curve-v1'
        ),
        'status': 'COMPLETE_3X2',
        'matrix_contract': {
            'tasks': list(TASKS),
            'protocols': list(PROTOCOLS),
            'pairs_per_cell': 100,
            'total_paired_endpoints': 600,
            'pooling': 'none; each task/protocol panel is separate',
        },
        'predicate_sweep': 'endpoint_latent_distance <= epsilon',
        'curve_semantics': (
            'empirical paired accuracy on fixed CLEAR endpoints; diagnostic '
            'only and not a threshold-selection or calibration artifact'
        ),
        'plot': plot_path.name,
        'plot_sha256': sha256_file(plot_path),
        'cells': ordered_cells,
    }
    if any(
        not math.isfinite(point['pair_accuracy'])
        for cell in ordered_cells
        for point in cell['curve_points']
    ):
        raise ValueError('non-finite accuracy in curve payload')
    write_json(output_dir / 'curve_manifest.json', payload)
    return payload


def plot_tpr_fpr_matrix(paths: list[Path], output_dir: Path) -> dict:
    """Render evaluator-relative TPR/FPR with requested task x-ranges."""
    cells = _load_matrix(paths, 'epsilon-TPR/FPR curve')

    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=False)
    plot_path = output_dir / 'epsilon_endpoint_tpr_fpr_3x2.png'
    figure, axes = plt.subplots(
        len(TASKS),
        len(PROTOCOLS),
        figsize=(12, 11),
        sharey=True,
        constrained_layout=True,
    )
    plot_x_limits = {}
    for row, task in enumerate(TASKS):
        for column, protocol in enumerate(PROTOCOLS):
            axis = axes[row, column]
            cell = cells[(task, protocol)]
            epsilon = np.asarray(
                [point['epsilon'] for point in cell['curve_points']]
            )
            tpr = np.asarray([point['tpr'] for point in cell['curve_points']])
            fpr = np.asarray([point['fpr'] for point in cell['curve_points']])
            locked = cell['locked_epsilon']
            automatic_xmax = max(float(epsilon[-1]), float(locked or 0))
            automatic_xmax = (
                automatic_xmax * 1.05 if automatic_xmax > 0 else 1.0
            )
            xmax = TPR_FPR_X_LIMITS.get(task, automatic_xmax)
            if float(epsilon[-1]) > xmax:
                raise ValueError(
                    f'{task}/{protocol} endpoint distances exceed requested '
                    f'x-axis maximum {xmax}: {epsilon[-1]}'
                )
            display_epsilon = (
                np.append(epsilon, xmax)
                if float(epsilon[-1]) < xmax
                else epsilon
            )
            display_tpr = (
                np.append(tpr, tpr[-1])
                if len(display_epsilon) > len(tpr)
                else tpr
            )
            display_fpr = (
                np.append(fpr, fpr[-1])
                if len(display_epsilon) > len(fpr)
                else fpr
            )
            axis.step(
                display_epsilon,
                display_tpr,
                where='post',
                linewidth=1.8,
                color='tab:blue',
                label='TPR',
            )
            axis.step(
                display_epsilon,
                display_fpr,
                where='post',
                linewidth=1.8,
                color='tab:orange',
                label='FPR',
            )
            if locked is not None:
                locked_tpr = cell['locked_epsilon_tpr']
                locked_fpr = cell['locked_epsilon_fpr']
                axis.axvline(
                    locked,
                    color='black',
                    linestyle='--',
                    linewidth=1.2,
                    label=f'locked epsilon = {locked:.4g}',
                )
                axis.scatter(
                    [locked, locked],
                    [locked_tpr, locked_fpr],
                    color=['tab:blue', 'tab:orange'],
                    edgecolor='black',
                    linewidth=0.4,
                    s=28,
                    zorder=3,
                )
                status = (
                    f'locked-epsilon TPR/FPR = '
                    f'{locked_tpr:.2f}/{locked_fpr:.2f}'
                )
            else:
                status = 'score-only; no promoted epsilon'
            axis.set(
                xlim=(0, xmax),
                ylim=(-0.02, 1.02),
                xlabel='epsilon (mean-D latent MSE)',
                ylabel='rate against CLEAR evaluator S/F',
                title=(
                    f'{TASK_LABELS[task]} / {protocol.title()} '
                    f'(actual SR {cell["actual_success_rate_percent"]:.0f}%)\n'
                    f'{status}'
                ),
            )
            axis.grid(alpha=0.2)
            axis.legend(loc='best', fontsize=8)
            plot_x_limits[f'{task}/{protocol}'] = [0.0, float(xmax)]
    figure.suptitle(
        'find-goal-threshold: epsilon vs CLEAR endpoint TPR/FPR\n'
        'Post-lock diagnostic sweep only; no epsilon is selected from this figure',
        fontsize=14,
    )
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    ordered_cells = [
        cells[(task, protocol)] for task in TASKS for protocol in PROTOCOLS
    ]
    payload = {
        'artifact_schema_version': (
            'find-goal-threshold-epsilon-endpoint-tpr-fpr-curve-v1'
        ),
        'status': 'COMPLETE_3X2',
        'matrix_contract': {
            'tasks': list(TASKS),
            'protocols': list(PROTOCOLS),
            'pairs_per_cell': 100,
            'total_paired_endpoints': 600,
            'pooling': 'none; each task/protocol panel is separate',
        },
        'predicate_sweep': 'endpoint_latent_distance <= epsilon',
        'rate_reference': 'CLEAR evaluator S/F per fixed endpoint pair',
        'curve_semantics': (
            'empirical endpoint TPR/FPR on fixed CLEAR pairs; diagnostic only '
            'and not a threshold-selection or calibration artifact'
        ),
        'plot_x_limits': plot_x_limits,
        'plot': plot_path.name,
        'plot_sha256': sha256_file(plot_path),
        'cells': ordered_cells,
    }
    if any(
        not math.isfinite(point[metric])
        for cell in ordered_cells
        for point in cell['curve_points']
        for metric in ('tpr', 'fpr')
    ):
        raise ValueError('non-finite TPR/FPR in curve payload')
    write_json(output_dir / 'curve_manifest.json', payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, action='append', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument(
        '--metric', choices=('accuracy', 'tpr-fpr'), default='accuracy'
    )
    args = parser.parse_args()
    renderer = (
        plot_matrix if args.metric == 'accuracy' else plot_tpr_fpr_matrix
    )
    renderer(args.result, args.output_dir.expanduser().resolve())


if __name__ == '__main__':
    main()
