"""Plot epsilon against fit-split macro TPR and FPR."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .io_utils import read_json, sha256_file, write_json
from .metrics import macro_curve


def _plot_indices(length: int, limit: int = 20_000) -> np.ndarray:
    if length <= limit:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, limit).astype(np.int64))


def plot_epsilon_tpr_fpr(
    stratified: dict[str, np.ndarray],
    output_path: str | Path,
    *,
    task: str,
    selected_epsilon: float | None,
    min_positive_recall: float,
    max_negative_fpr: float,
    status: str,
) -> dict:
    """Create the requested epsilon-versus-TPR/FPR operating curve."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    curve = macro_curve(
        stratified['latent_distance'],
        stratified['label'],
        stratified['analysis_weight'],
        stratified['anchor_group'],
    )
    finite = np.flatnonzero(np.isfinite(curve['epsilon']))
    chosen = finite[_plot_indices(len(finite))]
    epsilon = curve['epsilon'][chosen]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    display_status = status.removeprefix('THRESHOLD_CALIBRATION_').replace(
        '_', ' '
    )
    figure, axis = plt.subplots(figsize=(8, 5.5))
    axis.plot(
        epsilon,
        curve['macro_tpr'][chosen],
        label='fit macro TPR',
        linewidth=1.8,
    )
    axis.plot(
        epsilon,
        curve['macro_fpr'][chosen],
        label='fit macro FPR',
        linewidth=1.8,
    )
    axis.axhline(
        min_positive_recall,
        color='tab:blue',
        linestyle=':',
        linewidth=1,
        label=f'min TPR = {min_positive_recall:g}',
    )
    axis.axhline(
        max_negative_fpr,
        color='tab:orange',
        linestyle=':',
        linewidth=1,
        label=f'max FPR = {max_negative_fpr:g}',
    )
    if selected_epsilon is not None:
        axis.axvline(
            selected_epsilon,
            color='black',
            linestyle='--',
            linewidth=1.3,
            label=f'locked epsilon = {selected_epsilon:.6g}',
        )
    axis.set(
        xlabel='epsilon (mean-D latent MSE)',
        ylabel='rate',
        ylim=(-0.02, 1.02),
        title=f'{task}: epsilon vs fit TPR/FPR\n{display_status}',
    )
    axis.grid(alpha=0.2)
    axis.legend(loc='best')
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return {
        'task': task,
        'status': status,
        'curve_partition': 'threshold_fit',
        'curve_estimator': 'anchor-group macro',
        'curve_points_exact': len(curve['epsilon']),
        'curve_points_plotted': len(chosen),
        'epsilon_min': float(epsilon[0]),
        'epsilon_max': float(epsilon[-1]),
        'selected_epsilon': selected_epsilon,
        'min_positive_recall': float(min_positive_recall),
        'max_negative_fpr': float(max_negative_fpr),
        'plot': output.name,
        'plot_sha256': sha256_file(output),
    }


def _load_fit_stratified(run_dir: Path) -> dict[str, np.ndarray]:
    import pyarrow.parquet as pq

    columns = [
        'anchor_group',
        'label',
        'analysis_weight',
        'latent_distance',
    ]
    parts = {column: [] for column in columns}
    score_dir = run_dir / 'pair_scores' / 'threshold_fit'
    paths = sorted(score_dir.glob('task_stratified-*.parquet'))
    if not paths:
        raise FileNotFoundError(
            f'No fit task-stratified scores in {score_dir}'
        )
    for path in paths:
        table = pq.read_table(path, columns=columns)
        for column in columns:
            parts[column].append(table[column].to_numpy())
    return {column: np.concatenate(values) for column, values in parts.items()}


def plot_existing_run(run_dir: Path, output_dir: Path) -> dict:
    """Create a derived curve without mutating an immutable formal run."""
    config_path = run_dir / 'pre_registered_config.json'
    config = read_json(config_path)
    threshold_path = run_dir / 'selected_threshold.json'
    status_path = run_dir / 'status.json'
    threshold = read_json(threshold_path) if threshold_path.is_file() else None
    status = read_json(status_path)['status']
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = plot_epsilon_tpr_fpr(
        _load_fit_stratified(run_dir),
        output_dir / 'epsilon_tpr_fpr_curve.png',
        task=config['task'],
        selected_epsilon=(
            float(threshold['epsilon']) if threshold is not None else None
        ),
        min_positive_recall=float(config['selection']['min_positive_recall']),
        max_negative_fpr=float(config['selection']['max_negative_fpr']),
        status=status,
    )
    manifest.update(
        {
            'artifact_schema_version': 'find-goal-threshold-curve-v1',
            'source_run_dir': str(run_dir),
            'source_config_sha256': sha256_file(config_path),
            'source_status_sha256': sha256_file(status_path),
            'source_threshold_sha256': (
                sha256_file(threshold_path) if threshold is not None else None
            ),
            'source_fit_score_shards': [
                {'file': path.name, 'sha256': sha256_file(path)}
                for path in sorted(
                    (run_dir / 'pair_scores' / 'threshold_fit').glob(
                        'task_stratified-*.parquet'
                    )
                )
            ],
        }
    )
    write_json(output_dir / 'curve_manifest.json', manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    plot_existing_run(args.run_dir.resolve(), args.output_dir.resolve())


if __name__ == '__main__':
    main()
