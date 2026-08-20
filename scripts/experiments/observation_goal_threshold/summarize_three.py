"""Build the final three-task Experiment T summary from immutable runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from .io_utils import read_json, sha256_file, write_json

TASKS = ('pusht', 'cube', 'tworoom')


def summarize(root: Path) -> dict:
    rows = []
    for task in TASKS:
        run = root / task
        status = read_json(run / 'status.json')
        row = {
            'task': task,
            'status': status['status'],
            'formal_evidence': status['formal_evidence'],
            'run_dir': str(run),
            'report_sha256': sha256_file(run / 'report.md'),
        }
        threshold_path = run / 'selected_threshold.json'
        if threshold_path.is_file():
            threshold = read_json(threshold_path)
            row.update(
                {
                    'variant': threshold['pointwise_label_variant'],
                    'epsilon': threshold['epsilon'],
                    'checkpoint_sha256': threshold[
                        'encoder_checkpoint_sha256'
                    ],
                    'dataset_sha256': threshold['dataset_version']['sha256'],
                    'selected_threshold_sha256': sha256_file(threshold_path),
                    'fit': read_json(run / 'fit_metrics.json'),
                    'validation': read_json(run / 'validation_metrics.json'),
                    'audit': read_json(run / 'audit_metrics.json'),
                }
            )
        else:
            row['variant'] = read_json(run / 'pre_registered_config.json')[
                'task_label'
            ]['variant']
            row['epsilon'] = None
        rows.append(row)
    summary = {'tasks': rows}
    write_json(root / 'three_task_summary.json', summary)
    lines = [
        '# Experiment T: three-task calibrated goal-threshold report',
        '',
        '| Task | Pointwise variant | Status | Epsilon | Audit TPR | Audit FPR | Audit precision |',
        '|---|---|---|---:|---:|---:|---:|',
    ]
    for row in rows:
        if row['epsilon'] is None:
            lines.append(
                f'| {row["task"]} | `{row["variant"]}` | {row["status"]} | - | - | - | - |'
            )
        else:
            audit = row['audit']
            lines.append(
                f'| {row["task"]} | `{row["variant"]}` | {row["status"]} | '
                f'{row["epsilon"]:.10g} | {audit["macro_tpr"]:.6f} | '
                f'{audit["macro_fpr"]:.6f} | '
                f'{audit["population_precision_direct"]:.6f} |'
            )
    lines.extend(
        [
            '',
            (
                'Each epsilon is task-, variant-, checkpoint-, dataset-, '
                'preprocessing-, residual-, and dtype-specific. These '
                'pointwise encoder-geometry results are not predictor, '
                'reachability, planner, or official CLEAR Moderate/Strict '
                'results.'
            ),
            '',
        ]
    )
    (root / 'THREE_TASK_GOAL_THRESHOLD_REPORT.md').write_text('\n'.join(lines))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.root.resolve())


if __name__ == '__main__':
    main()
