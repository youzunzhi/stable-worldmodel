"""Summarize paired CLEAR endpoint self-evaluation result JSONs."""

from __future__ import annotations

import argparse
from pathlib import Path

from .io_utils import read_json, sha256_file, write_json

TASKS = ('pusht', 'cube', 'tworoom')
PROTOCOLS = ('moderate', 'strict')


def _validate_result(path: Path) -> dict:
    result = read_json(path)
    clear = result.get('clear_lewm')
    self_eval = result.get('find_goal_threshold_self_eval')
    if clear is None or self_eval is None:
        raise ValueError(f'{path} is not a CLEAR self-eval result')
    if result.get('requested_trajectories') != 100:
        raise ValueError(f'{path} did not request 100 trajectories')
    if result.get('completed_trajectories') != 100:
        raise ValueError(f'{path} did not complete 100 trajectories')
    if clear.get('cpu_threads') != 1:
        raise ValueError(f'{path} did not use one Torch CPU thread')
    if not clear.get('solver_contract_matched'):
        raise ValueError(f'{path} did not match the CLEAR CEM contract')
    protocol = clear['protocol']['name']
    task = clear['task']
    if task != self_eval['task'] or protocol != self_eval['clear_protocol']:
        raise ValueError(f'{path} has inconsistent task/protocol identity')
    if (
        result['checkpoint_sha256']
        != self_eval['threshold']['encoder_checkpoint_sha256']
    ):
        raise ValueError(f'{path} threshold/checkpoint hashes differ')
    records = self_eval['pairs']
    if len(records) != 100 or len({row['pair_id'] for row in records}) != 100:
        raise ValueError(f'{path} lacks 100 unique paired records')
    if self_eval['summary']['pairs'] != 100:
        raise ValueError(f'{path} summary pair count is not 100')
    return result


def summarize(paths: list[Path], output_dir: Path) -> dict:
    cells = {}
    for path in paths:
        result = _validate_result(path)
        clear = result['clear_lewm']
        key = (clear['task'], clear['protocol']['name'])
        if key in cells:
            raise ValueError(f'duplicate self-eval cell: {key}')
        cells[key] = {
            'task': key[0],
            'protocol': key[1],
            'result_path': str(path.resolve()),
            'result_sha256': sha256_file(path),
            'checkpoint_sha256': result['checkpoint_sha256'],
            'manifest_sha256': clear['manifest_sha256'],
            'selected_threshold_sha256': result[
                'find_goal_threshold_self_eval'
            ]['threshold']['selected_threshold_sha256'],
            'epsilon': result['find_goal_threshold_self_eval']['threshold'][
                'epsilon'
            ],
            'summary': result['find_goal_threshold_self_eval']['summary'],
        }
    expected = {(task, protocol) for task in TASKS for protocol in PROTOCOLS}
    missing = sorted(expected - set(cells))
    payload = {
        'artifact_schema_version': 'find-goal-threshold-self-eval-summary-v1',
        'status': 'COMPLETE' if not missing else 'INCOMPLETE',
        'matrix_contract': {
            'tasks': list(TASKS),
            'protocols': list(PROTOCOLS),
            'pairs_per_cell': 100,
            'total_pairs_when_complete': 600,
            'pooling': 'none; report every task/protocol cell separately',
        },
        'missing_cells': [
            {'task': task, 'protocol': protocol} for task, protocol in missing
        ],
        'cells': [cells[key] for key in sorted(cells)],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / 'self_eval_summary.json', payload)
    lines = [
        '# find-goal-threshold CLEAR endpoint self-eval',
        '',
        f'- Matrix status: **{payload["status"]}**',
        '- Contract: 3 tasks x Moderate/Strict x 100 fixed CLEAR pairs.',
        '- Prediction: final endpoint latent distance `<= epsilon`.',
        '',
        '| Task | CLEAR rule | Epsilon | Actual SR | Predicted SR | SR error [paired 95% CI] | Accuracy [Wilson 95% CI] | TP/TN/FP/FN |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for cell in payload['cells']:
        summary = cell['summary']
        confusion = summary['confusion']
        error_ci = summary['success_rate_error_paired_bootstrap_95ci']
        accuracy_ci = summary['accuracy_wilson_95ci']
        lines.append(
            f'| {cell["task"]} | {cell["protocol"]} | '
            f'{cell["epsilon"]:.10g} | '
            f'{summary["actual_success_rate_percent"]:.1f}% | '
            f'{summary["predicted_success_rate_percent"]:.1f}% | '
            f'{summary["success_rate_error_percentage_points"]:+.1f} '
            f'[{error_ci["low"]:+.1f}, {error_ci["high"]:+.1f}] pp | '
            f'{summary["accuracy"]:.3f} '
            f'[{accuracy_ci["low"]:.3f}, {accuracy_ci["high"]:.3f}] | '
            f'{confusion["tp"]}/{confusion["tn"]}/'
            f'{confusion["fp"]}/{confusion["fn"]} |'
        )
    if missing:
        lines.extend(
            [
                '',
                '## Missing cells',
                '',
                *[f'- `{task}/{protocol}`' for task, protocol in missing],
            ]
        )
    lines.extend(
        [
            '',
            (
                'Accuracy is paired endpoint agreement with the installed '
                'CLEAR evaluator. It does not turn the pointwise predicate '
                'into a reachability, route, pose, or sustained-success rule.'
            ),
            '',
        ]
    )
    (output_dir / 'SELF_EVAL_REPORT.md').write_text('\n'.join(lines))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, action='append', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    summarize(
        [path.expanduser().resolve() for path in args.result],
        args.output_dir.expanduser().resolve(),
    )


if __name__ == '__main__':
    main()
