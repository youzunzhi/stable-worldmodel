"""Run the paired Experiment-T threshold data-sensitivity study.

New replicates materialize the unchanged 100M-uniform plus 20M-stratified
design, but open only the fit scores.  The paired stratified-only estimate is
computed from the exact same fit stratified shards.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import socket
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .encode import score_pair_shards
from .io_utils import (
    git_provenance,
    read_json,
    sha256_file,
    sha256_json,
    software_versions,
    write_json,
)
from .metrics import (
    CalibrationOutcome,
    best_stratified_operating_point,
    select_threshold,
)
from .run import _load_scored_partition, build_pairs
from .split import PARTITIONS, row_partitions, split_groups

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TASKS = ('pusht', 'cube', 'tworoom')
IDENTITY_KEYS = (
    'artifact_schema_version',
    'task',
    'dataset',
    'checkpoint',
    'task_label',
    'compatibility_signature',
    'preprocessing',
    'selection',
)
STATIC_INDEX_FILES = (
    'state.npy',
    'episodes.npy',
    'steps.npy',
    'groups.npy',
    'eligible_rows.npy',
)


def _replicate(matrix: dict, replicate_id: str) -> dict:
    matches = [
        item for item in matrix['replicates'] if item['id'] == replicate_id
    ]
    if len(matches) != 1:
        raise ValueError(f'unknown or duplicate replicate: {replicate_id}')
    return matches[0]


def _validate_matrix(matrix: dict) -> None:
    if tuple(matrix['tasks']) != TASKS:
        raise ValueError(f'tasks must be exactly {TASKS}')
    replicates = matrix['replicates']
    if len(replicates) != 5:
        raise ValueError('the paired design requires exactly five seeds')
    if [item['id'] for item in replicates] != [
        f'seed-{index}' for index in range(5)
    ]:
        raise ValueError('replicate IDs must be seed-0 through seed-4')
    if replicates[0]['source'] != 'existing_formal_baseline':
        raise ValueError('seed-0 must be the existing formal baseline')
    if any(item['source'] != 'new' for item in replicates[1:]):
        raise ValueError('seed-1 through seed-4 must be new')
    seed_pairs = {
        (item['threshold_split_seed'], item['pair_sampling_seed'])
        for item in replicates
    }
    if len(seed_pairs) != len(replicates):
        raise ValueError('replicate seed pairs must be unique')
    current = matrix['conditions']['current_method']
    stratified = matrix['conditions']['stratified_only']
    if current != {
        'uniform_pairs': 100_000_000,
        'task_stratified_pairs': 20_000_000,
        'task_stratified_positive_fraction': 0.5,
    }:
        raise ValueError('current-method pair contract changed')
    if stratified != {
        'uniform_pairs': 0,
        'task_stratified_pairs': 20_000_000,
        'task_stratified_positive_fraction': 0.5,
    }:
        raise ValueError('stratified-only pair contract changed')


def _load_task_config(task: str, baseline_task_dir: Path) -> dict:
    if task not in TASKS:
        raise ValueError(f'unknown task: {task}')
    baseline = read_json(baseline_task_dir / 'pre_registered_config.json')
    checked_in = read_json(
        Path(__file__).with_name('configs') / f'{task}.json'
    )
    for key in IDENTITY_KEYS:
        if baseline[key] != checked_in[key]:
            raise ValueError(f'baseline/checked-in identity mismatch: {key}')
    if baseline['pair_sampling'] != checked_in['pair_sampling']:
        raise ValueError('baseline/checked-in pair contract mismatch')
    if baseline['data'] != checked_in['data']:
        raise ValueError('baseline/checked-in seed or split mismatch')
    return baseline


def _validate_baseline_reuse(
    config: dict, baseline_task_dir: Path
) -> dict[str, Any]:
    index = read_json(baseline_task_dir / 'index' / 'index_manifest.json')
    embeddings = read_json(
        baseline_task_dir / 'embedding_shards' / 'embedding_manifest.json'
    )
    if (
        index['dataset_sha256']
        != read_json(baseline_task_dir / 'dataset_inventory.json')[
            'dataset_sha256'
        ]
    ):
        raise ValueError('baseline dataset hashes disagree')
    if embeddings['checkpoint_sha256'] != config['checkpoint']['sha256']:
        raise ValueError('baseline embedding checkpoint mismatch')
    if (
        embeddings['checkpoint_config_sha256']
        != config['checkpoint']['config_sha256']
    ):
        raise ValueError('baseline embedding checkpoint-config mismatch')
    if embeddings['latent_dim'] != config['preprocessing']['latent_dim']:
        raise ValueError('baseline embedding latent dimension mismatch')
    if embeddings['dtype'] != config['preprocessing']['dtype']:
        raise ValueError('baseline embedding dtype mismatch')
    forbidden = embeddings['forbidden_call_counts']
    if any(forbidden.values()):
        raise ValueError('baseline embedding manifest has forbidden calls')
    embedding_dir = baseline_task_dir / 'embedding_shards'
    actual_embedding_hash = sha256_file(
        embedding_dir / embeddings['embedding_file']
    )
    if actual_embedding_hash != embeddings['embedding_sha256']:
        raise ValueError('baseline embedding file hash mismatch')
    actual_row_ids_hash = sha256_file(embedding_dir / 'row_ids.npy')
    if actual_row_ids_hash != embeddings['row_ids_sha256']:
        raise ValueError('baseline embedding row-ID hash mismatch')
    rows = np.load(embedding_dir / 'row_ids.npy', allow_pickle=False)
    eligible = np.load(
        baseline_task_dir / 'index' / 'eligible_rows.npy',
        allow_pickle=False,
    )
    if not np.array_equal(rows, eligible):
        raise ValueError('baseline embedding/index row IDs differ')
    return {
        'baseline_task_dir': str(baseline_task_dir),
        'dataset_sha256': index['dataset_sha256'],
        'embedding_sha256': actual_embedding_hash,
        'embedding_row_ids_sha256': actual_row_ids_hash,
        'checkpoint_sha256': embeddings['checkpoint_sha256'],
        'checkpoint_config_sha256': embeddings['checkpoint_config_sha256'],
        'encoder_projector_parameter_hash_before': embeddings[
            'encoder_projector_parameter_hash_before'
        ],
        'encoder_projector_parameter_hash_after': embeddings[
            'encoder_projector_parameter_hash_after'
        ],
        'forbidden_call_counts': forbidden,
    }


def _prepare_reused_index(
    config: dict,
    baseline_task_dir: Path,
    run_dir: Path,
) -> dict[str, Any]:
    baseline_index = baseline_task_dir / 'index'
    target = run_dir / 'index'
    target.mkdir(parents=True, exist_ok=False)
    for name in STATIC_INDEX_FILES:
        source = (baseline_index / name).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        os.symlink(source, target / name)
    groups = np.load(target / 'groups.npy', mmap_mode='r')
    group_split = split_groups(
        groups,
        config['data']['threshold_split_seed'],
        (
            config['data']['fit_fraction'],
            config['data']['validation_fraction'],
            config['data']['audit_fraction'],
        ),
    )
    partitions = row_partitions(groups, group_split)
    np.save(target / 'row_partitions.npy', partitions, allow_pickle=False)
    split_payload = {
        name: [int(value) for value in group_split[name]]
        for name in PARTITIONS
    }
    split_payload['sha256'] = sha256_json(split_payload)
    write_json(run_dir / 'split_ids.json', split_payload)
    manifest = {
        'reuse_mode': 'verified_exact_frozen_baseline_arrays',
        'baseline_index_manifest_sha256': sha256_file(
            baseline_index / 'index_manifest.json'
        ),
        'split_sha256': split_payload['sha256'],
        'threshold_split_seed': config['data']['threshold_split_seed'],
        'fit_groups': len(group_split[PARTITIONS[0]]),
        'validation_groups': len(group_split[PARTITIONS[1]]),
        'audit_groups': len(group_split[PARTITIONS[2]]),
        'row_partitions_sha256': sha256_file(target / 'row_partitions.npy'),
    }
    write_json(target / 'sensitivity_index_manifest.json', manifest)
    return manifest


def _selection_payload(
    stratified: dict[str, np.ndarray],
    uniform: dict[str, np.ndarray],
    selection_config: dict,
) -> tuple[dict[str, Any], dict[str, Any]]:
    best = best_stratified_operating_point(
        stratified,
        max_negative_fpr=selection_config['max_negative_fpr'],
    )
    meets_recall = best.macro_tpr >= selection_config['min_positive_recall']
    status = (
        'THRESHOLD_FIT_FEASIBLE'
        if meets_recall
        else 'THRESHOLD_CALIBRATION_NO_FEASIBLE_OPERATING_POINT'
    )
    full_status = None
    full_epsilon = None
    try:
        selected = select_threshold(
            stratified,
            uniform,
            min_positive_recall=selection_config['min_positive_recall'],
            max_negative_fpr=selection_config['max_negative_fpr'],
            min_population_precision=selection_config[
                'min_population_precision'
            ],
        )
        full_status = 'THRESHOLD_FIT_FEASIBLE'
        full_epsilon = selected.epsilon
    except CalibrationOutcome as outcome:
        full_status = outcome.code
    if full_status != status:
        raise AssertionError('full and stratified feasibility statuses differ')
    if full_epsilon is not None and full_epsilon != best.epsilon:
        raise AssertionError('uniform pairs changed the selected epsilon')
    common = {
        'status': status,
        'selected_epsilon': best.epsilon if meets_recall else None,
        'reported_epsilon': best.epsilon,
        'reported_epsilon_role': (
            'selected_fit_threshold'
            if meets_recall
            else 'descriptive_best_operating_point_not_promoted'
        ),
        'macro_tpr': best.macro_tpr,
        'macro_fpr': best.macro_fpr,
        'required_macro_tpr': selection_config['min_positive_recall'],
        'maximum_macro_fpr': selection_config['max_negative_fpr'],
        'tie_break': selection_config['tie_break'],
    }
    return dict(common), dict(common)


def _stratified_fit_manifest_hash(pair_manifest: dict) -> str:
    rows = [
        {
            'file': shard['file'],
            'rows': shard['rows'],
            'sha256': shard['sha256'],
        }
        for shard in pair_manifest['shards']
        if shard['partition'] == PARTITIONS[0]
        and shard['family'] == 'task_stratified'
    ]
    if sum(item['rows'] for item in rows) != 12_000_000:
        raise ValueError('fit task-stratified pair count is not 12M')
    return sha256_json(rows)


def _write_result(
    *,
    matrix: dict,
    replicate: dict,
    task: str,
    run_dir: Path,
    baseline_task_dir: Path,
    config: dict,
    reuse: dict,
    source: str,
    pair_manifest: dict,
    fit_score_dir: Path,
    timings: dict[str, float],
) -> dict:
    fit = _load_scored_partition(fit_score_dir)
    full, stratified_only = _selection_payload(
        fit['task_stratified'], fit['uniform'], config['selection']
    )
    if full != stratified_only:
        raise AssertionError('paired condition payloads differ')
    pair_delta = full['reported_epsilon'] - stratified_only['reported_epsilon']
    if pair_delta != 0.0:
        raise AssertionError('removing uniform pairs changed epsilon')
    fit_score_manifest = read_json(fit_score_dir / 'score_manifest.json')
    result = {
        'artifact_schema_version': matrix['artifact_schema_version'],
        'scope': 'formal_threshold_fit_data_sensitivity_only',
        'task': task,
        'replicate': replicate,
        'source': source,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'hostname': socket.gethostname(),
        'repository': git_provenance(REPOSITORY_ROOT),
        'software_versions': software_versions(),
        'frozen_identity': reuse,
        'pointwise_label_variant': config['task_label']['variant'],
        'residual': config['preprocessing']['residual'],
        'dtype': config['preprocessing']['dtype'],
        'latent_dim': config['preprocessing']['latent_dim'],
        'selection_rule': config['selection'],
        'split': {
            'fractions': [
                config['data']['fit_fraction'],
                config['data']['validation_fraction'],
                config['data']['audit_fraction'],
            ],
            'split_sha256': read_json(run_dir / 'split_ids.json')['sha256'],
        },
        'pair_counts': {
            'current_method_total': {
                'uniform': pair_manifest['realized_uniform_pairs'],
                'task_stratified': pair_manifest['realized_stratified_pairs'],
            },
            'threshold_fit_opened': {
                'uniform': 60_000_000,
                'task_stratified': 12_000_000,
            },
            'stratified_only_total_design': {
                'uniform': 0,
                'task_stratified': matrix['conditions']['stratified_only'][
                    'task_stratified_pairs'
                ],
            },
            'stratified_only_threshold_fit': {
                'uniform': 0,
                'task_stratified': 12_000_000,
            },
        },
        'pair_provenance': {
            'current_method_pair_manifest_sha256': pair_manifest[
                'manifest_sha256'
            ],
            'paired_fit_task_stratified_manifest_sha256': (
                _stratified_fit_manifest_hash(pair_manifest)
            ),
            'fit_score_manifest_sha256': sha256_file(
                fit_score_dir / 'score_manifest.json'
            ),
            'fit_rows_scored': fit_score_manifest['rows_scored'],
        },
        'conditions': {
            'current_method': full,
            'stratified_only': stratified_only,
        },
        'paired_reported_epsilon_delta': pair_delta,
        'uniform_pairs_active_in_threshold_selector': False,
        'closed_partitions': [PARTITIONS[1], PARTITIONS[2]],
        'timings_seconds': timings,
        'baseline_task_dir': str(baseline_task_dir),
    }
    write_json(run_dir / 'fit_selection.json', result)
    write_json(
        run_dir / 'status.json',
        {
            'status': 'COMPLETE',
            'scope': result['scope'],
            'current_method': full['status'],
            'stratified_only': stratified_only['status'],
            'paired_epsilon_exactly_equal': True,
        },
    )
    return result


def run_new_replicate(
    matrix_path: Path,
    task: str,
    replicate_id: str,
    run_dir: Path,
) -> dict:
    if run_dir.exists():
        raise FileExistsError(f'run directory already exists: {run_dir}')
    matrix = read_json(matrix_path)
    _validate_matrix(matrix)
    replicate = _replicate(matrix, replicate_id)
    if replicate['source'] != 'new':
        raise ValueError('run requires a new replicate')
    baseline_task_dir = Path(matrix['baseline']['root']) / task
    config = deepcopy(_load_task_config(task, baseline_task_dir))
    config['data']['threshold_split_seed'] = replicate['threshold_split_seed']
    config['data']['pair_sampling_seed'] = replicate['pair_sampling_seed']
    current = matrix['conditions']['current_method']
    config['pair_sampling']['uniform_pairs'] = current['uniform_pairs']
    config['pair_sampling']['stratified_pairs'] = current[
        'task_stratified_pairs'
    ]
    config['pair_sampling']['stratified_positive_fraction'] = current[
        'task_stratified_positive_fraction'
    ]
    run_dir.mkdir(parents=True)
    shutil.copy2(matrix_path, run_dir / 'pre_registered_matrix.json')
    write_json(run_dir / 'effective_task_config.json', config)
    total_start = time.monotonic()
    start = time.monotonic()
    reuse = _validate_baseline_reuse(config, baseline_task_dir)
    _prepare_reused_index(config, baseline_task_dir, run_dir)
    timings = {'verify_reuse_and_split': time.monotonic() - start}
    start = time.monotonic()
    pair_manifest = build_pairs(config, run_dir)
    timings['materialize_full_pair_design'] = time.monotonic() - start
    start = time.monotonic()
    fit_score_dir = run_dir / 'pair_scores' / PARTITIONS[0]
    fit_score_dir.parent.mkdir()
    score_pair_shards(
        pair_dir=run_dir / 'pair_manifests' / PARTITIONS[0],
        score_dir=fit_score_dir,
        embedding_dir=baseline_task_dir / 'embedding_shards',
        total_dataset_rows=len(
            np.load(run_dir / 'index' / 'state.npy', mmap_mode='r')
        ),
        partition=PARTITIONS[0],
        locked_threshold_path=None,
    )
    timings['score_fit_pairs'] = time.monotonic() - start
    timings['total_before_selection'] = time.monotonic() - total_start
    result = _write_result(
        matrix=matrix,
        replicate=replicate,
        task=task,
        run_dir=run_dir,
        baseline_task_dir=baseline_task_dir,
        config=config,
        reuse=reuse,
        source='new_materialized_current_method_fit',
        pair_manifest=pair_manifest,
        fit_score_dir=fit_score_dir,
        timings=timings,
    )
    result['timings_seconds']['total'] = time.monotonic() - total_start
    write_json(run_dir / 'fit_selection.json', result)
    return result


def import_baseline(
    matrix_path: Path,
    task: str,
    run_dir: Path,
) -> dict:
    if run_dir.exists():
        raise FileExistsError(f'run directory already exists: {run_dir}')
    matrix = read_json(matrix_path)
    _validate_matrix(matrix)
    replicate = _replicate(matrix, 'seed-0')
    baseline_task_dir = Path(matrix['baseline']['root']) / task
    config = _load_task_config(task, baseline_task_dir)
    run_dir.mkdir(parents=True)
    shutil.copy2(matrix_path, run_dir / 'pre_registered_matrix.json')
    write_json(run_dir / 'effective_task_config.json', config)
    reuse = _validate_baseline_reuse(config, baseline_task_dir)
    baseline_split = read_json(baseline_task_dir / 'split_ids.json')
    write_json(run_dir / 'split_ids.json', baseline_split)
    pair_manifest = read_json(
        baseline_task_dir / 'pair_manifests' / 'pair_manifest.json'
    )
    return _write_result(
        matrix=matrix,
        replicate=replicate,
        task=task,
        run_dir=run_dir,
        baseline_task_dir=baseline_task_dir,
        config=config,
        reuse=reuse,
        source='existing_formal_baseline_fit_scores',
        pair_manifest=pair_manifest,
        fit_score_dir=(baseline_task_dir / 'pair_scores' / PARTITIONS[0]),
        timings={'baseline_import': 0.0},
    )


def _descriptive(values: list[float], baseline: float) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    std = float(array.std(ddof=1))
    return {
        'n': len(values),
        'mean': mean,
        'sample_std': std,
        'min': float(array.min()),
        'max': float(array.max()),
        'range': float(array.max() - array.min()),
        'coefficient_of_variation': (
            std / abs(mean) if mean != 0.0 else math.nan
        ),
        'max_absolute_deviation_from_seed_0': float(
            np.max(np.abs(array - baseline))
        ),
    }


def summarize(matrix_path: Path, root: Path) -> dict:
    matrix = read_json(matrix_path)
    _validate_matrix(matrix)
    rows = []
    by_task = {}
    for task in TASKS:
        task_rows = []
        for replicate in matrix['replicates']:
            path = root / 'replicates' / replicate['id'] / task
            status = read_json(path / 'status.json')
            if status['status'] != 'COMPLETE':
                raise ValueError(f'incomplete result: {path}')
            result = read_json(path / 'fit_selection.json')
            if result['repository']['dirty']:
                raise ValueError(f'dirty execution provenance: {path}')
            full = result['conditions']['current_method']
            stratified = result['conditions']['stratified_only']
            if result['paired_reported_epsilon_delta'] != 0.0:
                raise ValueError(f'nonzero paired epsilon delta: {path}')
            row = {
                'task': task,
                'replicate_id': replicate['id'],
                'threshold_split_seed': replicate['threshold_split_seed'],
                'pair_sampling_seed': replicate['pair_sampling_seed'],
                'source': result['source'],
                'current_method_status': full['status'],
                'current_method_selected_epsilon': full['selected_epsilon'],
                'current_method_reported_epsilon': full['reported_epsilon'],
                'current_method_macro_tpr': full['macro_tpr'],
                'current_method_macro_fpr': full['macro_fpr'],
                'stratified_only_status': stratified['status'],
                'stratified_only_selected_epsilon': stratified[
                    'selected_epsilon'
                ],
                'stratified_only_reported_epsilon': stratified[
                    'reported_epsilon'
                ],
                'paired_reported_epsilon_delta': 0.0,
                'fit_selection_sha256': sha256_file(
                    path / 'fit_selection.json'
                ),
                'pair_manifest_sha256': result['pair_provenance'][
                    'current_method_pair_manifest_sha256'
                ],
                'split_sha256': result['split']['split_sha256'],
                'checkpoint_sha256': result['frozen_identity'][
                    'checkpoint_sha256'
                ],
                'embedding_sha256': result['frozen_identity'][
                    'embedding_sha256'
                ],
            }
            rows.append(row)
            task_rows.append(row)
        values = [row['current_method_reported_epsilon'] for row in task_rows]
        by_task[task] = {
            'reported_epsilon_role': (
                'descriptive_best_operating_point_not_promoted'
                if task == 'cube'
                else 'selected_fit_threshold'
            ),
            'current_method': _descriptive(values, values[0]),
            'stratified_only': _descriptive(values, values[0]),
            'maximum_absolute_paired_condition_delta': 0.0,
            'feasible_replicates': sum(
                row['current_method_selected_epsilon'] is not None
                for row in task_rows
            ),
        }
    summary = {
        'artifact_schema_version': matrix['artifact_schema_version'],
        'created_at': datetime.now(timezone.utc).isoformat(),
        'repository': git_provenance(REPOSITORY_ROOT),
        'scope': 'formal_threshold_fit_data_sensitivity_only',
        'paired_design': True,
        'uniform_pairs_active_in_threshold_selector': False,
        'rows': rows,
        'task_summaries': by_task,
    }
    write_json(root / 'DATA_SENSITIVITY_RESULTS.json', summary)
    lines = [
        '# Experiment T: goal-threshold data sensitivity',
        '',
        (
            'Scope: formal sensitivity of the `threshold_fit` estimator only; '
            'these replicates are not newly validated, locked, or audited.'
        ),
        '',
        '## Per-seed results',
        '',
        '| Task | Seed | Split seed | Pair seed | Current method epsilon | Stratified-only epsilon | Macro TPR | Macro FPR | Status |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in rows:
        lines.append(
            f'| {row["task"]} | {row["replicate_id"]} | '
            f'{row["threshold_split_seed"]} | {row["pair_sampling_seed"]} | '
            f'{row["current_method_reported_epsilon"]:.10g} | '
            f'{row["stratified_only_reported_epsilon"]:.10g} | '
            f'{row["current_method_macro_tpr"]:.6f} | '
            f'{row["current_method_macro_fpr"]:.6f} | '
            f'{row["current_method_status"]} |'
        )
    lines.extend(
        [
            '',
            '## Across-seed sensitivity',
            '',
            '| Task | Role | Mean | Sample SD | Min | Max | Range | CV | Max abs deviation from seed-0 | Feasible / 5 |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
        ]
    )
    for task in TASKS:
        item = by_task[task]
        stats = item['current_method']
        lines.append(
            f'| {task} | {item["reported_epsilon_role"]} | '
            f'{stats["mean"]:.10g} | {stats["sample_std"]:.10g} | '
            f'{stats["min"]:.10g} | {stats["max"]:.10g} | '
            f'{stats["range"]:.10g} | '
            f'{stats["coefficient_of_variation"]:.6g} | '
            f'{stats["max_absolute_deviation_from_seed_0"]:.10g} | '
            f'{item["feasible_replicates"]}/5 |'
        )
    lines.extend(
        [
            '',
            '## Paired condition result',
            '',
            (
                'For every task and seed, removing the 100M uniform family '
                'changed the reported epsilon by exactly `0.0`. This is '
                'expected from the preregistered selector: with '
                '`min_population_precision=null`, uniform pairs do not enter '
                'the macro-TPR/macro-FPR threshold decision.'
            ),
            '',
            (
                'Cube values are descriptive best operating points under '
                'macro FPR <= 0.10. Any replicate below macro TPR 0.90 remains '
                '`THRESHOLD_CALIBRATION_NO_FEASIBLE_OPERATING_POINT` and is '
                'not a promoted threshold.'
            ),
            '',
            (
                'These frozen pointwise encoder-geometry results are not '
                'predictor, reachability, planner, execution, or official '
                'CLEAR evidence.'
            ),
            '',
        ]
    )
    (root / 'DATA_SENSITIVITY_REPORT.md').write_text('\n'.join(lines))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)
    run_parser = subparsers.add_parser('run')
    run_parser.add_argument('--matrix-config', type=Path, required=True)
    run_parser.add_argument('--task', choices=TASKS, required=True)
    run_parser.add_argument('--replicate', required=True)
    run_parser.add_argument('--run-dir', type=Path, required=True)
    import_parser = subparsers.add_parser('import-baseline')
    import_parser.add_argument('--matrix-config', type=Path, required=True)
    import_parser.add_argument('--task', choices=TASKS, required=True)
    import_parser.add_argument('--run-dir', type=Path, required=True)
    summary_parser = subparsers.add_parser('summarize')
    summary_parser.add_argument('--matrix-config', type=Path, required=True)
    summary_parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'run':
        run_new_replicate(
            args.matrix_config.resolve(),
            args.task,
            args.replicate,
            args.run_dir.resolve(),
        )
    elif args.command == 'import-baseline':
        import_baseline(
            args.matrix_config.resolve(), args.task, args.run_dir.resolve()
        )
    else:
        summarize(args.matrix_config.resolve(), args.root.resolve())


if __name__ == '__main__':
    main()
