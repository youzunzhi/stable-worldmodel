"""Run Experiment T with fit/validation/lock/audit access boundaries.

Example smoke:

    python -m scripts.experiments.observation_goal_threshold.run \
      --config scripts/experiments/observation_goal_threshold/configs/pusht.json \
      --run-dir /tmp/experiment-t-pusht-smoke --smoke
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import LABEL_NAMES, TaskContract
from .encode import encode_observations, score_pair_shards
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
    attach_bootstrap_cis,
    bootstrap_epsilon_stability,
    bootstrap_uniform_cis,
    enforce_validation,
    metrics_at_threshold,
    select_threshold,
)
from .sample_pairs import (
    sample_stratified_partition,
    uniform_ordered_pairs,
)
from .split import PARTITIONS, row_partitions, split_groups

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _partition_dictionary(name: str, length: int):
    import pyarrow as pa

    return pa.DictionaryArray.from_arrays(
        pa.array(np.zeros(length, dtype=np.int8)), pa.array([name])
    )


def _pair_table(data: dict[str, np.ndarray], partition: str, family: str):
    import pyarrow as pa

    length = len(data['pair_id'])
    columns = {
        'pair_id': pa.array(data['pair_id'], type=pa.uint64()),
        'partition': _partition_dictionary(partition, length),
        'sample_family': _partition_dictionary(family, length),
        'anchor_row': pa.array(data['anchor_row'], type=pa.int64()),
        'goal_row': pa.array(data['goal_row'], type=pa.int64()),
        'anchor_group': pa.array(data['anchor_group'], type=pa.int64()),
        'goal_group': pa.array(data['goal_group'], type=pa.int64()),
        'anchor_episode': pa.array(data['anchor_episode'], type=pa.int64()),
        'goal_episode': pa.array(data['goal_episode'], type=pa.int64()),
        'task_error': pa.array(data['task_error'], type=pa.float32()),
        'label': pa.array(data['label'], type=pa.uint8()),
        'negative_stratum': pa.array(data['negative_stratum'], type=pa.int8()),
        'sampling_probability': pa.array(
            data['sampling_probability'], type=pa.float64()
        ),
        'analysis_weight': pa.array(
            data['analysis_weight'], type=pa.float64()
        ),
        'same_trajectory': pa.array(
            data['anchor_episode'] == data['goal_episode']
        ),
    }
    return pa.table(columns)


def _write_pair_shard(
    path: Path,
    data: dict[str, np.ndarray],
    partition: str,
    family: str,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    table = _pair_table(data, partition, family)
    pq.write_table(table, path, compression='zstd')
    return {
        'file': path.name,
        'rows': len(table),
        'sha256': sha256_file(path),
    }


def _allocate(total: int, fractions: tuple[float, float, float]) -> list[int]:
    first = round(total * fractions[0])
    second = round(total * fractions[1])
    return [first, second, total - first - second]


def _read_config(path: Path, smoke: bool) -> dict:
    config = read_json(path)
    if smoke:
        config = json.loads(json.dumps(config))
        config['pair_sampling']['uniform_pairs'] = 12000
        config['pair_sampling']['stratified_pairs'] = 6000
        config['pair_sampling']['pair_shard_rows'] = 4000
        config['analysis']['bootstrap_replicates'] = 100
        config['runtime']['encode_batch_size'] = 64
        config['smoke'] = {
            'enabled': True,
            'groups_per_partition': 8,
            'formal_evidence': False,
        }
    else:
        config['smoke'] = {'enabled': False, 'formal_evidence': True}
    return config


def _dataset_full_hash(path: Path, mode: str) -> str:
    if mode != 'full_file':
        raise ValueError('formal configs require full_file dataset hashing')
    return sha256_file(path)


def prepare_index(config: dict, run_dir: Path) -> dict:
    import h5py
    import hdf5plugin  # noqa: F401
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml

    started = time.monotonic()
    dataset_config = config['dataset']
    dataset_path = Path(dataset_config['path'])
    checkpoint_path = Path(config['checkpoint']['path'])
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if dataset_path.stat().st_size != dataset_config['expected_bytes']:
        raise ValueError('dataset byte-size mismatch')
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if sha256_file(checkpoint_path) != config['checkpoint']['sha256']:
        raise ValueError('checkpoint SHA-256 mismatch')
    if (
        sha256_file(checkpoint_path.parent / 'config.json')
        != config['checkpoint']['config_sha256']
    ):
        raise ValueError('checkpoint config SHA-256 mismatch')

    with h5py.File(dataset_path, 'r') as dataset:
        state_column = dataset_config['state_column']
        columns = {
            key: {
                'shape': list(value.shape),
                'dtype': str(value.dtype),
                'chunks': list(value.chunks) if value.chunks else None,
                'compression': str(value.compression),
            }
            for key, value in dataset.items()
        }
        required = {
            dataset_config['pixels_column'],
            state_column,
            dataset_config['episode_column'],
            dataset_config['step_column'],
        }
        missing = required - set(dataset.keys())
        if missing:
            raise ValueError(f'missing required dataset columns: {missing}')
        row_count = len(dataset[state_column])
        if dataset[dataset_config['pixels_column']].shape != (
            row_count,
            224,
            224,
            3,
        ):
            raise ValueError('unexpected pixel shape')
        state = dataset[state_column][:]
        indices = dataset_config['state_indices']
        state = np.asarray(state[:, indices], dtype=np.float32)
        episodes = np.asarray(
            dataset[dataset_config['episode_column']][:], dtype=np.int64
        )
        steps = np.asarray(
            dataset[dataset_config['step_column']][:], dtype=np.int64
        )
    if not np.isfinite(state).all():
        raise ValueError('required task state has non-finite rows')
    if dataset_config['group_rule'] == 'episode_idx // 101':
        groups = episodes // int(dataset_config['variants_per_source'])
        for group in np.unique(groups):
            group_episodes = np.unique(episodes[groups == group])
            if len(group_episodes) != dataset_config['variants_per_source']:
                raise ValueError(
                    'PushT source group does not contain 101 variants'
                )
    elif dataset_config['group_rule'] == 'episode':
        groups = episodes.copy()
    else:
        raise ValueError(f'unknown group rule {dataset_config["group_rule"]}')

    fractions = (
        config['data']['fit_fraction'],
        config['data']['validation_fraction'],
        config['data']['audit_fraction'],
    )
    group_split = split_groups(
        groups, config['data']['threshold_split_seed'], fractions
    )
    if config['smoke']['enabled']:
        limit = config['smoke']['groups_per_partition']
        group_split = {
            key: value[:limit] for key, value in group_split.items()
        }
        keep_groups = np.concatenate(list(group_split.values()))
        eligible = np.isin(groups, keep_groups)
        selected_rows = np.flatnonzero(eligible)
        selected_partition = np.full(row_count, -1, dtype=np.int8)
        for index, name in enumerate(PARTITIONS):
            selected_partition[np.isin(groups, group_split[name])] = index
    else:
        selected_rows = np.arange(row_count, dtype=np.int64)
        selected_partition = row_partitions(groups, group_split)
    if np.any(selected_partition[selected_rows] < 0):
        raise ValueError('eligible row without partition')

    dataset_hash = _dataset_full_hash(
        dataset_path, dataset_config['hash_mode']
    )
    index_dir = run_dir / 'index'
    index_dir.mkdir(parents=True, exist_ok=False)
    np.save(index_dir / 'state.npy', state, allow_pickle=False)
    np.save(index_dir / 'episodes.npy', episodes, allow_pickle=False)
    np.save(index_dir / 'steps.npy', steps, allow_pickle=False)
    np.save(index_dir / 'groups.npy', groups, allow_pickle=False)
    np.save(
        index_dir / 'row_partitions.npy',
        selected_partition,
        allow_pickle=False,
    )
    np.save(index_dir / 'eligible_rows.npy', selected_rows, allow_pickle=False)

    split_payload = {
        name: [int(v) for v in group_split[name]] for name in PARTITIONS
    }
    split_payload['sha256'] = sha256_json(split_payload)
    write_json(run_dir / 'split_ids.json', split_payload)
    partition_names = np.asarray(PARTITIONS, dtype=object)[
        selected_partition[selected_rows]
    ]
    observation_table = pa.table(
        {
            'observation_id': pa.array(selected_rows, type=pa.int64()),
            'partition': pa.array(partition_names).dictionary_encode(),
            'raw_group_id': pa.array(groups[selected_rows], type=pa.int64()),
            'episode_id': pa.array(episodes[selected_rows], type=pa.int64()),
            'row_id': pa.array(selected_rows, type=pa.int64()),
            'step_id': pa.array(steps[selected_rows], type=pa.int64()),
            'observation_ref_column': _partition_dictionary(
                dataset_config['pixels_column'], len(selected_rows)
            ),
            'checkpoint_hash': _partition_dictionary(
                config['checkpoint']['sha256'], len(selected_rows)
            ),
            'dataset_hash': _partition_dictionary(
                dataset_hash, len(selected_rows)
            ),
        }
    )
    observation_path = run_dir / 'observation_records.parquet'
    pq.write_table(observation_table, observation_path, compression='zstd')
    observation_manifest_hash = sha256_file(observation_path)
    inventory = {
        'path': str(dataset_path),
        'bytes': dataset_path.stat().st_size,
        'rows': row_count,
        'eligible_rows': len(selected_rows),
        'episodes': len(np.unique(episodes)),
        'raw_groups': len(np.unique(groups)),
        'pixel_shape': [224, 224, 3],
        'columns': columns,
        'dataset_sha256': dataset_hash,
        'dataset_hash_mode': dataset_config['hash_mode'],
        'observation_manifest_sha256': observation_manifest_hash,
        'exclusions': {},
    }
    write_json(run_dir / 'dataset_inventory.json', inventory)
    mapping = {
        'checkpoint_loading': 'stable_worldmodel.wm.utils.load_pretrained',
        'encoder_forward': 'model.encoder(..., interpolate_pos_encoding=True)',
        'projector_forward': 'model.projector(last_hidden_state[:, 0])',
        'observation_preprocessing': config['preprocessing'],
        'latent_dim': config['preprocessing']['latent_dim'],
        'dtype': config['preprocessing']['dtype'],
        'dataset_reader': 'h5py.File read-only',
        'row_id': 'HDF5 global row index',
        'episode_column': dataset_config['episode_column'],
        'source_group_rule': dataset_config['group_rule'],
        'task_state_columns': config['task_label']['state_fields'],
        'task_error': config['task_label']['metric'],
        'compatibility_signature': config['compatibility_signature'],
        'compatibility_signature_sha256': sha256_json(
            config['compatibility_signature']
        ),
        'split_builder': 'split.split_groups then pair construction',
        'downstream_validator': 'selected threshold full-tuple equality',
    }
    write_json(run_dir / 'repository_mapping.json', mapping)
    manifest = {
        **config,
        'run': {
            'created_at': datetime.now(timezone.utc).isoformat(),
            'hostname': socket.gethostname(),
            'repository': git_provenance(REPOSITORY_ROOT),
            'config_sha256': sha256_json(config),
        },
    }
    with (run_dir / 'manifest.yaml').open('w') as stream:
        yaml.safe_dump(manifest, stream, sort_keys=True)
    index_manifest = {
        'dataset_sha256': dataset_hash,
        'observation_manifest_sha256': observation_manifest_hash,
        'split_sha256': split_payload['sha256'],
        'eligible_rows': len(selected_rows),
        'state_sha256': sha256_file(index_dir / 'state.npy'),
        'groups_sha256': sha256_file(index_dir / 'groups.npy'),
        'elapsed_seconds': time.monotonic() - started,
    }
    write_json(index_dir / 'index_manifest.json', index_manifest)
    return index_manifest


def _complete_pair_data(
    base: dict[str, np.ndarray],
    *,
    episodes: np.ndarray,
    groups: np.ndarray,
    probability: float,
) -> dict[str, np.ndarray]:
    count = len(base['pair_id'])
    result = dict(base)
    result.setdefault('anchor_group', groups[result['anchor_row']])
    result.setdefault('goal_group', groups[result['goal_row']])
    result['anchor_episode'] = episodes[result['anchor_row']]
    result['goal_episode'] = episodes[result['goal_row']]
    result['sampling_probability'] = np.full(
        count, probability, dtype=np.float64
    )
    return result


def build_pairs(config: dict, run_dir: Path) -> dict:
    started = time.monotonic()
    index_dir = run_dir / 'index'
    state = np.load(index_dir / 'state.npy', mmap_mode='r')
    episodes = np.load(index_dir / 'episodes.npy', mmap_mode='r')
    groups = np.load(index_dir / 'groups.npy', mmap_mode='r')
    partitions = np.load(index_dir / 'row_partitions.npy', mmap_mode='r')
    eligible = np.load(index_dir / 'eligible_rows.npy', mmap_mode='r')
    contract = TaskContract.from_config(config)
    fractions = (
        config['data']['fit_fraction'],
        config['data']['validation_fraction'],
        config['data']['audit_fraction'],
    )
    uniform_counts = _allocate(
        config['pair_sampling']['uniform_pairs'], fractions
    )
    stratified_counts = _allocate(
        config['pair_sampling']['stratified_pairs'], fractions
    )
    shard_rows = config['pair_sampling']['pair_shard_rows']
    pair_root = run_dir / 'pair_manifests'
    pair_root.mkdir(parents=True, exist_ok=False)
    all_shards = []
    sampling_audit = {}
    for partition_index, partition_name in enumerate(PARTITIONS):
        partition_rows = eligible[partitions[eligible] == partition_index]
        population = len(partition_rows) * (len(partition_rows) - 1)
        partition_dir = pair_root / partition_name
        partition_dir.mkdir()
        uniform_target = uniform_counts[partition_index]
        permutation_record = None
        uniform_label_counts = {name: 0 for name in LABEL_NAMES.values()}
        for shard_index, start in enumerate(
            range(0, uniform_target, shard_rows)
        ):
            count = min(shard_rows, uniform_target - start)
            anchor, goal, permutation = uniform_ordered_pairs(
                partition_rows,
                start=start,
                count=count,
                total_count=uniform_target,
                seed=config['data']['pair_sampling_seed']
                + partition_index * 1_000_003,
            )
            error = contract.task_error(state[anchor], state[goal])
            labels = contract.classify(error)
            strata = contract.negative_strata(error)
            for value, name in LABEL_NAMES.items():
                uniform_label_counts[name] += int((labels == value).sum())
            pair_id = anchor.astype(np.uint64) * np.uint64(
                len(state)
            ) + goal.astype(np.uint64)
            data = _complete_pair_data(
                {
                    'pair_id': pair_id,
                    'anchor_row': anchor,
                    'goal_row': goal,
                    'task_error': error,
                    'label': labels,
                    'negative_stratum': strata,
                    'analysis_weight': np.full(
                        count, population / uniform_target, dtype=np.float64
                    ),
                },
                episodes=episodes,
                groups=groups,
                probability=uniform_target / population,
            )
            path = partition_dir / f'uniform-{shard_index:05d}.parquet'
            all_shards.append(
                {
                    'partition': partition_name,
                    'family': 'uniform',
                    **_write_pair_shard(path, data, partition_name, 'uniform'),
                }
            )
            permutation_record = {
                'domain': permutation.domain,
                'multiplier': permutation.multiplier,
                'offset': permutation.offset,
                'prefix': permutation.max_items,
            }

        stratified, stratum_audit = sample_stratified_partition(
            partition_rows=partition_rows,
            all_states=state,
            all_groups=groups,
            contract=contract,
            total_count=stratified_counts[partition_index],
            seed=config['data']['pair_sampling_seed']
            + 7_000_021
            + partition_index * 1_000_003,
            total_dataset_rows=len(state),
        )
        stratified = _complete_pair_data(
            stratified,
            episodes=episodes,
            groups=groups,
            probability=float('nan'),
        )
        for shard_index, start in enumerate(
            range(0, len(stratified['pair_id']), shard_rows)
        ):
            stop = min(start + shard_rows, len(stratified['pair_id']))
            data = {
                key: value[start:stop] for key, value in stratified.items()
            }
            path = partition_dir / f'task_stratified-{shard_index:05d}.parquet'
            all_shards.append(
                {
                    'partition': partition_name,
                    'family': 'task_stratified',
                    **_write_pair_shard(
                        path, data, partition_name, 'task_stratified'
                    ),
                }
            )
        sampling_audit[partition_name] = {
            'eligible_rows': len(partition_rows),
            'ordered_pair_population': int(population),
            'uniform_target': uniform_target,
            'uniform_realized': uniform_target,
            'uniform_sampling_probability': uniform_target / population,
            'uniform_label_counts': uniform_label_counts,
            'uniform_permutation': permutation_record,
            'stratified_target': stratified_counts[partition_index],
            'stratified_realized': stratified_counts[partition_index],
            'strata': stratum_audit,
        }
    manifest = {
        'task': config['task'],
        'target_uniform_pairs': config['pair_sampling']['uniform_pairs'],
        'target_stratified_pairs': config['pair_sampling']['stratified_pairs'],
        'realized_uniform_pairs': sum(uniform_counts),
        'realized_stratified_pairs': sum(stratified_counts),
        'sampling': sampling_audit,
        'shards': all_shards,
        'elapsed_seconds': time.monotonic() - started,
    }
    manifest['manifest_sha256'] = sha256_json(manifest)
    write_json(pair_root / 'pair_manifest.json', manifest)
    prevalence = {
        partition: audit['uniform_label_counts']
        for partition, audit in sampling_audit.items()
    }
    write_json(run_dir / 'task_label_prevalence.json', prevalence)
    return manifest


def _load_scored_partition(
    score_dir: Path,
) -> dict[str, dict[str, np.ndarray]]:
    import pyarrow.parquet as pq

    columns = [
        'anchor_group',
        'anchor_episode',
        'goal_episode',
        'task_error',
        'label',
        'negative_stratum',
        'analysis_weight',
        'latent_distance',
    ]
    by_family: dict[str, dict[str, list[np.ndarray]]] = {
        'uniform': {key: [] for key in columns},
        'task_stratified': {key: [] for key in columns},
    }
    for path in sorted(score_dir.glob('*.parquet')):
        family = (
            'task_stratified'
            if path.name.startswith('task_stratified-')
            else 'uniform'
        )
        table = pq.read_table(path, columns=columns)
        for key in columns:
            by_family[family][key].append(table[key].to_numpy())
    return {
        family: {key: np.concatenate(parts) for key, parts in values.items()}
        for family, values in by_family.items()
    }


def _write_candidates(path: Path, candidates: dict[str, np.ndarray]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    writer = None
    chunk = 1_000_000
    try:
        for start in range(0, len(candidates['epsilon']), chunk):
            stop = min(start + chunk, len(candidates['epsilon']))
            table = pa.table(
                {
                    key: pa.array(value[start:stop])
                    for key, value in candidates.items()
                }
            )
            if writer is None:
                writer = pq.ParquetWriter(
                    path, table.schema, compression='zstd'
                )
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def _clean_metrics(metrics: dict) -> dict:
    return {
        key: value for key, value in metrics.items() if not key.startswith('_')
    }


def _selection_kwargs(config: dict) -> dict[str, float | None]:
    return {
        'min_positive_recall': config['selection']['min_positive_recall'],
        'max_negative_fpr': config['selection']['max_negative_fpr'],
        'min_population_precision': config['selection'][
            'min_population_precision'
        ],
    }


def _analyze_partition(
    scored: dict[str, dict[str, np.ndarray]],
    epsilon: float,
    config: dict,
    *,
    seed_offset: int,
) -> dict:
    metrics = metrics_at_threshold(
        scored['task_stratified'], scored['uniform'], epsilon
    )
    metrics = attach_bootstrap_cis(
        metrics,
        config['analysis']['bootstrap_replicates'],
        config['analysis']['analysis_seed'] + seed_offset,
    )
    metrics['group_clustered_95ci'].update(
        bootstrap_uniform_cis(
            scored['uniform'],
            epsilon,
            config['analysis']['bootstrap_replicates'],
            config['analysis']['analysis_seed'] + seed_offset + 17,
        )
    )
    return _clean_metrics(metrics)


def _lock_threshold(
    config: dict,
    run_dir: Path,
    epsilon: float,
    fit_metrics: dict,
    validation_metrics: dict,
) -> tuple[dict, str]:
    index = read_json(run_dir / 'index' / 'index_manifest.json')
    embeddings = read_json(
        run_dir / 'embedding_shards' / 'embedding_manifest.json'
    )
    pairs = read_json(run_dir / 'pair_manifests' / 'pair_manifest.json')
    artifact = {
        'artifact_schema_version': config['artifact_schema_version'],
        'epsilon': epsilon,
        'task': config['task'],
        'pointwise_label_variant': config['task_label']['variant'],
        'task_state_fields': config['task_label']['state_fields'],
        'task_error_definition_and_unit': {
            'metric': config['task_label']['metric'],
            'unit': config['task_label']['unit'],
        },
        'positive_and_negative_thresholds_with_boundary_semantics': {
            'positive': f'task_error < {config["task_label"]["positive_if_lt"]}',
            'negative': f'task_error > {config["task_label"]["negative_if_gt"]}',
        },
        'ignored_region_definition': (
            f'{config["task_label"]["positive_if_lt"]} <= task_error <= '
            f'{config["task_label"]["negative_if_gt"]}'
        ),
        'residual_definition': config['preprocessing']['residual'],
        'D': config['preprocessing']['latent_dim'],
        'dtype': config['preprocessing']['dtype'],
        'encoder_checkpoint_path': config['checkpoint']['path'],
        'encoder_checkpoint_sha256': config['checkpoint']['sha256'],
        'encoder_checkpoint_provenance': config['checkpoint']['provenance'],
        'encoder_projector_parameter_hashes': {
            'before': embeddings['encoder_projector_parameter_hash_before'],
            'after': embeddings['encoder_projector_parameter_hash_after'],
        },
        'dataset_version': {
            'repository': config['dataset']['repository'],
            'revision': config['dataset']['revision'],
            'sha256': index['dataset_sha256'],
        },
        'observation_manifest_hash': index['observation_manifest_sha256'],
        'fit_and_validation_split_hashes': index['split_sha256'],
        'uniform_and_stratified_pair_sample_hashes': pairs['manifest_sha256'],
        'selection_rule_and_constraints': config['selection'],
        'fit_metrics': fit_metrics,
        'validation_metrics': validation_metrics,
        'observation_preprocessing': config['preprocessing'],
        'encoder_projector_symbol_mapping': read_json(
            run_dir / 'repository_mapping.json'
        ),
        'compatibility_signature': config['compatibility_signature'],
        'git_commit_and_dirty_status': git_provenance(REPOSITORY_ROOT),
        'software_versions': software_versions(),
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }
    path = run_dir / 'selected_threshold.json'
    write_json(path, artifact)
    digest = sha256_file(path)
    write_json(
        run_dir / 'threshold_lock.json',
        {
            'selected_threshold_sha256': digest,
            'locked_at': artifact['timestamp'],
        },
    )
    return artifact, digest


def _verify_lock(run_dir: Path, expected: str) -> None:
    actual = sha256_file(run_dir / 'selected_threshold.json')
    lock = read_json(run_dir / 'threshold_lock.json')
    if actual != expected or lock['selected_threshold_sha256'] != expected:
        raise ValueError('selected threshold changed after lock')


def _plot_outputs(
    run_dir: Path,
    selection,
    fit: dict,
    audit: dict,
    epsilon_distribution: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    candidates = selection.candidates
    finite = np.isfinite(candidates['epsilon'])
    index = np.flatnonzero(finite)
    if len(index) > 20_000:
        index = index[np.linspace(0, len(index) - 1, 20_000).astype(int)]
    plt.figure(figsize=(7, 5))
    plt.plot(candidates['macro_fpr'][index], candidates['macro_tpr'][index])
    plt.xlabel('anchor-group macro FPR')
    plt.ylabel('anchor-group macro TPR')
    plt.tight_layout()
    plt.savefig(run_dir / 'roc_curve.png', dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(
        candidates['macro_tpr'][index],
        candidates['population_precision'][index],
    )
    plt.xlabel('anchor-group macro TPR')
    plt.ylabel('uniform population precision')
    plt.tight_layout()
    plt.savefig(run_dir / 'precision_recall_curve.png', dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    for name, data in (('fit', fit), ('audit', audit)):
        values = data['uniform']['latent_distance']
        plt.hist(values, bins=200, density=True, histtype='step', label=name)
    plt.axvline(selection.epsilon, color='black', linestyle='--')
    plt.xlabel('mean-D latent MSE')
    plt.ylabel('density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / 'score_distributions.png', dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    finite_epsilon = epsilon_distribution[np.isfinite(epsilon_distribution)]
    if len(finite_epsilon):
        plt.hist(finite_epsilon, bins=80)
    plt.axvline(selection.epsilon, color='black', linestyle='--')
    plt.xlabel('bootstrap-selected epsilon (histogram approximation)')
    plt.ylabel('replicates')
    plt.tight_layout()
    plt.savefig(run_dir / 'threshold_stability.png', dpi=160)
    plt.close()


def _report_success(
    config: dict,
    run_dir: Path,
    artifact: dict,
    fit_metrics: dict,
    validation_metrics: dict,
    audit_metrics: dict,
    stability: dict,
    timings: dict,
) -> str:
    task = config['task']
    lines = [
        f'# Experiment T report: {task}',
        '',
        f'- Status: **PROMOTABLE for `{config["task_label"]["variant"]}` only**',
        f'- Locked threshold epsilon: `{artifact["epsilon"]:.10g}` mean-D latent MSE',
        f'- Checkpoint SHA-256: `{config["checkpoint"]["sha256"]}`',
        f'- Dataset SHA-256: `{artifact["dataset_version"]["sha256"]}`',
        f'- Residual: `{config["preprocessing"]["residual"]}` in float32, D=192',
        f'- Label gap: T `< {config["task_label"]["positive_if_lt"]}`; F `> {config["task_label"]["negative_if_gt"]}` {config["task_label"]["unit"]}',
        '',
        '## Metrics',
        '',
        '| Partition | Macro TPR | Macro FPR | Population precision | U occupancy |',
        '|---|---:|---:|---:|---:|',
    ]
    for name, metrics in (
        ('fit', fit_metrics),
        ('validation', validation_metrics),
        ('audit', audit_metrics),
    ):
        lines.append(
            f'| {name} | {metrics["macro_tpr"]:.6f} | {metrics["macro_fpr"]:.6f} | '
            f'{metrics["population_precision_direct"]:.6f} | '
            f'{metrics["ignored_band"]["ball_occupancy"]:.6f} |'
        )
    lines.extend(
        [
            '',
            (
                'Direct and prevalence-reconstructed audit precision differ '
                'by '
                f'`{audit_metrics["precision_reconstruction_abs_error"]:.3g}`.'
            ),
            '',
            '## Uncertainty and stability',
            '',
            f'Audit macro TPR 95% CI: `{audit_metrics["group_clustered_95ci"]["macro_tpr"]}`.',
            f'Audit macro FPR 95% CI: `{audit_metrics["group_clustered_95ci"]["macro_fpr"]}`.',
            f'Fit epsilon bootstrap stability: `{stability}`.',
            '',
            '## Boundaries of interpretation',
            '',
            (
                'This is frozen encoder geometry between recorded '
                'observations. It does not test predictor error, reachability, '
                'CEM, environment execution, sustained success, or official '
                'CLEAR Moderate/Strict.'
            ),
            '',
            (
                'The selected checkpoint is the canonical seed-3072 M0 '
                'fallback named in the immutable artifact; it is not the '
                'unavailable former `/ssd` full-demo checkpoint.'
            ),
            '',
            '## Runtime',
            '',
            f'`{json.dumps(timings, sort_keys=True)}`',
            '',
        ]
    )
    report = '\n'.join(lines)
    (run_dir / 'report.md').write_text(report)
    return report


def _report_failure(
    config: dict, run_dir: Path, outcome: CalibrationOutcome, stage: str
) -> None:
    report = '\n'.join(
        [
            f'# Experiment T report: {config["task"]}',
            '',
            f'- Status: **{outcome.code}**',
            f'- Stage: `{stage}`',
            f'- Details: `{json.dumps(outcome.details, sort_keys=True, default=str)}`',
            '',
            (
                'No threshold was promoted, and audit scores were not opened. '
                'The pre-registered task gap and operating constraints were '
                'not relaxed.'
            ),
            '',
        ]
    )
    (run_dir / 'report.md').write_text(report)
    write_json(
        run_dir / 'status.json',
        {
            'status': outcome.code,
            'stage': stage,
            'details': outcome.details,
            'formal_evidence': not config['smoke']['enabled'],
        },
    )


def run_experiment(config_path: Path, run_dir: Path, smoke: bool) -> None:
    if run_dir.exists():
        raise FileExistsError(f'run directory already exists: {run_dir}')
    run_dir.mkdir(parents=True)
    config = _read_config(config_path, smoke)
    shutil.copy2(config_path, run_dir / 'pre_registered_config.json')
    timings: dict[str, float] = {}
    total_started = time.monotonic()

    start = time.monotonic()
    prepare_index(config, run_dir)
    timings['provenance_schema_preflight'] = time.monotonic() - start

    start = time.monotonic()
    build_pairs(config, run_dir)
    timings['pair_construction'] = time.monotonic() - start

    eligible_rows = np.load(
        run_dir / 'index' / 'eligible_rows.npy', allow_pickle=False
    )
    start = time.monotonic()
    encode_observations(
        dataset_path=config['dataset']['path'],
        pixels_column=config['dataset']['pixels_column'],
        row_ids=eligible_rows,
        checkpoint_path=config['checkpoint']['path'],
        expected_checkpoint_sha256=config['checkpoint']['sha256'],
        expected_config_sha256=config['checkpoint']['config_sha256'],
        output_dir=run_dir / 'embedding_shards',
        batch_size=config['runtime']['encode_batch_size'],
        latent_dim=config['preprocessing']['latent_dim'],
    )
    timings['encode_all_observations'] = time.monotonic() - start

    start = time.monotonic()
    score_root = run_dir / 'pair_scores'
    score_root.mkdir()
    fit_score = score_root / PARTITIONS[0]
    score_pair_shards(
        pair_dir=run_dir / 'pair_manifests' / PARTITIONS[0],
        score_dir=fit_score,
        embedding_dir=run_dir / 'embedding_shards',
        total_dataset_rows=len(
            np.load(run_dir / 'index' / 'state.npy', mmap_mode='r')
        ),
        partition=PARTITIONS[0],
        locked_threshold_path=None,
    )
    fit = _load_scored_partition(fit_score)
    try:
        selection = select_threshold(
            fit['task_stratified'], fit['uniform'], **_selection_kwargs(config)
        )
    except CalibrationOutcome as outcome:
        timings['fit_score_and_select'] = time.monotonic() - start
        timings['total'] = time.monotonic() - total_started
        write_json(run_dir / 'timing.json', timings)
        _report_failure(config, run_dir, outcome, 'fit')
        return
    _write_candidates(
        run_dir / 'threshold_candidates.parquet', selection.candidates
    )
    fit_metrics = _analyze_partition(
        fit, selection.epsilon, config, seed_offset=0
    )
    write_json(run_dir / 'fit_metrics.json', fit_metrics)
    stability, epsilon_distribution = bootstrap_epsilon_stability(
        fit['task_stratified'],
        min_positive_recall=config['selection']['min_positive_recall'],
        max_negative_fpr=config['selection']['max_negative_fpr'],
        replicates=config['analysis']['bootstrap_replicates'],
        seed=config['analysis']['analysis_seed'] + 101,
    )
    np.save(
        run_dir / 'threshold_stability_bootstrap.npy',
        epsilon_distribution,
        allow_pickle=False,
    )
    write_json(run_dir / 'threshold_stability.json', stability)
    timings['fit_score_select_analyze'] = time.monotonic() - start

    start = time.monotonic()
    validation_score = score_root / PARTITIONS[1]
    score_pair_shards(
        pair_dir=run_dir / 'pair_manifests' / PARTITIONS[1],
        score_dir=validation_score,
        embedding_dir=run_dir / 'embedding_shards',
        total_dataset_rows=len(
            np.load(run_dir / 'index' / 'state.npy', mmap_mode='r')
        ),
        partition=PARTITIONS[1],
        locked_threshold_path=None,
    )
    validation = _load_scored_partition(validation_score)
    validation_metrics = _analyze_partition(
        validation, selection.epsilon, config, seed_offset=1000
    )
    write_json(run_dir / 'validation_metrics.json', validation_metrics)
    try:
        enforce_validation(validation_metrics, **_selection_kwargs(config))
    except CalibrationOutcome as outcome:
        timings['validation'] = time.monotonic() - start
        timings['total'] = time.monotonic() - total_started
        write_json(run_dir / 'timing.json', timings)
        _report_failure(config, run_dir, outcome, 'validation')
        return
    artifact, lock_hash = _lock_threshold(
        config,
        run_dir,
        selection.epsilon,
        fit_metrics,
        validation_metrics,
    )
    timings['validation_and_lock'] = time.monotonic() - start

    start = time.monotonic()
    audit_score = score_root / PARTITIONS[2]
    _verify_lock(run_dir, lock_hash)
    score_pair_shards(
        pair_dir=run_dir / 'pair_manifests' / PARTITIONS[2],
        score_dir=audit_score,
        embedding_dir=run_dir / 'embedding_shards',
        total_dataset_rows=len(
            np.load(run_dir / 'index' / 'state.npy', mmap_mode='r')
        ),
        partition=PARTITIONS[2],
        locked_threshold_path=run_dir / 'selected_threshold.json',
    )
    _verify_lock(run_dir, lock_hash)
    audit = _load_scored_partition(audit_score)
    audit_metrics = _analyze_partition(
        audit, selection.epsilon, config, seed_offset=2000
    )
    write_json(run_dir / 'audit_metrics.json', audit_metrics)
    timings['one_time_audit'] = time.monotonic() - start

    start = time.monotonic()
    _plot_outputs(run_dir, selection, fit, audit, epsilon_distribution)
    timings['plots_and_report'] = time.monotonic() - start
    timings['total'] = time.monotonic() - total_started
    write_json(run_dir / 'timing.json', timings)
    _report_success(
        config,
        run_dir,
        artifact,
        fit_metrics,
        validation_metrics,
        audit_metrics,
        stability,
        timings,
    )
    formal = not config['smoke']['enabled']
    within_budget = (
        timings['total'] <= config['runtime']['max_task_hours'] * 3600
    )
    write_json(
        run_dir / 'status.json',
        {
            'status': 'PROMOTABLE'
            if within_budget
            else 'FAILED_RUNTIME_CONTRACT',
            'formal_evidence': formal,
            'threshold_locked': True,
            'selected_threshold_sha256': lock_hash,
            'within_five_hour_contract': within_budget,
            'total_seconds': timings['total'],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run_experiment(args.config.resolve(), args.run_dir.resolve(), args.smoke)


if __name__ == '__main__':
    main()
