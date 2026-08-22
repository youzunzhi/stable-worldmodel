"""Leakage-safe v2 pairs, action statistics, and action-effect geometry basis."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from scripts.experiments.self_supervised_plannability.pairs import (
    _load_model,
    _materialize_split,
    _read_clear_contracts,
    candidate_pairs,
    episode_balanced_order,
    hdf5_metadata_fingerprint,
    split_groups,
)

from .contracts import (
    FORMAL_TASKS,
    LOCKED,
    PROTOCOL_ID,
    SSPV2Failure,
    create_root,
    environment_identity,
    git_source,
    sha256_file,
    write_json,
)
from .geometry import (
    ClipConsistentActionTransform,
    TrajectoryHitDiagnostic,
    build_action_effect_basis,
    save_action_effect_basis,
)
from .planner import model_state_sha256

SPLITS = ('train', 'validation', 'test')


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        for row in rows:
            stream.write(
                json.dumps(row, sort_keys=True, separators=(',', ':'))
            )
            stream.write('\n')


def _action_statistics(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(actions)
    if values.ndim < 2:
        raise SSPV2Failure(
            'SSP_V2_INPUT_HASH_MISMATCH', 'action column must be at least 2D'
        )
    values = values.reshape(values.shape[0], -1)
    valid = values[~np.isnan(values).any(axis=1)]
    mean = valid.mean(axis=0, keepdims=False)
    std = valid.std(axis=0, keepdims=False)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise SSPV2Failure(
            'SSP_V2_INPUT_HASH_MISMATCH', 'non-finite action statistics'
        )
    if np.any(std <= 0):
        raise SSPV2Failure(
            'SSP_V2_INPUT_HASH_MISMATCH', 'action standard deviation is zero'
        )
    return mean.astype(np.float32), std.astype(np.float32)


def _expert_action_plans(
    records: list[dict],
    actions: np.ndarray,
    episodes: np.ndarray,
    steps: np.ndarray,
) -> np.ndarray:
    lookup = {
        (int(episode), int(step)): row
        for row, (episode, step) in enumerate(zip(episodes, steps))
    }
    rows = []
    for record in records:
        episode = int(record['episode_id'])
        start = int(record['start_step'])
        try:
            rows.append(
                [lookup[(episode, start + offset)] for offset in range(25)]
            )
        except KeyError as error:
            raise SSPV2Failure(
                'SSP_V2_INPUT_HASH_MISMATCH',
                f'missing expert action in pair {record["pair_id"]}',
            ) from error
    values = np.asarray(actions)[np.asarray(rows, dtype=np.int64)]
    return values.reshape(len(records), 25, -1).astype(np.float32)


@torch.inference_mode()
def _action_effects(
    *,
    model: torch.nn.Module,
    starts: torch.Tensor,
    goals: torch.Tensor,
    expert_raw: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    epsilon: float,
    seed: int,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, dict, torch.Tensor]:
    mean_np = action_mean.reshape(1, 1, -1)
    std_np = action_std.reshape(1, 1, -1)
    normalized = (expert_raw - mean_np) / std_np
    plans = torch.from_numpy(normalized).reshape(
        len(expert_raw),
        LOCKED['horizon'],
        LOCKED['action_block'] * expert_raw.shape[-1],
    )
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    perturbation = torch.from_numpy(
        rng.standard_normal(plans.shape).astype(np.float32)
    ) * float(LOCKED['action_effect_sigma'])
    transform = ClipConsistentActionTransform(
        torch.from_numpy(action_mean),
        torch.from_numpy(action_std),
        action_block=LOCKED['action_block'],
        raw_low=LOCKED['raw_action_low'],
        raw_high=LOCKED['raw_action_high'],
    )
    baseline = transform(plans)
    perturbed = transform(plans + perturbation)
    clip_record = transform.record()
    effects = []
    expert_hits = []
    diagnostic = TrajectoryHitDiagnostic(
        epsilon, LOCKED['trajectory_hit_horizon']
    )
    for offset in range(0, len(starts), batch_size):
        end = min(offset + batch_size, len(starts))
        start = starts[offset:end].to(device).reshape(end - offset, 1, 1, 192)
        goal = goals[offset:end].to(device).reshape(end - offset, 1, 1, 192)
        pixels = torch.empty(
            end - offset, 1, 1, 0, device=device, dtype=start.dtype
        )
        baseline_actions = baseline[offset:end].to(device).unsqueeze(1)
        perturbed_actions = perturbed[offset:end].to(device).unsqueeze(1)
        base_info = model.rollout(
            {'pixels': pixels, 'emb': start, 'goal_emb': goal},
            baseline_actions,
        )
        perturbed_info = model.rollout(
            {'pixels': pixels, 'emb': start, 'goal_emb': goal},
            perturbed_actions,
        )
        base_terminal = base_info['predicted_emb'][..., -1, :]
        perturbed_terminal = perturbed_info['predicted_emb'][..., -1, :]
        effects.append(
            (perturbed_terminal - base_terminal)[:, 0].float().cpu()
        )
        expert_hits.append(diagnostic(base_info)['hit_bits'][:, 0].cpu())
    effect_tensor = torch.cat(effects)
    hit_tensor = torch.cat(expert_hits)
    summary = {
        'effect_rows': len(effect_tensor),
        'effect_l2_mean': float(effect_tensor.norm(dim=1).mean()),
        'effect_l2_median': float(effect_tensor.norm(dim=1).median()),
        'expert_action_trajectory_hit_count': int(hit_tensor.sum()),
        'expert_action_trajectory_hit_rate': float(hit_tensor.float().mean()),
        'perturbation_sigma': LOCKED['action_effect_sigma'],
        'clip_projection': clip_record,
    }
    return effect_tensor.numpy(), summary, baseline.cpu()


def prepare_task(
    *,
    config_path: Path,
    config: dict,
    output_dir: str | Path,
    repo_root: Path,
    device: str,
    formal: bool,
    full_dataset_hash: bool,
) -> dict:
    """Create immutable v2 pair, action, latent, and rotated-basis artifacts."""
    import h5py
    import hdf5plugin  # noqa: F401

    if formal and config['task'] not in FORMAL_TASKS:
        raise SSPV2Failure(
            'SSP_V2_INPUT_HASH_MISMATCH',
            f'{config["task"]} is diagnostic-only in the v2 contract',
        )
    started = time.monotonic()
    root = create_root(output_dir, formal=formal)
    source = git_source(repo_root)
    if formal and source['dirty']:
        raise SSPV2Failure(
            'SSP_V2_INPUT_HASH_MISMATCH',
            'formal preparation requires a clean committed source tree',
        )
    write_json(root / 'source.json', source)
    write_json(root / 'environment.json', environment_identity())
    shutil.copyfile(config_path, root / 'pre_registered_config.json')
    write_json(
        root / 'protocol.json',
        {
            'protocol_id': PROTOCOL_ID,
            'formal_evidence': bool(formal),
            'task': config['task'],
            'epsilon_task': config['epsilon_task'],
            'verifier': 'trajectory-aware float32 sum-SSE strict binary',
            'planner_return': 'verified-hit-archive-else-best-evaluated',
            'locked': config['locked'],
        },
    )

    dataset = Path(config['dataset']['path']).expanduser().resolve()
    checkpoint = Path(config['checkpoint']['path']).expanduser().resolve()
    checkpoint_config = checkpoint.parent / 'config.json'
    for path in (dataset, checkpoint, checkpoint_config):
        if not path.is_file():
            raise SSPV2Failure(
                'SSP_V2_INPUT_HASH_MISMATCH', f'required input missing: {path}'
            )
    if dataset.stat().st_size != int(config['dataset']['expected_bytes']):
        raise SSPV2Failure(
            'SSP_V2_INPUT_HASH_MISMATCH', 'dataset byte size mismatch'
        )
    checkpoint_hash = sha256_file(checkpoint)
    checkpoint_config_hash = sha256_file(checkpoint_config)
    if checkpoint_hash != config['checkpoint']['sha256']:
        raise SSPV2Failure(
            'SSP_V2_INPUT_HASH_MISMATCH', 'checkpoint SHA-256 mismatch'
        )
    if checkpoint_config_hash != config['checkpoint']['config_sha256']:
        raise SSPV2Failure(
            'SSP_V2_INPUT_HASH_MISMATCH', 'checkpoint config SHA-256 mismatch'
        )
    clear_rows, clear_episodes, clear_identities = _read_clear_contracts(
        repo_root, config
    )
    input_hashes = {
        'dataset': {
            **config['dataset'],
            'absolute_path': str(dataset),
            'bytes': dataset.stat().st_size,
            'metadata_sha256': hdf5_metadata_fingerprint(dataset),
            'full_file_sha256': (
                sha256_file(dataset) if full_dataset_hash else None
            ),
            'full_file_hash_computed': bool(full_dataset_hash),
        },
        'checkpoint': {
            **config['checkpoint'],
            'absolute_path': str(checkpoint),
            'actual_sha256': checkpoint_hash,
            'config_path': str(checkpoint_config),
            'actual_config_sha256': checkpoint_config_hash,
        },
        'clear_manifests': clear_identities,
        'pre_registered_config_sha256': sha256_file(config_path),
    }
    write_json(root / 'input_hashes.json', input_hashes)

    model = _load_model(checkpoint, device)
    before_model_hash = model_state_sha256(model)
    with h5py.File(dataset, 'r') as h5:
        dataset_cfg = config['dataset']
        episodes = np.asarray(h5[dataset_cfg['episode_column']])
        steps = np.asarray(h5[dataset_cfg['step_column']])
        actions = np.asarray(h5[dataset_cfg['action_column']])
        action_mean, action_std = _action_statistics(actions)
        records, counts = candidate_pairs(
            task=config['task'],
            episodes=episodes,
            steps=steps,
            clear_start_rows=clear_rows,
        )
        group_split = split_groups(
            np.asarray([row['group_id'] for row in records]),
            LOCKED['split_seed'],
        )
        write_json(
            root / 'pair_manifests' / 'group_split.json',
            {
                name: [int(value) for value in values]
                for name, values in group_split.items()
            },
        )
        group_to_split = {
            int(group): split
            for split, groups in group_split.items()
            for group in groups
        }
        split_records = {name: [] for name in SPLITS}
        for record in records:
            split_records[group_to_split[record['group_id']]].append(record)
        required = {
            'train': LOCKED['train_pairs'],
            'validation': LOCKED['validation_pairs'],
            'test': LOCKED['test_pairs'],
        }
        chosen = {}
        latent_payload = {}
        filter_counts = {}
        pixels = h5[dataset_cfg['pixels_column']]
        for index, split in enumerate(SPLITS):
            ordered = episode_balanced_order(
                split_records[split], LOCKED['split_seed'] + index + 1
            )
            selected, start, goal, split_filter = _materialize_split(
                ordered=ordered,
                count=required[split],
                epsilon=float(config['epsilon_task']),
                model=model,
                pixels=pixels,
                device=device,
                encode_batch_size=int(config['runtime']['encode_batch_size']),
            )
            chosen[split] = selected
            latent_payload[split] = {
                'pair_ids': [row['pair_id'] for row in selected],
                'start': start,
                'goal': goal,
            }
            filter_counts[split] = split_filter
            _write_jsonl(root / 'pair_manifests' / f'{split}.jsonl', selected)

        expert_raw = _expert_action_plans(
            chosen['train'], actions, episodes, steps
        )

    effects, effect_summary, expert_normalized = _action_effects(
        model=model,
        starts=latent_payload['train']['start'],
        goals=latent_payload['train']['goal'],
        expert_raw=expert_raw,
        action_mean=action_mean,
        action_std=action_std,
        epsilon=float(config['epsilon_task']),
        seed=int(config['action_effect_seed']),
        device=device,
        batch_size=int(config['runtime']['action_effect_batch_size']),
    )
    basis, basis_metadata = build_action_effect_basis(
        effects, LOCKED['parameter_dim']
    )
    basis_summary = save_action_effect_basis(
        root,
        basis,
        {**basis_metadata, **effect_summary},
        seed=int(config['action_effect_seed']),
    )
    action_stats = {
        'method': 'dataset-full-column-zscore',
        'mean': action_mean.tolist(),
        'std': action_std.tolist(),
        'raw_low': LOCKED['raw_action_low'],
        'raw_high': LOCKED['raw_action_high'],
        'action_dim': int(action_mean.size),
    }
    write_json(root / 'action_stats.json', action_stats)
    torch.save(latent_payload, root / 'pair_latents.pt')
    torch.save(
        {
            'protocol_id': PROTOCOL_ID,
            'task': config['task'],
            'pair_ids': latent_payload['train']['pair_ids'],
            'normalized_expert_plans': expert_normalized,
        },
        root / 'action_effect_inputs.pt',
    )

    after_model_hash = model_state_sha256(model)
    if before_model_hash != after_model_hash:
        raise SSPV2Failure(
            'SSP_V2_FROZEN_MODEL_MUTATION',
            'model changed during v2 preparation',
        )
    selected_pair_ids = [
        row['pair_id'] for split in SPLITS for row in chosen[split]
    ]
    if len(selected_pair_ids) != len(set(selected_pair_ids)):
        raise AssertionError('pair IDs overlap across v2 splits')
    split_group_sets = [
        {row['group_id'] for row in chosen[split]} for split in SPLITS
    ]
    if any(
        split_group_sets[left].intersection(split_group_sets[right])
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        raise AssertionError('leakage groups overlap across v2 splits')
    overlap = {
        split: sorted(
            {row['episode_id'] for row in chosen[split]}.intersection(
                clear_episodes
            )
        )
        for split in SPLITS
    }
    summary = {
        'protocol_id': PROTOCOL_ID,
        'task': config['task'],
        'formal_evidence': bool(formal),
        'counts': counts,
        'filter_counts': filter_counts,
        'selected_counts': {split: len(chosen[split]) for split in SPLITS},
        'clear_episode_overlap': overlap,
        'clear_episode_overlap_is_not_episode_level_isolation': True,
        'model_state_sha256_before': before_model_hash,
        'model_state_sha256_after': after_model_hash,
        'action_statistics': action_stats,
        'action_effect_basis': basis_summary,
        'pair_latents_sha256': sha256_file(root / 'pair_latents.pt'),
        'action_effect_inputs_sha256': sha256_file(
            root / 'action_effect_inputs.pt'
        ),
        'elapsed_seconds': time.monotonic() - started,
        'forbidden_call_counts': {
            'environment_constructor': 0,
            'environment_step': 0,
            'simulator_reward': 0,
        },
    }
    write_json(root / 'preparation.completed.json', summary)
    return summary


__all__ = ['_load_model', 'prepare_task']
