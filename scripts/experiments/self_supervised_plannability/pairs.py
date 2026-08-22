"""Leakage-safe SSP pair manifests and frozen observation embeddings."""

from __future__ import annotations

import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .contracts import (
    LOCKED,
    PROTOCOL_ID,
    SSPFailure,
    create_root,
    environment_identity,
    git_source,
    resolve_repo_path,
    sha256_file,
    sha256_json,
    write_json,
)
from .geometry import build_basis, save_basis
from .planner import model_state_sha256

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SPLITS = ('train', 'validation', 'test')


def hdf5_metadata_fingerprint(path: str | Path) -> str:
    import h5py
    import hdf5plugin  # noqa: F401

    source = Path(path)
    with h5py.File(source, 'r') as dataset:
        schema = []
        for key in sorted(dataset.keys()):
            value = dataset[key]
            if isinstance(value, h5py.Dataset):
                schema.append(
                    {
                        'key': key,
                        'shape': list(value.shape),
                        'dtype': str(value.dtype),
                    }
                )
    payload = {'size_bytes': source.stat().st_size, 'schema': schema}
    return sha256_json(payload)


def preprocess_pixels(pixels: np.ndarray) -> torch.Tensor:
    from torchvision.transforms.v2 import functional as tvf

    array = np.asarray(pixels)
    if array.ndim != 4 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise SSPFailure(
            'SSP_LATENT_CONTRACT_MISMATCH',
            f'pixels must be uint8 NHWC RGB, got {array.shape} {array.dtype}',
        )
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2)
    tensor = tvf.to_dtype(tensor, dtype=torch.float32, scale=True)
    tensor = tvf.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return tvf.resize(tensor, size=[224, 224], antialias=True)


def split_groups(group_ids: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    unique = np.unique(np.asarray(group_ids, dtype=np.int64))
    if len(unique) < 3:
        raise SSPFailure(
            'SSP_INSUFFICIENT_ELIGIBLE_PAIRS',
            'at least three independent leakage groups are required',
        )
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    ordered = unique.copy()
    rng.shuffle(ordered)
    n_train = round(len(ordered) * 0.70)
    n_validation = round(len(ordered) * 0.15)
    n_train = min(max(n_train, 1), len(ordered) - 2)
    n_validation = min(max(n_validation, 1), len(ordered) - n_train - 1)
    return {
        'train': ordered[:n_train],
        'validation': ordered[n_train : n_train + n_validation],
        'test': ordered[n_train + n_validation :],
    }


def episode_balanced_order(records: list[dict], seed: int) -> list[dict]:
    """Return a deterministic without-replacement round-robin ordering."""
    by_episode: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        by_episode[int(record['episode_id'])].append(record)
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    episode_ids = np.asarray(sorted(by_episode), dtype=np.int64)
    rng.shuffle(episode_ids)
    for episode_id in episode_ids:
        rng.shuffle(by_episode[int(episode_id)])
    ordered = []
    depth = 0
    while True:
        added = False
        for episode_id in episode_ids:
            rows = by_episode[int(episode_id)]
            if depth < len(rows):
                ordered.append(rows[depth])
                added = True
        if not added:
            break
        depth += 1
    if len(ordered) != len(records):
        raise AssertionError('episode-balanced ordering lost a pair')
    return ordered


def candidate_pairs(
    *,
    task: str,
    episodes: np.ndarray,
    steps: np.ndarray,
    clear_start_rows: set[int],
) -> tuple[list[dict], dict]:
    episodes = np.asarray(episodes)
    steps = np.asarray(steps)
    if episodes.shape != steps.shape:
        raise ValueError('episode and step columns must have equal shape')
    lookup = {
        (int(episode), int(step)): row
        for row, (episode, step) in enumerate(zip(episodes, steps))
    }
    output = []
    missing_goal = 0
    excluded_clear = 0
    for row, (episode, step) in enumerate(zip(episodes, steps)):
        goal_row = lookup.get((int(episode), int(step) + 25))
        if goal_row is None:
            missing_goal += 1
            continue
        if row in clear_start_rows:
            excluded_clear += 1
            continue
        group_id = int(episode) // 101 if task == 'pusht' else int(episode)
        output.append(
            {
                'pair_id': (f'ssp-{task}-ep{int(episode)}-step{int(step)}'),
                'episode_id': int(episode),
                'group_id': group_id,
                'start_row': int(row),
                'goal_row': int(goal_row),
                'start_step': int(step),
                'goal_step': int(step) + 25,
            }
        )
    return output, {
        'dataset_rows': len(episodes),
        'missing_goal_offset': missing_goal,
        'excluded_clear_start_rows': excluded_clear,
        'pre_initial_hit_candidates': len(output),
    }


def _read_clear_contracts(
    repo_root: Path, config: dict
) -> tuple[set[int], set[int], list[dict]]:
    start_rows: set[int] = set()
    episodes: set[int] = set()
    identities = []
    for value in config['clear_manifests']:
        path = resolve_repo_path(repo_root, value)
        manifest = json.loads(path.read_text())
        if manifest.get('task') != config['task']:
            raise ValueError(f'CLEAR manifest task mismatch: {path}')
        for pair in manifest['pairs']:
            start_rows.add(int(pair['start_row']))
            episodes.add(int(pair['episode_id']))
        identities.append(
            {
                'path': str(path),
                'sha256': sha256_file(path),
                'pairs': len(manifest['pairs']),
                'protocol': manifest['protocol']['name'],
            }
        )
    return start_rows, episodes, identities


def _load_model(checkpoint: Path, device: str) -> torch.nn.Module:
    import stable_worldmodel as swm

    model = swm.wm.utils.load_pretrained(str(checkpoint))
    model = model.to(device).eval().requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise SSPFailure(
            'SSP_FROZEN_MODEL_MUTATION', 'checkpoint parameters are trainable'
        )
    return model


def _encode_pair_batch(
    *,
    model: torch.nn.Module,
    pixels,
    records: list[dict],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    def read_rows(column, rows: np.ndarray) -> np.ndarray:
        order = np.argsort(rows, kind='stable')
        sorted_rows = rows[order]
        if len(np.unique(sorted_rows)) != len(sorted_rows):
            return np.stack([column[int(row)] for row in rows])
        sorted_values = np.asarray(column[sorted_rows])
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        return sorted_values[inverse]

    start_rows = np.asarray(
        [record['start_row'] for record in records], dtype=np.int64
    )
    goal_rows = np.asarray(
        [record['goal_row'] for record in records], dtype=np.int64
    )
    raw = np.concatenate(
        [read_rows(pixels, start_rows), read_rows(pixels, goal_rows)], axis=0
    )
    tensor = preprocess_pixels(raw).to(device, non_blocking=True)
    with torch.inference_mode():
        encoded = model.encoder(tensor, interpolate_pos_encoding=True)
        projected = model.projector(encoded.last_hidden_state[:, 0]).float()
    if projected.shape != (2 * len(records), 192):
        raise SSPFailure(
            'SSP_LATENT_CONTRACT_MISMATCH',
            f'encoded latent shape is {tuple(projected.shape)}',
        )
    if not torch.isfinite(projected).all():
        raise SSPFailure(
            'SSP_LATENT_CONTRACT_MISMATCH', 'non-finite encoded latent'
        )
    start, goal = projected.split(len(records), dim=0)
    return start.cpu(), goal.cpu()


def _materialize_split(
    *,
    ordered: list[dict],
    count: int,
    epsilon: float,
    model: torch.nn.Module,
    pixels,
    device: str,
    encode_batch_size: int,
) -> tuple[list[dict], torch.Tensor, torch.Tensor, dict]:
    selected: list[dict] = []
    start_values = []
    goal_values = []
    initial_hits = 0
    inspected = 0
    for offset in range(0, len(ordered), encode_batch_size):
        batch = ordered[offset : offset + encode_batch_size]
        start, goal = _encode_pair_batch(
            model=model,
            pixels=pixels,
            records=batch,
            device=device,
        )
        distances = (start - goal).square().sum(dim=1, dtype=torch.float32)
        keep = distances >= float(epsilon)
        inspected += len(batch)
        initial_hits += int((~keep).sum())
        for index in torch.nonzero(keep, as_tuple=False).flatten().tolist():
            selected.append(batch[index])
            start_values.append(start[index])
            goal_values.append(goal[index])
            if len(selected) == count:
                break
        if len(selected) == count:
            break
    if len(selected) != count:
        raise SSPFailure(
            'SSP_INSUFFICIENT_ELIGIBLE_PAIRS',
            f'needed {count} pairs but found {len(selected)} after '
            f'initial-hit filtering',
        )
    return (
        selected,
        torch.stack(start_values),
        torch.stack(goal_values),
        {'inspected': inspected, 'initial_hits_excluded': initial_hits},
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        for row in rows:
            stream.write(
                json.dumps(row, sort_keys=True, separators=(',', ':'))
            )
            stream.write('\n')


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
    """Create immutable pair/basis/latent preparation artifacts."""
    import h5py
    import hdf5plugin  # noqa: F401

    started = time.monotonic()
    root = create_root(output_dir, formal=formal)
    source = git_source(repo_root)
    if formal and source['dirty']:
        raise SSPFailure(
            'SSP_INPUT_HASH_MISMATCH',
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
            'original_distance': 'float32 sum_D(residual^2)',
            'hit_comparison': '<',
            'locked': config['locked'],
        },
    )

    dataset = Path(config['dataset']['path']).expanduser().resolve()
    checkpoint = Path(config['checkpoint']['path']).expanduser().resolve()
    checkpoint_config = checkpoint.parent / 'config.json'
    for path in (dataset, checkpoint, checkpoint_config):
        if not path.is_file():
            raise SSPFailure(
                'SSP_INPUT_HASH_MISMATCH', f'required input missing: {path}'
            )
    if dataset.stat().st_size != int(config['dataset']['expected_bytes']):
        raise SSPFailure(
            'SSP_INPUT_HASH_MISMATCH', 'dataset byte size mismatch'
        )
    checkpoint_hash = sha256_file(checkpoint)
    config_hash = sha256_file(checkpoint_config)
    if checkpoint_hash != config['checkpoint']['sha256']:
        raise SSPFailure(
            'SSP_INPUT_HASH_MISMATCH', 'checkpoint SHA-256 mismatch'
        )
    if config_hash != config['checkpoint']['config_sha256']:
        raise SSPFailure(
            'SSP_INPUT_HASH_MISMATCH', 'checkpoint config SHA-256 mismatch'
        )
    clear_rows, clear_episodes, clear_identities = _read_clear_contracts(
        repo_root, config
    )
    metadata_hash = hdf5_metadata_fingerprint(dataset)
    dataset_hash = sha256_file(dataset) if full_dataset_hash else None
    input_hashes = {
        'dataset': {
            **config['dataset'],
            'absolute_path': str(dataset),
            'bytes': dataset.stat().st_size,
            'metadata_sha256': metadata_hash,
            'full_file_sha256': dataset_hash,
            'full_file_hash_computed': bool(full_dataset_hash),
        },
        'checkpoint': {
            **config['checkpoint'],
            'absolute_path': str(checkpoint),
            'actual_sha256': checkpoint_hash,
            'config_path': str(checkpoint_config),
            'actual_config_sha256': config_hash,
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
    after_model_hash = model_state_sha256(model)
    if before_model_hash != after_model_hash:
        raise SSPFailure(
            'SSP_FROZEN_MODEL_MUTATION',
            'model state changed during pair preparation',
        )
    torch.save(latent_payload, root / 'pair_latents.pt')
    basis = build_basis(192, 16, int(config['basis_seed']))
    basis_metadata = save_basis(root, basis, int(config['basis_seed']))

    selected_pair_ids = [
        row['pair_id'] for split in SPLITS for row in chosen[split]
    ]
    if len(selected_pair_ids) != len(set(selected_pair_ids)):
        raise AssertionError('pair IDs overlap across SSP splits')
    split_group_sets = [
        {row['group_id'] for row in chosen[split]} for split in SPLITS
    ]
    if any(
        split_group_sets[left].intersection(split_group_sets[right])
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        raise AssertionError('leakage groups overlap across SSP splits')
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
        'basis': basis_metadata,
        'pair_latents_sha256': sha256_file(root / 'pair_latents.pt'),
        'elapsed_seconds': time.monotonic() - started,
        'forbidden_call_counts': {
            'planner': 0,
            'environment_constructor': 0,
            'environment_step': 0,
        },
    }
    write_json(root / 'preparation.completed.json', summary)
    return summary
