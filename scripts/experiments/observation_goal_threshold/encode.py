"""Frozen encoder/projector extraction and latent pair scoring."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

from .io_utils import sha256_file, write_json

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess_pixels(pixels, image_size: tuple[int, int] = (224, 224)):
    """Match eval_wm.img_transform for an HWC RGB uint8 batch."""
    import torch
    from torchvision.transforms.v2 import functional as tvf

    array = np.asarray(pixels)
    if array.ndim != 4 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise ValueError('pixels must be a uint8 NHWC RGB batch')
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2)
    tensor = tvf.to_dtype(tensor, dtype=torch.float32, scale=True)
    tensor = tvf.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return tvf.resize(tensor, size=list(image_size), antialias=True)


def encode_projected(model, pixels):
    """Use only the frozen encoder and projector; never model.encode/predict."""
    output = model.encoder(pixels, interpolate_pos_encoding=True)
    return model.projector(output.last_hidden_state[:, 0])


def parameter_hash(model) -> str:
    """Hash encoder/projector symbols, names, shapes, dtypes, and bytes."""
    digest = hashlib.sha256()
    for prefix, module in (
        ('encoder', model.encoder),
        ('projector', model.projector),
    ):
        for name, tensor in sorted(module.state_dict().items()):
            value = tensor.detach().cpu().contiguous()
            digest.update(f'{prefix}.{name}'.encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(str(value.dtype).encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_frozen_model(checkpoint: str | Path, device: str):
    import stable_worldmodel as swm

    model = swm.wm.utils.load_pretrained(str(checkpoint))
    model = model.to(device).eval()
    model.requires_grad_(False)
    for parameter in model.parameters():
        if parameter.requires_grad:
            raise AssertionError('checkpoint parameter was not frozen')
    return model


def pair_partition_matches(table, expected: str) -> bool:
    """Validate one dictionary-encoded partition column across pyarrow versions."""
    import pyarrow as pa
    import pyarrow.compute as pc

    if len(table) == 0:
        return False
    values = pc.unique(table['partition'].cast(pa.string())).to_pylist()
    return values == [expected]


def encode_observations(
    *,
    dataset_path: str | Path,
    pixels_column: str,
    row_ids: np.ndarray,
    checkpoint_path: str | Path,
    expected_checkpoint_sha256: str,
    expected_config_sha256: str,
    output_dir: str | Path,
    batch_size: int,
    latent_dim: int,
    device: str = 'cuda',
) -> dict:
    """Encode each resolved eligible observation exactly once."""
    import h5py
    import hdf5plugin  # noqa: F401
    import torch

    start_time = time.monotonic()
    checkpoint = Path(checkpoint_path)
    actual_checkpoint_hash = sha256_file(checkpoint)
    if actual_checkpoint_hash != expected_checkpoint_sha256:
        raise ValueError('checkpoint SHA-256 mismatch')
    config_hash = sha256_file(checkpoint.parent / 'config.json')
    if config_hash != expected_config_sha256:
        raise ValueError('checkpoint config SHA-256 mismatch')
    model = load_frozen_model(checkpoint, device)
    before = parameter_hash(model)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    rows = np.asarray(row_ids, dtype=np.int64)
    if len(np.unique(rows)) != len(rows):
        raise ValueError('observation row IDs must be unique')
    embedding_path = output / 'embeddings.float32.npy'
    embeddings = np.lib.format.open_memmap(
        embedding_path,
        mode='w+',
        dtype=np.float32,
        shape=(len(rows), latent_dim),
    )
    deterministic_reference = None
    with h5py.File(dataset_path, 'r') as dataset, torch.inference_mode():
        pixels = dataset[pixels_column]
        for start in range(0, len(rows), batch_size):
            stop = min(start + batch_size, len(rows))
            selected = rows[start:stop]
            if len(selected) > 1 and np.all(np.diff(selected) == 1):
                raw = pixels[int(selected[0]) : int(selected[-1]) + 1]
            else:
                raw = pixels[selected]
            tensor = preprocess_pixels(raw).to(device, non_blocking=True)
            encoded = encode_projected(model, tensor).float().cpu().numpy()
            if encoded.shape != (stop - start, latent_dim):
                raise ValueError(
                    f'latent shape mismatch: {encoded.shape}, '
                    f'expected {(stop - start, latent_dim)}'
                )
            if not np.isfinite(encoded).all():
                raise ValueError('non-finite projected latent')
            embeddings[start:stop] = encoded
            if start == 0:
                second = encode_projected(model, tensor).float().cpu().numpy()
                if not np.array_equal(encoded, second):
                    raise ValueError('encoder determinism check failed')
                deterministic_reference = hashlib.sha256(
                    encoded.tobytes()
                ).hexdigest()
    embeddings.flush()
    after = parameter_hash(model)
    if before != after:
        raise ValueError('encoder/projector parameter hash changed')
    np.save(output / 'row_ids.npy', rows, allow_pickle=False)
    manifest = {
        'rows': len(rows),
        'latent_dim': int(latent_dim),
        'dtype': 'float32',
        'checkpoint_path': str(checkpoint),
        'checkpoint_sha256': actual_checkpoint_hash,
        'checkpoint_config_sha256': config_hash,
        'encoder_projector_parameter_hash_before': before,
        'encoder_projector_parameter_hash_after': after,
        'determinism_batch_sha256': deterministic_reference,
        'embedding_file': embedding_path.name,
        'embedding_sha256': sha256_file(embedding_path),
        'row_ids_sha256': sha256_file(output / 'row_ids.npy'),
        'elapsed_seconds': time.monotonic() - start_time,
        'forbidden_call_counts': {
            'predictor': 0,
            'action_encoder': 0,
            'planner': 0,
            'environment_constructor': 0,
            'environment_step': 0,
        },
    }
    write_json(output / 'embedding_manifest.json', manifest)
    return manifest


def score_pair_shards(
    *,
    pair_dir: str | Path,
    score_dir: str | Path,
    embedding_dir: str | Path,
    total_dataset_rows: int,
    partition: str,
    locked_threshold_path: str | Path | None,
    device: str = 'cuda',
) -> dict:
    """Append mean-D squared latent distance to materialized pair shards."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    if partition == 'threshold_audit' and (
        locked_threshold_path is None
        or not Path(locked_threshold_path).is_file()
    ):
        raise PermissionError('audit scoring requires a locked threshold')
    source = Path(pair_dir)
    target = Path(score_dir)
    target.mkdir(parents=True, exist_ok=False)
    emb_dir = Path(embedding_dir)
    embeddings_np = np.load(emb_dir / 'embeddings.float32.npy', mmap_mode='r')
    row_ids = np.load(emb_dir / 'row_ids.npy', allow_pickle=False)
    lookup = np.full(total_dataset_rows, -1, dtype=np.int64)
    lookup[row_ids] = np.arange(len(row_ids), dtype=np.int64)
    embeddings = torch.from_numpy(np.asarray(embeddings_np)).to(device)
    rows_scored = 0
    start_time = time.monotonic()
    hashes = []
    for path in sorted(source.glob('*.parquet')):
        table = pq.read_table(path)
        if not pair_partition_matches(table, partition):
            raise ValueError(f'pair shard partition mismatch: {path}')
        anchor_row = table['anchor_row'].to_numpy()
        goal_row = table['goal_row'].to_numpy()
        anchor_index = lookup[anchor_row]
        goal_index = lookup[goal_row]
        if np.any(anchor_index < 0) or np.any(goal_index < 0):
            raise ValueError('pair references an unencoded observation')
        with torch.inference_mode():
            anchor = embeddings[
                torch.from_numpy(anchor_index).to(device=device)
            ]
            goal = embeddings[torch.from_numpy(goal_index).to(device=device)]
            distance = (anchor - goal).square().mean(dim=1).cpu().numpy()
        if not np.isfinite(distance).all():
            raise ValueError('non-finite latent distance')
        scored = table.append_column(
            'latent_distance', pa.array(distance.astype(np.float32))
        )
        output_path = target / path.name
        pq.write_table(scored, output_path, compression='zstd')
        hashes.append({'file': path.name, 'sha256': sha256_file(output_path)})
        rows_scored += len(table)
    if rows_scored == 0:
        raise ValueError(f'no pair shards found in {source}')
    manifest = {
        'partition': partition,
        'rows_scored': rows_scored,
        'residual_definition': 'mean_D((z_i-z_j)^2)',
        'latent_dim': int(embeddings.shape[1]),
        'dtype': 'float32',
        'elapsed_seconds': time.monotonic() - start_time,
        'shards': hashes,
    }
    write_json(target / 'score_manifest.json', manifest)
    return manifest
