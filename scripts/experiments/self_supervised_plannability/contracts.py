"""Locked protocol constants and auditable artifact helpers for SSP v1."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

PROTOCOL_ID = 'self-supervised-plannability-v1'
TASKS = ('pusht', 'cube', 'tworoom')
THRESHOLDS = {'pusht': 1.5, 'cube': 1.0, 'tworoom': 1.5}
BASIS_SEEDS = {'pusht': 26082201, 'cube': 26082202, 'tworoom': 26082203}
REPLICATE_SEEDS = (260822, 260823, 260824)
PROFILE_SEEDS = (42, 43, 44, 45, 46)

LOCKED = {
    'latent_dim': 192,
    'parameter_dim': 16,
    'goal_offset': 25,
    'split_seed': 260822,
    'split_fractions': [0.70, 0.15, 0.15],
    'train_pairs': 800,
    'validation_pairs': 256,
    'test_pairs': 512,
    'batch_size': 1,
    'num_samples': 300,
    'n_steps': 30,
    'topk': 30,
    'var_scale': 1.0,
    'horizon': 5,
    'receding_horizon': 5,
    'action_block': 5,
    'outer_pair_batch': 16,
    'directions': 8,
    'sigma': 0.25,
    'learning_rate': 0.05,
    'adam_betas': [0.9, 0.999],
    'adam_epsilon': 1e-8,
    'outer_steps': 50,
    'checkpoint_interval': 5,
    'validation_interval': 5,
    'validation_seed': 26082290,
    'bootstrap_replicates': 10000,
    'analysis_seed': 20260822,
}


class SSPFailure(RuntimeError):
    """Structured terminal failure with a stable protocol code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f'{code}: {message}')
        self.code = code
        self.message = message


def sha256_file(path: str | Path, chunk_size: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(',', ':'), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f'.{target.name}.tmp-{os.getpid()}')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    temporary.replace(target)


def append_jsonl(path: str | Path, value: Any) -> None:
    with Path(path).open('a') as stream:
        stream.write(
            json.dumps(
                value, sort_keys=True, separators=(',', ':'), allow_nan=False
            )
        )
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def load_config(path: str | Path) -> tuple[Path, dict]:
    config_path = Path(path).expanduser().resolve()
    config = json.loads(config_path.read_text())
    validate_config(config)
    return config_path, config


def validate_config(config: dict) -> None:
    if config.get('protocol_id') != PROTOCOL_ID:
        raise SSPFailure(
            'SSP_LATENT_CONTRACT_MISMATCH',
            f'protocol_id must be {PROTOCOL_ID}',
        )
    task = config.get('task')
    if task not in TASKS:
        raise ValueError(f'task must be one of {TASKS}, got {task!r}')
    if float(config.get('epsilon_task')) != THRESHOLDS[task]:
        raise SSPFailure(
            'SSP_LATENT_CONTRACT_MISMATCH', 'task threshold is not locked'
        )
    if int(config.get('basis_seed')) != BASIS_SEEDS[task]:
        raise ValueError('basis seed is not locked')
    for name, expected in LOCKED.items():
        actual = config.get('locked', {}).get(name)
        if actual != expected:
            raise ValueError(
                f'locked config mismatch for {name}: {actual!r} != '
                f'{expected!r}'
            )
    checkpoint = config.get('checkpoint', {})
    for name in ('path', 'sha256', 'config_sha256'):
        if not checkpoint.get(name):
            raise ValueError(f'checkpoint.{name} is required')
    dataset = config.get('dataset', {})
    for name in (
        'path',
        'repository',
        'revision',
        'expected_bytes',
        'pixels_column',
        'episode_column',
        'step_column',
    ):
        if dataset.get(name) in (None, ''):
            raise ValueError(f'dataset.{name} is required')


def resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (
        path.resolve() if path.is_absolute() else (repo_root / path).resolve()
    )


def create_root(path: str | Path, *, formal: bool) -> Path:
    root = Path(path).expanduser().resolve()
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        code = 'SSP_FORMAL_ROOT_EXISTS' if formal else 'SSP_OUTPUT_ROOT_EXISTS'
        raise SSPFailure(
            code, f'output root already exists: {root}'
        ) from error
    return root


def git_source(repo_root: Path) -> dict:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ['git', *args], cwd=repo_root, text=True
        ).strip()

    status = run('status', '--porcelain')
    return {
        'repo_root': str(repo_root.resolve()),
        'commit': run('rev-parse', 'HEAD'),
        'branch': run('branch', '--show-current'),
        'dirty': bool(status),
        'status_porcelain': status.splitlines(),
    }


def environment_identity() -> dict:
    import numpy as np
    import torch

    cuda = torch.cuda.is_available()
    return {
        'python': sys.version,
        'platform': platform.platform(),
        'numpy': np.__version__,
        'torch': torch.__version__,
        'cuda_available': cuda,
        'cuda_runtime': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0) if cuda else None,
        'cudnn': torch.backends.cudnn.version() if cuda else None,
    }


def hash_inventory(root: str | Path) -> dict[str, str]:
    base = Path(root)
    output = {}
    for path in sorted(value for value in base.rglob('*') if value.is_file()):
        if path.name in {'audit.json', 'training.completed.json'}:
            continue
        output[str(path.relative_to(base))] = sha256_file(path)
    return output
