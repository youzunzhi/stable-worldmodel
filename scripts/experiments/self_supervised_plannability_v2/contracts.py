"""Frozen SSP-v2 constants and auditable artifact helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

PROTOCOL_ID = 'self-supervised-plannability-v2'
FORMAL_TASKS = ('tworoom', 'pusht')
DIAGNOSTIC_TASKS = ('cube',)
TASKS = FORMAL_TASKS + DIAGNOSTIC_TASKS
THRESHOLDS = {'pusht': 1.5, 'cube': 1.0, 'tworoom': 1.5}
ACTION_EFFECT_SEEDS = {
    'pusht': 26082211,
    'cube': 26082212,
    'tworoom': 26082213,
}
REPLICATE_SEEDS = (260822, 260823, 260824)
VALIDATION_SEEDS = (42, 43, 44, 45, 46)
PROFILE_SEEDS = VALIDATION_SEEDS

LOCKED = {
    'latent_dim': 192,
    'parameter_dim': 32,
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
    'trajectory_hit_horizon': 5,
    'raw_action_low': -1.0,
    'raw_action_high': 1.0,
    'action_effect_sigma': 0.5,
    'late_hit_iterations': 5,
    'hit_mass_beta': 0.25,
    'outer_pair_batch': 32,
    'directions': 16,
    'planner_tapes': 2,
    'sigma': 0.5,
    'learning_rate': 0.03,
    'adam_betas': [0.9, 0.999],
    'adam_epsilon': 1e-8,
    'epochs': 4,
    'outer_steps': 100,
    'checkpoint_interval': 10,
    'validation_interval': 10,
    'validation_seeds': list(VALIDATION_SEEDS),
    'selection_standard_errors': 1.0,
    'bootstrap_replicates': 10000,
    'analysis_seed': 20260822,
}


class SSPV2Failure(RuntimeError):
    """Structured terminal failure with a stable v2 protocol code."""

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
        raise SSPV2Failure(
            'SSP_V2_INPUT_HASH_MISMATCH',
            f'protocol_id must be {PROTOCOL_ID}',
        )
    task = config.get('task')
    if task not in TASKS:
        raise ValueError(f'task must be one of {TASKS}, got {task!r}')
    if float(config.get('epsilon_task')) != THRESHOLDS[task]:
        raise SSPV2Failure(
            'SSP_V2_INPUT_HASH_MISMATCH', 'task threshold is not locked'
        )
    if int(config.get('action_effect_seed')) != ACTION_EFFECT_SEEDS[task]:
        raise ValueError('action-effect seed is not locked')
    if list(config.get('replicate_seeds', [])) != list(REPLICATE_SEEDS):
        raise ValueError('replicate seeds are not locked')
    for name, expected in LOCKED.items():
        actual = config.get('locked', {}).get(name)
        if actual != expected:
            raise ValueError(
                f'locked config mismatch for {name}: {actual!r} != '
                f'{expected!r}'
            )
    for section, names in (
        ('checkpoint', ('path', 'sha256', 'config_sha256')),
        (
            'dataset',
            (
                'path',
                'repository',
                'revision',
                'expected_bytes',
                'pixels_column',
                'action_column',
                'episode_column',
                'step_column',
            ),
        ),
    ):
        for name in names:
            if config.get(section, {}).get(name) in (None, ''):
                raise ValueError(f'{section}.{name} is required')


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
        code = (
            'SSP_V2_FORMAL_ROOT_EXISTS'
            if formal
            else 'SSP_V2_OUTPUT_ROOT_EXISTS'
        )
        raise SSPV2Failure(
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
