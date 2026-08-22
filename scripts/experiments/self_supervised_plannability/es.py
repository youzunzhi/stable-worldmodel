"""Antithetic ES training and real-checkpoint smoke gates for SSP."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .contracts import (
    LOCKED,
    PROTOCOL_ID,
    SSPFailure,
    append_jsonl,
    create_root,
    hash_inventory,
    sha256_file,
    write_json,
)
from .geometry import project_parameter
from .pairs import _load_model
from .planner import FrozenSSPPlanner, PlanningResult, model_state_sha256


def select_validation_checkpoint(validations: list[dict]) -> dict:
    """Select by validation AUC only, breaking exact ties toward step zero."""
    if not validations:
        raise ValueError('at least one validation checkpoint is required')
    required = {'checkpoint_step', 'auc', 'center'}
    if any(not required.issubset(row) for row in validations):
        raise ValueError('validation record is missing selection fields')
    return max(
        validations, key=lambda row: (row['auc'], -row['checkpoint_step'])
    )


def _load_preparation(preparation_dir: str | Path, config: dict) -> dict:
    root = Path(preparation_dir).expanduser().resolve()
    summary_path = root / 'preparation.completed.json'
    if not summary_path.is_file():
        raise SSPFailure('SSP_INCOMPLETE', f'missing {summary_path}')
    summary = json.loads(summary_path.read_text())
    if summary.get('protocol_id') != PROTOCOL_ID:
        raise SSPFailure(
            'SSP_INPUT_HASH_MISMATCH', 'preparation protocol mismatch'
        )
    if summary.get('task') != config['task']:
        raise SSPFailure(
            'SSP_INPUT_HASH_MISMATCH', 'preparation task mismatch'
        )
    if sha256_file(root / 'pair_latents.pt') != summary['pair_latents_sha256']:
        raise SSPFailure(
            'SSP_INPUT_HASH_MISMATCH', 'prepared latent cache hash mismatch'
        )
    return {'root': root, 'summary': summary}


def _load_latents(path: Path) -> dict:
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')


def _pair_lookup(latents: dict, split: str) -> dict[str, tuple]:
    payload = latents[split]
    return {
        pair_id: (payload['start'][index], payload['goal'][index])
        for index, pair_id in enumerate(payload['pair_ids'])
    }


def _copy_preparation(preparation: Path, root: Path) -> None:
    for name in (
        'protocol.json',
        'source.json',
        'environment.json',
        'input_hashes.json',
        'pre_registered_config.json',
        'geometry_basis.npy',
        'geometry_basis.json',
        'pair_latents.pt',
        'preparation.completed.json',
    ):
        shutil.copy2(preparation / name, root / name)
    shutil.copytree(preparation / 'pair_manifests', root / 'pair_manifests')


def _action_dim(config: dict) -> int:
    import h5py
    import hdf5plugin  # noqa: F401

    with h5py.File(config['dataset']['path'], 'r') as dataset:
        return int(np.prod(dataset['action'].shape[1:]))


def _planner(
    *,
    config: dict,
    model: torch.nn.Module,
    basis: torch.Tensor,
    device: str,
    num_samples: int,
    n_steps: int,
) -> FrozenSSPPlanner:
    topk = min(LOCKED['topk'], num_samples)
    return FrozenSSPPlanner(
        model=model,
        basis=basis,
        epsilon=float(config['epsilon_task']),
        action_dim=_action_dim(config),
        device=device,
        num_samples=num_samples,
        n_steps=n_steps,
        topk=topk,
        horizon=LOCKED['horizon'],
        action_block=LOCKED['action_block'],
        var_scale=LOCKED['var_scale'],
    )


def _evaluate(
    *,
    planner: FrozenSSPPlanner,
    lookup: dict[str, tuple],
    pair_ids: list[str],
    psi: torch.Tensor,
    key_base: dict[str, Any],
    noise_path: Path,
    sign: str,
) -> list[PlanningResult]:
    output = []
    for pair_slot, pair_id in enumerate(pair_ids):
        start, goal = lookup[pair_id]
        key = {**key_base, 'pair_slot': pair_slot, 'pair_id': pair_id}
        result = planner.run(
            pair_id=pair_id,
            start_embedding=start,
            goal_embedding=goal,
            psi=psi,
            noise_key=key,
        )
        append_jsonl(
            noise_path,
            {
                'sign': sign,
                'pair_id': pair_id,
                **result.noise,
            },
        )
        output.append(result)
    return output


def _validate_crn(
    plus: list[PlanningResult], minus: list[PlanningResult]
) -> None:
    if [row.pair_id for row in plus] != [row.pair_id for row in minus]:
        raise SSPFailure('SSP_CRN_MISMATCH', 'antithetic pair order differs')
    for left, right in zip(plus, minus):
        left_noise = left.noise
        right_noise = right.noise
        fields = (
            'noise_schedule_id',
            'pre_state_sha256',
            'post_state_sha256',
            'standard_normal_blocks',
            'standard_normal_sha256',
        )
        if any(left_noise[name] != right_noise[name] for name in fields):
            raise SSPFailure(
                'SSP_CRN_MISMATCH',
                f'antithetic noise mismatch for {left.pair_id}',
            )


def _result_payload(results: list[PlanningResult]) -> dict:
    return {
        'pair_ids': [row.pair_id for row in results],
        'first_hit_iteration': [row.first_hit_iteration for row in results],
        'hit_bits': [row.hit_bits for row in results],
        'auc_rewards': [row.auc_reward for row in results],
        'mean_auc_reward': float(np.mean([row.auc_reward for row in results])),
        'noise_schedule_ids': [
            row.noise['noise_schedule_id'] for row in results
        ],
        'noise_content_sha256': [
            row.noise['standard_normal_sha256'] for row in results
        ],
    }


def _validation(
    *,
    planner: FrozenSSPPlanner,
    lookup: dict[str, tuple],
    pair_ids: list[str],
    psi: torch.Tensor,
    task: str,
    replicate: int,
    checkpoint_step: int,
    noise_path: Path,
) -> dict:
    results = _evaluate(
        planner=planner,
        lookup=lookup,
        pair_ids=pair_ids,
        psi=psi,
        key_base={
            'protocol_id': PROTOCOL_ID,
            'task': task,
            'replicate': replicate,
            'phase': 'validation',
            'planner_seed': LOCKED['validation_seed'],
        },
        noise_path=noise_path,
        sign=f'validation-step-{checkpoint_step:03d}',
    )
    payload = _result_payload(results)
    return {
        'checkpoint_step': checkpoint_step,
        'center': psi.detach().cpu().tolist(),
        'auc': payload['mean_auc_reward'],
        'endpoint_solve_rate': float(
            np.mean([row.first_hit_iteration is not None for row in results])
        ),
        'first_hit_iteration': payload['first_hit_iteration'],
    }


def run_smoke(
    *,
    config: dict,
    preparation_dir: str | Path,
    output_dir: str | Path,
    device: str,
) -> dict:
    """Run the locked short and four-pair full-budget smoke gates."""
    preparation = _load_preparation(preparation_dir, config)
    root = create_root(output_dir, formal=False)
    _copy_preparation(preparation['root'], root)
    protocol = json.loads((root / 'protocol.json').read_text())
    protocol['formal_evidence'] = False
    protocol['smoke_only'] = True
    write_json(root / 'protocol.json', protocol)
    checkpoint = Path(config['checkpoint']['path'])
    model = _load_model(checkpoint, device)
    before = model_state_sha256(model)
    basis = torch.from_numpy(np.load(root / 'geometry_basis.npy'))
    latents = _load_latents(root / 'pair_latents.pt')
    lookup = _pair_lookup(latents, 'train')
    pair_ids = list(latents['train']['pair_ids'])
    zero = torch.zeros(16)
    eta_rng = np.random.Generator(np.random.PCG64DXSM(260822))
    eta = torch.from_numpy(eta_rng.standard_normal(16).astype(np.float32))
    plus, _, _ = project_parameter(basis, zero + 0.25 * eta)
    minus, _, _ = project_parameter(basis, zero - 0.25 * eta)
    noise_path = root / 'noise_schedule.jsonl'

    short_planner = _planner(
        config=config,
        model=model,
        basis=basis,
        device=device,
        num_samples=300,
        n_steps=3,
    )
    short_key = {
        'protocol_id': PROTOCOL_ID,
        'task': config['task'],
        'replicate': 'smoke',
        'outer_step': 0,
        'direction': 0,
        'phase': 'short-smoke',
    }
    identity_result = _evaluate(
        planner=short_planner,
        lookup=lookup,
        pair_ids=pair_ids[:1],
        psi=zero,
        key_base={**short_key, 'geometry': 'identity'},
        noise_path=noise_path,
        sign='identity',
    )
    short_plus = _evaluate(
        planner=short_planner,
        lookup=lookup,
        pair_ids=pair_ids[:1],
        psi=plus,
        key_base=short_key,
        noise_path=noise_path,
        sign='plus',
    )
    short_minus = _evaluate(
        planner=short_planner,
        lookup=lookup,
        pair_ids=pair_ids[:1],
        psi=minus,
        key_base=short_key,
        noise_path=noise_path,
        sign='minus',
    )
    _validate_crn(short_plus, short_minus)

    full_planner = _planner(
        config=config,
        model=model,
        basis=basis,
        device=device,
        num_samples=LOCKED['num_samples'],
        n_steps=LOCKED['n_steps'],
    )
    full_key = {
        'protocol_id': PROTOCOL_ID,
        'task': config['task'],
        'replicate': 'smoke',
        'outer_step': 0,
        'direction': 0,
        'phase': 'full-budget-smoke',
    }
    full_plus = _evaluate(
        planner=full_planner,
        lookup=lookup,
        pair_ids=pair_ids[:4],
        psi=plus,
        key_base=full_key,
        noise_path=noise_path,
        sign='plus',
    )
    full_minus = _evaluate(
        planner=full_planner,
        lookup=lookup,
        pair_ids=pair_ids[:4],
        psi=minus,
        key_base=full_key,
        noise_path=noise_path,
        sign='minus',
    )
    _validate_crn(full_plus, full_minus)
    after = model_state_sha256(model)
    if before != after:
        raise SSPFailure(
            'SSP_FROZEN_MODEL_MUTATION', 'model changed during smoke'
        )
    summary = {
        'protocol_id': PROTOCOL_ID,
        'formal_evidence': False,
        'short_identity': _result_payload(identity_result),
        'short_plus': _result_payload(short_plus),
        'short_minus': _result_payload(short_minus),
        'full_plus': _result_payload(full_plus),
        'full_minus': _result_payload(full_minus),
        'model_state_sha256_before': before,
        'model_state_sha256_after': after,
        'environment_constructor_calls': 0,
        'environment_step_calls': 0,
    }
    write_json(root / 'smoke.completed.json', summary)
    return summary


def run_training(
    *,
    config: dict,
    preparation_dir: str | Path,
    output_dir: str | Path,
    replicate_seed: int,
    device: str,
) -> dict:
    """Run one create-only formal SSP training replicate."""
    if replicate_seed not in config['replicate_seeds']:
        raise ValueError(f'unregistered replicate seed {replicate_seed}')
    started = time.monotonic()
    preparation = _load_preparation(preparation_dir, config)
    root = create_root(output_dir, formal=True)
    _copy_preparation(preparation['root'], root)
    for name in ('checkpoints', 'profile', 'clear'):
        (root / name).mkdir()
    protocol = json.loads((root / 'protocol.json').read_text())
    protocol['formal_evidence'] = True
    protocol['replicate_seed'] = replicate_seed
    write_json(root / 'protocol.json', protocol)

    checkpoint = Path(config['checkpoint']['path'])
    model = _load_model(checkpoint, device)
    before_model_hash = model_state_sha256(model)
    expected_model_hash = preparation['summary']['model_state_sha256_before']
    if before_model_hash != expected_model_hash:
        raise SSPFailure(
            'SSP_INPUT_HASH_MISMATCH',
            'loaded model state differs from prepare',
        )
    basis = torch.from_numpy(np.load(root / 'geometry_basis.npy'))
    latents = _load_latents(root / 'pair_latents.pt')
    train_lookup = _pair_lookup(latents, 'train')
    validation_lookup = _pair_lookup(latents, 'validation')
    train_ids = list(latents['train']['pair_ids'])
    validation_ids = list(latents['validation']['pair_ids'])
    if len(train_ids) != 800 or len(validation_ids) != 256:
        raise SSPFailure(
            'SSP_INPUT_HASH_MISMATCH', 'prepared pair counts are not locked'
        )
    rng = np.random.Generator(np.random.PCG64DXSM(replicate_seed))
    permutation = np.asarray(train_ids, dtype=object)
    rng.shuffle(permutation)
    train_batches = [
        permutation[index : index + 16].tolist()
        for index in range(0, len(permutation), 16)
    ]
    if len(train_batches) != 50 or any(
        len(batch) != 16 for batch in train_batches
    ):
        raise AssertionError('formal outer batches are not 50 x 16')

    planner = _planner(
        config=config,
        model=model,
        basis=basis,
        device=device,
        num_samples=LOCKED['num_samples'],
        n_steps=LOCKED['n_steps'],
    )
    center = torch.nn.Parameter(torch.zeros(16, dtype=torch.float32))
    optimizer = torch.optim.Adam(
        [center],
        lr=LOCKED['learning_rate'],
        betas=tuple(LOCKED['adam_betas']),
        eps=LOCKED['adam_epsilon'],
        maximize=True,
    )
    noise_path = root / 'noise_schedule.jsonl'
    validation_path = root / 'validation.jsonl'
    validations = []
    initial_validation = _validation(
        planner=planner,
        lookup=validation_lookup,
        pair_ids=validation_ids,
        psi=center.detach(),
        task=config['task'],
        replicate=replicate_seed,
        checkpoint_step=0,
        noise_path=noise_path,
    )
    validations.append(initial_validation)
    append_jsonl(validation_path, initial_validation)
    torch.save(
        {
            'protocol_id': PROTOCOL_ID,
            'task': config['task'],
            'replicate_seed': replicate_seed,
            'step': 0,
            'center': center.detach().cpu(),
            'optimizer': optimizer.state_dict(),
        },
        root / 'checkpoints' / 'step-000.pt',
    )

    no_signal_streak = 0
    projection_count = 0
    terminal_code = 'SSP_COMPLETED'
    completed_steps = 0
    no_signal_hit_classification = None
    for outer_step, pair_ids in enumerate(train_batches, start=1):
        center_before = center.detach().clone()
        directions = rng.standard_normal((8, 16)).astype(np.float32)
        direction_records = []
        observed_step_hits = []
        gradient = torch.zeros(16, dtype=torch.float32)
        deltas = []
        for direction_index, raw_eta in enumerate(directions):
            eta = torch.from_numpy(raw_eta)
            plus_psi, plus_projected, plus_scale = project_parameter(
                basis, center_before + LOCKED['sigma'] * eta
            )
            minus_psi, minus_projected, minus_scale = project_parameter(
                basis, center_before - LOCKED['sigma'] * eta
            )
            projection_count += int(plus_projected) + int(minus_projected)
            key = {
                'protocol_id': PROTOCOL_ID,
                'task': config['task'],
                'replicate': replicate_seed,
                'outer_step': outer_step,
                'direction': direction_index,
                'phase': 'outer-training',
            }
            plus_results = _evaluate(
                planner=planner,
                lookup=train_lookup,
                pair_ids=pair_ids,
                psi=plus_psi,
                key_base=key,
                noise_path=noise_path,
                sign='plus',
            )
            minus_results = _evaluate(
                planner=planner,
                lookup=train_lookup,
                pair_ids=pair_ids,
                psi=minus_psi,
                key_base=key,
                noise_path=noise_path,
                sign='minus',
            )
            _validate_crn(plus_results, minus_results)
            plus_payload = _result_payload(plus_results)
            minus_payload = _result_payload(minus_results)
            observed_step_hits.extend(
                value for rows in plus_payload['hit_bits'] for value in rows
            )
            observed_step_hits.extend(
                value for rows in minus_payload['hit_bits'] for value in rows
            )
            delta = (
                plus_payload['mean_auc_reward']
                - minus_payload['mean_auc_reward']
            )
            deltas.append(delta)
            gradient.add_(eta, alpha=delta)
            direction_records.append(
                {
                    'direction': direction_index,
                    'eta': eta.tolist(),
                    'psi_plus': plus_psi.tolist(),
                    'psi_minus': minus_psi.tolist(),
                    'plus_projected': plus_projected,
                    'minus_projected': minus_projected,
                    'plus_projection_scale': plus_scale,
                    'minus_projection_scale': minus_scale,
                    'plus': plus_payload,
                    'minus': minus_payload,
                    'raw_delta': delta,
                }
            )
        gradient.div_(2 * LOCKED['directions'] * LOCKED['sigma'])
        optimizer.zero_grad(set_to_none=True)
        center.grad = gradient
        optimizer.step()
        with torch.no_grad():
            projected_center, active, scale = project_parameter(basis, center)
            center.copy_(projected_center)
        projection_count += int(active)
        state = optimizer.state[center]
        record = {
            'outer_step': outer_step,
            'pair_ids': pair_ids,
            'center_before': center_before.tolist(),
            'directions': direction_records,
            'raw_es_gradient': gradient.tolist(),
            'center_projection_active': active,
            'center_projection_scale': scale,
            'center_after': center.detach().tolist(),
            'adam': {
                'step': int(state['step'].item()),
                'exp_avg_norm': float(state['exp_avg'].norm()),
                'exp_avg_sq_norm': float(state['exp_avg_sq'].norm()),
            },
            'basis_sha256': sha256_file(root / 'geometry_basis.npy'),
        }
        append_jsonl(root / 'outer_steps.jsonl', record)
        completed_steps = outer_step
        no_signal_streak = (
            no_signal_streak + 1
            if all(delta == 0.0 for delta in deltas)
            else 0
        )

        if outer_step % LOCKED['checkpoint_interval'] == 0:
            torch.save(
                {
                    'protocol_id': PROTOCOL_ID,
                    'task': config['task'],
                    'replicate_seed': replicate_seed,
                    'step': outer_step,
                    'center': center.detach().cpu(),
                    'optimizer': optimizer.state_dict(),
                },
                root / 'checkpoints' / f'step-{outer_step:03d}.pt',
            )
        if outer_step % LOCKED['validation_interval'] == 0:
            validation = _validation(
                planner=planner,
                lookup=validation_lookup,
                pair_ids=validation_ids,
                psi=center.detach(),
                task=config['task'],
                replicate=replicate_seed,
                checkpoint_step=outer_step,
                noise_path=noise_path,
            )
            validations.append(validation)
            append_jsonl(validation_path, validation)
        if no_signal_streak == 10:
            terminal_code = 'SSP_NO_LEARNING_SIGNAL'
            no_signal_hit_classification = (
                'all-one'
                if all(observed_step_hits)
                else 'all-zero'
                if not any(observed_step_hits)
                else 'mixed'
            )
            break

    selected = select_validation_checkpoint(validations)
    selected_step = int(selected['checkpoint_step'])
    selected_checkpoint = root / 'checkpoints' / f'step-{selected_step:03d}.pt'
    shutil.copy2(selected_checkpoint, root / 'selected_geometry.pt')
    selected_payload = {
        'selection_metric': 'validation_auc',
        'tie_break': 'earliest_outer_step',
        'selected_step': selected_step,
        'selected_validation_auc': selected['auc'],
        'selected_center': selected['center'],
        'step_zero_selected': selected_step == 0,
        'learned_improvement_claimed': selected_step != 0,
        'checkpoint_sha256': sha256_file(root / 'selected_geometry.pt'),
    }
    write_json(root / 'selected_geometry.json', selected_payload)
    after_model_hash = model_state_sha256(model)
    if before_model_hash != after_model_hash:
        raise SSPFailure(
            'SSP_FROZEN_MODEL_MUTATION', 'model changed during formal training'
        )
    completion = {
        'protocol_id': PROTOCOL_ID,
        'task': config['task'],
        'replicate_seed': replicate_seed,
        'terminal_code': terminal_code,
        'completed_outer_steps': completed_steps,
        'projection_incidence_count': projection_count,
        'no_signal_streak': no_signal_streak,
        'no_signal_hit_classification': no_signal_hit_classification,
        'selected_geometry': selected_payload,
        'model_state_sha256_before': before_model_hash,
        'model_state_sha256_after': after_model_hash,
        'environment_constructor_calls': 0,
        'environment_step_calls': 0,
        'elapsed_seconds': time.monotonic() - started,
    }
    write_json(root / 'audit.json', {'sha256': hash_inventory(root)})
    write_json(root / 'training.completed.json', completion)
    return completion
