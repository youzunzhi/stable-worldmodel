"""Orthogonal antithetic ES, diagnostics, and selection for SSP-v2."""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .contracts import (
    LOCKED,
    PROTOCOL_ID,
    REPLICATE_SEEDS,
    VALIDATION_SEEDS,
    SSPV2Failure,
    append_jsonl,
    create_root,
    hash_inventory,
    sha256_file,
    write_json,
)
from .geometry import orthogonal_directions
from .pairs import _load_model
from .planner import FrozenSSPV2Planner, PlanningResult, model_state_sha256


def _load_preparation(preparation_dir: str | Path, config: dict) -> dict:
    root = Path(preparation_dir).expanduser().resolve()
    summary_path = root / 'preparation.completed.json'
    if not summary_path.is_file():
        raise SSPV2Failure('SSP_V2_INCOMPLETE', f'missing {summary_path}')
    summary = json.loads(summary_path.read_text())
    if summary.get('protocol_id') != PROTOCOL_ID:
        raise SSPV2Failure(
            'SSP_V2_INPUT_HASH_MISMATCH', 'preparation protocol mismatch'
        )
    if summary.get('task') != config['task']:
        raise SSPV2Failure(
            'SSP_V2_INPUT_HASH_MISMATCH', 'preparation task mismatch'
        )
    if sha256_file(root / 'pair_latents.pt') != summary['pair_latents_sha256']:
        raise SSPV2Failure(
            'SSP_V2_INPUT_HASH_MISMATCH', 'pair latent hash mismatch'
        )
    if (
        sha256_file(root / 'action_effect_basis.npy')
        != summary['action_effect_basis']['sha256']
    ):
        raise SSPV2Failure(
            'SSP_V2_INPUT_HASH_MISMATCH', 'action-effect basis hash mismatch'
        )
    return {'root': root, 'summary': summary}


def _copy_preparation(source: Path, target: Path) -> None:
    for name in (
        'source.json',
        'environment.json',
        'pre_registered_config.json',
        'protocol.json',
        'input_hashes.json',
        'preparation.completed.json',
        'pair_latents.pt',
        'action_effect_basis.npy',
        'action_effect_basis.json',
        'action_stats.json',
        'action_effect_inputs.pt',
    ):
        shutil.copy2(source / name, target / name)
    shutil.copytree(source / 'pair_manifests', target / 'pair_manifests')


def _load_latents(path: str | Path) -> dict:
    return torch.load(path, map_location='cpu', weights_only=True)


def _pair_lookup(
    latents: dict, split: str
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    payload = latents[split]
    return {
        pair_id: (payload['start'][index], payload['goal'][index])
        for index, pair_id in enumerate(payload['pair_ids'])
    }


def _planner(
    *,
    config: dict,
    model: torch.nn.Module,
    preparation_root: Path,
    device: str,
    num_samples: int | None = None,
    n_steps: int | None = None,
) -> FrozenSSPV2Planner:
    action_stats = json.loads(
        (preparation_root / 'action_stats.json').read_text()
    )
    basis = torch.from_numpy(
        np.load(preparation_root / 'action_effect_basis.npy')
    )
    return FrozenSSPV2Planner(
        model=model,
        basis=basis,
        epsilon=float(config['epsilon_task']),
        action_mean=torch.tensor(action_stats['mean']),
        action_std=torch.tensor(action_stats['std']),
        device=device,
        num_samples=num_samples or LOCKED['num_samples'],
        n_steps=n_steps or LOCKED['n_steps'],
        topk=min(LOCKED['topk'], num_samples or LOCKED['num_samples']),
        horizon=LOCKED['horizon'],
        action_block=LOCKED['action_block'],
        var_scale=LOCKED['var_scale'],
        late_hit_iterations=min(
            LOCKED['late_hit_iterations'], n_steps or LOCKED['n_steps']
        ),
        hit_mass_beta=LOCKED['hit_mass_beta'],
    )


def _validate_crn(
    left: list[PlanningResult], right: list[PlanningResult]
) -> None:
    if len(left) != len(right):
        raise SSPV2Failure('SSP_V2_CRN_MISMATCH', 'result lengths differ')
    for plus, minus in zip(left, right):
        left_noise = plus.noise
        right_noise = minus.noise
        fields = (
            'noise_schedule_id',
            'seed',
            'pre_state_sha256',
            'post_state_sha256',
            'standard_normal_blocks',
            'standard_normal_sha256',
        )
        if any(left_noise[field] != right_noise[field] for field in fields):
            raise SSPV2Failure(
                'SSP_V2_CRN_MISMATCH',
                f'noise mismatch for pair {plus.pair_id}',
            )


def _evaluate(
    *,
    planner: FrozenSSPV2Planner,
    lookup: dict[str, tuple[torch.Tensor, torch.Tensor]],
    pair_ids: list[str],
    theta: torch.Tensor,
    key_base: dict[str, Any],
    planner_seeds: tuple[int, ...] | list[int],
    noise_path: Path | None,
    geometry_label: str,
) -> list[PlanningResult]:
    results = []
    for pair_slot, pair_id in enumerate(pair_ids):
        start, goal = lookup[pair_id]
        for planner_seed in planner_seeds:
            key = {
                **key_base,
                'pair_slot': pair_slot,
                'pair_id': pair_id,
                'planner_seed': int(planner_seed),
            }
            result = planner.run(
                pair_id=pair_id,
                start_embedding=start,
                goal_embedding=goal,
                theta=theta,
                noise_key=key,
            )
            results.append(result)
            if noise_path is not None:
                append_jsonl(
                    noise_path,
                    {'geometry': geometry_label, **result.noise},
                )
    return results


def _result_payload(results: list[PlanningResult]) -> dict:
    rewards = [row.fixed_budget_reward for row in results]
    returned = [row.returned_verified_hit for row in results]
    late_mass = [
        float(np.mean(row.population_hit_fractions[-5:])) for row in results
    ]
    return {
        'observations': len(results),
        'mean_fixed_budget_reward': float(np.mean(rewards)),
        'returned_hit_rate': float(np.mean(returned)),
        'mean_late_hit_mass': float(np.mean(late_mass)),
        'mean_first_hit_auc': float(
            np.mean([row.first_hit_auc for row in results])
        ),
        'fixed_budget_rewards': rewards,
        'returned_verified_hits': returned,
        'first_hit_iterations': [row.first_hit_iteration for row in results],
        'return_modes': [row.return_mode for row in results],
        'clip_fraction': float(
            sum(row.action_clip['clipped_values'] for row in results)
            / max(sum(row.action_clip['total_values'] for row in results), 1)
        ),
    }


def _standard_error(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return (
        float(values.std(ddof=1) / math.sqrt(len(values)))
        if len(values) > 1
        else 0.0
    )


def select_validation_checkpoint(validations: list[dict]) -> dict:
    """One-SE best candidate plus paired one-SE identity promotion gate."""
    if not validations or validations[0]['checkpoint_step'] != 0:
        raise ValueError('validation sequence must start at exact identity')
    for row in validations:
        rewards = np.asarray(row['fixed_budget_rewards'], dtype=np.float64)
        row['reward_se'] = _standard_error(rewards)
    maximum = max(
        validations,
        key=lambda row: (
            row['mean_fixed_budget_reward'],
            -row['checkpoint_step'],
        ),
    )
    threshold = maximum['mean_fixed_budget_reward'] - maximum['reward_se']
    candidate = min(
        (
            row
            for row in validations
            if row['mean_fixed_budget_reward'] >= threshold
        ),
        key=lambda row: row['checkpoint_step'],
    )
    identity_hits = np.asarray(
        validations[0]['returned_verified_hits'], dtype=np.float64
    )
    candidate_hits = np.asarray(
        candidate['returned_verified_hits'], dtype=np.float64
    )
    if identity_hits.shape != candidate_hits.shape:
        raise ValueError('validation observations are not paired')
    delta = candidate_hits - identity_hits
    delta_mean = float(delta.mean())
    delta_se = _standard_error(delta)
    promoted = candidate['checkpoint_step'] != 0 and delta_mean > delta_se
    selected = candidate if promoted else validations[0]
    return {
        'best_candidate': candidate,
        'promoted': promoted,
        'promoted_checkpoint': selected,
        'paired_returned_hit_delta': delta_mean,
        'paired_returned_hit_delta_se': delta_se,
        'promotion_rule': 'paired mean returned-hit gain > one standard error',
        'candidate_rule': 'earliest checkpoint within one SE of max reward',
    }


def _validation(
    *,
    planner: FrozenSSPV2Planner,
    lookup: dict[str, tuple[torch.Tensor, torch.Tensor]],
    pair_ids: list[str],
    theta: torch.Tensor,
    task: str,
    replicate: int,
    checkpoint_step: int,
    noise_path: Path,
) -> dict:
    results = _evaluate(
        planner=planner,
        lookup=lookup,
        pair_ids=pair_ids,
        theta=theta,
        key_base={
            'protocol_id': PROTOCOL_ID,
            'task': task,
            'replicate': replicate,
            'phase': 'validation',
        },
        planner_seeds=VALIDATION_SEEDS,
        noise_path=noise_path,
        geometry_label=f'step-{checkpoint_step:03d}',
    )
    return {
        'checkpoint_step': checkpoint_step,
        'center': theta.detach().cpu().tolist(),
        **_result_payload(results),
    }


def _outer_gradient(
    *,
    planner: FrozenSSPV2Planner,
    lookup: dict[str, tuple[torch.Tensor, torch.Tensor]],
    pair_ids: list[str],
    center: torch.Tensor,
    directions: np.ndarray,
    sigma: float,
    key_base: dict[str, Any],
    planner_seeds: tuple[int, ...] | list[int],
    noise_path: Path | None,
) -> tuple[torch.Tensor, list[dict]]:
    gradient = torch.zeros_like(center)
    records = []
    for direction_index, raw_eta in enumerate(directions):
        eta = torch.from_numpy(raw_eta).to(dtype=center.dtype)
        key = {**key_base, 'direction': direction_index}
        plus = _evaluate(
            planner=planner,
            lookup=lookup,
            pair_ids=pair_ids,
            theta=center + float(sigma) * eta,
            key_base=key,
            planner_seeds=planner_seeds,
            noise_path=noise_path,
            geometry_label='plus',
        )
        minus = _evaluate(
            planner=planner,
            lookup=lookup,
            pair_ids=pair_ids,
            theta=center - float(sigma) * eta,
            key_base=key,
            planner_seeds=planner_seeds,
            noise_path=noise_path,
            geometry_label='minus',
        )
        _validate_crn(plus, minus)
        plus_payload = _result_payload(plus)
        minus_payload = _result_payload(minus)
        delta = (
            plus_payload['mean_fixed_budget_reward']
            - minus_payload['mean_fixed_budget_reward']
        )
        gradient.add_(eta, alpha=delta)
        records.append(
            {
                'direction': direction_index,
                'eta': eta.tolist(),
                'plus': plus_payload,
                'minus': minus_payload,
                'raw_delta': delta,
            }
        )
    gradient.div_(2 * len(directions) * float(sigma))
    return gradient, records


def _mini_overfit(
    *,
    planner: FrozenSSPV2Planner,
    lookup: dict[str, tuple[torch.Tensor, torch.Tensor]],
    pair_ids: list[str],
    seed: int,
    label: str,
    steps: int = 10,
) -> dict:
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    center = torch.nn.Parameter(torch.zeros(LOCKED['parameter_dim']))
    optimizer = torch.optim.Adam(
        [center],
        lr=LOCKED['learning_rate'],
        betas=tuple(LOCKED['adam_betas']),
        eps=LOCKED['adam_epsilon'],
        maximize=True,
    )
    trace = []
    for step in range(1, steps + 1):
        directions = orthogonal_directions(
            rng, LOCKED['directions'], LOCKED['parameter_dim']
        )
        gradient, records = _outer_gradient(
            planner=planner,
            lookup=lookup,
            pair_ids=pair_ids,
            center=center.detach(),
            directions=directions,
            sigma=LOCKED['sigma'],
            key_base={
                'protocol_id': PROTOCOL_ID,
                'phase': f'diagnostic-{label}',
                'outer_step': step,
            },
            planner_seeds=(0,),
            noise_path=None,
        )
        optimizer.zero_grad(set_to_none=True)
        center.grad = gradient
        optimizer.step()
        trace.append(
            {
                'step': step,
                'gradient_norm': float(gradient.norm()),
                'nonzero_direction_deltas': sum(
                    row['raw_delta'] != 0.0 for row in records
                ),
                'center_norm': float(center.detach().norm()),
            }
        )
    return {
        'pair_ids': pair_ids,
        'trace': trace,
        'center': center.detach().tolist(),
    }


def run_smoke(
    *,
    config: dict,
    preparation_dir: str | Path,
    output_dir: str | Path,
    device: str,
) -> dict:
    """Run real-checkpoint local-response and progressive-overfit diagnostics."""
    preparation = _load_preparation(preparation_dir, config)
    root = create_root(output_dir, formal=False)
    protocol = json.loads((preparation['root'] / 'protocol.json').read_text())
    protocol['formal_evidence'] = False
    protocol['diagnostic_only'] = True
    write_json(root / 'protocol.json', protocol)
    model = _load_model(Path(config['checkpoint']['path']), device)
    before = model_state_sha256(model)
    planner = _planner(
        config=config,
        model=model,
        preparation_root=preparation['root'],
        device=device,
    )
    latents = _load_latents(preparation['root'] / 'pair_latents.pt')
    lookup = _pair_lookup(latents, 'train')
    pair_ids = list(latents['train']['pair_ids'])
    zero = torch.zeros(LOCKED['parameter_dim'])
    identity = _evaluate(
        planner=planner,
        lookup=lookup,
        pair_ids=pair_ids[:8],
        theta=zero,
        key_base={
            'protocol_id': PROTOCOL_ID,
            'task': config['task'],
            'phase': 'diagnostic-identity-funnel',
        },
        planner_seeds=(0, 1),
        noise_path=root / 'noise_schedule.jsonl',
        geometry_label='identity',
    )
    rng = np.random.Generator(np.random.PCG64DXSM(26082299))
    response = []
    for sigma in (0.25, 0.5, 1.0):
        directions = orthogonal_directions(rng, 8, LOCKED['parameter_dim'])
        gradient, records = _outer_gradient(
            planner=planner,
            lookup=lookup,
            pair_ids=pair_ids[:4],
            center=zero,
            directions=directions,
            sigma=sigma,
            key_base={
                'protocol_id': PROTOCOL_ID,
                'task': config['task'],
                'phase': 'diagnostic-local-response',
                'sigma': sigma,
            },
            planner_seeds=(0, 1),
            noise_path=root / 'noise_schedule.jsonl',
        )
        response.append(
            {
                'sigma': sigma,
                'gradient_norm': float(gradient.norm()),
                'nonzero_direction_deltas': sum(
                    row['raw_delta'] != 0.0 for row in records
                ),
                'direction_deltas': [row['raw_delta'] for row in records],
            }
        )
    overfit = {
        'single_pair': _mini_overfit(
            planner=planner,
            lookup=lookup,
            pair_ids=pair_ids[:1],
            seed=26082301,
            label='single-pair-overfit',
        ),
        'eight_pairs': _mini_overfit(
            planner=planner,
            lookup=lookup,
            pair_ids=pair_ids[:8],
            seed=26082302,
            label='eight-pair-overfit',
        ),
    }
    after = model_state_sha256(model)
    if before != after:
        raise SSPV2Failure(
            'SSP_V2_FROZEN_MODEL_MUTATION', 'model changed during diagnostic'
        )
    summary = {
        'protocol_id': PROTOCOL_ID,
        'formal_evidence': False,
        'task': config['task'],
        'identity_funnel': _result_payload(identity),
        'local_response_locked_sigma_not_selected_here': response,
        'progressive_overfit': overfit,
        'model_state_sha256_before': before,
        'model_state_sha256_after': after,
    }
    write_json(root / 'smoke.completed.json', summary)
    write_json(root / 'audit.json', {'sha256': hash_inventory(root)})
    return summary


def run_training(
    *,
    config: dict,
    preparation_dir: str | Path,
    output_dir: str | Path,
    replicate_seed: int,
    device: str,
) -> dict:
    """Run one create-only formal SSP-v2 training replicate."""
    if replicate_seed not in REPLICATE_SEEDS:
        raise ValueError(f'unregistered replicate seed {replicate_seed}')
    started = time.monotonic()
    preparation = _load_preparation(preparation_dir, config)
    if not preparation['summary'].get('formal_evidence'):
        raise SSPV2Failure(
            'SSP_V2_INPUT_HASH_MISMATCH',
            'formal training requires formal preparation',
        )
    root = create_root(output_dir, formal=True)
    _copy_preparation(preparation['root'], root)
    for name in ('checkpoints', 'profile', 'clear'):
        (root / name).mkdir()
    protocol = json.loads((root / 'protocol.json').read_text())
    protocol['formal_evidence'] = True
    protocol['replicate_seed'] = replicate_seed
    write_json(root / 'protocol.json', protocol)

    model = _load_model(Path(config['checkpoint']['path']), device)
    before_model_hash = model_state_sha256(model)
    if (
        before_model_hash
        != preparation['summary']['model_state_sha256_before']
    ):
        raise SSPV2Failure(
            'SSP_V2_INPUT_HASH_MISMATCH',
            'model state differs from preparation',
        )
    planner = _planner(
        config=config,
        model=model,
        preparation_root=root,
        device=device,
    )
    latents = _load_latents(root / 'pair_latents.pt')
    train_lookup = _pair_lookup(latents, 'train')
    validation_lookup = _pair_lookup(latents, 'validation')
    train_ids = list(latents['train']['pair_ids'])
    validation_ids = list(latents['validation']['pair_ids'])
    if len(train_ids) != 800 or len(validation_ids) != 256:
        raise SSPV2Failure(
            'SSP_V2_INPUT_HASH_MISMATCH', 'prepared pair counts are not locked'
        )
    rng = np.random.Generator(np.random.PCG64DXSM(replicate_seed))
    train_batches = []
    for epoch in range(LOCKED['epochs']):
        permutation = np.asarray(train_ids, dtype=object)
        rng.shuffle(permutation)
        for index in range(0, len(permutation), LOCKED['outer_pair_batch']):
            train_batches.append(
                {
                    'epoch': epoch + 1,
                    'pair_ids': permutation[
                        index : index + LOCKED['outer_pair_batch']
                    ].tolist(),
                }
            )
    if len(train_batches) != LOCKED['outer_steps']:
        raise AssertionError('outer batch schedule does not match 100 steps')

    center = torch.nn.Parameter(torch.zeros(LOCKED['parameter_dim']))
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
    initial = _validation(
        planner=planner,
        lookup=validation_lookup,
        pair_ids=validation_ids,
        theta=center.detach(),
        task=config['task'],
        replicate=replicate_seed,
        checkpoint_step=0,
        noise_path=noise_path,
    )
    validations.append(initial)
    append_jsonl(validation_path, initial)
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

    for outer_step, batch in enumerate(train_batches, start=1):
        center_before = center.detach().clone()
        directions = orthogonal_directions(
            rng, LOCKED['directions'], LOCKED['parameter_dim']
        )
        gradient, direction_records = _outer_gradient(
            planner=planner,
            lookup=train_lookup,
            pair_ids=batch['pair_ids'],
            center=center_before,
            directions=directions,
            sigma=LOCKED['sigma'],
            key_base={
                'protocol_id': PROTOCOL_ID,
                'task': config['task'],
                'replicate': replicate_seed,
                'outer_step': outer_step,
                'epoch': batch['epoch'],
                'phase': 'outer-training',
            },
            planner_seeds=tuple(range(LOCKED['planner_tapes'])),
            noise_path=noise_path,
        )
        optimizer.zero_grad(set_to_none=True)
        center.grad = gradient
        optimizer.step()
        if not torch.isfinite(center).all():
            raise SSPV2Failure(
                'SSP_V2_INCOMPLETE', 'non-finite optimizer center'
            )
        state = optimizer.state[center]
        record = {
            'outer_step': outer_step,
            'epoch': batch['epoch'],
            'pair_ids': batch['pair_ids'],
            'center_before': center_before.tolist(),
            'directions': direction_records,
            'raw_es_gradient': gradient.tolist(),
            'center_after': center.detach().tolist(),
            'adam': {
                'step': int(state['step'].item()),
                'exp_avg_norm': float(state['exp_avg'].norm()),
                'exp_avg_sq_norm': float(state['exp_avg_sq'].norm()),
            },
            'action_effect_basis_sha256': sha256_file(
                root / 'action_effect_basis.npy'
            ),
        }
        append_jsonl(root / 'outer_steps.jsonl', record)
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
                theta=center.detach(),
                task=config['task'],
                replicate=replicate_seed,
                checkpoint_step=outer_step,
                noise_path=noise_path,
            )
            validations.append(validation)
            append_jsonl(validation_path, validation)

    selection = select_validation_checkpoint(validations)
    candidate = selection['best_candidate']
    promoted = selection['promoted_checkpoint']
    candidate_checkpoint = (
        root / 'checkpoints' / f'step-{candidate["checkpoint_step"]:03d}.pt'
    )
    promoted_checkpoint = (
        root / 'checkpoints' / f'step-{promoted["checkpoint_step"]:03d}.pt'
    )
    shutil.copy2(candidate_checkpoint, root / 'best_candidate_geometry.pt')
    shutil.copy2(promoted_checkpoint, root / 'selected_geometry.pt')
    selected_payload = {
        'selection_metric': 'validation fixed-budget reward',
        'best_candidate_step': candidate['checkpoint_step'],
        'best_candidate_reward': candidate['mean_fixed_budget_reward'],
        'promotion_passed': selection['promoted'],
        'promoted_step': promoted['checkpoint_step'],
        'promoted_is_identity': promoted['checkpoint_step'] == 0,
        'paired_returned_hit_delta': selection['paired_returned_hit_delta'],
        'paired_returned_hit_delta_se': selection[
            'paired_returned_hit_delta_se'
        ],
        'candidate_rule': selection['candidate_rule'],
        'promotion_rule': selection['promotion_rule'],
        'best_candidate_sha256': sha256_file(
            root / 'best_candidate_geometry.pt'
        ),
        'selected_geometry_sha256': sha256_file(root / 'selected_geometry.pt'),
    }
    write_json(root / 'selected_geometry.json', selected_payload)
    after_model_hash = model_state_sha256(model)
    if before_model_hash != after_model_hash:
        raise SSPV2Failure(
            'SSP_V2_FROZEN_MODEL_MUTATION', 'model changed during training'
        )
    completion = {
        'protocol_id': PROTOCOL_ID,
        'task': config['task'],
        'replicate_seed': replicate_seed,
        'terminal_code': 'SSP_V2_COMPLETED',
        'completed_outer_steps': LOCKED['outer_steps'],
        'completed_epochs': LOCKED['epochs'],
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
