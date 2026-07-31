"""Script to evaluate a World Model using MPC on a dataset of episodes."""

import json
import os

os.environ['MUJOCO_GL'] = 'egl'

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm

from clear_protocol import (
    CLEAR_LEWM_REVISION,
    install_success_criterion,
    load_manifest,
    manifest_sha256,
    resolve_manifest_pairs,
    seed_runtime,
    validate_dataset,
    validate_solver_config,
)


def img_transform(cfg, dtype=torch.float32):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(dtype, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def episode_col(dataset):
    """Name of the episode-index column, robust across dataset formats.

    HDF5 lists every column (including index columns) in ``column_names``,
    but the Lance reader deliberately excludes its index columns
    (``episode_idx``/``step_idx``) from ``column_names`` and only exposes
    them via ``_schema_names``. Consult both so ``episode_idx`` is found
    regardless of format.
    """
    names = set(dataset.column_names)
    names |= set(getattr(dataset, '_schema_names', ()))
    return 'episode_idx' if 'episode_idx' in names else 'ep_idx'


def get_episodes_length(dataset, episodes):
    col_name = episode_col(dataset)

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data('step_idx')
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name, keys_to_load=None):
    kwargs = {}
    if keys_to_load is not None:
        kwargs['keys_to_load'] = keys_to_load
    dataset = swm.data.load_dataset(
        dataset_name,
        cache_dir=cfg.get('cache_dir', None),
        keys_to_cache=list(cfg.dataset.keys_to_cache),
        **kwargs,
    )
    return dataset


def non_pixel_hdf5_keys(dataset_name):
    """List non-image columns without reading any HDF5 pixel payload."""
    import h5py

    path = Path(dataset_name).expanduser()
    if not path.is_file():
        return None
    with h5py.File(path, 'r') as dataset:
        return [
            key
            for key in dataset.keys()
            if key not in ('ep_len', 'ep_offset')
            and not key.startswith('pixels')
        ]


def prepare_results_path(cfg):
    """Choose the output path and reserve explicit evaluation directories."""
    if cfg.output.dir:
        results_path = Path(cfg.output.dir).expanduser().resolve()
        try:
            results_path.mkdir(parents=True)
        except FileExistsError as error:
            raise FileExistsError(
                f'Evaluation output directory already exists: {results_path}'
            ) from error
        return results_path

    if cfg.policy != 'random':
        results_path = Path(
            swm.data.utils.get_cache_dir(sub_folder='checkpoints'), cfg.policy
        ).parent
    else:
        results_path = Path(__file__).parent
    results_path.mkdir(parents=True, exist_ok=True)
    return results_path


def sample_evaluation_starts(
    dataset, episode_ids, goal_offset_steps, num_samples, seed
):
    """Sample dataset rows whose future goal remains in the same episode."""
    episode_lengths = get_episodes_length(dataset, episode_ids)
    max_start_steps = episode_lengths - goal_offset_steps - 1
    max_start_by_episode = dict(zip(episode_ids, max_start_steps))

    episode_column = episode_col(dataset)
    row_episode_ids = dataset.get_col_data(episode_column)
    row_max_start_steps = np.array(
        [max_start_by_episode[episode_id] for episode_id in row_episode_ids]
    )
    valid_mask = dataset.get_col_data('step_idx') <= row_max_start_steps
    valid_indices = np.flatnonzero(valid_mask)
    print(len(valid_indices), 'valid starting points found for evaluation.')

    if len(valid_indices) < num_samples:
        raise ValueError(
            f'Requested {num_samples} trajectories, but the dataset has only '
            f'{len(valid_indices)} valid starting points.'
        )

    generator = np.random.default_rng(seed)
    sampled_positions = generator.choice(
        len(valid_indices), size=num_samples, replace=False
    )
    return np.sort(valid_indices[sampled_positions])


@hydra.main(version_base=None, config_path='./config', config_name='pusht')
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    clear_manifest_path = cfg.eval.get('manifest')
    solver_ablation = bool(cfg.eval.get('solver_ablation', False))
    clear_solver_contract_matched = None
    clear_manifest = (
        load_manifest(clear_manifest_path) if clear_manifest_path else None
    )
    if clear_manifest is not None:
        expected_env = {
            'pusht': 'swm/PushT-v1',
            'cube': 'swm/OGBCube-v0',
        }[clear_manifest['task']]
        if cfg.world.env_name != expected_env:
            raise ValueError(
                f"CLEAR manifest task {clear_manifest['task']!r} requires "
                f'{expected_env}, got {cfg.world.env_name}'
            )
        protocol = clear_manifest['protocol']
        cfg.eval.num_eval = len(clear_manifest['pairs'])
        cfg.eval.goal_offset_steps = int(protocol['goal_offset'])
        cfg.eval.eval_budget = int(protocol['eval_budget'])
        try:
            validate_solver_config(cfg.solver)
        except ValueError:
            if not solver_ablation:
                raise
            clear_solver_contract_matched = False
        else:
            clear_solver_contract_matched = True
        seed_runtime(int(cfg.seed))

    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block
        <= cfg.eval.eval_budget
    ), 'Planning horizon must be smaller than or equal to eval_budget'
    results_path = prepare_results_path(cfg)

    # create world environment
    policy_name = cfg.get('policy', 'random')
    use_pixels = policy_name != 'random' or bool(cfg.eval.video)
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(
        **cfg.world,
        image_shape=(224, 224) if use_pixels else None,
        add_pixels=use_pixels,
    )

    # create the transform
    img_dtype = torch.bfloat16 if cfg.get('bf16', False) else torch.float32
    transform = {
        'pixels': img_transform(cfg, img_dtype),
        'goal': img_transform(cfg, img_dtype),
    }

    keys_to_load = (
        None
        if use_pixels
        else non_pixel_hdf5_keys(cfg.eval.dataset_name)
    )
    dataset = get_dataset(
        cfg, cfg.eval.dataset_name, keys_to_load=keys_to_load
    )
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = episode_col(dataset)
    if clear_manifest is not None:
        validate_dataset(dataset, clear_manifest)
    ep_indices, _ = np.unique(
        stats_dataset.get_col_data(col_name), return_index=True
    )

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ['pixels']:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != 'action':
            process[f'goal_{col}'] = process[col]

    # -- run evaluation
    policy = policy_name

    if policy != 'random':
        model = swm.wm.utils.load_pretrained(cfg.policy)
        if cfg.get('bf16', False):
            model = model.to(torch.bfloat16)
        model = model.to('cuda')
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        if cfg.get('compile', False):
            encoder_attr = (
                'backbone' if hasattr(model, 'backbone') else 'encoder'
            )
            setattr(
                model,
                encoder_attr,
                torch.compile(getattr(model, encoder_attr)),
            )
            model.predictor = torch.compile(model.predictor)
        config = swm.PlanConfig(**cfg.plan_config)
        objective = hydra.utils.instantiate(cfg.objective)
        cost = swm.planning.ShootingCostEvaluator(model, objective)
        solver = hydra.utils.instantiate(cfg.solver, cost=cost)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy(seed=int(cfg.seed))

    if clear_manifest is None:
        random_episode_indices = sample_evaluation_starts(
            dataset=dataset,
            episode_ids=ep_indices,
            goal_offset_steps=cfg.eval.goal_offset_steps,
            num_samples=cfg.eval.num_eval,
            seed=cfg.seed,
        )
        eval_episodes = dataset.get_col_data(col_name)[
            random_episode_indices
        ]
        eval_start_idx = dataset.get_col_data('step_idx')[
            random_episode_indices
        ]
    else:
        (
            random_episode_indices,
            eval_episodes,
            eval_start_idx,
        ) = resolve_manifest_pairs(dataset, clear_manifest)
    print(random_episode_indices)

    # Index the full index columns directly: the Lance reader excludes
    # index columns (episode_idx/step_idx) from get_row_data, but get_col_data
    # exposes them (and both are already cached from the checks above).
    if clear_manifest is not None:
        install_success_criterion(world, clear_manifest)

    # The planner operates in normalized dataset coordinates and may produce
    # values outside the environment's declared action space after inverse
    # normalization. Clip only the final action sent to the environment.
    original_get_action = policy.get_action
    action_low = world.envs.single_action_space.low
    action_high = world.envs.single_action_space.high

    def clipped_get_action(info_dict, **kwargs):
        action = original_get_action(info_dict, **kwargs)
        return np.clip(action, action_low, action_high)

    policy.get_action = clipped_get_action
    world.set_policy(policy)

    video_path = results_path if cfg.eval.video else None
    if video_path is None:
        print('[eval] video output disabled')
    else:
        print(
            f'[eval] saving videos to {video_path.resolve()} '
            '(one env_{i}.mp4 per env)'
        )

    autocast_ctx = torch.autocast(
        device_type='cuda',
        dtype=torch.bfloat16,
        enabled=cfg.get('bf16', False),
    )

    if cfg.get('compile', False):
        print('Warming up compiled model...')
        warmup_autocast_ctx = torch.autocast(
            device_type='cuda',
            dtype=torch.bfloat16,
            enabled=cfg.get('bf16', False),
        )
        with warmup_autocast_ctx:
            n = world.num_envs
            world.evaluate(
                dataset=dataset,
                start_steps=eval_start_idx.tolist()[:n],
                goal_offset=cfg.eval.goal_offset_steps,
                eval_budget=cfg.eval.eval_budget,
                episodes_idx=eval_episodes.tolist()[:n],
                callables=OmegaConf.to_container(
                    cfg.eval.get('callables'), resolve=True
                ),
                video=video_path,
            )
        print('Warmup done.')

    start_time = time.time()
    with autocast_ctx:
        metrics = world.evaluate(
            dataset=dataset,
            start_steps=eval_start_idx.tolist(),
            goal_offset=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_episodes.tolist(),
            callables=OmegaConf.to_container(
                cfg.eval.get('callables'), resolve=True
            ),
            video=video_path,
        )
    end_time = time.time()

    print(metrics)
    if video_path is not None:
        print(f'[eval] videos saved to {video_path.resolve()}')

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open('x') as stream:
        stream.write('==== CONFIG ====\n')
        stream.write(OmegaConf.to_yaml(cfg))
        stream.write('\n')

        stream.write('==== RESULTS ====\n')
        stream.write(f'metrics: {metrics}\n')
        stream.write(f'evaluation_time: {end_time - start_time} seconds\n')

    def jsonable(value):
        if isinstance(value, dict):
            return {str(key): jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [jsonable(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    completed = len(metrics['episode_successes'])
    structured = {
        'checkpoint': str(Path(cfg.policy).resolve()),
        'dataset': str(Path(cfg.eval.dataset_name).resolve()),
        'seed': int(cfg.seed),
        'requested_trajectories': int(cfg.eval.num_eval),
        'completed_trajectories': completed,
        'sampled_flat_indices': random_episode_indices.tolist(),
        'sampled_episode_indices': eval_episodes.tolist(),
        'sampled_start_steps': eval_start_idx.tolist(),
        'metrics': jsonable(metrics),
        'evaluation_time_seconds': end_time - start_time,
        'evaluation_time_per_trajectory_seconds': (
            (end_time - start_time) / completed
        ),
        'clear_lewm': (
            {
                'source_revision': CLEAR_LEWM_REVISION,
                'manifest_path': str(
                    Path(clear_manifest_path).expanduser().resolve()
                ),
                'manifest_sha256': manifest_sha256(clear_manifest_path),
                'task': clear_manifest['task'],
                'protocol': clear_manifest['protocol'],
                'cpu_threads': torch.get_num_threads(),
                'solver_contract_matched': clear_solver_contract_matched,
                'solver_ablation_opt_in': solver_ablation,
            }
            if clear_manifest is not None
            else None
        ),
        'resolved_config': OmegaConf.to_container(cfg, resolve=True),
    }
    structured_path = results_path.with_suffix(results_path.suffix + '.json')
    with structured_path.open('x') as stream:
        json.dump(structured, stream, indent=2, sort_keys=True)
        stream.write('\n')

    world.close()


if __name__ == '__main__':
    run()
