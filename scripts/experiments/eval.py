import argparse
from pathlib import Path

from cli import add_experiment_arguments, validate_experiment_arguments


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate a world model.')
    add_experiment_arguments(parser)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument(
        '--num-trajectories', type=int, help='Number of trajectories.'
    )
    parser.add_argument(
        'overrides', nargs='*', help='Extra Hydra overrides.'
    )
    args = parser.parse_args(argv)
    experiment, data_root = validate_experiment_arguments(parser, args)

    if args.num_trajectories is not None and args.num_trajectories < 1:
        parser.error('--num-trajectories must be at least 1')

    dataset = data_root / experiment.eval_dataset
    if not dataset.is_file():
        parser.error(f'Evaluation dataset does not exist: {dataset}')

    checkpoint = args.checkpoint.expanduser().resolve()
    if checkpoint.suffix != '.pt' or not checkpoint.is_file():
        parser.error(f'Checkpoint must be an existing .pt file: {checkpoint}')
    config = checkpoint.parent / 'config.json'
    if not config.is_file():
        parser.error(f'Checkpoint config does not exist: {config}')

    args.experiment = experiment
    args.dataset = dataset
    args.checkpoint = checkpoint
    args.overrides = tuple(args.overrides)
    return args
