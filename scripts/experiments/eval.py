import argparse
import os
from pathlib import Path
import re

from registry import get_experiment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate a world model.')
    parser.add_argument('--model', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--run-name', required=True)
    parser.add_argument(
        '--num-trajectories', type=int, help='Number of trajectories.'
    )
    parser.add_argument(
        'overrides', nargs='*', help='Extra Hydra overrides.'
    )
    args = parser.parse_args(argv)

    try:
        experiment = get_experiment(args.model, args.task)
    except ValueError as error:
        parser.error(str(error))

    if args.num_trajectories is not None and args.num_trajectories < 1:
        parser.error('--num-trajectories must be at least 1')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.run_name):
        parser.error(
            'Run name must contain only letters, digits, dot, dash, '
            'or underscore'
        )

    data_root = os.environ.get('DATA_ROOT')
    if not data_root:
        parser.error('DATA_ROOT is not set')
    dataset = (Path(data_root) / experiment.eval_dataset).resolve()
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
