import argparse
import os
from pathlib import Path

from registry import get_experiment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a world model.')
    parser.add_argument('--model', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--run-name', required=True)
    parser.add_argument(
        '--max-steps', type=int, help='Maximum optimizer steps.'
    )
    parser.add_argument(
        'overrides', nargs='*', help='Extra Hydra overrides.'
    )
    args = parser.parse_args(argv)

    try:
        experiment = get_experiment(args.model, args.task)
    except ValueError as error:
        parser.error(str(error))

    if args.max_steps is not None and args.max_steps < 1:
        parser.error('--max-steps must be at least 1')
    if not args.run_name.strip():
        parser.error('Run name cannot be empty')

    data_root = os.environ.get('DATA_ROOT')
    if not data_root:
        parser.error('DATA_ROOT is not set')
    dataset = (Path(data_root) / experiment.train_dataset).resolve()
    if not dataset.exists():
        parser.error(f'Training dataset does not exist: {dataset}')

    args.experiment = experiment
    args.dataset = dataset
    args.overrides = tuple(args.overrides)
    return args
