import argparse
import os
from pathlib import Path
import sys

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


def build_command(args: argparse.Namespace) -> list[str]:
    """Build the upstream training command without executing it."""
    repo_root_value = os.environ.get('REPO_ROOT')
    if not repo_root_value:
        raise RuntimeError('REPO_ROOT is not set')

    repo_root = Path(repo_root_value).resolve()
    train_script = repo_root / args.experiment.train_script
    if not train_script.is_file():
        raise FileNotFoundError(f'Training script not found: {train_script}')

    command = [
        sys.executable,
        str(train_script),
        *args.experiment.train_defaults,
        f'seed={args.seed}',
        f'run_name={args.run_name}',
        f'data.dataset.name={args.dataset}',
    ]
    if args.max_steps is not None:
        command.extend(
            (
                'trainer.max_epochs=1',
                f'+trainer.limit_train_batches={args.max_steps}',
                '+trainer.limit_val_batches=1',
            )
        )
    command.extend(args.overrides)
    return command
