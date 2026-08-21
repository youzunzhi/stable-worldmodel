import argparse
import os
from pathlib import Path
import subprocess
import sys

from cli import add_experiment_arguments, validate_experiment_arguments


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a world model.')
    add_experiment_arguments(parser)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument(
        '--max-steps', type=int, help='Maximum optimizer steps.'
    )
    parser.add_argument('overrides', nargs='*', help='Extra Hydra overrides.')
    args = parser.parse_args(argv)
    experiment, data_root = validate_experiment_arguments(parser, args)

    if args.max_steps is not None and args.max_steps < 1:
        parser.error('--max-steps must be at least 1')

    dataset = data_root / experiment.train_dataset
    if not dataset.exists():
        parser.error(f'Training dataset does not exist: {dataset}')

    args.experiment = experiment
    args.dataset = dataset
    args.overrides = tuple(args.overrides)
    return args


def build_train_command(args: argparse.Namespace) -> list[str]:
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


def run_training(args: argparse.Namespace) -> None:
    """Run training in the foreground using an isolated runtime cache."""
    run_root_value = os.environ.get('RUN_ROOT')
    if not run_root_value:
        raise RuntimeError('RUN_ROOT is not set')

    run_root = Path(run_root_value).resolve()
    environment = os.environ.copy()
    environment['STABLEWM_HOME'] = str(run_root)
    environment['SPT_CACHE_DIR'] = str(run_root / 'spt_cache' / args.run_name)

    subprocess.run(
        build_train_command(args),
        cwd=os.environ['REPO_ROOT'],
        env=environment,
        check=True,
    )


def main() -> None:
    run_training(parse_args())


if __name__ == '__main__':
    main()
