import argparse
import os
from pathlib import Path
import subprocess
import sys

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


def build_eval_command(args: argparse.Namespace) -> list[str]:
    """Build the upstream evaluation command without executing it."""
    repo_root_value = os.environ.get('REPO_ROOT')
    run_root_value = os.environ.get('RUN_ROOT')
    if not repo_root_value:
        raise RuntimeError('REPO_ROOT is not set')
    if not run_root_value:
        raise RuntimeError('RUN_ROOT is not set')

    repo_root = Path(repo_root_value).resolve()
    eval_script = repo_root / args.experiment.eval_script
    if not eval_script.is_file():
        raise FileNotFoundError(f'Evaluation script not found: {eval_script}')

    output_dir = (
        Path(run_root_value).resolve() / 'evaluations' / args.run_name
    )
    command = [
        sys.executable,
        str(eval_script),
        *args.experiment.eval_defaults,
        f'policy={args.checkpoint}',
        f'eval.dataset_name={args.dataset}',
        f'seed={args.seed}',
        f'output.dir={output_dir}',
        'output.filename=results.txt',
    ]
    if args.num_trajectories is not None:
        command.append(f'eval.num_eval={args.num_trajectories}')
    command.extend(args.overrides)
    return command


def run_evaluation(args: argparse.Namespace) -> None:
    """Run evaluation in the foreground using the experiment output root."""
    run_root_value = os.environ.get('RUN_ROOT')
    if not run_root_value:
        raise RuntimeError('RUN_ROOT is not set')

    environment = os.environ.copy()
    environment['STABLEWM_HOME'] = str(Path(run_root_value).resolve())
    subprocess.run(
        build_eval_command(args),
        cwd=os.environ['REPO_ROOT'],
        env=environment,
        check=True,
    )


def main() -> None:
    run_evaluation(parse_args())


if __name__ == '__main__':
    main()
