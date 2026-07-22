import argparse
import os
from pathlib import Path
import re

from registry import ExperimentSpec, get_experiment


RUN_NAME_PATTERN = r'[A-Za-z0-9][A-Za-z0-9._-]*'


def add_experiment_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--model', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--run-name', required=True)


def validate_experiment_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> tuple[ExperimentSpec, Path]:
    try:
        experiment = get_experiment(args.model, args.task)
    except ValueError as error:
        parser.error(str(error))

    if not re.fullmatch(RUN_NAME_PATTERN, args.run_name):
        parser.error(
            'Run name must contain only letters, digits, dot, dash, '
            'or underscore'
        )

    data_root = os.environ.get('DATA_ROOT')
    if not data_root:
        parser.error('DATA_ROOT is not set')
    return experiment, Path(data_root).resolve()
