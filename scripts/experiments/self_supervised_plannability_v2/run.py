"""Command-line entry point for SSP-v2 preparation, diagnostics, and training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import SSPV2Failure, load_config
from .es import run_smoke, run_training
from .evaluate_profile import evaluate_profile
from .pairs import prepare_task


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument('--config', required=True)
    shared.add_argument('--device', default='cuda')

    prepare = commands.add_parser('prepare', parents=[shared])
    prepare.add_argument('--output-dir', required=True)
    prepare.add_argument('--repo-root')
    prepare.add_argument('--formal', action='store_true')
    prepare.add_argument('--full-dataset-hash', action='store_true')

    smoke = commands.add_parser('smoke', parents=[shared])
    smoke.add_argument('--preparation-dir', required=True)
    smoke.add_argument('--output-dir', required=True)

    train = commands.add_parser('train', parents=[shared])
    train.add_argument('--preparation-dir', required=True)
    train.add_argument('--output-dir', required=True)
    train.add_argument('--replicate-seed', required=True, type=int)

    profile = commands.add_parser('profile', parents=[shared])
    profile.add_argument('--preparation-dir', required=True)
    profile.add_argument('--replicate-dir', required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config_path, config = load_config(args.config)
        if args.command == 'prepare':
            repo_root = (
                Path(args.repo_root).expanduser().resolve()
                if args.repo_root
                else Path(__file__).resolve().parents[3]
            )
            result = prepare_task(
                config_path=config_path,
                config=config,
                output_dir=args.output_dir,
                repo_root=repo_root,
                device=args.device,
                formal=args.formal,
                full_dataset_hash=args.full_dataset_hash,
            )
        elif args.command == 'smoke':
            result = run_smoke(
                config=config,
                preparation_dir=args.preparation_dir,
                output_dir=args.output_dir,
                device=args.device,
            )
        elif args.command == 'train':
            result = run_training(
                config=config,
                preparation_dir=args.preparation_dir,
                output_dir=args.output_dir,
                replicate_seed=args.replicate_seed,
                device=args.device,
            )
        else:
            result = evaluate_profile(
                config=config,
                preparation_dir=args.preparation_dir,
                replicate_dir=args.replicate_dir,
                device=args.device,
            )
    except SSPV2Failure as error:
        print(
            json.dumps(
                {
                    'status': 'failed',
                    'code': error.code,
                    'message': error.message,
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
