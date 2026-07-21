from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentSpec:
    train_script: Path
    eval_script: Path
    train_defaults: tuple[str, ...]
    eval_defaults: tuple[str, ...]


EXPERIMENTS = {
    ('lewm', 'pusht'): ExperimentSpec(
        train_script=Path('scripts/train/lewm.py'),
        eval_script=Path('scripts/plan/eval_wm.py'),
        train_defaults=('data=pusht', 'launcher=local'),
        eval_defaults=('--config-name=pusht',),
    ),
}


def get_experiment(model: str, task: str) -> ExperimentSpec:
    """Return the requested experiment or fail before launching any work."""
    try:
        return EXPERIMENTS[(model, task)]
    except KeyError as error:
        supported = ', '.join(
            f'{supported_model}/{supported_task}'
            for supported_model, supported_task in EXPERIMENTS
        )
        raise ValueError(
            f'Unsupported model/task {model}/{task}. Supported: {supported}'
        ) from error
