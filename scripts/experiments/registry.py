from dataclasses import dataclass
from pathlib import Path


PUSHT_REVISION = '655cd446b9929369d7d406001da85c15d1457850'
PUSHT_TRAIN_DATASET = (
    Path('converted') / PUSHT_REVISION / 'pusht_expert_train.lance'
)
PUSHT_EVAL_DATASET = (
    Path('hf/datasets/quentinll--lewm-pusht')
    / PUSHT_REVISION
    / 'pusht_expert_train.h5'
)


@dataclass(frozen=True)
class ExperimentSpec:
    train_script: Path
    eval_script: Path
    train_dataset: Path
    eval_dataset: Path
    train_defaults: tuple[str, ...]
    eval_defaults: tuple[str, ...]


EXPERIMENTS = {
    ('lewm', 'pusht'): ExperimentSpec(
        train_script=Path('scripts/train/lewm.py'),
        eval_script=Path('scripts/plan/eval_wm.py'),
        train_dataset=PUSHT_TRAIN_DATASET,
        eval_dataset=PUSHT_EVAL_DATASET,
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
