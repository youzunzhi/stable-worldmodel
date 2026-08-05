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

CUBE_DATA_REVISION = '02a19a67a0dc8c9d6215f89c19e0a597691e152a'
# Cube ships as one immutable HDF5 dataset, so both entrypoints use the
# same pinned artifact rather than a derived training conversion.
CUBE_DATASET = (
    Path('hf/datasets/quentinll--lewm-cube')
    / CUBE_DATA_REVISION
    / 'cube_single_expert.h5'
)

# The official TwoRoom dataset is distributed as one HDF5 snapshot. CLEAR
# v0.5 identifies it by a metadata fingerprint rather than a repository
# revision; both training and evaluation use the same immutable local file.
TWOROOM_DATASET = Path('tworoom') / 'tworoom.h5'


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
    ('lewm', 'cube'): ExperimentSpec(
        train_script=Path('scripts/train/lewm.py'),
        eval_script=Path('scripts/plan/eval_wm.py'),
        train_dataset=CUBE_DATASET,
        eval_dataset=CUBE_DATASET,
        train_defaults=('data=ogb', 'launcher=local'),
        eval_defaults=(
            '--config-name=cube',
            'solver.n_steps=10',
        ),
    ),
    ('lewm', 'tworoom'): ExperimentSpec(
        train_script=Path('scripts/train/lewm.py'),
        eval_script=Path('scripts/plan/eval_wm.py'),
        train_dataset=TWOROOM_DATASET,
        eval_dataset=TWOROOM_DATASET,
        train_defaults=(
            'data=tworoom',
            'launcher=local',
            'trainer.max_epochs=10',
        ),
        eval_defaults=('--config-name=tworoom',),
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
