from pathlib import Path

from omegaconf import OmegaConf

from scripts.experiments.registry import get_experiment


def test_lewm_tworoom_uses_one_pinned_hdf5_for_train_and_eval():
    experiment = get_experiment('lewm', 'tworoom')
    assert experiment.train_dataset == Path('tworoom/tworoom.h5')
    assert experiment.eval_dataset == experiment.train_dataset
    assert experiment.train_defaults == (
        'data=tworoom',
        'launcher=local',
        'trainer.max_epochs=10',
    )
    assert experiment.eval_defaults == ('--config-name=tworoom',)


def test_lewm_reacher_uses_pinned_hdf5_and_ten_epochs():
    experiment = get_experiment('lewm', 'reacher')
    assert experiment.train_dataset == Path(
        'hf/datasets/quentinll--lewm-reacher/'
        'e70a080d0d04c6072123c9ebd343acf7fff28dbf/reacher.h5'
    )
    assert experiment.eval_dataset == experiment.train_dataset
    assert experiment.train_defaults == (
        'data=dmc',
        'launcher=local',
        'trainer.max_epochs=10',
    )
    assert experiment.eval_defaults == ('--config-name=reacher',)


def test_reacher_eval_config_accepts_isolated_output_directory():
    config = OmegaConf.load(
        Path(__file__).parents[2] / 'scripts/plan/config/reacher.yaml'
    )

    assert 'dir' in config.output
    assert config.output.dir is None
