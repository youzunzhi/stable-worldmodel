from pathlib import Path

from scripts.experiments.registry import get_experiment


def test_lewm_tworoom_uses_one_pinned_hdf5_for_train_and_eval():
    experiment = get_experiment('lewm', 'tworoom')
    assert experiment.train_dataset == Path('tworoom/tworoom.h5')
    assert experiment.eval_dataset == experiment.train_dataset
    assert experiment.train_defaults == ('data=tworoom', 'launcher=local')
    assert experiment.eval_defaults == ('--config-name=tworoom',)
