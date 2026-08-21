from pathlib import Path

import numpy as np
import pytest

from scripts.experiments.observation_goal_threshold.contracts import (
    LABEL_F,
    LABEL_T,
    LABEL_U,
    TaskContract,
    classify_cube_pose,
    cube_symmetry_angle_deg,
    cube_symmetry_quaternions,
)
from scripts.experiments.observation_goal_threshold.curve_plot import (
    plot_epsilon_tpr_fpr,
)
from scripts.experiments.observation_goal_threshold.encode import (
    encode_projected,
    pair_partition_matches,
    parameter_hash,
    preprocess_pixels,
    score_pair_shards,
)
from scripts.experiments.observation_goal_threshold.io_utils import (
    read_json,
    write_json,
)
from scripts.experiments.observation_goal_threshold.metrics import (
    CalibrationOutcome,
    attach_bootstrap_cis,
    enforce_validation,
    metrics_at_threshold,
    select_threshold,
)
from scripts.experiments.observation_goal_threshold.sample_pairs import (
    AffinePermutation,
    DenseSpatialGrid,
    sample_stratified_partition,
    sample_task_stratum,
    uniform_ordered_pairs,
)
from scripts.experiments.observation_goal_threshold.self_eval import (
    compare_predictions,
    endpoint_distances,
    load_and_validate_threshold,
)
from scripts.experiments.observation_goal_threshold.split import (
    PARTITIONS,
    row_partitions,
    split_groups,
)
from scripts.experiments.observation_goal_threshold.summarize_self_eval import (
    summarize as summarize_self_eval,
)


def contract(positive=1.0, negative=2.0):
    return TaskContract(
        task='fixture',
        variant='fixture_gap',
        state_fields=('x',),
        positive_if_lt=positive,
        negative_if_gt=negative,
        unit='unit',
        negative_edges=(3.0, 4.0),
    )


@pytest.mark.parametrize(
    ('positive', 'negative', 'errors', 'expected'),
    [
        (20.0, 30.0, [19.9, 20.0, 30.0, 30.1], [0, 2, 2, 1]),
        (0.03, 0.04, [0.029, 0.03, 0.04, 0.041], [0, 2, 2, 1]),
        (8.0, 16.0, [7.9, 8.0, 16.0, 16.1], [0, 2, 2, 1]),
    ],
)
def test_exact_gap_boundaries(positive, negative, errors, expected):
    c = TaskContract(
        'task',
        'variant',
        ('x',),
        positive,
        negative,
        'unit',
        (negative * 2, negative * 4),
    )
    assert c.classify(np.array(errors)).tolist() == expected


def test_task_error_is_plain_joint_l2():
    c = contract()
    anchor = np.array([[0.0, 0.0], [3.0, 4.0]])
    goal = np.array([[3.0, 4.0], [3.0, 4.0]])
    assert np.allclose(c.task_error(anchor, goal), [5.0, 0.0])


def test_task_error_rejects_nonfinite_state():
    with pytest.raises(ValueError, match='non-finite'):
        contract().task_error(np.array([[np.nan]]), np.array([[0.0]]))


def test_negative_strata_exact_edges():
    result = contract().negative_strata(
        np.array([2.0, 2.1, 3.0, 3.1, 4.0, 4.1])
    )
    assert result.tolist() == [-1, 0, 0, 1, 1, 2]


def test_cube_has_exactly_24_proper_symmetries():
    symmetries = cube_symmetry_quaternions()
    assert symmetries.shape == (24, 4)
    assert np.allclose(np.linalg.norm(symmetries, axis=1), 1.0)


def test_cube_symmetry_and_quaternion_sign_aliases_have_zero_angle():
    identity = np.array([[1.0, 0.0, 0.0, 0.0]])
    for symmetry in cube_symmetry_quaternions():
        goal = symmetry[None]
        assert cube_symmetry_angle_deg(identity, goal)[0] < 1e-5
        assert cube_symmetry_angle_deg(identity, -goal)[0] < 1e-5


def test_cube_pose_t_f_u_boolean_contract():
    identity = np.tile([1.0, 0.0, 0.0, 0.0], (4, 1))
    angles = np.deg2rad(np.array([0.0, 15.0, 30.0, 31.0])) / 2
    rotated = np.stack(
        [np.cos(angles), np.zeros(4), np.zeros(4), np.sin(angles)], axis=1
    )
    anchor_pos = np.zeros((4, 3))
    goal_pos = np.array(
        [[0.02, 0, 0], [0.02, 0, 0], [0.04, 0, 0], [0.02, 0, 0]]
    )
    labels, _, _ = classify_cube_pose(anchor_pos, goal_pos, identity, rotated)
    assert labels.tolist() == [LABEL_T, LABEL_U, LABEL_U, LABEL_F]


@pytest.mark.parametrize('bad', ['xyzw', 'scalar-last', ''])
def test_cube_pose_rejects_non_wxyz_conventions(bad):
    q = np.array([[1.0, 0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match='wxyz'):
        cube_symmetry_angle_deg(q, q, convention=bad)


def test_cube_pose_rejects_bad_quaternions():
    good = np.array([[1.0, 0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match='zero-norm'):
        cube_symmetry_angle_deg(good, np.zeros((1, 4)))
    with pytest.raises(ValueError, match='finite'):
        cube_symmetry_angle_deg(good, np.array([[np.nan, 0, 0, 0]]))


def test_group_split_is_reproducible_disjoint_and_complete():
    groups = np.repeat(np.arange(185), 101)
    first = split_groups(groups, 260820)
    second = split_groups(groups, 260820)
    assert [len(first[name]) for name in PARTITIONS] == [111, 37, 37]
    assert all(
        np.array_equal(first[name], second[name]) for name in PARTITIONS
    )
    assert len(np.unique(np.concatenate(list(first.values())))) == 185
    assigned = row_partitions(groups, first)
    for group in range(185):
        assert len(np.unique(assigned[groups == group])) == 1


def test_row_partitions_reject_overlap():
    split = {
        PARTITIONS[0]: np.array([0, 1]),
        PARTITIONS[1]: np.array([1, 2]),
        PARTITIONS[2]: np.array([3]),
    }
    with pytest.raises(ValueError, match='overlap'):
        row_partitions(np.arange(4), split)


def test_affine_permutation_prefix_has_no_replacement():
    permutation = AffinePermutation.create(997, 700, 42)
    values = np.concatenate(
        [permutation.values(0, 333), permutation.values(333, 367)]
    )
    assert len(np.unique(values)) == 700
    assert values.min() >= 0 and values.max() < 997


def test_affine_permutation_bounds_are_enforced():
    with pytest.raises(ValueError):
        AffinePermutation.create(10, 11, 0)
    permutation = AffinePermutation.create(10, 5, 0)
    with pytest.raises(ValueError):
        permutation.values(4, 2)


def test_uniform_ordered_pairs_are_unique_deterministic_and_not_self():
    rows = np.arange(20, 70)
    a1, g1, _ = uniform_ordered_pairs(
        rows, start=0, count=1000, total_count=1000, seed=8
    )
    a2, g2, _ = uniform_ordered_pairs(
        rows, start=0, count=1000, total_count=1000, seed=8
    )
    assert np.array_equal(a1, a2) and np.array_equal(g1, g2)
    assert not np.any(a1 == g1)
    assert len(np.unique(a1 * 100 + g1)) == 1000


def test_sparse_grid_empty_terminal_cell_is_rejected_without_index_error():
    grid = DenseSpatialGrid(np.array([[0.0, 10.0], [10.0, 0.0]]), 10.0)

    class DeterministicRng:
        @staticmethod
        def integers(low, high, size):
            return np.zeros(size, dtype=np.int64)

        @staticmethod
        def random(size):
            return np.zeros(size)

    goal, count, valid = grid.propose(
        np.array([0]), np.array([[1, 0]]), DeterministicRng()
    )
    assert goal.shape == count.shape == valid.shape == (1,)
    assert count[0] == 0
    assert not valid[0]


def test_stratified_sampler_is_latent_blind_reproducible_and_balanced():
    values = np.tile(np.arange(11, dtype=np.float32), 40)[:, None]
    groups = np.repeat(np.arange(8), 55)
    rows = np.arange(len(values))
    kwargs = {
        'partition_rows': rows,
        'all_states': values,
        'all_groups': groups,
        'contract': contract(),
        'total_count': 600,
        'seed': 123,
        'total_dataset_rows': len(values),
    }
    first, audit = sample_stratified_partition(**kwargs)
    second, _ = sample_stratified_partition(**kwargs)
    assert all(np.array_equal(first[key], second[key]) for key in first)
    assert len(np.unique(first['pair_id'])) == 600
    assert (first['anchor_row'] != first['goal_row']).all()
    assert (first['label'] == LABEL_T).sum() == 300
    assert (first['label'] == LABEL_F).sum() == 300
    assert set(first['negative_stratum'][first['label'] == LABEL_F]) == {
        0,
        1,
        2,
    }
    assert set(first['anchor_group']) == set(groups)
    assert audit['positive']['realized'] == 300


def test_stratum_sampler_refills_after_unique_pair_collisions():
    # Only cross-cluster pairs are in the far stratum: 19,800 pairs exist, but
    # the first 3% overdraw has too many collisions to fill this 15,000 sample.
    # The sampler must continue its seeded stream until the no-replacement
    # contract is satisfied.
    states = np.concatenate(
        [np.zeros(990, dtype=np.float32), np.full(10, 100.0, np.float32)]
    )[:, None]
    rows = np.arange(len(states))
    result = sample_task_stratum(
        partition_rows=rows,
        all_states=states,
        all_groups=np.zeros(len(states), dtype=np.int64),
        contract=contract(),
        target_count=15_000,
        lower_exclusive=50.0,
        upper_inclusive=None,
        label=LABEL_F,
        seed=260820,
        total_dataset_rows=len(states),
    )
    assert len(result['pair_id']) == 15_000
    assert len(np.unique(result['pair_id'])) == 15_000
    assert (result['task_error'] > 50.0).all()


def synthetic_samples():
    stratified = {
        'latent_distance': np.array([0.1, 0.2, 0.8, 0.9], np.float32),
        'label': np.array([LABEL_T, LABEL_T, LABEL_F, LABEL_F]),
        'analysis_weight': np.ones(4),
        'anchor_group': np.array([0, 1, 0, 1]),
        'negative_stratum': np.array([-1, -1, 0, 0]),
        'task_error': np.array([0.1, 0.2, 3.0, 3.0]),
        'anchor_episode': np.array([0, 1, 0, 1]),
        'goal_episode': np.array([2, 3, 2, 3]),
    }
    uniform = {
        'latent_distance': np.array([0.1, 0.15, 0.8, 1.0, 0.3], np.float32),
        'label': np.array([LABEL_T, LABEL_T, LABEL_F, LABEL_F, LABEL_U]),
        'analysis_weight': np.ones(5),
        'anchor_group': np.array([0, 1, 0, 1, 0]),
        'task_error': np.array([0.1, 0.2, 3.0, 3.0, 1.5]),
        'anchor_episode': np.array([0, 1, 0, 1, 0]),
        'goal_episode': np.array([2, 3, 2, 3, 0]),
        'negative_stratum': np.array([-1, -1, 0, 0, -1]),
    }
    return stratified, uniform


def test_selector_maximizes_macro_tpr_then_uses_smallest_epsilon():
    stratified, uniform = synthetic_samples()
    result = select_threshold(
        stratified,
        uniform,
        min_positive_recall=0.9,
        max_negative_fpr=0.1,
        min_population_precision=None,
    )
    assert result.epsilon == pytest.approx(0.2)
    assert result.macro_tpr == pytest.approx(1.0)
    assert result.macro_fpr == pytest.approx(0.0)


def test_selector_reports_registered_no_feasible_outcome():
    stratified, uniform = synthetic_samples()
    stratified['latent_distance'] = np.array([0.8, 0.9, 0.1, 0.2])
    with pytest.raises(CalibrationOutcome) as error:
        select_threshold(
            stratified,
            uniform,
            min_positive_recall=0.9,
            max_negative_fpr=0.1,
            min_population_precision=None,
        )
    assert (
        error.value.code == 'THRESHOLD_CALIBRATION_NO_FEASIBLE_OPERATING_POINT'
    )


def test_uniform_population_precision_excludes_u_and_is_not_balanced_precision():
    stratified, uniform = synthetic_samples()
    metrics = metrics_at_threshold(stratified, uniform, 0.2)
    assert metrics['population_precision_direct'] == pytest.approx(1.0)
    assert metrics['uniform_prevalence'] == pytest.approx(
        {'T': 0.4, 'F': 0.4, 'U': 0.2}
    )
    assert metrics['ignored_band']['ball_occupancy'] == pytest.approx(0.0)


def test_direct_and_reconstructed_uniform_precision_agree():
    stratified, uniform = synthetic_samples()
    metrics = metrics_at_threshold(stratified, uniform, 0.8)
    assert metrics['population_precision_direct'] == pytest.approx(
        metrics['population_precision_reconstructed']
    )


def test_validation_constraint_failure_does_not_relax_contract():
    metrics = {
        'macro_tpr': 0.89,
        'macro_fpr': 0.05,
        'population_precision_direct': 1.0,
    }
    with pytest.raises(CalibrationOutcome) as error:
        enforce_validation(
            metrics,
            min_positive_recall=0.9,
            max_negative_fpr=0.1,
            min_population_precision=None,
        )
    assert error.value.code == 'THRESHOLD_CALIBRATION_FAILED_VALIDATION'


def test_cluster_bootstrap_is_seed_reproducible():
    metrics = {
        'macro_tpr': 0.5,
        'macro_fpr': 0.5,
        '_group_tpr': np.array([0.0, 1.0]),
        '_group_fpr': np.array([1.0, 0.0]),
    }
    first = attach_bootstrap_cis(metrics, 100, 7)
    second = attach_bootstrap_cis(metrics, 100, 7)
    assert first == second


def test_preprocessing_shape_scale_and_finiteness():
    torch = pytest.importorskip('torch')
    spt = pytest.importorskip('stable_pretraining')
    transforms = pytest.importorskip('torchvision.transforms.v2')
    pixels = np.arange(2 * 224 * 224 * 3, dtype=np.uint8).reshape(
        2, 224, 224, 3
    )
    output = preprocess_pixels(pixels)
    assert tuple(output.shape) == (2, 3, 224, 224)
    assert output.dtype.is_floating_point
    assert output.isfinite().all()
    reference = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=224),
        ]
    )
    expected = torch.stack([reference(image) for image in pixels])
    assert torch.equal(output, expected)


def test_encode_projected_never_calls_predictor_or_action_encoder():
    torch = pytest.importorskip('torch')

    class Output:
        last_hidden_state = torch.ones(2, 2, 3)

    class Encoder(torch.nn.Module):
        def forward(self, pixels, interpolate_pos_encoding=False):
            assert interpolate_pos_encoding
            return Output()

    class Forbidden(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError('forbidden path called')

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.projector = torch.nn.Identity()
            self.predictor = Forbidden()
            self.action_encoder = Forbidden()

    result = encode_projected(Model(), torch.zeros(2, 3, 224, 224))
    assert tuple(result.shape) == (2, 3)


def test_parameter_hash_changes_only_when_frozen_symbols_change():
    torch = pytest.importorskip('torch')

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2)
            self.projector = torch.nn.Linear(2, 2)
            self.predictor = torch.nn.Linear(2, 2)

    model = Model()
    original = parameter_hash(model)
    with torch.no_grad():
        model.predictor.weight.add_(1)
    assert parameter_hash(model) == original
    with torch.no_grad():
        model.encoder.weight.add_(1)
    assert parameter_hash(model) != original


def test_audit_scoring_requires_locked_threshold(tmp_path):
    pytest.importorskip('pyarrow')
    with pytest.raises(PermissionError, match='locked threshold'):
        score_pair_shards(
            pair_dir=tmp_path / 'pairs',
            score_dir=tmp_path / 'scores',
            embedding_dir=tmp_path / 'embeddings',
            total_dataset_rows=10,
            partition='threshold_audit',
            locked_threshold_path=None,
            device='cpu',
        )


def test_dictionary_encoded_partition_guard_supports_pyarrow_25():
    pa = pytest.importorskip('pyarrow')
    dictionary = pa.DictionaryArray.from_arrays(
        pa.array([0, 0], type=pa.int8()), pa.array(['threshold_fit'])
    )
    table = pa.table({'partition': dictionary})
    assert pair_partition_matches(table, 'threshold_fit')
    assert not pair_partition_matches(table, 'threshold_validation')


@pytest.mark.parametrize(
    ('task', 'variant'),
    [
        ('pusht', 'pusht_joint_xy_pointwise_gap20_30'),
        ('cube', 'cube_block_xyz_pointwise_gap03_04'),
        ('tworoom', 'tworoom_agent_xy_pointwise_gap8_16'),
    ],
)
def test_formal_configs_pin_main_variant_and_full_counts(task, variant):
    root = Path('scripts/experiments/observation_goal_threshold/configs')
    config = read_json(root / f'{task}.json')
    assert config['task_label']['variant'] == variant
    assert config['pair_sampling']['uniform_pairs'] == 100_000_000
    assert config['pair_sampling']['stratified_pairs'] == 20_000_000
    assert config['analysis']['bootstrap_replicates'] == 10_000
    assert config['selection']['tie_break'] == 'smallest_epsilon'


def test_sampler_source_imports_no_model_or_latent_code():
    source = Path(
        'scripts/experiments/observation_goal_threshold/sample_pairs.py'
    ).read_text()
    assert 'import torch' not in source
    assert 'stable_worldmodel' not in source
    assert 'latent_distance' not in source


def test_epsilon_tpr_fpr_curve_is_written_for_failed_selection(tmp_path):
    pytest.importorskip('matplotlib')
    stratified, _ = synthetic_samples()
    manifest = plot_epsilon_tpr_fpr(
        stratified,
        tmp_path / 'curve.png',
        task='fixture',
        selected_epsilon=None,
        min_positive_recall=0.9,
        max_negative_fpr=0.1,
        status='THRESHOLD_CALIBRATION_NO_FEASIBLE_OPERATING_POINT',
    )
    assert (tmp_path / 'curve.png').is_file()
    assert manifest['selected_epsilon'] is None
    assert manifest['curve_points_exact'] > 0


def test_locked_threshold_validation_checks_checkpoint_and_contract(tmp_path):
    path = tmp_path / 'selected_threshold.json'
    path.write_text(
        """{
          "epsilon": 0.25,
          "task": "pusht",
          "pointwise_label_variant": "fixture",
          "residual_definition": "mean_D((z_i-z_j)^2)",
          "D": 192,
          "dtype": "float32",
          "encoder_checkpoint_sha256": "abc",
          "encoder_checkpoint_config_sha256": "cfg",
          "encoder_projector_parameter_hashes": {
            "before": "params",
            "after": "params"
          },
          "observation_preprocessing": {
            "residual": "mean_D((z_i-z_j)^2)",
            "dtype": "float32",
            "latent_dim": 192,
            "resize": [224, 224],
            "normalization": "ImageNet mean/std from stable_pretraining"
          }
        }"""
    )
    _, provenance = load_and_validate_threshold(
        path,
        task='pusht',
        checkpoint_sha256='abc',
        checkpoint_config_sha256='cfg',
    )
    assert provenance['epsilon'] == 0.25
    with pytest.raises(ValueError, match='identity mismatch'):
        load_and_validate_threshold(
            path,
            task='pusht',
            checkpoint_sha256='different',
            checkpoint_config_sha256='cfg',
        )


def test_self_eval_confusion_and_sr_error_are_pair_aligned():
    result = compare_predictions(
        np.array([0.1, 0.7, 0.2, 0.8], dtype=np.float32),
        np.array([True, True, False, False]),
        epsilon=0.5,
        pair_ids=['a', 'b', 'c', 'd'],
    )
    summary = result['summary']
    assert summary['confusion'] == {'tp': 1, 'tn': 1, 'fp': 1, 'fn': 1}
    assert summary['accuracy'] == pytest.approx(0.5)
    assert summary['actual_success_rate_percent'] == pytest.approx(50.0)
    assert summary['predicted_success_rate_percent'] == pytest.approx(50.0)
    assert summary['success_rate_error_paired_bootstrap_95ci'] == {
        'low': -75.0,
        'high': 75.0,
        'replicates': 10_000,
        'seed': 20260821,
        'unit': 'percentage_points',
    }
    assert [row['pair_id'] for row in result['pairs']] == [
        'a',
        'b',
        'c',
        'd',
    ]


def test_endpoint_self_eval_uses_encoder_projector_only():
    torch = pytest.importorskip('torch')

    class Output:
        def __init__(self, value):
            self.last_hidden_state = value

    class Encoder(torch.nn.Module):
        def forward(self, pixels, interpolate_pos_encoding=False):
            assert interpolate_pos_encoding
            pooled = pixels.mean(dim=(-2, -1))[:, None, :]
            return Output(pooled)

    class Forbidden(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError('forbidden path called')

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.encoder = Encoder()
            self.projector = torch.nn.Identity()
            self.predictor = Forbidden()
            self.action_encoder = Forbidden()

    endpoint = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    goal = endpoint.copy()
    goal[1] = 255
    distance = endpoint_distances(Model(), endpoint, goal, batch_size=1)
    assert distance[0] == pytest.approx(0.0)
    assert distance[1] > 0


def test_self_eval_summary_keeps_missing_matrix_cells_explicit(tmp_path):
    result_path = tmp_path / 'results.txt.json'
    write_json(
        result_path,
        {
            'requested_trajectories': 100,
            'completed_trajectories': 100,
            'checkpoint_sha256': 'checkpoint',
            'clear_lewm': {
                'task': 'pusht',
                'protocol': {'name': 'moderate'},
                'manifest_sha256': 'manifest',
                'cpu_threads': 1,
                'solver_contract_matched': True,
            },
            'find_goal_threshold_self_eval': {
                'task': 'pusht',
                'clear_protocol': 'moderate',
                'threshold': {
                    'encoder_checkpoint_sha256': 'checkpoint',
                    'selected_threshold_sha256': 'threshold',
                    'epsilon': 0.5,
                },
                'summary': {
                    'pairs': 100,
                    'actual_success_rate_percent': 50.0,
                    'predicted_success_rate_percent': 45.0,
                    'accuracy': 0.8,
                    'accuracy_wilson_95ci': {
                        'low': 0.711,
                        'high': 0.866,
                    },
                    'success_rate_error_percentage_points': -5.0,
                    'success_rate_error_paired_bootstrap_95ci': {
                        'low': -10.0,
                        'high': 0.0,
                        'replicates': 10_000,
                        'seed': 20260821,
                        'unit': 'percentage_points',
                    },
                    'confusion': {
                        'tp': 40,
                        'tn': 40,
                        'fp': 10,
                        'fn': 10,
                    },
                },
                'pairs': [
                    {'pair_id': f'pair-{index}'} for index in range(100)
                ],
            },
        },
    )
    summary = summarize_self_eval([result_path], tmp_path / 'summary')
    assert summary['status'] == 'INCOMPLETE'
    assert len(summary['missing_cells']) == 5
    assert summary['cells'][0]['task'] == 'pusht'
