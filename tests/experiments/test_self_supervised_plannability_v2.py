"""Protocol and planner invariants for self-supervised-plannability-v2."""

from __future__ import annotations

import numpy as np
import torch
from gymnasium.spaces import Box

from scripts.experiments.self_supervised_plannability_v2.es import (
    select_validation_checkpoint,
)
from scripts.experiments.self_supervised_plannability_v2.geometry import (
    ClipConsistentActionTransform,
    RotatedSearchCost,
    TrajectoryHitDiagnostic,
    build_action_effect_basis,
    fixed_budget_reward,
    orthogonal_directions,
)
from scripts.experiments.self_supervised_plannability_v2.noise import (
    HitFunnelObserver,
)
from stable_worldmodel.planning import GoalMSE
from stable_worldmodel.planning.solver import CEMSolver
from stable_worldmodel.policy import PlanConfig


def _basis(seed: int = 1) -> torch.Tensor:
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    effects = rng.standard_normal((96, 192))
    basis, _ = build_action_effect_basis(effects, 32)
    return torch.from_numpy(basis)


def test_rotated_identity_is_exact_goalmse_parity():
    generator = torch.Generator().manual_seed(7)
    predicted = torch.randn(2, 5, 6, 192, generator=generator)
    goal = torch.randn(2, 5, 192, generator=generator)
    info = {'predicted_emb': predicted, 'goal_emb': goal}
    actual = RotatedSearchCost(_basis(), torch.zeros(32))(info)
    expected = GoalMSE()(info)
    assert torch.equal(actual, expected)


def test_rotated_metric_is_orthonormal_and_smoothly_bounded():
    basis = _basis(2)
    cost = RotatedSearchCost(basis, torch.full((32,), 100.0))
    torch.testing.assert_close(
        basis.T @ basis, torch.eye(32), rtol=0, atol=3e-6
    )
    assert float(cost.eigenvalues.min()) >= 4.0 - 1e-6
    assert float(cost.eigenvalues.max()) <= 4.0 + 1e-6
    negative = RotatedSearchCost(basis, torch.full((32,), -100.0))
    assert float(negative.eigenvalues.min()) >= 0.25 - 1e-6
    assert float(negative.eigenvalues.max()) <= 0.25 + 1e-6


def test_clip_transform_matches_raw_clip_then_renormalize():
    transform = ClipConsistentActionTransform(
        torch.tensor([0.5, -0.5]),
        torch.tensor([0.5, 0.25]),
        action_block=2,
    )
    actions = torch.tensor([[[[4.0, -4.0, 0.0, 0.0]]]])
    actual = transform(actions)
    raw = actions * torch.tensor([0.5, 0.25, 0.5, 0.25]) + torch.tensor(
        [0.5, -0.5, 0.5, -0.5]
    )
    expected = (
        raw.clamp(-1, 1) - torch.tensor([0.5, -0.5, 0.5, -0.5])
    ) / torch.tensor([0.5, 0.25, 0.5, 0.25])
    torch.testing.assert_close(actual, expected)
    assert transform.record()['clipped_values'] == 2


def test_trajectory_hit_accepts_prefix_without_distance_leakage():
    predicted = torch.full((1, 2, 6, 192), 10.0)
    predicted[0, 0, 2] = 0.0
    goal = torch.zeros(1, 2, 1, 192)
    result = TrajectoryHitDiagnostic(1.5, horizon=5)(
        {'predicted_emb': predicted, 'goal_emb': goal}
    )
    assert result.keys() == {'hit_bits', 'first_hit_horizon'}
    assert result['hit_bits'].tolist() == [[True, False]]
    assert result['first_hit_horizon'].tolist() == [[2, 0]]


def test_binary_fixed_budget_reward_uses_archive_and_late_hit_mass():
    fractions = [0.0] * 25 + [0.1, 0.2, 0.3, 0.4, 0.5]
    assert fixed_budget_reward(True, fractions) == 1.0 + 0.25 * 0.3
    assert fixed_budget_reward(False, fractions) == 0.25 * 0.3


def test_orthogonal_directions_have_isotropic_norm():
    rng = np.random.Generator(np.random.PCG64DXSM(4))
    directions = orthogonal_directions(rng, 16, 32)
    with np.errstate(all='ignore'):
        gram = directions @ directions.T
    np.testing.assert_allclose(gram, np.eye(16) * 32, rtol=0, atol=1e-5)


class _VerifiedDirectCost(torch.nn.Module):
    def __init__(self, hit_index: int | None) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.hit_index = hit_index

    def get_cost_and_diagnostics(self, info, candidates):
        del info
        costs = candidates[..., 0, 0].abs()
        bits = torch.zeros_like(costs, dtype=torch.bool)
        horizons = torch.zeros_like(costs, dtype=torch.long)
        if self.hit_index is not None:
            bits[:, self.hit_index] = True
            horizons[:, self.hit_index] = 1
        return costs, {'hit_bits': bits, 'first_hit_horizon': horizons}


def _fixed_noise(values: list[float]):
    tensor = torch.tensor(values, dtype=torch.float32).reshape(1, -1, 1, 1)

    def generate(**kwargs):
        assert tuple(kwargs['shape']) == tuple(tensor.shape)
        return tensor.to(kwargs['device'], kwargs['dtype'])

    return generate


def _configure(solver: CEMSolver, samples: int) -> None:
    solver.configure(
        action_space=Box(
            low=-np.inf, high=np.inf, shape=(1, 1), dtype=np.float32
        ),
        n_envs=1,
        config=PlanConfig(horizon=1, receding_horizon=1, action_block=1),
    )
    assert solver.num_samples == samples


def test_verified_hit_is_elite_archived_and_returned_even_with_high_cost():
    observer = HitFunnelObserver(expected_candidates=4, expected_elites=2)
    solver = CEMSolver(
        cost=_VerifiedDirectCost(hit_index=3),
        num_samples=4,
        n_steps=1,
        topk=2,
        candidate_noise=_fixed_noise([0.0, 1.0, 2.0, 3.0]),
        iteration_observer=observer,
        verified_hit_key='hit_bits',
        return_best_evaluated=True,
        log_timing=False,
    )
    _configure(solver, 4)
    output = solver.solve({'dummy': torch.zeros(1, 1)})
    assert output['returned_verified_hit'] == [True]
    assert output['return_mode'] == ['verified_hit_archive']
    assert output['actions'].item() == 3.0
    observed = output['iteration_observer']
    assert observed['population_hit_counts_by_iteration'] == [[1]]
    assert observed['elite_hit_counts_by_iteration'] == [[1]]


def test_no_hit_returns_best_evaluated_candidate_not_final_mean():
    observer = HitFunnelObserver(expected_candidates=4, expected_elites=2)
    solver = CEMSolver(
        cost=_VerifiedDirectCost(hit_index=None),
        num_samples=4,
        n_steps=1,
        topk=2,
        candidate_noise=_fixed_noise([0.0, 1.0, 2.0, 3.0]),
        iteration_observer=observer,
        verified_hit_key='hit_bits',
        return_best_evaluated=True,
        log_timing=False,
    )
    _configure(solver, 4)
    output = solver.solve({'dummy': torch.zeros(1, 1)})
    assert output['return_mode'] == ['best_evaluated_candidate']
    assert output['actions'].item() == 0.0
    assert output['distribution_mean'].item() == 0.5


def _validation_row(step, reward, hits):
    return {
        'checkpoint_step': step,
        'mean_fixed_budget_reward': reward,
        'fixed_budget_rewards': [reward - 0.01, reward + 0.01],
        'returned_verified_hits': hits,
    }


def test_validation_selection_keeps_identity_without_one_se_promotion():
    rows = [
        _validation_row(0, 0.4, [False, True, False, True]),
        _validation_row(10, 0.5, [True, True, False, True]),
        _validation_row(20, 0.49, [True, True, False, True]),
    ]
    selection = select_validation_checkpoint(rows)
    assert selection['best_candidate']['checkpoint_step'] == 10
    assert not selection['promoted']
    assert selection['promoted_checkpoint']['checkpoint_step'] == 0
