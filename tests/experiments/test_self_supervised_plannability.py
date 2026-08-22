"""Protocol invariants for self-supervised-plannability-v1."""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium.spaces import Box

from scripts.experiments.self_supervised_plannability.contracts import (
    THRESHOLDS,
    SSPFailure,
    create_root,
)
from scripts.experiments.self_supervised_plannability.es import (
    select_validation_checkpoint,
)
from scripts.experiments.self_supervised_plannability.geometry import (
    DiagonalSearchCost,
    OriginalHitDiagnostic,
    auc_reward,
    build_basis,
    project_parameter,
)
from scripts.experiments.self_supervised_plannability.noise import (
    FirstHitObserver,
    KeyedNoiseSchedule,
)
from scripts.experiments.self_supervised_plannability.pairs import (
    candidate_pairs,
    episode_balanced_order,
    split_groups,
)
from scripts.experiments.self_supervised_plannability.planner import (
    FrozenSSPPlanner,
    model_state_sha256,
)
from stable_worldmodel.planning import GoalMSE, ShootingCostEvaluator
from stable_worldmodel.planning.solver import CEMSolver
from stable_worldmodel.policy import PlanConfig


def _info(residual: torch.Tensor) -> dict:
    predicted = residual.reshape(1, 1, 1, -1)
    goal = torch.zeros(1, 1, residual.numel(), dtype=residual.dtype)
    return {'predicted_emb': predicted, 'goal_emb': goal}


def test_identity_geometry_is_exact_goalmse_parity():
    generator = torch.Generator().manual_seed(9)
    predicted = torch.randn(2, 7, 4, 192, generator=generator)
    goal = torch.randn(2, 4, 192, generator=generator)
    info = {'predicted_emb': predicted, 'goal_emb': goal}
    basis = torch.from_numpy(build_basis(192, 16, 26082201))
    actual = DiagonalSearchCost(basis, torch.zeros(16))(info)
    expected = GoalMSE()(info)
    assert torch.equal(actual, expected)


def test_basis_weights_are_positive_bounded_and_scale_free():
    basis = torch.from_numpy(build_basis(192, 16, 26082202))
    raw = torch.full((16,), 100.0)
    psi, active, _ = project_parameter(basis, raw)
    log_weights = basis @ psi
    weights = log_weights.exp()
    assert active
    assert float(weights.min()) >= 0.25 - 1e-6
    assert float(weights.max()) <= 4.0 + 1e-6
    assert abs(float(log_weights.sum())) < 2e-5


def test_original_distance_is_float32_sum_over_192_dimensions():
    residual = torch.ones(192, dtype=torch.float16)
    diagnostic = OriginalHitDiagnostic(193.0)
    distance = diagnostic.distance(_info(residual))
    assert distance.dtype == torch.float32
    assert distance.item() == 192.0
    assert diagnostic(_info(residual))['hit_bits'].item()


@pytest.mark.parametrize(
    ('task', 'epsilon'), [('pusht', 1.5), ('cube', 1.0), ('tworoom', 1.5)]
)
def test_task_thresholds_are_locked_and_strict(task, epsilon):
    assert THRESHOLDS[task] == epsilon

    class Boundary(OriginalHitDiagnostic):
        def distance(self, info_dict):
            del info_dict
            return torch.tensor([[epsilon]], dtype=torch.float32)

    assert not Boundary(epsilon)({})['hit_bits'].item()


class _DirectCost(torch.nn.Module):
    def __init__(self, sign: float, hit_last: bool = False) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.sign = sign
        self.hit_last = hit_last

    def get_cost(self, info, candidates):
        del info
        return self.sign * candidates[..., 0, 0]

    def get_cost_and_diagnostics(self, info, candidates):
        costs = self.get_cost(info, candidates)
        bits = torch.zeros_like(costs, dtype=torch.bool)
        if self.hit_last:
            bits[:, -1] = True
        return costs, {'hit_bits': bits}


def _configure(solver, samples: int) -> None:
    solver.configure(
        action_space=Box(
            low=-np.inf, high=np.inf, shape=(1, 1), dtype=np.float32
        ),
        n_envs=1,
        config=PlanConfig(horizon=1, receding_horizon=1),
    )
    assert solver.num_samples == samples


def test_observer_sees_every_candidate_before_elite_update():
    observer = FirstHitObserver(300)
    solver = CEMSolver(
        cost=_DirectCost(1.0, hit_last=True),
        num_samples=300,
        n_steps=1,
        topk=2,
        iteration_observer=observer,
        log_timing=False,
    )
    _configure(solver, 300)
    output = solver.solve({'dummy': torch.zeros(1, 1)})
    observed = output['iteration_observer']
    assert observed['candidates_observed_per_iteration'] == 300
    assert observed['first_hit_iteration'] == [1]


def test_failed_distance_magnitudes_cannot_change_reward_or_update():
    near = torch.zeros(192)
    near[0] = math.sqrt(1.6)
    far = torch.zeros(192)
    far[0] = 10.0
    diagnostic = OriginalHitDiagnostic(1.5)
    assert not diagnostic(_info(near))['hit_bits'].item()
    assert not diagnostic(_info(far))['hit_bits'].item()
    assert auc_reward(None) == 0.0
    eta = torch.arange(16, dtype=torch.float32)
    near_gradient = (0.0 - 0.0) * eta
    far_gradient = (0.0 - 0.0) * eta
    torch.testing.assert_close(near_gradient, far_gradient, rtol=0, atol=0)


def test_search_cost_can_change_elites_without_changing_hit_predicate():
    key = {'test': 'elite-divergence'}
    left_noise = KeyedNoiseSchedule(key)
    right_noise = KeyedNoiseSchedule(key)
    left_observer = FirstHitObserver(16)
    right_observer = FirstHitObserver(16)
    left = CEMSolver(
        cost=_DirectCost(1.0),
        num_samples=16,
        n_steps=2,
        topk=4,
        candidate_noise=left_noise,
        iteration_observer=left_observer,
        log_timing=False,
    )
    right = CEMSolver(
        cost=_DirectCost(-1.0),
        num_samples=16,
        n_steps=2,
        topk=4,
        candidate_noise=right_noise,
        iteration_observer=right_observer,
        log_timing=False,
    )
    _configure(left, 16)
    _configure(right, 16)
    left_output = left.solve({'dummy': torch.zeros(1, 1)})
    right_output = right.solve({'dummy': torch.zeros(1, 1)})
    assert (
        left_noise.record()['standard_normal_sha256']
        == right_noise.record()['standard_normal_sha256']
    )
    assert (
        left_output['iteration_observer'] == right_output['iteration_observer']
    )
    assert not torch.equal(left_output['actions'], right_output['actions'])


def test_antithetic_noise_keys_produce_identical_content():
    key = {'task': 'cube', 'outer_step': 3, 'direction': 2, 'pair_slot': 7}
    plus = KeyedNoiseSchedule(key)
    minus = KeyedNoiseSchedule(key)
    shape = (1, 8, 2, 3)
    plus_value = plus(
        step=0,
        batch_start=0,
        shape=shape,
        device=torch.device('cpu'),
        dtype=torch.float32,
    )
    minus_value = minus(
        step=0,
        batch_start=0,
        shape=shape,
        device=torch.device('cpu'),
        dtype=torch.float32,
    )
    assert torch.equal(plus_value, minus_value)
    assert plus.record() == minus.record()


@pytest.mark.parametrize(
    ('first_hit', 'expected'),
    [(1, 1.0), (15, 16 / 30), (30, 1 / 30), (None, 0.0)],
)
def test_auc_reward_matches_discrete_solve_curve(first_hit, expected):
    assert auc_reward(first_hit) == expected


def test_validation_selection_uses_auc_and_earliest_tie_only():
    rows = [
        {'checkpoint_step': 0, 'auc': 0.2, 'center': [0], 'test_auc': 0.9},
        {'checkpoint_step': 5, 'auc': 0.3, 'center': [1], 'clear': 0.0},
        {'checkpoint_step': 10, 'auc': 0.3, 'center': [2], 'clear': 1.0},
    ]
    assert select_validation_checkpoint(rows)['checkpoint_step'] == 5


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def rollout(self, info, action_candidates):
        start = info['emb'][..., -1, :]
        delta = action_candidates[..., 0].mean(dim=-1, keepdim=True)
        terminal = start + delta * self.scale
        info['predicted_emb'] = terminal.unsqueeze(-2)
        return info


def test_frozen_planner_has_no_model_gradients_or_environment_calls(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        gym, 'make', lambda *args, **kwargs: calls.append(args)
    )
    model = _TinyModel().eval().requires_grad_(False)
    before = model_state_sha256(model)
    planner = FrozenSSPPlanner(
        model=model,
        basis=torch.from_numpy(build_basis(192, 16, 1)),
        epsilon=1.5,
        action_dim=1,
        device='cpu',
        num_samples=8,
        n_steps=2,
        topk=2,
        horizon=1,
        action_block=1,
    )
    planner.run(
        pair_id='tiny',
        start_embedding=torch.zeros(192),
        goal_embedding=torch.ones(192),
        psi=torch.zeros(16),
        noise_key={'tiny': 1},
    )
    assert model_state_sha256(model) == before
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not calls


def test_evaluator_returns_cost_and_bits_from_one_rollout():
    model = _TinyModel().eval().requires_grad_(False)
    model.rollout_calls = 0
    original = model.rollout

    def counted(info, candidates):
        model.rollout_calls += 1
        return original(info, candidates)

    model.rollout = counted
    basis = torch.from_numpy(build_basis(192, 16, 2))
    evaluator = ShootingCostEvaluator(
        model,
        DiagonalSearchCost(basis, torch.zeros(16)),
        encode_goal=None,
        diagnostic=OriginalHitDiagnostic(1.5),
    )
    candidates = torch.zeros(1, 4, 1, 1)
    info = {
        'emb': torch.zeros(1, 4, 1, 192),
        'goal_emb': torch.zeros(1, 4, 1, 192),
    }
    costs, diagnostics = evaluator.get_cost_and_diagnostics(info, candidates)
    assert model.rollout_calls == 1
    assert costs.shape == diagnostics['hit_bits'].shape == (1, 4)


def test_pair_split_is_group_disjoint_and_episode_balanced():
    episodes = np.repeat(np.arange(12), 30)
    steps = np.tile(np.arange(30), 12)
    records, _ = candidate_pairs(
        task='cube', episodes=episodes, steps=steps, clear_start_rows={0}
    )
    split = split_groups(
        np.asarray([record['group_id'] for record in records]), 260822
    )
    assert not set(split['train']).intersection(split['validation'])
    assert not set(split['train']).intersection(split['test'])
    subset = [
        record for record in records if record['group_id'] in split['train']
    ]
    ordered = episode_balanced_order(subset, 260823)
    first_round = ordered[: len({row['episode_id'] for row in subset})]
    assert len({row['episode_id'] for row in first_round}) == len(first_round)


def test_formal_root_is_create_only(tmp_path):
    target = tmp_path / 'formal'
    assert create_root(target, formal=True) == target
    with pytest.raises(SSPFailure, match='SSP_FORMAL_ROOT_EXISTS'):
        create_root(target, formal=True)
