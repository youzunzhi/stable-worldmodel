"""Regression tests for solver warm-start device placement."""

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box, Discrete

from stable_worldmodel.planning.solver.cem import CEMSolver
from stable_worldmodel.planning.solver.gd import GradientSolver
from stable_worldmodel.planning.solver.icem import ICEMSolver
from stable_worldmodel.planning.solver.lagrangian import LagrangianSolver
from stable_worldmodel.planning.solver.mppi import MPPISolver
from stable_worldmodel.planning.solver.pgd import PGDSolver
from stable_worldmodel.planning.solver.predictive_sampling import (
    PredictiveSamplingSolver,
)
from stable_worldmodel.planning.solver.utils import prepare_init_action
from stable_worldmodel.policy import PlanConfig


class DummyCost:
    """Non-Actionable cost stand-in for initialization-only tests."""


def _foreign_devices() -> list[str]:
    devices = ['meta']
    if torch.cuda.is_available():
        devices.append('cuda')
    return devices


FOREIGN_DEVICES = _foreign_devices()


def _box_setup(horizon: int, action_dim: int):
    action_space = Box(low=-1, high=1, shape=(1, action_dim), dtype=np.float32)
    config = PlanConfig(
        horizon=horizon, receding_horizon=horizon, action_block=1
    )
    return action_space, config


def _discrete_setup(horizon: int, n_categories: int):
    action_space = Discrete(n_categories)
    config = PlanConfig(
        horizon=horizon, receding_horizon=horizon, action_block=1
    )
    return action_space, config


@pytest.mark.parametrize('other_device', FOREIGN_DEVICES)
@pytest.mark.parametrize('solver_cls', [CEMSolver, MPPISolver, ICEMSolver])
def test_distribution_warm_start_stays_on_input_device(
    solver_cls, other_device
):
    n_envs, horizon, action_dim = 2, 5, 4
    action_space, config = _box_setup(horizon, action_dim)
    solver = solver_cls(cost=DummyCost(), batch_size=n_envs, num_samples=8)
    solver.configure(action_space=action_space, n_envs=n_envs, config=config)
    warm_start = torch.zeros(
        n_envs, horizon - 1, solver.action_dim, device=other_device
    )

    mean, var = solver.init_action_distrib(n_envs, actions=warm_start)

    assert mean.shape == (n_envs, horizon, solver.action_dim)
    assert mean.device.type == torch.device(other_device).type
    assert var.shape == (n_envs, horizon, solver.action_dim)


@pytest.mark.parametrize('other_device', FOREIGN_DEVICES)
def test_predictive_sampling_warm_start_stays_on_input_device(other_device):
    n_envs, horizon, action_dim = 2, 5, 4
    action_space, config = _box_setup(horizon, action_dim)
    solver = PredictiveSamplingSolver(
        cost=DummyCost(), batch_size=n_envs, num_samples=8
    )
    solver.configure(action_space=action_space, n_envs=n_envs, config=config)
    warm_start = torch.zeros(
        n_envs, horizon - 1, solver.action_dim, device=other_device
    )

    nominal = solver.init_nominal(n_envs, actions=warm_start)

    assert nominal.shape == (n_envs, horizon, solver.action_dim)
    assert nominal.device.type == torch.device(other_device).type


@pytest.mark.parametrize('other_device', FOREIGN_DEVICES)
@pytest.mark.parametrize(
    'solver_cls,ctor_kwargs',
    [
        (GradientSolver, {'n_steps': 3}),
        (LagrangianSolver, {'n_steps': 3}),
    ],
)
def test_gradient_based_partial_warm_start_stays_on_solver_device(
    solver_cls, ctor_kwargs, other_device
):
    n_envs, horizon, action_dim = 2, 5, 4
    action_space, config = _box_setup(horizon, action_dim)
    solver = solver_cls(cost=DummyCost(), num_samples=3, **ctor_kwargs)
    solver.device = other_device
    solver.configure(action_space=action_space, n_envs=n_envs, config=config)
    warm_start = torch.zeros(
        n_envs, horizon - 1, solver.action_dim, device=other_device
    )

    if solver_cls is GradientSolver:
        solver.init_action(n_envs, actions=warm_start)
    else:
        solver.init_action(actions=warm_start)

    assert solver.init.shape[2] == horizon
    assert solver.init.device.type == torch.device(other_device).type


@pytest.mark.parametrize('other_device', FOREIGN_DEVICES)
def test_pgd_partial_warm_start_stays_on_solver_device(other_device):
    n_envs, horizon, n_categories = 2, 5, 4
    action_space, config = _discrete_setup(horizon, n_categories)
    solver = PGDSolver(cost=DummyCost(), n_steps=3, num_samples=3)
    solver.device = other_device
    solver.configure(action_space=action_space, n_envs=n_envs, config=config)
    warm_start = torch.zeros(
        n_envs,
        horizon - 1,
        solver.action_simplex_dim,
        device=other_device,
    )

    solver.init_action(actions=warm_start)

    assert solver.init.shape[2] == horizon
    assert solver.init.device.type == torch.device(other_device).type


@pytest.mark.parametrize('other_device', FOREIGN_DEVICES)
def test_prepare_init_action_cold_start_respects_device(other_device):
    n_envs, horizon, action_dim = 2, 5, 4
    result = prepare_init_action(
        DummyCost(),
        {'dummy': torch.zeros(n_envs)},
        None,
        horizon,
        n_envs=n_envs,
        action_dim=action_dim,
        device=other_device,
    )

    assert result.shape == (n_envs, horizon, action_dim)
    assert result.device.type == torch.device(other_device).type


@pytest.mark.parametrize('other_device', FOREIGN_DEVICES)
def test_prepare_init_action_moves_cpu_warm_start(other_device):
    n_envs, horizon, action_dim = 2, 5, 4
    warm_start = torch.zeros(n_envs, horizon - 2, action_dim)
    result = prepare_init_action(
        DummyCost(),
        {'dummy': torch.zeros(n_envs)},
        warm_start,
        horizon,
        n_envs=n_envs,
        action_dim=action_dim,
        device=other_device,
    )

    assert result.shape == (n_envs, horizon, action_dim)
    assert result.device.type == torch.device(other_device).type


def test_prepare_init_action_without_device_keeps_cpu_default():
    result = prepare_init_action(
        DummyCost(),
        {'dummy': torch.zeros(2)},
        None,
        5,
        n_envs=2,
        action_dim=4,
    )

    assert result.shape == (2, 5, 4)
    assert result.device.type == 'cpu'


@pytest.mark.parametrize(
    'solver_cls,ctor_kwargs',
    [
        (GradientSolver, {'n_steps': 3}),
        (LagrangianSolver, {'n_steps': 3}),
    ],
)
def test_gradient_based_full_warm_start_moves_to_solver_device(
    solver_cls, ctor_kwargs
):
    n_envs, horizon, action_dim = 2, 5, 4
    action_space, config = _box_setup(horizon, action_dim)
    solver = solver_cls(cost=DummyCost(), num_samples=3, **ctor_kwargs)
    solver.device = 'meta'
    solver.configure(action_space=action_space, n_envs=n_envs, config=config)
    warm_start = torch.zeros(n_envs, horizon, solver.action_dim)

    if solver_cls is GradientSolver:
        solver.init_action(n_envs, actions=warm_start)
    else:
        solver.init_action(actions=warm_start)

    assert solver.init.shape[2] == horizon
    assert solver.init.device.type == 'meta'


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires a CUDA device'
)
def test_cem_end_to_end_cuda_warm_start():
    class DummyCostModel:
        def get_cost(self, info_dict, action_candidates):
            batch_size, num_samples = action_candidates.shape[:2]
            return torch.rand(
                batch_size, num_samples, device=action_candidates.device
            )

    solver = CEMSolver(
        cost=DummyCostModel(),
        batch_size=1,
        num_samples=16,
        n_steps=1,
        topk=4,
        device='cuda',
    )
    action_space, config = _box_setup(horizon=5, action_dim=4)
    solver.configure(action_space=action_space, n_envs=1, config=config)
    info_dict = {'dummy': torch.zeros(1, device='cuda')}

    first = solver.solve(info_dict)
    leftover = first['actions'][:, 1:].to('cuda')
    second = solver.solve(info_dict, init_action=leftover)

    assert second['actions'].shape == (1, 5, solver.action_dim)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires a CUDA device'
)
def test_gradient_end_to_end_cuda_cold_start():
    class DiffCost:
        def get_cost(self, info_dict, action_candidates):
            return action_candidates.pow(2).sum(dim=(-2, -1))

    solver = GradientSolver(
        cost=DiffCost(),
        n_steps=2,
        num_samples=3,
        action_noise=0.1,
        device='cuda',
    )
    action_space, config = _box_setup(horizon=5, action_dim=4)
    solver.configure(action_space=action_space, n_envs=1, config=config)

    output = solver.solve({'dummy': torch.zeros(1, device='cuda')})

    assert output['actions'].shape == (1, 5, solver.action_dim)
