"""Frozen-model verified-hit planner used by every SSP-v2 arm."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from gymnasium.spaces import Box

from stable_worldmodel.planning import ShootingCostEvaluator
from stable_worldmodel.planning.solver import CEMSolver
from stable_worldmodel.policy import PlanConfig

from .geometry import (
    ClipConsistentActionTransform,
    RotatedSearchCost,
    TrajectoryHitDiagnostic,
    fixed_budget_reward,
)
from .noise import HitFunnelObserver, KeyedNoiseSchedule


def model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class PlanningResult:
    pair_id: str
    first_hit_iteration: int | None
    returned_verified_hit: bool
    return_mode: str
    population_hit_fractions: list[float]
    elite_hit_fractions: list[float]
    first_hit_horizons: list[int | None]
    fixed_budget_reward: float
    first_hit_auc: float
    noise: dict[str, Any]
    action_clip: dict[str, Any]


class FrozenSSPV2Planner:
    """Execute one clip-consistent verified CEM solve for cached latents."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        basis: torch.Tensor,
        epsilon: float,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        device: str,
        num_samples: int = 300,
        n_steps: int = 30,
        topk: int = 30,
        horizon: int = 5,
        action_block: int = 5,
        var_scale: float = 1.0,
        late_hit_iterations: int = 5,
        hit_mass_beta: float = 0.25,
    ) -> None:
        self.model = model.eval().requires_grad_(False)
        self.basis = basis.detach().clone().float()
        self.epsilon = float(epsilon)
        self.action_mean = action_mean.detach().clone().float().reshape(-1)
        self.action_std = action_std.detach().clone().float().reshape(-1)
        self.raw_action_dim = self.action_mean.numel()
        self.device = torch.device(device)
        self.num_samples = int(num_samples)
        self.n_steps = int(n_steps)
        self.topk = int(topk)
        self.horizon = int(horizon)
        self.action_block = int(action_block)
        self.var_scale = float(var_scale)
        self.late_hit_iterations = int(late_hit_iterations)
        self.hit_mass_beta = float(hit_mass_beta)
        self._plan_config = PlanConfig(
            horizon=self.horizon,
            receding_horizon=self.horizon,
            action_block=self.action_block,
            warm_start=False,
        )
        self._action_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(1, self.raw_action_dim),
            dtype=np.float32,
        )

    def run(
        self,
        *,
        pair_id: str,
        start_embedding: torch.Tensor,
        goal_embedding: torch.Tensor,
        theta: torch.Tensor,
        noise_key: dict[str, Any],
    ) -> PlanningResult:
        objective = RotatedSearchCost(self.basis, theta)
        diagnostic = TrajectoryHitDiagnostic(self.epsilon, self.horizon)
        evaluator = ShootingCostEvaluator(
            self.model,
            objective,
            encode_goal=None,
            diagnostic=diagnostic,
        )
        noise = KeyedNoiseSchedule(noise_key)
        observer = HitFunnelObserver(self.num_samples, self.topk)
        transform = ClipConsistentActionTransform(
            self.action_mean,
            self.action_std,
            action_block=self.action_block,
        )
        solver = CEMSolver(
            cost=evaluator,
            batch_size=1,
            num_samples=self.num_samples,
            var_scale=self.var_scale,
            n_steps=self.n_steps,
            topk=self.topk,
            device=self.device,
            seed=0,
            candidate_noise=noise,
            candidate_transform=transform,
            iteration_observer=observer,
            verified_hit_key='hit_bits',
            return_best_evaluated=True,
            log_timing=False,
        )
        solver.configure(
            action_space=self._action_space,
            n_envs=1,
            config=self._plan_config,
        )
        start = (
            start_embedding.detach().float().to(self.device).reshape(1, 1, -1)
        )
        goal = (
            goal_embedding.detach().float().to(self.device).reshape(1, 1, -1)
        )
        info = {
            'pixels': torch.empty(
                1, 1, 0, device=self.device, dtype=start.dtype
            ),
            'emb': start,
            'goal_emb': goal,
        }
        output = solver.solve(info)
        observed = output['iteration_observer']
        first_hit = observed['first_hit_iteration'][0]
        population = [
            row[0] / self.num_samples
            for row in observed['population_hit_counts_by_iteration']
        ]
        elites = [
            row[0] / self.topk
            for row in observed['elite_hit_counts_by_iteration']
        ]
        horizons = [
            row[0] for row in observed['first_hit_horizons_by_iteration']
        ]
        returned_hit = bool(output['returned_verified_hit'][0])
        reward = fixed_budget_reward(
            returned_hit,
            population,
            late_iterations=self.late_hit_iterations,
            beta=self.hit_mass_beta,
        )
        first_hit_auc = (
            0.0
            if first_hit is None
            else (self.n_steps - first_hit + 1) / self.n_steps
        )
        return PlanningResult(
            pair_id=pair_id,
            first_hit_iteration=first_hit,
            returned_verified_hit=returned_hit,
            return_mode=output['return_mode'][0],
            population_hit_fractions=population,
            elite_hit_fractions=elites,
            first_hit_horizons=horizons,
            fixed_budget_reward=reward,
            first_hit_auc=first_hit_auc,
            noise=noise.record(),
            action_clip=transform.record(),
        )
