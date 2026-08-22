"""Frozen-LeWM SSP planning without simulator construction or stepping."""

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
    DiagonalSearchCost,
    OriginalHitDiagnostic,
    auc_reward,
)
from .noise import FirstHitObserver, KeyedNoiseSchedule


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
    hit_bits: list[bool]
    auc_reward: float
    noise: dict[str, Any]


class FrozenSSPPlanner:
    """Execute one frozen-model CEM solve for one cached observation pair."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        basis: torch.Tensor,
        epsilon: float,
        action_dim: int,
        device: str,
        num_samples: int = 300,
        n_steps: int = 30,
        topk: int = 30,
        horizon: int = 5,
        action_block: int = 5,
        var_scale: float = 1.0,
    ) -> None:
        self.model = model.eval().requires_grad_(False)
        self.basis = basis.detach().clone().float()
        self.epsilon = float(epsilon)
        self.raw_action_dim = int(action_dim)
        self.device = torch.device(device)
        self.num_samples = int(num_samples)
        self.n_steps = int(n_steps)
        self.topk = int(topk)
        self.horizon = int(horizon)
        self.action_block = int(action_block)
        self.var_scale = float(var_scale)
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
        psi: torch.Tensor,
        noise_key: dict[str, Any],
    ) -> PlanningResult:
        objective = DiagonalSearchCost(self.basis, psi)
        diagnostic = OriginalHitDiagnostic(self.epsilon)
        evaluator = ShootingCostEvaluator(
            self.model,
            objective,
            encode_goal=None,
            diagnostic=diagnostic,
        )
        noise = KeyedNoiseSchedule(noise_key)
        observer = FirstHitObserver(self.num_samples)
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
            iteration_observer=observer,
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
        hit_bits = [row[0] for row in observed['hit_bits_by_iteration']]
        return PlanningResult(
            pair_id=pair_id,
            first_hit_iteration=first_hit,
            hit_bits=hit_bits,
            auc_reward=auc_reward(first_hit, self.n_steps),
            noise=noise.record(),
        )
