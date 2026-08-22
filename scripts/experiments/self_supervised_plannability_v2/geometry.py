"""Action-aligned rotated geometry and immutable binary verifier for SSP-v2."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .contracts import SSPV2Failure, sha256_file, write_json


def terminal_residual(info_dict: dict) -> torch.Tensor:
    """Return predicted terminal minus goal terminal as ``(B,S,D)``."""
    predicted = info_dict['predicted_emb'][..., -1, :]
    goal = info_dict['goal_emb']
    if goal.ndim == predicted.ndim + 1:
        goal_terminal = goal[..., -1, :]
    elif goal.ndim == predicted.ndim:
        goal_terminal = goal[..., -1, :].unsqueeze(1)
    else:
        raise SSPV2Failure(
            'SSP_V2_INPUT_HASH_MISMATCH',
            f'incompatible goal/predicted shapes {tuple(goal.shape)} and '
            f'{tuple(predicted.shape)}',
        )
    residual = predicted - goal_terminal
    if residual.shape[-1] != 192:
        raise SSPV2Failure(
            'SSP_V2_INPUT_HASH_MISMATCH',
            f'latent dimension is {residual.shape[-1]}, expected 192',
        )
    return residual


def build_action_effect_basis(
    effects: np.ndarray, parameter_dim: int = 32
) -> tuple[np.ndarray, dict]:
    """Build a deterministic uncentered PCA basis from model action effects."""
    matrix = np.asarray(effects, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 192:
        raise ValueError(
            f'action effects must have shape (N,192), got {matrix.shape}'
        )
    if matrix.shape[0] < parameter_dim or not np.isfinite(matrix).all():
        raise ValueError('insufficient or non-finite action effects')
    _, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
    basis = right[:parameter_dim].T.copy()
    for column in range(parameter_dim):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0:
            basis[:, column] *= -1
    with np.errstate(all='ignore'):
        gram = basis.T @ basis
    if not np.allclose(gram, np.eye(parameter_dim), rtol=0, atol=1e-10):
        raise AssertionError('action-effect basis is not orthonormal')
    with np.errstate(all='ignore'):
        energy = np.square(singular_values)
    metadata = {
        'algorithm': 'uncentered-action-effect-svd-sign-v1',
        'source_rows': int(matrix.shape[0]),
        'shape': [192, parameter_dim],
        'singular_values': singular_values[:parameter_dim].tolist(),
        'explained_second_moment_fraction': float(
            energy[:parameter_dim].sum()
            / max(energy.sum(), np.finfo(float).tiny)
        ),
        'max_abs_gram_error_float64': float(
            np.max(np.abs(gram - np.eye(parameter_dim)))
        ),
    }
    return basis.astype(np.float32), metadata


def save_action_effect_basis(
    root: str | Path,
    basis: np.ndarray,
    metadata: dict,
    *,
    seed: int,
) -> dict:
    path = Path(root) / 'action_effect_basis.npy'
    np.save(path, np.asarray(basis, dtype=np.float32), allow_pickle=False)
    payload = {
        **metadata,
        'seed': int(seed),
        'dtype': 'float32',
        'sha256': sha256_file(path),
        'max_abs_gram_error_float32': float(
            np.max(
                np.abs(
                    basis.T @ basis - np.eye(basis.shape[1], dtype=np.float32)
                )
            )
        ),
    }
    write_json(Path(root) / 'action_effect_basis.json', payload)
    return payload


class RotatedSearchCost(nn.Module):
    """Identity-residual SPD metric in an action-effect subspace."""

    def __init__(self, basis: torch.Tensor, theta: torch.Tensor) -> None:
        super().__init__()
        locked_basis = basis.detach().clone().float()
        locked_theta = theta.detach().clone().float()
        if locked_basis.shape != (192, 32) or locked_theta.shape != (32,):
            raise SSPV2Failure(
                'SSP_V2_INPUT_HASH_MISMATCH',
                'rotated basis/theta shape must be (192,32)/(32,)',
            )
        if not torch.allclose(
            locked_basis.T @ locked_basis,
            torch.eye(32),
            rtol=0,
            atol=3e-6,
        ):
            raise SSPV2Failure(
                'SSP_V2_INPUT_HASH_MISMATCH',
                'rotated basis is not orthonormal',
            )
        if not torch.isfinite(locked_theta).all():
            raise SSPV2Failure(
                'SSP_V2_INPUT_HASH_MISMATCH',
                'theta contains non-finite values',
            )
        self.register_buffer('basis', locked_basis)
        self.register_buffer('theta', locked_theta)

    @property
    def log_eigenvalues(self) -> torch.Tensor:
        return math.log(4.0) * torch.tanh(self.theta)

    @property
    def eigenvalues(self) -> torch.Tensor:
        return self.log_eigenvalues.exp()

    def forward(self, info_dict: dict) -> torch.Tensor:
        residual = terminal_residual(info_dict)
        basis = self.basis.to(device=residual.device, dtype=residual.dtype)
        eigenvalues = self.eigenvalues.to(
            device=residual.device, dtype=residual.dtype
        )
        projected = residual @ basis
        base = residual.square().sum(dim=-1)
        adjustment = (projected.square() * (eigenvalues - 1.0)).sum(dim=-1)
        return base + adjustment


class TrajectoryHitDiagnostic(nn.Module):
    """Return only binary any-prefix hits and their first model horizon."""

    def __init__(self, epsilon: float, horizon: int = 5) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.horizon = int(horizon)

    def forward(self, info_dict: dict) -> dict[str, torch.Tensor]:
        predicted = info_dict['predicted_emb']
        if predicted.ndim != 4 or predicted.shape[-1] != 192:
            raise SSPV2Failure(
                'SSP_V2_INPUT_HASH_MISMATCH',
                f'predicted trajectory shape is {tuple(predicted.shape)}',
            )
        trajectory = predicted[..., -self.horizon :, :].float()
        goal = info_dict['goal_emb']
        if goal.ndim == 4:
            goal_terminal = goal[..., -1, :]
        elif goal.ndim == 3:
            goal_terminal = goal[..., -1, :].unsqueeze(1)
        else:
            raise SSPV2Failure(
                'SSP_V2_INPUT_HASH_MISMATCH',
                f'goal trajectory shape is {tuple(goal.shape)}',
            )
        distances = (
            (trajectory - goal_terminal.unsqueeze(-2))
            .square()
            .sum(dim=-1, dtype=torch.float32)
        )
        per_horizon = distances < self.epsilon
        hit_bits = per_horizon.any(dim=-1)
        first = torch.argmax(per_horizon.to(torch.int8), dim=-1) + 1
        first = torch.where(hit_bits, first, torch.zeros_like(first))
        return {'hit_bits': hit_bits, 'first_hit_horizon': first}


class ClipConsistentActionTransform:
    """Project normalized plans through raw action bounds and back."""

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        *,
        action_block: int,
        raw_low: float = -1.0,
        raw_high: float = 1.0,
    ) -> None:
        mean = mean.detach().clone().float().reshape(-1)
        std = std.detach().clone().float().reshape(-1)
        if mean.shape != std.shape or torch.any(std <= 0):
            raise ValueError('action mean/std must be same-shape and positive')
        self.mean = mean.repeat(int(action_block))
        self.std = std.repeat(int(action_block))
        self.raw_low = float(raw_low)
        self.raw_high = float(raw_high)
        self.reset()

    def reset(self) -> None:
        self.clipped_values = 0
        self.total_values = 0

    def __call__(self, actions: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=actions.device, dtype=actions.dtype)
        std = self.std.to(device=actions.device, dtype=actions.dtype)
        if actions.shape[-1] != mean.numel():
            raise ValueError(
                f'action dimension {actions.shape[-1]} != {mean.numel()}'
            )
        raw = actions * std + mean
        clipped = raw.clamp(self.raw_low, self.raw_high)
        self.clipped_values += int((clipped != raw).sum().detach().cpu())
        self.total_values += raw.numel()
        return (clipped - mean) / std

    def record(self) -> dict:
        return {
            'clipped_values': self.clipped_values,
            'total_values': self.total_values,
            'clip_fraction': (
                self.clipped_values / self.total_values
                if self.total_values
                else 0.0
            ),
            'raw_low': self.raw_low,
            'raw_high': self.raw_high,
        }


def fixed_budget_reward(
    returned_hit: bool,
    population_hit_fractions: list[float],
    *,
    late_iterations: int = 5,
    beta: float = 0.25,
) -> float:
    if len(population_hit_fractions) < late_iterations:
        raise ValueError('not enough CEM iterations for late hit mass')
    late_mass = float(np.mean(population_hit_fractions[-late_iterations:]))
    return float(bool(returned_hit)) + float(beta) * late_mass


def orthogonal_directions(
    rng: np.random.Generator, count: int, dimension: int
) -> np.ndarray:
    """Return randomized orthogonal rows with isotropic norm sqrt(dimension)."""
    if count > dimension:
        raise ValueError('orthogonal direction count exceeds dimension')
    raw = rng.standard_normal((dimension, dimension), dtype=np.float64)
    orthogonal, _ = np.linalg.qr(raw)
    directions = orthogonal[:, :count].T * math.sqrt(dimension)
    return directions.astype(np.float32)
