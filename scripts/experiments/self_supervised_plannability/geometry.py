"""Bounded diagonal SSP geometry and immutable hit predicate."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .contracts import SSPFailure, sha256_file, write_json


def terminal_residual(info_dict: dict) -> torch.Tensor:
    """Return predicted-terminal minus goal-terminal as ``(B, S, D)``."""
    predicted = info_dict['predicted_emb'][..., -1, :]
    goal = info_dict['goal_emb']
    if goal.ndim == predicted.ndim + 1:
        goal_terminal = goal[..., -1, :]
    elif goal.ndim == predicted.ndim:
        goal_terminal = goal[..., -1, :].unsqueeze(1)
    else:
        raise SSPFailure(
            'SSP_LATENT_CONTRACT_MISMATCH',
            f'goal shape {tuple(goal.shape)} is incompatible with '
            f'predicted shape {tuple(predicted.shape)}',
        )
    residual = predicted - goal_terminal
    if residual.shape[-1] != 192:
        raise SSPFailure(
            'SSP_LATENT_CONTRACT_MISMATCH',
            f'latent dimension is {residual.shape[-1]}, expected 192',
        )
    return residual


def build_basis(latent_dim: int, parameter_dim: int, seed: int) -> np.ndarray:
    """Construct the locked CPU float64 basis, then cast to float32."""
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    raw = rng.standard_normal((latent_dim, parameter_dim), dtype=np.float64)
    ones = np.ones(latent_dim, dtype=np.float64) / math.sqrt(latent_dim)
    with np.errstate(all='ignore'):
        raw = raw - ones[:, None] * (ones @ raw)[None, :]
    if not np.isfinite(raw).all():
        raise AssertionError('SSP basis draw or projection is non-finite')
    basis, _ = np.linalg.qr(raw, mode='reduced')
    for column in range(parameter_dim):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0:
            basis[:, column] *= -1
    with np.errstate(all='ignore'):
        gram = basis.T @ basis
        scale_component = basis.T @ ones
    if not np.allclose(gram, np.eye(parameter_dim), rtol=0, atol=1e-12):
        raise AssertionError('float64 SSP basis is not orthonormal')
    if not np.allclose(scale_component, 0, rtol=0, atol=1e-12):
        raise AssertionError('SSP basis contains a global-scale direction')
    result = basis.astype(np.float32)
    with np.errstate(all='ignore'):
        result_gram = result.T @ result
    if not np.allclose(
        result_gram,
        np.eye(parameter_dim, dtype=np.float32),
        rtol=0,
        atol=2e-6,
    ):
        raise AssertionError('float32 SSP basis is not orthonormal')
    return result


def save_basis(root: str | Path, basis: np.ndarray, seed: int) -> dict:
    path = Path(root) / 'geometry_basis.npy'
    np.save(path, np.asarray(basis, dtype=np.float32), allow_pickle=False)
    with np.errstate(all='ignore'):
        gram = basis.T @ basis
        scale_component = basis.T @ np.ones(basis.shape[0])
    metadata = {
        'algorithm': 'numpy.PCG64DXSM-normal-project-qr-sign-v1',
        'seed': int(seed),
        'shape': list(basis.shape),
        'dtype': str(basis.dtype),
        'sha256': sha256_file(path),
        'max_abs_gram_error': float(
            np.max(np.abs(gram - np.eye(basis.shape[1])))
        ),
        'max_abs_global_scale_component': float(
            np.max(np.abs(scale_component))
        ),
    }
    write_json(Path(root) / 'geometry_basis.json', metadata)
    return metadata


def project_parameter(
    basis: torch.Tensor, psi: torch.Tensor
) -> tuple[torch.Tensor, bool, float]:
    """Apply the locked single radial rescale in log-weight space."""
    ell = basis.to(device=psi.device, dtype=psi.dtype) @ psi
    maximum = float(ell.abs().max().detach().cpu())
    bound = math.log(4.0)
    if maximum <= bound:
        return psi.clone(), False, 1.0
    scale = bound / maximum
    return psi * scale, True, scale


class DiagonalSearchCost(nn.Module):
    """SSP search cost with positive weights in the fixed basis."""

    def __init__(self, basis: torch.Tensor, psi: torch.Tensor) -> None:
        super().__init__()
        locked_basis = basis.detach().clone().float()
        locked_psi = psi.detach().clone().float()
        if locked_basis.shape != (192, 16) or locked_psi.shape != (16,):
            raise SSPFailure(
                'SSP_LATENT_CONTRACT_MISMATCH',
                'SSP basis/parameter shape is not (192,16)/(16,)',
            )
        gram = locked_basis.T @ locked_basis
        if not torch.allclose(
            gram, torch.eye(16), rtol=0, atol=2e-6
        ) or not torch.allclose(
            locked_basis.T @ torch.ones(192),
            torch.zeros(16),
            rtol=0,
            atol=2e-5,
        ):
            raise SSPFailure(
                'SSP_LATENT_CONTRACT_MISMATCH', 'invalid SSP geometry basis'
            )
        if float((locked_basis @ locked_psi).abs().max()) > math.log(4) + 1e-6:
            raise SSPFailure(
                'SSP_LATENT_CONTRACT_MISMATCH', 'SSP geometry exceeds bounds'
            )
        self.register_buffer('basis', locked_basis)
        self.register_buffer('psi', locked_psi)

    @property
    def log_weights(self) -> torch.Tensor:
        return self.basis @ self.psi

    @property
    def weights(self) -> torch.Tensor:
        return self.log_weights.exp()

    def forward(self, info_dict: dict) -> torch.Tensor:
        residual = terminal_residual(info_dict)
        weights = self.weights.to(device=residual.device, dtype=residual.dtype)
        return (residual.square() * weights).sum(dim=-1)


class OriginalHitDiagnostic(nn.Module):
    """Float32 sum-SSE hit predicate; exposes bits, never magnitudes."""

    def __init__(self, epsilon: float) -> None:
        super().__init__()
        self.epsilon = float(epsilon)

    def distance(self, info_dict: dict) -> torch.Tensor:
        residual = terminal_residual(info_dict).float()
        return residual.square().sum(dim=-1, dtype=torch.float32)

    def forward(self, info_dict: dict) -> dict[str, torch.Tensor]:
        return {'hit_bits': self.distance(info_dict) < self.epsilon}


def auc_reward(first_hit_iteration: int | None, iterations: int = 30) -> float:
    if first_hit_iteration is None:
        return 0.0
    if not 1 <= int(first_hit_iteration) <= iterations:
        raise ValueError('first hit lies outside the planner budget')
    return (iterations - int(first_hit_iteration) + 1) / iterations
