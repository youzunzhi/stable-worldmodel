"""Keyed candidate-noise schedules and read-only hit observers."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch

from .contracts import SSPFailure, sha256_json


def _state_hash(generator: torch.Generator) -> str:
    state = generator.get_state().detach().cpu().contiguous().numpy()
    return hashlib.sha256(state.tobytes()).hexdigest()


class KeyedNoiseSchedule:
    """Generate and hash a deterministic Torch standard-normal stream."""

    def __init__(self, key: dict[str, Any]) -> None:
        self.key = key
        self.schedule_id = sha256_json(key)
        raw_seed = int.from_bytes(
            hashlib.sha256(
                json.dumps(key, sort_keys=True, separators=(',', ':')).encode()
            ).digest()[:8],
            'big',
        )
        self.seed = raw_seed % (2**63 - 1)
        self.generator: torch.Generator | None = None
        self.device: torch.device | None = None
        self.before_state_sha256: str | None = None
        self.after_state_sha256: str | None = None
        self._digest = hashlib.sha256()
        self.blocks = 0

    def __call__(
        self,
        *,
        step: int,
        batch_start: int,
        shape: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        del batch_start
        if step != self.blocks:
            raise SSPFailure(
                'SSP_CRN_MISMATCH',
                f'noise block {step} requested after {self.blocks} blocks',
            )
        if self.generator is None:
            self.device = device
            self.generator = torch.Generator(device=device).manual_seed(
                self.seed
            )
            self.before_state_sha256 = _state_hash(self.generator)
        elif device != self.device:
            raise SSPFailure(
                'SSP_CRN_MISMATCH', 'noise schedule device changed mid-solve'
            )
        value = torch.randn(
            *shape,
            generator=self.generator,
            device=device,
            dtype=dtype,
        )
        raw = value.detach().cpu().contiguous().numpy().tobytes()
        self._digest.update(raw)
        self.blocks += 1
        self.after_state_sha256 = _state_hash(self.generator)
        return value

    def record(self) -> dict:
        return {
            'noise_schedule_id': self.schedule_id,
            'key': self.key,
            'seed': self.seed,
            'generator': 'torch.Generator',
            'torch_version': torch.__version__,
            'device': str(self.device) if self.device is not None else None,
            'pre_state_sha256': self.before_state_sha256,
            'post_state_sha256': self.after_state_sha256,
            'standard_normal_blocks': self.blocks,
            'standard_normal_sha256': self._digest.hexdigest(),
        }


class FirstHitObserver:
    """Observe all candidate hit bits before elite selection."""

    def __init__(self, expected_candidates: int) -> None:
        self.expected_candidates = int(expected_candidates)
        self.reset()

    def reset(self) -> None:
        self.hit_bits_by_iteration: list[list[bool]] = []
        self.first_hit: list[int | None] | None = None

    def start_batch(self, *, start_idx: int, end_idx: int) -> None:
        if start_idx != 0 or self.hit_bits_by_iteration:
            raise ValueError('SSP planner requires one batch per solve')
        self.first_hit = [None] * (end_idx - start_idx)

    def __call__(
        self, *, step: int, batch_start: int, diagnostics: dict
    ) -> None:
        if batch_start != 0 or step != len(self.hit_bits_by_iteration):
            raise ValueError('unexpected SSP observer ordering')
        bits = diagnostics.get('hit_bits')
        if not torch.is_tensor(bits) or bits.dtype != torch.bool:
            raise TypeError('hit_bits must be a boolean tensor')
        if bits.ndim != 2 or bits.shape[1] != self.expected_candidates:
            raise ValueError(
                f'hit_bits shape {tuple(bits.shape)} does not observe all '
                f'{self.expected_candidates} candidates'
            )
        any_hit = bits.any(dim=1).detach().cpu().tolist()
        if self.first_hit is None or len(any_hit) != len(self.first_hit):
            raise ValueError('observer batch size changed')
        iteration = step + 1
        for row, hit in enumerate(any_hit):
            if hit and self.first_hit[row] is None:
                self.first_hit[row] = iteration
        self.hit_bits_by_iteration.append([bool(value) for value in any_hit])

    def end_solve(self) -> dict:
        return {
            'first_hit_iteration': list(self.first_hit or []),
            'hit_bits_by_iteration': list(self.hit_bits_by_iteration),
            'iterations': len(self.hit_bits_by_iteration),
            'candidates_observed_per_iteration': self.expected_candidates,
        }
