"""Keyed common-random-number schedules and v2 hit-funnel observer."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch

from .contracts import SSPV2Failure, sha256_json


def _state_hash(generator: torch.Generator) -> str:
    state = generator.get_state().detach().cpu().contiguous().numpy()
    return hashlib.sha256(state.tobytes()).hexdigest()


class KeyedNoiseSchedule:
    """Generate and hash a deterministic Torch standard-normal stream."""

    def __init__(self, key: dict[str, Any]) -> None:
        self.key = key
        self.schedule_id = sha256_json(key)
        encoded = json.dumps(
            key, sort_keys=True, separators=(',', ':')
        ).encode()
        self.seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8], 'big')
        self.seed %= 2**63 - 1
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
            raise SSPV2Failure(
                'SSP_V2_CRN_MISMATCH',
                f'noise block {step} requested after {self.blocks} blocks',
            )
        if self.generator is None:
            self.device = device
            self.generator = torch.Generator(device=device).manual_seed(
                self.seed
            )
            self.before_state_sha256 = _state_hash(self.generator)
        elif device != self.device:
            raise SSPV2Failure(
                'SSP_V2_CRN_MISMATCH', 'noise device changed mid-solve'
            )
        value = torch.randn(
            *shape,
            generator=self.generator,
            device=device,
            dtype=dtype,
        )
        self._digest.update(
            value.detach().cpu().contiguous().numpy().tobytes()
        )
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


class HitFunnelObserver:
    """Record candidate and elite binary hit flow without distances."""

    def __init__(self, expected_candidates: int, expected_elites: int) -> None:
        self.expected_candidates = int(expected_candidates)
        self.expected_elites = int(expected_elites)
        self.reset()

    def reset(self) -> None:
        self.first_hit: list[int | None] | None = None
        self.population_hit_counts: list[list[int]] = []
        self.elite_hit_counts: list[list[int]] = []
        self.first_hit_horizons: list[list[int | None]] = []
        self._diagnostics_by_step: dict[int, dict] = {}

    def start_batch(self, *, start_idx: int, end_idx: int) -> None:
        if start_idx != 0 or self.population_hit_counts:
            raise ValueError('SSP-v2 planner requires one batch per solve')
        self.first_hit = [None] * (end_idx - start_idx)

    def __call__(
        self, *, step: int, batch_start: int, diagnostics: dict
    ) -> None:
        if batch_start != 0 or step != len(self.population_hit_counts):
            raise ValueError('unexpected v2 observer ordering')
        bits = diagnostics.get('hit_bits')
        horizons = diagnostics.get('first_hit_horizon')
        if (
            not torch.is_tensor(bits)
            or bits.dtype != torch.bool
            or bits.ndim != 2
            or bits.shape[1] != self.expected_candidates
        ):
            raise TypeError('hit_bits must cover every candidate')
        if not torch.is_tensor(horizons) or horizons.shape != bits.shape:
            raise TypeError('first_hit_horizon must match hit_bits')
        counts = bits.sum(dim=1).detach().cpu().tolist()
        any_hit = bits.any(dim=1).detach().cpu().tolist()
        if self.first_hit is None or len(counts) != len(self.first_hit):
            raise ValueError('observer batch size changed')
        for row, value in enumerate(any_hit):
            if value and self.first_hit[row] is None:
                self.first_hit[row] = step + 1
        first_horizons = []
        for row in range(bits.shape[0]):
            row_values = horizons[row][bits[row]]
            first_horizons.append(
                int(row_values.min().detach().cpu())
                if len(row_values)
                else None
            )
        self.population_hit_counts.append([int(value) for value in counts])
        self.first_hit_horizons.append(first_horizons)
        self._diagnostics_by_step[step] = diagnostics

    def after_selection(
        self,
        *,
        step: int,
        batch_start: int,
        diagnostics: dict,
        topk_inds: torch.Tensor,
    ) -> None:
        if (
            batch_start != 0
            or diagnostics is not self._diagnostics_by_step[step]
        ):
            raise ValueError('selection diagnostics do not match observation')
        bits = diagnostics['hit_bits']
        selected = bits.gather(1, topk_inds)
        if selected.shape[1] != self.expected_elites:
            raise ValueError('elite count changed')
        counts = selected.sum(dim=1).detach().cpu().tolist()
        self.elite_hit_counts.append([int(value) for value in counts])

    def end_solve(self) -> dict:
        if len(self.elite_hit_counts) != len(self.population_hit_counts):
            raise ValueError('missing elite hit observations')
        return {
            'first_hit_iteration': list(self.first_hit or []),
            'population_hit_counts_by_iteration': self.population_hit_counts,
            'elite_hit_counts_by_iteration': self.elite_hit_counts,
            'first_hit_horizons_by_iteration': self.first_hit_horizons,
            'iterations': len(self.population_hit_counts),
            'candidates_observed_per_iteration': self.expected_candidates,
            'elites_observed_per_iteration': self.expected_elites,
        }
