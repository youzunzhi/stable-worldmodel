"""Latent-blind uniform and task-space-stratified pair sampling."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from itertools import product

import numpy as np

from .contracts import LABEL_F, LABEL_T, TaskContract


@dataclass(frozen=True)
class AffinePermutation:
    """A random affine permutation prefix over ``range(domain)``.

    The random offset makes every population member have equal first-order
    inclusion probability.  A coprime step guarantees no replacement.
    ``max_items`` also bounds the step so vectorized uint64 arithmetic cannot
    overflow while producing the requested prefix.
    """

    domain: int
    multiplier: int
    offset: int
    max_items: int

    @classmethod
    def create(
        cls, domain: int, max_items: int, seed: int
    ) -> AffinePermutation:
        if not 0 < max_items <= domain:
            raise ValueError('sample count must be in [1, domain]')
        digest = hashlib.sha256(f'affine:{seed}:{domain}'.encode()).digest()
        raw_a = int.from_bytes(digest[:8], 'little')
        raw_b = int.from_bytes(digest[8:16], 'little')
        uint64_max = np.iinfo(np.uint64).max
        safe_max = (uint64_max - (domain - 1)) // max(max_items, 1)
        max_a = max(1, min(domain - 1, safe_max))
        a = 1 + raw_a % max_a
        while math.gcd(a, domain) != 1:
            a += 1
            if a > max_a:
                a = 1
        return cls(domain, a, raw_b % domain, max_items)

    def values(self, start: int, count: int) -> np.ndarray:
        if start < 0 or count < 0 or start + count > self.max_items:
            raise ValueError('requested affine prefix is out of range')
        indices = np.arange(start, start + count, dtype=np.uint64)
        values = (
            np.uint64(self.offset) + np.uint64(self.multiplier) * indices
        ) % np.uint64(self.domain)
        return values


def uniform_ordered_pairs(
    partition_rows: np.ndarray,
    *,
    start: int,
    count: int,
    total_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, AffinePermutation]:
    """Draw an equal-probability no-replacement prefix of ordered i!=j pairs."""
    rows = np.asarray(partition_rows, dtype=np.int64)
    n_rows = len(rows)
    population = n_rows * (n_rows - 1)
    permutation = AffinePermutation.create(population, total_count, seed)
    flat = permutation.values(start, count)
    anchors_local = flat // np.uint64(n_rows - 1)
    goals_local = flat % np.uint64(n_rows - 1)
    goals_local += goals_local >= anchors_local
    anchor = rows[anchors_local.astype(np.int64)]
    goal = rows[goals_local.astype(np.int64)]
    if np.any(anchor == goal):
        raise AssertionError('uniform sampler emitted a self-pair')
    return anchor, goal, permutation


class DenseSpatialGrid:
    """Dense cell index used only over simulator task state."""

    def __init__(self, states: np.ndarray, cell_size: float):
        state = np.asarray(states, dtype=np.float64)
        if state.ndim != 2 or not np.isfinite(state).all():
            raise ValueError('grid states must be finite (N, D)')
        if cell_size <= 0:
            raise ValueError('cell_size must be positive')
        raw = np.floor(state / cell_size).astype(np.int64)
        self.minimum = raw.min(axis=0)
        self.coordinates = raw - self.minimum
        self.dimensions = self.coordinates.max(axis=0) + 1
        population = int(np.prod(self.dimensions, dtype=np.int64))
        if population > 50_000_000:
            raise ValueError(f'spatial grid is too large: {population} cells')
        self.cell_ids = np.ravel_multi_index(
            self.coordinates.T, tuple(self.dimensions)
        )
        self.order = np.argsort(self.cell_ids, kind='stable')
        self.counts = np.bincount(self.cell_ids, minlength=population)
        self.starts = np.zeros(population, dtype=np.int64)
        self.starts[1:] = np.cumsum(self.counts[:-1], dtype=np.int64)

    def propose(
        self,
        anchor_local: np.ndarray,
        offsets: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        anchor_local = np.asarray(anchor_local, dtype=np.int64)
        offset_index = rng.integers(0, len(offsets), size=len(anchor_local))
        target_coord = self.coordinates[anchor_local] + offsets[offset_index]
        valid = np.all(
            (target_coord >= 0) & (target_coord < self.dimensions), axis=1
        )
        safe_coord = np.clip(target_coord, 0, self.dimensions - 1)
        cell = np.ravel_multi_index(safe_coord.T, tuple(self.dimensions))
        count = self.counts[cell]
        valid &= count > 0
        safe_count = np.maximum(count, 1)
        slot = (rng.random(len(anchor_local)) * safe_count).astype(np.int64)
        position = self.starts[cell] + slot
        # A multidimensional bounding grid can end in an empty flattened cell
        # even though every coordinate-wise maximum is occupied somewhere.
        # Invalid proposals are rejected by the returned mask, but must first
        # be redirected to a safe row so indexing itself remains total.
        position[~valid] = 0
        goal_local = self.order[position]
        return goal_local, count.astype(np.float64), valid


def _offsets(dimension: int, outer: float, cell_size: float) -> np.ndarray:
    radius = int(np.ceil(outer / cell_size))
    return np.asarray(
        list(product(range(-radius, radius + 1), repeat=dimension)),
        dtype=np.int64,
    )


def sample_task_stratum(
    *,
    partition_rows: np.ndarray,
    all_states: np.ndarray,
    all_groups: np.ndarray,
    contract: TaskContract,
    target_count: int,
    lower_exclusive: float | None,
    upper_inclusive: float | None,
    label: np.uint8,
    seed: int,
    total_dataset_rows: int,
) -> dict[str, np.ndarray | int | float]:
    """Draw one task-space band without reading pixels or model outputs.

    Finite bands propose a uniformly random anchor, a uniformly random nearby
    grid cell, then a uniformly random row inside that cell.  The cell count is
    the inverse-proposal design weight up to constants that cancel in weighted
    ratios.  The far band uses a uniform goal and constant weight.
    """
    if target_count <= 0:
        raise ValueError('target_count must be positive')
    rows = np.asarray(partition_rows, dtype=np.int64)
    states = np.asarray(all_states)[rows]
    groups = np.asarray(all_groups)[rows]
    rng = np.random.default_rng(seed)
    finite_band = upper_inclusive is not None
    grid = None
    offsets = None
    if finite_band:
        width = float(upper_inclusive) - float(lower_exclusive or 0.0)
        cell_size = max(width, float(upper_inclusive) / 4.0)
        grid = DenseSpatialGrid(states, cell_size)
        offsets = _offsets(states.shape[1], float(upper_inclusive), cell_size)

    accepted: list[dict[str, np.ndarray]] = []
    accepted_raw = 0
    proposals = 0
    max_proposals = max(20_000_000, target_count * 300)
    while accepted_raw < int(np.ceil(target_count * 1.03)):
        remaining = target_count - accepted_raw
        batch = min(4_000_000, max(250_000, remaining * 4))
        anchor_local = rng.integers(0, len(rows), size=batch, dtype=np.int64)
        if finite_band:
            assert grid is not None and offsets is not None
            goal_local, design_weight, valid = grid.propose(
                anchor_local, offsets, rng
            )
        else:
            goal_local = rng.integers(0, len(rows), size=batch, dtype=np.int64)
            design_weight = np.full(batch, len(rows), dtype=np.float64)
            valid = np.ones(batch, dtype=bool)
        valid &= anchor_local != goal_local
        error = contract.task_error(states[anchor_local], states[goal_local])
        if lower_exclusive is not None:
            valid &= error > lower_exclusive
        if upper_inclusive is not None:
            if int(label) == int(LABEL_T):
                valid &= error < upper_inclusive
            else:
                valid &= error <= upper_inclusive
        labels = contract.classify(error)
        valid &= labels == label
        selected = np.flatnonzero(valid)
        if len(selected):
            anchor_row = rows[anchor_local[selected]]
            goal_row = rows[goal_local[selected]]
            accepted.append(
                {
                    'anchor_row': anchor_row,
                    'goal_row': goal_row,
                    'anchor_group': groups[anchor_local[selected]],
                    'goal_group': groups[goal_local[selected]],
                    'task_error': error[selected],
                    'analysis_weight': design_weight[selected],
                }
            )
            accepted_raw += len(selected)
        proposals += batch
        if proposals > max_proposals:
            raise RuntimeError(
                f'stratum sampling shortfall: {accepted_raw}/{target_count} '
                f'after {proposals} proposals'
            )

    merged = {
        key: np.concatenate([part[key] for part in accepted])
        for key in accepted[0]
    }
    pair_key = merged['anchor_row'].astype(np.uint64) * np.uint64(
        total_dataset_rows
    ) + merged['goal_row'].astype(np.uint64)
    _, first = np.unique(pair_key, return_index=True)
    first.sort()
    if len(first) < target_count:
        # Extremely unlikely at formal scale; rerun with a distinct seed rather
        # than silently duplicating a pair.
        raise RuntimeError(
            f'unique-pair shortfall: {len(first)}/{target_count}; '
            'use a new pre-scoring config seed'
        )
    first = first[:target_count]
    result: dict[str, np.ndarray | int | float] = {
        key: value[first] for key, value in merged.items()
    }
    result['pair_id'] = pair_key[first]
    result['label'] = np.full(target_count, label, dtype=np.uint8)
    result['proposals'] = proposals
    result['acceptance_rate'] = accepted_raw / proposals
    covered = np.unique(np.asarray(result['anchor_group']))
    expected = np.unique(groups)
    if not np.array_equal(covered, expected):
        missing = np.setdiff1d(expected, covered)
        raise RuntimeError(
            f'anchor-group coverage shortfall: {missing.tolist()}'
        )
    return result


def sample_stratified_partition(
    *,
    partition_rows: np.ndarray,
    all_states: np.ndarray,
    all_groups: np.ndarray,
    contract: TaskContract,
    total_count: int,
    seed: int,
    total_dataset_rows: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float | int]]]:
    """Sample 50% T and equal thirds of three pre-registered F strata."""
    positive = total_count // 2
    negative = total_count - positive
    negative_counts = [negative // 3] * 3
    negative_counts[-1] += negative - sum(negative_counts)
    edge_1, edge_2 = contract.negative_edges
    definitions = [
        ('positive', None, contract.positive_if_lt, LABEL_T, positive),
        (
            'boundary_outside',
            contract.negative_if_gt,
            edge_1,
            LABEL_F,
            negative_counts[0],
        ),
        ('medium', edge_1, edge_2, LABEL_F, negative_counts[1]),
        ('far', edge_2, None, LABEL_F, negative_counts[2]),
    ]
    outputs = []
    audit = {}
    for index, (name, lower, upper, label, count) in enumerate(definitions):
        sampled = sample_task_stratum(
            partition_rows=partition_rows,
            all_states=all_states,
            all_groups=all_groups,
            contract=contract,
            target_count=count,
            lower_exclusive=lower,
            upper_inclusive=upper,
            label=label,
            seed=seed + 104729 * (index + 1),
            total_dataset_rows=total_dataset_rows,
        )
        sampled['negative_stratum'] = np.full(
            count, index - 1 if index else -1, dtype=np.int8
        )
        audit[name] = {
            'target': count,
            'realized': count,
            'proposals': int(sampled.pop('proposals')),
            'acceptance_rate': float(sampled.pop('acceptance_rate')),
        }
        outputs.append(sampled)
    keys = outputs[0].keys()
    merged = {
        key: np.concatenate([part[key] for part in outputs]) for key in keys
    }
    order = np.argsort(merged['pair_id'], kind='stable')
    merged = {key: value[order] for key, value in merged.items()}
    if len(np.unique(merged['pair_id'])) != total_count:
        raise RuntimeError('duplicate pair across stratified strata')
    return merged, audit
