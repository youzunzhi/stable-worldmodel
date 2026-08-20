"""Group-first split construction for Experiment T."""

from __future__ import annotations

import hashlib

import numpy as np

PARTITIONS = ('threshold_fit', 'threshold_validation', 'threshold_audit')


def split_groups(
    groups: np.ndarray,
    seed: int,
    fractions: tuple[float, float, float] = (0.60, 0.20, 0.20),
) -> dict[str, np.ndarray]:
    unique = np.unique(np.asarray(groups, dtype=np.int64))
    if len(unique) < 3:
        raise ValueError('at least three independent groups are required')
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError('split fractions must sum to one')

    def rank(group: int) -> bytes:
        return hashlib.sha256(f'{seed}:{int(group)}'.encode()).digest()

    ordered = np.array(sorted(unique.tolist(), key=rank), dtype=np.int64)
    n_fit = round(len(ordered) * fractions[0])
    n_validation = round(len(ordered) * fractions[1])
    n_fit = min(max(n_fit, 1), len(ordered) - 2)
    n_validation = min(max(n_validation, 1), len(ordered) - n_fit - 1)
    return {
        PARTITIONS[0]: ordered[:n_fit],
        PARTITIONS[1]: ordered[n_fit : n_fit + n_validation],
        PARTITIONS[2]: ordered[n_fit + n_validation :],
    }


def row_partitions(
    row_groups: np.ndarray, group_split: dict[str, np.ndarray]
) -> np.ndarray:
    groups = np.asarray(row_groups, dtype=np.int64)
    result = np.full(len(groups), -1, dtype=np.int8)
    seen: set[int] = set()
    for index, name in enumerate(PARTITIONS):
        selected = {int(v) for v in group_split[name]}
        if seen.intersection(selected):
            raise ValueError('raw-group overlap across partitions')
        seen.update(selected)
        result[np.isin(groups, list(selected))] = index
    if np.any(result < 0):
        raise ValueError('some rows were not assigned to a partition')
    return result
