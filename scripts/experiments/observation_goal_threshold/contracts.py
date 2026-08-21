"""Pure task-space label contracts for find-goal-threshold.

This module intentionally imports neither torch nor any world-model code.  Pair
IDs, compatibility, task errors, and T/F/U labels must be materialized before
latent scoring and cannot depend on pixels or embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

import numpy as np

LABEL_T = np.uint8(0)
LABEL_F = np.uint8(1)
LABEL_U = np.uint8(2)

LABEL_NAMES = {int(LABEL_T): 'T', int(LABEL_F): 'F', int(LABEL_U): 'U'}


@dataclass(frozen=True)
class TaskContract:
    task: str
    variant: str
    state_fields: tuple[str, ...]
    positive_if_lt: float
    negative_if_gt: float
    unit: str
    negative_edges: tuple[float, float]

    @classmethod
    def from_config(cls, config: dict) -> TaskContract:
        label = config['task_label']
        return cls(
            task=config['task'],
            variant=label['variant'],
            state_fields=tuple(label['state_fields']),
            positive_if_lt=float(label['positive_if_lt']),
            negative_if_gt=float(label['negative_if_gt']),
            unit=label['unit'],
            negative_edges=tuple(float(v) for v in label['negative_edges']),
        )

    def task_error(self, anchor: np.ndarray, goal: np.ndarray) -> np.ndarray:
        anchor = np.asarray(anchor)
        goal = np.asarray(goal)
        if anchor.shape != goal.shape:
            raise ValueError('anchor and goal state shapes must match')
        if anchor.ndim != 2:
            raise ValueError('state arrays must have shape (N, D)')
        if not np.isfinite(anchor).all() or not np.isfinite(goal).all():
            raise ValueError('task state contains a non-finite value')
        return np.linalg.norm(anchor - goal, axis=1).astype(np.float32)

    def classify(self, task_error: np.ndarray) -> np.ndarray:
        error = np.asarray(task_error)
        if not np.isfinite(error).all():
            raise ValueError('task error contains a non-finite value')
        labels = np.full(error.shape, LABEL_U, dtype=np.uint8)
        labels[error < self.positive_if_lt] = LABEL_T
        labels[error > self.negative_if_gt] = LABEL_F
        return labels

    def negative_strata(self, task_error: np.ndarray) -> np.ndarray:
        """Return -1 outside F, otherwise boundary/medium/far = 0/1/2."""
        error = np.asarray(task_error)
        edge_1, edge_2 = self.negative_edges
        if not self.negative_if_gt < edge_1 < edge_2:
            raise ValueError('negative edges must be above the F boundary')
        result = np.full(error.shape, -1, dtype=np.int8)
        result[(error > self.negative_if_gt) & (error <= edge_1)] = 0
        result[(error > edge_1) & (error <= edge_2)] = 1
        result[error > edge_2] = 2
        return result


def _rotation_matrix_to_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert one proper 3x3 rotation matrix to a normalized wxyz quat."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        quat = np.array(
            [
                0.25 * s,
                (m[2, 1] - m[1, 2]) / s,
                (m[0, 2] - m[2, 0]) / s,
                (m[1, 0] - m[0, 1]) / s,
            ]
        )
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            quat = np.array(
                [
                    (m[2, 1] - m[1, 2]) / s,
                    0.25 * s,
                    (m[0, 1] + m[1, 0]) / s,
                    (m[0, 2] + m[2, 0]) / s,
                ]
            )
        elif i == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            quat = np.array(
                [
                    (m[0, 2] - m[2, 0]) / s,
                    (m[0, 1] + m[1, 0]) / s,
                    0.25 * s,
                    (m[1, 2] + m[2, 1]) / s,
                ]
            )
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            quat = np.array(
                [
                    (m[1, 0] - m[0, 1]) / s,
                    (m[0, 2] + m[2, 0]) / s,
                    (m[1, 2] + m[2, 1]) / s,
                    0.25 * s,
                ]
            )
    return quat / np.linalg.norm(quat)


def cube_symmetry_quaternions() -> np.ndarray:
    """Return the 24 proper rotational symmetries of a cube as wxyz quats."""
    rotations = []
    for perm in permutations(range(3)):
        for signs in product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float64)
            for row, col in enumerate(perm):
                matrix[row, col] = signs[row]
            if np.linalg.det(matrix) > 0.5:
                rotations.append(_rotation_matrix_to_wxyz(matrix))
    result = np.stack(rotations)
    if result.shape != (24, 4):
        raise AssertionError('cube symmetry construction did not yield 24')
    return result


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def cube_symmetry_angle_deg(
    anchor_wxyz: np.ndarray,
    goal_wxyz: np.ndarray,
    *,
    convention: str = 'wxyz',
) -> np.ndarray:
    """Minimum geodesic angle under all 24 proper cube symmetries."""
    if convention != 'wxyz':
        raise ValueError('Cube pose calibration requires wxyz quaternions')
    anchor = np.asarray(anchor_wxyz, dtype=np.float64)
    goal = np.asarray(goal_wxyz, dtype=np.float64)
    if anchor.shape != goal.shape or anchor.ndim != 2 or anchor.shape[1] != 4:
        raise ValueError('quaternions must both have shape (N, 4)')
    if not np.isfinite(anchor).all() or not np.isfinite(goal).all():
        raise ValueError('quaternions must be finite')
    an = np.linalg.norm(anchor, axis=1, keepdims=True)
    gn = np.linalg.norm(goal, axis=1, keepdims=True)
    if np.any(an <= 1e-12) or np.any(gn <= 1e-12):
        raise ValueError('zero-norm quaternion')
    anchor = anchor / an
    goal = goal / gn
    sym = cube_symmetry_quaternions()
    goal_sym = _quat_multiply(goal[:, None, :], sym[None, :, :])
    dots = np.abs(np.einsum('nd,nsd->ns', anchor, goal_sym))
    best = np.clip(dots.max(axis=1), -1.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(best))


def classify_cube_pose(
    anchor_pos: np.ndarray,
    goal_pos: np.ndarray,
    anchor_wxyz: np.ndarray,
    goal_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return T/F/U labels, position error, and symmetry-aware angle."""
    pos_error = np.linalg.norm(
        np.asarray(anchor_pos) - np.asarray(goal_pos), axis=1
    )
    angle = cube_symmetry_angle_deg(anchor_wxyz, goal_wxyz)
    labels = np.full(pos_error.shape, LABEL_U, dtype=np.uint8)
    labels[(pos_error < 0.03) & (angle < 15.0)] = LABEL_T
    labels[(pos_error > 0.04) | (angle > 30.0)] = LABEL_F
    return labels, pos_error.astype(np.float32), angle.astype(np.float32)
