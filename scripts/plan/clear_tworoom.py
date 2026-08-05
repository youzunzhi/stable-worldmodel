"""CLEAR-LeWM v0.5 swept-disk physics and route audit for TwoRoom.

Derived from DavidSunok/CLEAR-LeWM at revision
df026185a36bd9997c69d94753854db0b1a46f54. CLEAR-LeWM is MIT licensed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import sqrt
from types import MethodType

import numpy as np


EPS = 1e-7
CONTACT_EPS = 1e-4


@dataclass(frozen=True)
class Rectangle:
    xmin: float
    xmax: float
    ymin: float
    ymax: float


@dataclass(frozen=True)
class SweepHit:
    time: float
    normal_x: float
    normal_y: float
    obstacle: int

    @property
    def normal(self) -> np.ndarray:
        return np.asarray([self.normal_x, self.normal_y], dtype=np.float64)


@dataclass(frozen=True)
class MotionResult:
    position: tuple[float, float]
    collisions: int
    blocked: bool


@dataclass(frozen=True)
class RouteCheck:
    valid: bool
    wall_collision: bool
    illegal_crossing: bool
    start_clear: bool
    end_clear: bool


@dataclass(frozen=True)
class TwoRoomGeometry:
    image_size: float
    border_size: float
    wall_center: float
    wall_axis: int
    wall_thickness: float
    agent_radius: float
    doors: tuple[tuple[float, float], ...]

    @classmethod
    def from_env(cls, env) -> TwoRoomGeometry:
        radius = float(env.variation_space['agent']['radius'].value.item())
        doors = tuple(
            (float(env.door_positions[index]), float(env.door_sizes[index]))
            for index in range(int(env.num_doors))
        )
        return cls(
            image_size=float(env.IMG_SIZE),
            border_size=float(env.BORDER_SIZE),
            wall_center=float(env.WALL_CENTER),
            wall_axis=int(env.wall_axis),
            wall_thickness=float(env.wall_thickness),
            agent_radius=radius,
            doors=doors,
        )

    @property
    def normal_axis(self) -> int:
        return 0 if self.wall_axis == 1 else 1

    @property
    def tangent_axis(self) -> int:
        return 1 - self.normal_axis

    @property
    def wall_bounds(self) -> tuple[float, float]:
        half = float(int(self.wall_thickness) // 2)
        return self.wall_center - half, self.wall_center + half

    @property
    def center_bounds(self) -> tuple[float, float]:
        low = self.border_size + self.agent_radius
        high = self.image_size - self.border_size - self.agent_radius
        return low, high

    def to_dict(self) -> dict:
        return asdict(self)

    def merged_door_intervals(self) -> tuple[tuple[float, float], ...]:
        intervals = sorted(
            (
                max(0.0, center - half_extent),
                min(self.image_size, center + half_extent),
            )
            for center, half_extent in self.doors
        )
        merged: list[list[float]] = []
        for low, high in intervals:
            if high <= low:
                continue
            if merged and low <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], high)
            else:
                merged.append([low, high])
        return tuple((low, high) for low, high in merged)

    def clear_door_intervals(self) -> tuple[tuple[float, float], ...]:
        radius = self.agent_radius
        return tuple(
            (low + radius, high - radius)
            for low, high in self.merged_door_intervals()
            if high - low >= 2.0 * radius
        )

    def solid_wall_rectangles(self) -> tuple[Rectangle, ...]:
        wall_low, wall_high = self.wall_bounds
        cursor = 0.0
        tangent_segments: list[tuple[float, float]] = []
        for door_low, door_high in self.merged_door_intervals():
            if door_low > cursor:
                tangent_segments.append((cursor, door_low))
            cursor = max(cursor, door_high)
        if cursor < self.image_size:
            tangent_segments.append((cursor, self.image_size))
        if self.normal_axis == 0:
            return tuple(
                Rectangle(wall_low, wall_high, low, high)
                for low, high in tangent_segments
            )
        return tuple(
            Rectangle(low, high, wall_low, wall_high)
            for low, high in tangent_segments
        )

    def clamp_to_borders(self, point: np.ndarray) -> np.ndarray:
        low, high = self.center_bounds
        return np.clip(np.asarray(point, dtype=np.float64), low, high)

    def room_side(self, point: np.ndarray) -> int:
        value = float(np.asarray(point)[self.normal_axis] - self.wall_center)
        return -1 if value < 0 else 1 if value > 0 else 0

    def is_cross_room(self, start: np.ndarray, goal: np.ndarray) -> bool:
        return self.room_side(start) * self.room_side(goal) < 0

    def crossing_coordinate(
        self, start: np.ndarray, end: np.ndarray
    ) -> float | None:
        axis = self.normal_axis
        delta = float(end[axis] - start[axis])
        if abs(delta) <= EPS:
            return None
        time = (self.wall_center - float(start[axis])) / delta
        if not EPS < time <= 1.0 + EPS:
            return None
        tangent = self.tangent_axis
        return float(start[tangent] + time * (end[tangent] - start[tangent]))

    def crossing_has_full_clearance(self, coordinate: float) -> bool:
        return any(
            low - EPS <= coordinate <= high + EPS
            for low, high in self.clear_door_intervals()
        )


def _aabb_entry(
    start: np.ndarray, delta: np.ndarray, rectangle: Rectangle
) -> tuple[float, np.ndarray] | None:
    lower = np.asarray([rectangle.xmin, rectangle.ymin], dtype=np.float64)
    upper = np.asarray([rectangle.xmax, rectangle.ymax], dtype=np.float64)
    entry = -np.inf
    exit_time = np.inf
    normal = np.zeros(2, dtype=np.float64)
    for axis in range(2):
        if abs(delta[axis]) <= EPS:
            if start[axis] < lower[axis] or start[axis] > upper[axis]:
                return None
            continue
        first = (lower[axis] - start[axis]) / delta[axis]
        second = (upper[axis] - start[axis]) / delta[axis]
        first_normal = np.zeros(2, dtype=np.float64)
        first_normal[axis] = -1.0 if delta[axis] > 0 else 1.0
        if first > second:
            first, second = second, first
        if first > entry:
            entry = first
            normal = first_normal
        exit_time = min(exit_time, second)
        if entry > exit_time:
            return None
    if exit_time < -EPS or entry > 1.0 + EPS:
        return None
    time = max(0.0, float(entry))
    if time <= EPS and float(np.dot(delta, normal)) >= -EPS:
        return None
    return time, normal


def _circle_entry(
    start: np.ndarray,
    delta: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> tuple[float, np.ndarray] | None:
    offset = start - center
    a = float(np.dot(delta, delta))
    if a <= EPS:
        return None
    b = 2.0 * float(np.dot(offset, delta))
    c = float(np.dot(offset, offset) - radius * radius)
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None
    root = sqrt(max(0.0, discriminant))
    for time in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)):
        if -EPS <= time <= 1.0 + EPS:
            time = max(0.0, float(time))
            point = start + time * delta
            normal = point - center
            norm = float(np.linalg.norm(normal))
            if norm <= EPS:
                continue
            normal /= norm
            if time <= EPS and float(np.dot(delta, normal)) >= -EPS:
                continue
            return time, normal
    return None


def _sweep_against_rectangle(
    start: np.ndarray,
    delta: np.ndarray,
    rectangle: Rectangle,
    radius: float,
) -> tuple[float, np.ndarray] | None:
    candidates: list[tuple[float, np.ndarray]] = []
    expanded_rectangles = (
        Rectangle(
            rectangle.xmin - radius,
            rectangle.xmax + radius,
            rectangle.ymin,
            rectangle.ymax,
        ),
        Rectangle(
            rectangle.xmin,
            rectangle.xmax,
            rectangle.ymin - radius,
            rectangle.ymax + radius,
        ),
    )
    for expanded in expanded_rectangles:
        hit = _aabb_entry(start, delta, expanded)
        if hit is not None:
            candidates.append(hit)
    for x in (rectangle.xmin, rectangle.xmax):
        for y in (rectangle.ymin, rectangle.ymax):
            hit = _circle_entry(
                start,
                delta,
                np.asarray([x, y], dtype=np.float64),
                radius,
            )
            if hit is not None:
                candidates.append(hit)
    return min(candidates, key=lambda item: item[0]) if candidates else None


def first_wall_hit(
    geometry: TwoRoomGeometry, start: np.ndarray, end: np.ndarray
) -> SweepHit | None:
    start = np.asarray(start, dtype=np.float64)
    delta = np.asarray(end, dtype=np.float64) - start
    best: SweepHit | None = None
    for index, rectangle in enumerate(geometry.solid_wall_rectangles()):
        result = _sweep_against_rectangle(
            start, delta, rectangle, geometry.agent_radius
        )
        if result is None:
            continue
        time, normal = result
        hit = SweepHit(float(time), float(normal[0]), float(normal[1]), index)
        if best is None or hit.time < best.time:
            best = hit
    return best


def point_is_clear(geometry: TwoRoomGeometry, point: np.ndarray) -> bool:
    """Return whether the complete agent disk is outside every wall solid."""
    point = np.asarray(point, dtype=np.float64)
    low, high = geometry.center_bounds
    if np.any(point < low - EPS) or np.any(point > high + EPS):
        return False
    for rectangle in geometry.solid_wall_rectangles():
        closest = np.asarray(
            [
                np.clip(point[0], rectangle.xmin, rectangle.xmax),
                np.clip(point[1], rectangle.ymin, rectangle.ymax),
            ]
        )
        distance = float(np.linalg.norm(point - closest))
        if distance < geometry.agent_radius - 1e-6:
            return False
    return True


def resolve_motion(
    geometry: TwoRoomGeometry,
    start: np.ndarray,
    desired_end: np.ndarray,
    max_contacts: int = 4,
) -> MotionResult:
    """Move a disk continuously, sliding along a wall after contact."""
    current = geometry.clamp_to_borders(np.asarray(start, dtype=np.float64))
    target = geometry.clamp_to_borders(
        np.asarray(desired_end, dtype=np.float64)
    )
    remaining = target - current
    collisions = 0
    for _ in range(max_contacts):
        if float(np.linalg.norm(remaining)) <= EPS:
            break
        hit = first_wall_hit(geometry, current, current + remaining)
        if hit is None:
            current = current + remaining
            remaining = np.zeros(2, dtype=np.float64)
            break
        collisions += 1
        length = max(float(np.linalg.norm(remaining)), EPS)
        safe_time = max(0.0, hit.time - CONTACT_EPS / length)
        current = current + safe_time * remaining
        remainder = (1.0 - hit.time) * remaining
        normal = hit.normal
        inward = min(0.0, float(np.dot(remainder, normal)))
        remaining = remainder - inward * normal
        current = current + CONTACT_EPS * normal
    current = geometry.clamp_to_borders(current)
    return MotionResult(
        position=(float(current[0]), float(current[1])),
        collisions=collisions,
        blocked=collisions > 0,
    )


def check_route_segment(
    geometry: TwoRoomGeometry, start: np.ndarray, end: np.ndarray
) -> RouteCheck:
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    hit = first_wall_hit(geometry, start, end)
    wall_collision = hit is not None and hit.time < 1.0 - 1e-6
    start_clear = point_is_clear(geometry, start)
    end_clear = point_is_clear(geometry, end)
    coordinate = geometry.crossing_coordinate(start, end)
    illegal_crossing = (
        coordinate is not None
        and not geometry.crossing_has_full_clearance(coordinate)
    )
    return RouteCheck(
        valid=(
            start_clear
            and end_clear
            and not wall_collision
            and not illegal_crossing
        ),
        wall_collision=wall_collision or not start_clear or not end_clear,
        illegal_crossing=illegal_crossing,
        start_clear=start_clear,
        end_clear=end_clear,
    )


@dataclass
class EpisodeAudit:
    env_index: int
    route_valid: bool = True
    collision_contacts: int = 0
    blocked_steps: int = 0
    valid_room_crossings: int = 0
    hold_count: int = 0
    initial_state_clear: bool = True
    goal_state_clear: bool = True
    cross_room_goal: bool = False
    goal_side_reached: bool = False
    positions: list[list[float]] = field(default_factory=list)
    goal_position: list[float] | None = None
    geometry: dict = field(default_factory=dict)

    def reset(self, position, geometry: TwoRoomGeometry) -> None:
        self.route_valid = point_is_clear(geometry, position)
        self.collision_contacts = 0
        self.blocked_steps = 0
        self.valid_room_crossings = 0
        self.hold_count = 0
        self.initial_state_clear = self.route_valid
        self.goal_state_clear = True
        self.cross_room_goal = False
        self.goal_side_reached = False
        self.positions = [np.asarray(position, dtype=np.float64).tolist()]
        self.goal_position = None
        self.geometry = geometry.to_dict()

    def to_dict(self) -> dict:
        return {
            'env_index': self.env_index,
            'route_valid': self.route_valid,
            'collision_contacts': self.collision_contacts,
            'blocked_steps': self.blocked_steps,
            'valid_room_crossings': self.valid_room_crossings,
            'hold_count': self.hold_count,
            'initial_state_clear': self.initial_state_clear,
            'goal_state_clear': self.goal_state_clear,
            'cross_room_goal': self.cross_room_goal,
            'goal_side_reached': self.goal_side_reached,
            'positions': self.positions,
            'goal_position': self.goal_position,
            'geometry': self.geometry,
        }


ACTIVE_AUDITS: list[EpisodeAudit] = []


def install_tworoom_success(world, protocol: dict) -> None:
    """Install swept-disk collision and protocol-selected TwoRoom success."""
    if protocol['tworoom_collision_mode'] != 'swept':
        raise ValueError(
            'CLEAR TwoRoom currently supports only swept collision'
        )

    task_hold_steps = protocol.get('tworoom_sustained_steps')
    hold_steps = int(
        task_hold_steps
        if task_hold_steps is not None
        else protocol.get('sustained_steps', 1)
    )
    ACTIVE_AUDITS.clear()
    for env_index, wrapped in enumerate(world.envs.envs):
        env = wrapped.unwrapped
        audit = EpisodeAudit(env_index=env_index)
        ACTIVE_AUDITS.append(audit)
        original_set_state = env._set_state
        original_set_goal = env._set_goal_state
        original_step = env.step

        def apply_collisions(self, position, desired, *, _audit=audit):
            geometry = TwoRoomGeometry.from_env(self)
            motion = resolve_motion(geometry, position, desired)
            _audit.collision_contacts += motion.collisions
            _audit.blocked_steps += int(motion.blocked)
            import torch

            return torch.as_tensor(
                motion.position,
                dtype=getattr(position, 'dtype', torch.float32),
                device=getattr(position, 'device', None),
            )

        def set_state(
            self, state, *, _audit=audit, _original=original_set_state
        ):
            result = _original(state)
            _audit.reset(self.agent_position, TwoRoomGeometry.from_env(self))
            self._clear_lewm_route_valid = _audit.route_valid
            self._clear_lewm_hold_count = 0
            return result

        def set_goal_state(
            self, goal_state, *, _audit=audit, _original=original_set_goal
        ):
            result = _original(goal_state)
            geometry = TwoRoomGeometry.from_env(self)
            _audit.goal_position = np.asarray(
                goal_state, dtype=np.float64
            ).tolist()
            _audit.goal_state_clear = point_is_clear(geometry, goal_state)
            _audit.cross_room_goal = geometry.is_cross_room(
                self.agent_position, goal_state
            )
            _audit.goal_side_reached = False
            _audit.route_valid &= _audit.goal_state_clear
            _audit.hold_count = 0
            self._clear_lewm_route_valid = _audit.route_valid
            self._clear_lewm_hold_count = 0
            return result

        def step(
            self, action, *, _audit=audit, _original=original_step
        ):
            before = np.asarray(self.agent_position, dtype=np.float64).copy()
            observation, reward, _, truncated, info = _original(action)
            after = np.asarray(self.agent_position, dtype=np.float64).copy()
            geometry = TwoRoomGeometry.from_env(self)
            route_check = check_route_segment(geometry, before, after)
            coordinate = geometry.crossing_coordinate(before, after)
            if (
                coordinate is not None
                and geometry.crossing_has_full_clearance(coordinate)
            ):
                _audit.valid_room_crossings += 1
            _audit.route_valid &= route_check.valid
            _audit.positions.append(after.tolist())
            self._clear_lewm_route_valid = _audit.route_valid

            distance = float(
                np.linalg.norm(
                    np.asarray(self.agent_position)
                    - np.asarray(self.target_position)
                )
            )
            crossing_ok = (
                not _audit.cross_room_goal
                or _audit.valid_room_crossings > 0
            )
            _audit.goal_side_reached = geometry.room_side(
                self.agent_position
            ) == geometry.room_side(self.target_position)
            success = distance < protocol['tworoom_distance_threshold']
            if protocol['tworoom_route_required']:
                success = bool(
                    success and _audit.route_valid and crossing_ok
                )
            if protocol.get('tworoom_goal_side_required', False):
                success = bool(success and _audit.goal_side_reached)
            _audit.hold_count = _audit.hold_count + 1 if success else 0
            self._clear_lewm_hold_count = _audit.hold_count
            terminated = _audit.hold_count >= hold_steps
            info['clear_lewm_route_valid'] = _audit.route_valid
            info['clear_lewm_valid_room_crossings'] = (
                _audit.valid_room_crossings
            )
            info['clear_lewm_hold_count'] = _audit.hold_count
            return observation, reward, terminated, truncated, info

        env._apply_collisions = MethodType(apply_collisions, env)
        env._set_state = MethodType(set_state, env)
        env._set_goal_state = MethodType(set_goal_state, env)
        env.step = MethodType(step, env)


def topology_audit_records() -> list[dict]:
    return [audit.to_dict() for audit in ACTIVE_AUDITS]
