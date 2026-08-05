from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.plan.clear_protocol import (
    install_success_criterion,
    load_manifest,
    topology_audit_records,
)
from scripts.plan.clear_tworoom import (
    TwoRoomGeometry,
    check_route_segment,
    point_is_clear,
    resolve_motion,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def geometry(door_half_extent: float = 14.0) -> TwoRoomGeometry:
    return TwoRoomGeometry(
        image_size=224.0,
        border_size=14.0,
        wall_center=112.0,
        wall_axis=1,
        wall_thickness=10.0,
        agent_radius=7.0,
        doors=((49.0, door_half_extent),),
    )


def _world(env):
    wrapped = SimpleNamespace(unwrapped=env)
    return SimpleNamespace(envs=SimpleNamespace(envs=[wrapped]))


def test_solid_wall_crossing_is_blocked():
    result = resolve_motion(
        geometry(), np.array([90.0, 100.0]), np.array([130.0, 100.0])
    )
    assert result.blocked
    assert result.position[0] < 112.0
    assert check_route_segment(
        geometry(), np.array([90.0, 100.0]), np.asarray(result.position)
    ).valid


def test_wide_door_crossing_is_allowed():
    result = resolve_motion(
        geometry(), np.array([90.0, 49.0]), np.array([130.0, 49.0])
    )
    assert not result.blocked
    assert result.position[0] == 130.0
    assert check_route_segment(
        geometry(), np.array([90.0, 49.0]), np.asarray(result.position)
    ).valid


def test_visible_door_edge_without_radius_clearance_is_blocked():
    start = np.array([90.0, 60.0])
    desired = np.array([130.0, 60.0])
    assert not check_route_segment(geometry(), start, desired).valid
    result = resolve_motion(geometry(), start, desired)
    assert result.blocked
    assert result.position[0] < 112.0


def test_narrow_door_cannot_pass_the_agent_disk():
    result = resolve_motion(
        geometry(door_half_extent=6.0),
        np.array([90.0, 49.0]),
        np.array([130.0, 49.0]),
    )
    assert result.blocked
    assert result.position[0] < 112.0


def test_diagonal_segment_cannot_tunnel_through_corner():
    start = np.array([90.0, 70.0])
    desired = np.array([130.0, 55.0])
    result = resolve_motion(geometry(), start, desired)
    assert result.blocked
    assert check_route_segment(
        geometry(), start, np.asarray(result.position)
    ).valid


def test_start_overlapping_doorframe_is_not_route_valid():
    start = np.array([107.9, 61.0])
    assert not point_is_clear(geometry(), start)
    check = check_route_segment(
        geometry(), start, start + np.array([0.1, 0.0])
    )
    assert not check.valid
    assert not check.start_clear


def test_strict_adapter_requires_a_legal_crossing_and_goal_side():
    pytest.importorskip('gymnasium')
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    manifest_path = (
        REPO_ROOT
        / 'results/clear_eval/v0.5/manifests/tworoom'
        / 'strict-seed42-n100.json'
    )
    env = TwoRoomEnv()
    env.reset(seed=0)
    install_success_criterion(_world(env), load_manifest(manifest_path))
    env._set_state(np.array([90.0, 49.0], dtype=np.float32))
    env._set_goal_state(np.array([130.0, 49.0], dtype=np.float32))

    outcomes = [
        env.step(np.array([1.0, 0.0], dtype=np.float32))[2]
        for _ in range(7)
    ]
    assert outcomes == [False] * 6 + [True]
    audit = topology_audit_records()[0]
    assert audit['route_valid']
    assert audit['valid_room_crossings'] == 1
    assert audit['goal_side_reached']
