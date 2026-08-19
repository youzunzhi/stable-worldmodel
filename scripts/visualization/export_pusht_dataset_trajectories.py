#!/usr/bin/env python3
"""Audit the canonical LeWM PushT dataset and render annotated trajectories.

The dataset has a non-obvious 185 x 101 structure: 185 released source
demonstrations, each followed by 100 noisy replays.  This exporter makes that
structure explicit instead of treating the misleading ``expert_train`` file
name as a data-quality guarantee.

Example:
    python scripts/visualization/export_pusht_dataset_trajectories.py \
        --dataset /path/to/pusht_expert_train.h5 \
        --output-dir outputs/analysis/pusht_training_data_260814
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401  # registers compressed HDF5 filters
import numpy as np

PUSHT_REVISION = "655cd446b9929369d7d406001da85c15d1457850"
CANONICAL_METADATA_SHA256 = (
    "8dabf107816dce322a4afb3f2c80bf2981e399b00bb37c52aac1fd068fa3dd00"
)
VARIANTS_PER_SOURCE = 101
CONTROL_HZ = 10
ACTION_SCALE_PX = 100.0
LEGACY_ANCHOR_POS = np.asarray([256.0, 256.0])
LEGACY_ANCHOR_ANGLE = np.pi / 4


def _angle_delta(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    """Return signed wrapped angle differences in [-pi, pi]."""
    return np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b))))


def _finite_float(value: Any) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _quantiles(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return {
        "min": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def summarize_episode(
    episode_idx: int,
    action: np.ndarray,
    state: np.ndarray,
    clean_action: np.ndarray,
    clean_state: np.ndarray,
) -> dict[str, Any]:
    """Reduce one numeric episode to metrics used for quality assessment."""
    agent = state[:, :2].astype(float)
    block = state[:, 2:4].astype(float)
    block_angle = state[:, 4].astype(float)
    velocity = state[:, 5:7].astype(float)
    action = action.astype(float)
    valid_action = np.isfinite(action).all(axis=1)

    agent_step = np.linalg.norm(np.diff(agent, axis=0), axis=1)
    block_step = np.linalg.norm(np.diff(block, axis=0), axis=1)
    block_angle_step = np.abs(_angle_delta(block_angle[1:], block_angle[:-1]))
    action_tv = np.linalg.norm(np.diff(action, axis=0), axis=1)
    valid_action_pair = valid_action[1:] & valid_action[:-1]

    command = agent + action * ACTION_SCALE_PX
    clean_command = clean_state[:, :2] + clean_action * ACTION_SCALE_PX
    command_deviation = np.linalg.norm(command - clean_command, axis=1)
    final_state_deviation = np.linalg.norm(
        np.concatenate(
            [
                state[-1, :4] - clean_state[-1, :4],
                [_angle_delta(state[-1, 4], clean_state[-1, 4])],
            ]
        )
    )

    anchor_position_error = np.linalg.norm(block - LEGACY_ANCHOR_POS, axis=1)
    anchor_angle_error = np.abs(_angle_delta(block_angle, LEGACY_ANCHOR_ANGLE))
    final_position_error = float(anchor_position_error[-1])
    final_angle_error = float(anchor_angle_error[-1])
    proximity = np.linalg.norm(agent - block, axis=1)

    return {
        "episode_idx": episode_idx,
        "source_demo_idx": episode_idx // VARIANTS_PER_SOURCE,
        "augmentation_variant_idx": episode_idx % VARIANTS_PER_SOURCE,
        "rows": len(state),
        "duration_seconds": float((len(state) - 1) / CONTROL_HZ),
        "valid_action_rows": int(valid_action.sum()),
        "invalid_action_rows": int((~valid_action).sum()),
        "initial_agent_pos": agent[0].tolist(),
        "initial_block_pose": [*block[0].tolist(), float(block_angle[0])],
        "final_agent_pos": agent[-1].tolist(),
        "final_block_pose": [*block[-1].tolist(), float(block_angle[-1])],
        "agent_path_length_px": float(agent_step.sum()),
        "block_path_length_px": float(block_step.sum()),
        "block_net_displacement_px": float(np.linalg.norm(block[-1] - block[0])),
        "block_rotation_path_deg": float(np.degrees(block_angle_step.sum())),
        "block_net_rotation_deg": float(
            np.degrees(abs(_angle_delta(block_angle[-1], block_angle[0])))
        ),
        "block_moving_frame_fraction": float(np.mean(block_step > 0.5)),
        "minimum_agent_block_center_distance_px": float(proximity.min()),
        "agent_velocity_norm_mean": float(np.linalg.norm(velocity, axis=1).mean()),
        "action_norm_mean": float(np.linalg.norm(action[valid_action], axis=1).mean()),
        "action_total_variation": float(action_tv[valid_action_pair].sum()),
        "action_saturation_fraction": float(
            np.mean(np.abs(action[valid_action]) >= 0.95)
        ),
        "command_out_of_bounds_fraction": float(
            np.mean((command < 0).any(axis=1) | (command > 512).any(axis=1))
        ),
        "command_deviation_from_clean_rms_px": float(
            np.sqrt(np.mean(np.square(command_deviation)))
        ),
        "final_state_deviation_from_clean": float(final_state_deviation),
        "final_legacy_anchor_position_error_px": final_position_error,
        "final_legacy_anchor_angle_error_deg": float(
            np.degrees(final_angle_error)
        ),
        "final_legacy_anchor_block_within_20px_20deg": bool(
            final_position_error < 20 and final_angle_error < np.deg2rad(20)
        ),
        "final_legacy_anchor_block_within_10px_10deg": bool(
            final_position_error < 10 and final_angle_error < np.deg2rad(10)
        ),
    }


def scan_metrics(h5: h5py.File) -> list[dict[str, Any]]:
    offsets = h5["ep_offset"][:].astype(int)
    lengths = h5["ep_len"][:].astype(int)
    metrics: list[dict[str, Any]] = []
    for group_start in range(0, len(lengths), VARIANTS_PER_SOURCE):
        clean_start = int(offsets[group_start])
        clean_end = clean_start + int(lengths[group_start])
        clean_action = h5["action"][clean_start:clean_end]
        clean_state = h5["state"][clean_start:clean_end]
        for episode_idx in range(
            group_start, min(group_start + VARIANTS_PER_SOURCE, len(lengths))
        ):
            start = int(offsets[episode_idx])
            end = start + int(lengths[episode_idx])
            if end - start != clean_end - clean_start:
                raise ValueError(
                    f"group {group_start // VARIANTS_PER_SOURCE} has unequal lengths"
                )
            metrics.append(
                summarize_episode(
                    episode_idx,
                    h5["action"][start:end],
                    h5["state"][start:end],
                    clean_action,
                    clean_state,
                )
            )
        if len(metrics) % 2020 == 0:
            print(f"Scanned {len(metrics):,} episodes", flush=True)
    return metrics


def aggregate_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = Counter(m["rows"] for m in metrics)
    clean = [m for m in metrics if m["augmentation_variant_idx"] == 0]
    noisy = [m for m in metrics if m["augmentation_variant_idx"] > 0]
    fields = (
        "rows",
        "agent_path_length_px",
        "block_path_length_px",
        "block_net_displacement_px",
        "block_rotation_path_deg",
        "block_moving_frame_fraction",
        "minimum_agent_block_center_distance_px",
        "action_norm_mean",
        "action_total_variation",
        "action_saturation_fraction",
        "command_out_of_bounds_fraction",
        "command_deviation_from_clean_rms_px",
        "final_legacy_anchor_position_error_px",
        "final_legacy_anchor_angle_error_deg",
    )

    def summarize_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "episodes": len(rows),
            "legacy_anchor_block_20px_20deg_fraction": float(
                np.mean(
                    [
                        m["final_legacy_anchor_block_within_20px_20deg"]
                        for m in rows
                    ]
                )
            ),
            "legacy_anchor_block_10px_10deg_fraction": float(
                np.mean(
                    [
                        m["final_legacy_anchor_block_within_10px_10deg"]
                        for m in rows
                    ]
                )
            ),
            "quantiles": {
                field: _quantiles([m[field] for m in rows]) for field in fields
            },
        }

    return {
        "episodes": len(metrics),
        "rows": int(sum(m["rows"] for m in metrics)),
        "episode_length_counts": {str(k): v for k, v in sorted(lengths.items())},
        "valid_action_rows": int(sum(m["valid_action_rows"] for m in metrics)),
        "invalid_action_rows": int(
            sum(m["invalid_action_rows"] for m in metrics)
        ),
        "source_demonstrations": len(clean),
        "noisy_replays": len(noisy),
        "all": summarize_subset(metrics),
        "clean_source_variant_0": summarize_subset(clean),
        "noisy_variants_1_to_100": summarize_subset(noisy),
    }


def augmentation_profile(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    profile = []
    for variant_idx in range(VARIANTS_PER_SOURCE):
        rows = [
            m for m in metrics if m["augmentation_variant_idx"] == variant_idx
        ]
        profile.append(
            {
                "augmentation_variant_idx": variant_idx,
                "episodes": len(rows),
                "command_deviation_from_clean_rms_px": _quantiles(
                    [m["command_deviation_from_clean_rms_px"] for m in rows]
                ),
                "final_legacy_anchor_position_error_px": _quantiles(
                    [m["final_legacy_anchor_position_error_px"] for m in rows]
                ),
                "final_legacy_anchor_angle_error_deg": _quantiles(
                    [m["final_legacy_anchor_angle_error_deg"] for m in rows]
                ),
                "legacy_anchor_block_20px_20deg_fraction": float(
                    np.mean(
                        [
                            m["final_legacy_anchor_block_within_20px_20deg"]
                            for m in rows
                        ]
                    )
                ),
                "legacy_anchor_block_10px_10deg_fraction": float(
                    np.mean(
                        [
                            m["final_legacy_anchor_block_within_10px_10deg"]
                            for m in rows
                        ]
                    )
                ),
                "block_path_length_px": _quantiles(
                    [m["block_path_length_px"] for m in rows]
                ),
            }
        )
    return profile


def augmentation_tier_profile(
    metrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize the clean source plus ten empirical 10-replay noise tiers."""
    profile = []
    for tier_idx in range(11):
        if tier_idx == 0:
            rows = [m for m in metrics if m["augmentation_variant_idx"] == 0]
            variant_range = [0, 0]
        else:
            first = (tier_idx - 1) * 10 + 1
            last = tier_idx * 10
            rows = [
                m
                for m in metrics
                if first <= m["augmentation_variant_idx"] <= last
            ]
            variant_range = [first, last]
        profile.append(
            {
                "empirical_tier_idx": tier_idx,
                "augmentation_variant_range": variant_range,
                "episodes": len(rows),
                "command_deviation_from_clean_rms_px": _quantiles(
                    [m["command_deviation_from_clean_rms_px"] for m in rows]
                ),
                "legacy_anchor_block_20px_20deg_fraction": float(
                    np.mean(
                        [
                            m["final_legacy_anchor_block_within_20px_20deg"]
                            for m in rows
                        ]
                    )
                ),
                "legacy_anchor_block_10px_10deg_fraction": float(
                    np.mean(
                        [
                            m["final_legacy_anchor_block_within_10px_10deg"]
                            for m in rows
                        ]
                    )
                ),
                "final_legacy_anchor_position_error_px": _quantiles(
                    [m["final_legacy_anchor_position_error_px"] for m in rows]
                ),
                "final_legacy_anchor_angle_error_deg": _quantiles(
                    [m["final_legacy_anchor_angle_error_deg"] for m in rows]
                ),
            }
        )
    return profile


def validate_group_structure(h5: h5py.File) -> dict[str, Any]:
    offsets = h5["ep_offset"][:].astype(int)
    lengths = h5["ep_len"][:].astype(int)
    if len(lengths) % VARIANTS_PER_SOURCE != 0:
        raise ValueError("episode count is not divisible by 101")
    group_count = len(lengths) // VARIANTS_PER_SOURCE
    initial_states = np.stack([h5["state"][int(offset)] for offset in offsets])
    equal_lengths = 0
    equal_initial_states = 0
    for group_idx in range(group_count):
        sl = slice(
            group_idx * VARIANTS_PER_SOURCE,
            (group_idx + 1) * VARIANTS_PER_SOURCE,
        )
        equal_lengths += int(np.all(lengths[sl] == lengths[sl][0]))
        equal_initial_states += int(
            np.all(initial_states[sl] == initial_states[sl][0])
        )
    return {
        "variants_per_source_demonstration": VARIANTS_PER_SOURCE,
        "source_demonstration_groups": group_count,
        "groups_with_identical_lengths": equal_lengths,
        "groups_with_identical_initial_states": equal_initial_states,
        "unique_initial_states": len(np.unique(initial_states, axis=0)),
        "empirical_interpretation": (
            "variant 0 is the clean released source demonstration; variants "
            "1-100 are same-initial-state noisy replays. Exact noise parameters "
            "are not stored in the artifact."
        ),
    }


def choose_noise_sweep_group(metrics: list[dict[str, Any]]) -> int:
    clean = [m for m in metrics if m["augmentation_variant_idx"] == 0]
    high = [m for m in metrics if m["augmentation_variant_idx"] == 100]
    median_length = np.median([m["rows"] for m in clean])
    median_clean_error = np.median(
        [m["final_legacy_anchor_position_error_px"] for m in clean]
    )
    median_high_error = np.median(
        [m["final_legacy_anchor_position_error_px"] for m in high]
    )
    scores = []
    for clean_row, high_row in zip(clean, high):
        score = (
            abs(clean_row["rows"] - median_length) / 25
            + abs(
                clean_row["final_legacy_anchor_position_error_px"]
                - median_clean_error
            )
            / 10
            + abs(
                high_row["final_legacy_anchor_position_error_px"]
                - median_high_error
            )
            / 100
        )
        if not clean_row["final_legacy_anchor_block_within_20px_20deg"]:
            score += 10
        scores.append(score)
    return int(np.argmin(scores))


def _font(size: int):
    from PIL import ImageFont

    candidates = (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def _plot_xy(point: np.ndarray, box: tuple[int, int, int, int]) -> tuple[int, int]:
    x, y, width, height = box
    return (
        int(x + np.clip(point[0], 0, 512) / 512 * width),
        int(y + height - np.clip(point[1], 0, 512) / 512 * height),
    )


def _draw_polyline(draw, points, box, fill, width=2):
    if len(points) > 1:
        draw.line([_plot_xy(p, box) for p in points], fill=fill, width=width)


def render_video(
    pixels: np.ndarray,
    action: np.ndarray,
    state: np.ndarray,
    clean_state: np.ndarray,
    summary: dict[str, Any],
    label: str,
    output_path: Path,
) -> Any:
    import imageio_ffmpeg
    from PIL import Image, ImageDraw

    width, height = 1280, 720
    image_box = (28, 78, 584, 584)
    plot_box = (680, 92, 548, 326)
    fonts = {size: _font(size) for size in (15, 17, 20, 27)}
    agent = state[:, :2]
    block = state[:, 2:4]
    angle = state[:, 4]
    velocity = state[:, 5:7]
    command = agent + action * ACTION_SCALE_PX
    preview_index = round((len(state) - 1) * 0.75)
    preview = None

    writer = imageio_ffmpeg.write_frames(
        str(output_path),
        (width, height),
        fps=CONTROL_HZ,
        codec="libx264",
        quality=7,
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    try:
        for idx, raw_pixels in enumerate(pixels):
            frame = Image.new("RGB", (width, height), "#0c111b")
            draw = ImageDraw.Draw(frame)
            source = Image.fromarray(raw_pixels).resize(
                (image_box[2], image_box[3]), Image.Resampling.LANCZOS
            )
            frame.paste(source, image_box[:2])

            draw.text(
                (28, 22), "PushT training trajectory", font=fonts[27], fill="#f5f7fa"
            )
            draw.text(
                (390, 27),
                f"{label.replace('_', ' ')}  |  episode {summary['episode_idx']}",
                font=fonts[17],
                fill="#9cc9ff",
            )
            draw.text(
                (680, 58),
                "Top-down state (512 px simulator coordinates)",
                font=fonts[17],
                fill="#d7deea",
            )

            x0, y0, plot_width, plot_height = plot_box
            draw.rectangle(
                (x0, y0, x0 + plot_width, y0 + plot_height),
                outline="#394455",
                width=2,
            )
            for fraction in (0.25, 0.5, 0.75):
                gx = int(x0 + plot_width * fraction)
                gy = int(y0 + plot_height * fraction)
                draw.line((gx, y0, gx, y0 + plot_height), fill="#202a38")
                draw.line((x0, gy, x0 + plot_width, gy), fill="#202a38")
            _draw_polyline(
                draw, clean_state[: idx + 1, 2:4], plot_box, "#8c96a6", 2
            )
            _draw_polyline(draw, agent[: idx + 1], plot_box, "#4cc9f0", 2)
            _draw_polyline(draw, block[: idx + 1], plot_box, "#ff6b7a", 3)
            anchor_xy = _plot_xy(LEGACY_ANCHOR_POS, plot_box)
            draw.ellipse(
                (
                    anchor_xy[0] - 6,
                    anchor_xy[1] - 6,
                    anchor_xy[0] + 6,
                    anchor_xy[1] + 6,
                ),
                outline="#75e6a4",
                width=3,
            )
            command_xy = _plot_xy(command[idx], plot_box)
            draw.line(
                (command_xy[0] - 7, command_xy[1], command_xy[0] + 7, command_xy[1]),
                fill="#ffd166",
                width=2,
            )
            draw.line(
                (command_xy[0], command_xy[1] - 7, command_xy[0], command_xy[1] + 7),
                fill="#ffd166",
                width=2,
            )
            draw.text((x0 + 8, y0 + 7), "agent", font=fonts[15], fill="#4cc9f0")
            draw.text((x0 + 70, y0 + 7), "block", font=fonts[15], fill="#ff6b7a")
            draw.text(
                (x0 + 130, y0 + 7), "clean block", font=fonts[15], fill="#8c96a6"
            )
            draw.text(
                (x0 + 235, y0 + 7), "anchor", font=fonts[15], fill="#75e6a4"
            )
            draw.text(
                (x0 + 300, y0 + 7), "command", font=fonts[15], fill="#ffd166"
            )

            info_y = 438
            pos_error = np.linalg.norm(block[idx] - LEGACY_ANCHOR_POS)
            angle_error = np.degrees(
                abs(_angle_delta(angle[idx], LEGACY_ANCHOR_ANGLE))
            )
            lines = (
                (
                    f"step {idx:3d}/{len(state)-1}    "
                    f"time {idx / CONTROL_HZ:5.1f}s    "
                    f"source group {summary['source_demo_idx']}    "
                    f"variant {summary['augmentation_variant_idx']}/100"
                ),
                (
                    f"agent xy [{agent[idx,0]:6.1f}, {agent[idx,1]:6.1f}]    "
                    f"velocity [{velocity[idx,0]:7.1f}, "
                    f"{velocity[idx,1]:7.1f}] px/s"
                ),
                (
                    f"block xy [{block[idx,0]:6.1f}, {block[idx,1]:6.1f}]    "
                    f"angle {np.degrees(angle[idx]):6.1f} deg"
                ),
                (
                    f"legacy green-anchor error: {pos_error:5.1f} px, "
                    f"{angle_error:5.1f} deg "
                    "(anchor is not DINO-WM evaluation goal)"
                ),
                (
                    f"action [{action[idx,0]:+.3f}, "
                    f"{action[idx,1]:+.3f}] -> "
                    f"PD target offset [{action[idx,0]*100:+.1f}, "
                    f"{action[idx,1]*100:+.1f}] px"
                ),
            )
            for line_idx, line in enumerate(lines):
                draw.text(
                    (680, info_y + 25 * line_idx),
                    line,
                    font=fonts[15],
                    fill="#d7deea",
                )

            bar_y = 584
            draw.text(
                (680, bar_y),
                "Normalized relative action (-1..1; physical scale: 100 px)",
                font=fonts[17],
                fill="#f5f7fa",
            )
            for action_idx, name in enumerate(("dx", "dy")):
                y = bar_y + 36 + action_idx * 32
                draw.text((680, y - 8), name, font=fonts[17], fill="#d7deea")
                center = 835
                draw.line((735, y, 935, y), fill="#6b7280", width=2)
                draw.line((center, y - 8, center, y + 8), fill="#d7deea")
                endpoint = int(center + 100 * np.clip(action[idx, action_idx], -1, 1))
                color = "#4cc9f0" if endpoint >= center else "#ff6b7a"
                draw.rectangle(
                    (min(center, endpoint), y - 6, max(center, endpoint), y + 6),
                    fill=color,
                )
                draw.text(
                    (955, y - 9),
                    f"{action[idx, action_idx]:+.3f}  ({action[idx, action_idx]*100:+.1f}px)",
                    font=fonts[15],
                    fill="#d7deea",
                )

            progress_y = 690
            draw.line((680, progress_y, 1228, progress_y), fill="#4b5563", width=3)
            progress_x = int(680 + 548 * idx / max(1, len(state) - 1))
            draw.ellipse(
                (progress_x - 5, progress_y - 5, progress_x + 5, progress_y + 5),
                fill="#75e6a4",
            )

            if idx == preview_index:
                preview = frame.copy()
            writer.send(np.asarray(frame))
    finally:
        writer.close()
    return preview


def make_contact_sheet(previews: list[tuple[str, Any]], output_path: Path) -> None:
    from PIL import Image, ImageDraw

    tile_width, tile_height = 640, 398
    sheet = Image.new("RGB", (tile_width * 2, tile_height * 2), "#10131a")
    for idx, (label, preview) in enumerate(previews):
        tile = Image.new("RGB", (tile_width, tile_height), "#10131a")
        tile.paste(
            preview.resize((tile_width, 360), Image.Resampling.LANCZOS),
            (0, 38),
        )
        ImageDraw.Draw(tile).text((14, 11), label, font=_font(17), fill="white")
        sheet.paste(tile, ((idx % 2) * tile_width, (idx // 2) * tile_height))
    sheet.save(output_path, optimize=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-videos", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.dataset, "r") as h5:
        metrics = scan_metrics(h5)
        structure = validate_group_structure(h5)
        aggregate = aggregate_metrics(metrics)
        profile = augmentation_profile(metrics)
        tier_profile = augmentation_tier_profile(metrics)
        group_idx = choose_noise_sweep_group(metrics)
        representative_indices = [
            group_idx * VARIANTS_PER_SOURCE + variant_idx
            for variant_idx in (0, 25, 50, 100)
        ]
        metric_by_episode = {m["episode_idx"]: m for m in metrics}
        representatives = {
            label: metric_by_episode[episode_idx]
            for label, episode_idx in zip(
                (
                    "clean_source",
                    "low_noise_replay",
                    "medium_noise_replay",
                    "high_noise_replay",
                ),
                representative_indices,
            )
        }

        audit = {
            "source": {
                "canonical_hf_repo": "quentinll/lewm-pusht",
                "canonical_revision": PUSHT_REVISION,
                "canonical_archive_name": "pusht_expert_train.h5.zst",
                "canonical_hdf5_name": args.dataset.name,
                "canonical_metadata_sha256": CANONICAL_METADATA_SHA256,
                "hdf5_bytes": args.dataset.stat().st_size,
            },
            "schema": {
                name: {"shape": list(dataset.shape), "dtype": str(dataset.dtype)}
                for name, dataset in sorted(h5.items())
            },
            "action_semantics": {
                "shape": [2],
                "normalized_range": [-1.0, 1.0],
                "meaning": "relative 2-D pusher position command to the PD controller",
                "physical_scale_px": ACTION_SCALE_PX,
                "control_hz": CONTROL_HZ,
            },
            "legacy_visual_anchor": {
                "block_position_px": LEGACY_ANCHOR_POS.tolist(),
                "block_angle_rad": LEGACY_ANCHOR_ANGLE,
                "note": (
                    "The fixed green T is a visual anchor, not the arbitrary "
                    "goal used by DINO-WM/CLEAR evaluation. Anchor alignment is "
                    "reported only to assess source-demonstration replay quality."
                ),
            },
            "group_structure": structure,
            "aggregate": aggregate,
            "augmentation_profile": profile,
            "augmentation_tier_profile": tier_profile,
            "representatives": representatives,
        }
        (args.output_dir / "pusht_training_data_audit.json").write_text(
            json.dumps(audit, indent=2) + "\n"
        )
        with (args.output_dir / "pusht_episode_metrics.jsonl").open("w") as file:
            for row in metrics:
                file.write(json.dumps(row) + "\n")

        if not args.skip_videos:
            offsets = h5["ep_offset"][:].astype(int)
            lengths = h5["ep_len"][:].astype(int)
            clean_episode_idx = group_idx * VARIANTS_PER_SOURCE
            clean_start = int(offsets[clean_episode_idx])
            clean_end = clean_start + int(lengths[clean_episode_idx])
            clean_state = h5["state"][clean_start:clean_end]
            previews = []
            for label, summary in representatives.items():
                episode_idx = summary["episode_idx"]
                start = int(offsets[episode_idx])
                end = start + int(lengths[episode_idx])
                output_path = args.output_dir / f"{label}_episode_{episode_idx}.mp4"
                preview = render_video(
                    h5["pixels"][start:end],
                    h5["action"][start:end],
                    h5["state"][start:end],
                    clean_state,
                    summary,
                    label,
                    output_path,
                )
                previews.append(
                    (
                        (
                            f"{label.replace('_', ' ')} - episode "
                            f"{episode_idx} - variant "
                            f"{summary['augmentation_variant_idx']}/100"
                        ),
                        preview,
                    )
                )
                print(f"Wrote {output_path}", flush=True)
            make_contact_sheet(
                previews,
                args.output_dir / "representative_noise_sweep_contact_sheet.png",
            )

    files = []
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            files.append(
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    (args.output_dir / "artifact_manifest.json").write_text(
        json.dumps({"files": files}, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "audit": str(args.output_dir / "pusht_training_data_audit.json"),
                "representative_group": group_idx,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
