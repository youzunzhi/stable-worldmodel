#!/usr/bin/env python3
"""Audit OGBench Cube training trajectories and export annotated videos.

The canonical LeWM artifact is ``cube_single_expert.h5``.  This script reads
the public, row-identical Lance conversion so that numeric columns and chosen
JPEG frames can be fetched with HTTP ranged reads instead of downloading the
46 GB compressed HDF5 release.

Example:
    python scripts/visualization/export_cube_dataset_trajectories.py \
        --output-dir outputs/analysis/cube_training_data_260809
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_URI = (
    "hf://datasets/galilai-group/ogb_cube_single/ogb_cube_single.lance"
)

# OGBench ManipSpace uses normalized actions.  The unnormalized maxima are
# [dx, dy, dz] in metres, yaw in radians, and relative gripper opening.
ACTION_NAMES = ("dx", "dy", "dz", "dyaw", "grip")
ACTION_SCALES = np.asarray([0.05, 0.05, 0.05, 0.3, 1.0])

SCAN_COLUMNS = (
    "episode_idx",
    "step_idx",
    "action",
    "privileged_block_0_pos",
    "privileged_block_0_yaw",
    "privileged_target_block_pos",
    "privileged_target_block_yaw",
    "proprio_effector_pos",
    "proprio_gripper_contact",
    "proprio_gripper_opening",
    "success",
    "reward",
    "terminated",
    "truncated",
    "time",
)

VIDEO_COLUMNS = SCAN_COLUMNS + ("pixels",)


def _as_numpy(array: Any) -> np.ndarray:
    """Convert an Arrow scalar/fixed-size-list array without Python rows."""
    import pyarrow as pa

    if pa.types.is_fixed_size_list(array.type):
        return array.values.to_numpy(zero_copy_only=False).reshape(
            len(array), array.type.list_size
        )
    return array.to_numpy(zero_copy_only=False)


def _record_batch_to_arrays(batch: Any) -> dict[str, np.ndarray]:
    return {
        name: _as_numpy(batch.column(batch.schema.get_field_index(name)))
        for name in SCAN_COLUMNS
    }


def _finite_float(value: Any) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _summarize_episode(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    ep = int(arrays["episode_idx"][0])
    step = arrays["step_idx"].astype(int)
    action = arrays["action"].astype(float)
    cube = arrays["privileged_block_0_pos"].astype(float)
    target = arrays["privileged_target_block_pos"].astype(float)
    effector = arrays["proprio_effector_pos"].astype(float)
    success = np.nan_to_num(
        arrays["success"].astype(float).reshape(-1), nan=0.0
    ) > 0.5
    success_event = success & ~np.concatenate([[False], success[:-1]])
    truncated = np.nan_to_num(
        arrays["truncated"].astype(float).reshape(-1), nan=0.0
    ) > 0.5
    terminated = np.nan_to_num(
        arrays["terminated"].astype(float).reshape(-1), nan=0.0
    ) > 0.5
    contact = np.nan_to_num(
        arrays["proprio_gripper_contact"].astype(float).reshape(-1), nan=0.0
    )
    opening = arrays["proprio_gripper_opening"].astype(float).reshape(-1)

    pos_error = np.linalg.norm(cube - target, axis=1)
    cube_delta = np.linalg.norm(np.diff(cube, axis=0), axis=1)
    eff_delta = np.linalg.norm(np.diff(effector, axis=0), axis=1)
    target_delta = np.linalg.norm(np.diff(target, axis=0), axis=1)
    target_yaw = arrays["privileged_target_block_yaw"].astype(float).reshape(-1)
    target_yaw_delta = np.abs(
        np.angle(np.exp(1j * np.diff(target_yaw)))
    )
    target_change = (target_delta > 1e-5) | (target_yaw_delta > 1e-5)
    valid_action = np.isfinite(action).all(axis=1)
    action_norm = np.linalg.norm(action[valid_action], axis=1)
    valid_action_pair = valid_action[1:] & valid_action[:-1]
    action_tv = np.linalg.norm(np.diff(action, axis=0)[valid_action_pair], axis=1)

    success_steps = step[success_event].tolist()
    return {
        "episode_idx": ep,
        "rows": int(len(step)),
        "first_step": int(step[0]),
        "last_step": int(step[-1]),
        "duration_seconds": _finite_float(
            arrays["time"].astype(float).reshape(-1)[-1]
        ),
        "success_event_count": int(success_event.sum()),
        "success_event_steps": [int(v) for v in success_steps],
        "success_frame_count": int(success.sum()),
        "success_frame_fraction": float(success.mean()),
        "first_success_step": int(success_steps[0]) if success_steps else None,
        "last_success_step": int(success_steps[-1]) if success_steps else None,
        "target_change_count": int(target_change.sum()),
        "initial_position_error_m": float(pos_error[0]),
        "minimum_position_error_m": float(pos_error.min()),
        "final_position_error_m": float(pos_error[-1]),
        "cube_path_length_m": float(cube_delta.sum()),
        "effector_path_length_m": float(eff_delta.sum()),
        "cube_net_displacement_m": float(np.linalg.norm(cube[-1] - cube[0])),
        "cube_max_height_m": float(cube[:, 2].max()),
        "cube_lift_fraction": float(np.mean(cube[:, 2] > 0.04)),
        "valid_action_rows": int(valid_action.sum()),
        "invalid_action_rows": int((~valid_action).sum()),
        "mean_action_norm": float(action_norm.mean()),
        "action_total_variation": float(action_tv.sum()),
        "action_saturation_fraction": float(
            np.mean(np.abs(action[valid_action]) >= 0.95)
        ),
        "gripper_contact_fraction": float(np.mean(contact > 0.5)),
        "gripper_opening_min": float(np.nanmin(opening)),
        "gripper_opening_max": float(np.nanmax(opening)),
        "terminated_last": bool(terminated[-1]),
        "truncated_last": bool(truncated[-1]),
    }


def scan_episode_metrics(
    dataset: Any,
    episode_start: int = 0,
    episode_end: int | None = None,
) -> list[dict[str, Any]]:
    """Reduce numeric columns into one row per episode with bounded memory.

    A full multi-column Lance scan can prefetch a large part of both remote
    fragments.  We first scan only the small episode-index column, then use
    explicit row ``take`` calls for 100 complete episodes at a time.
    """
    ep_parts = []
    scanner = dataset.scanner(columns=["episode_idx"], batch_size=65_536)
    for batch in scanner.to_batches():
        ep_parts.append(_as_numpy(batch.column(0)).astype(int))
    ep_ids = np.concatenate(ep_parts)
    boundaries = np.flatnonzero(np.diff(ep_ids) != 0) + 1
    offsets = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(ep_ids)]])
    lengths = ends - offsets
    episode_ids = ep_ids[offsets]
    episode_end = len(offsets) if episode_end is None else episode_end
    if not 0 <= episode_start < episode_end <= len(offsets):
        raise ValueError(
            f"invalid episode range [{episode_start}, {episode_end}) for "
            f"{len(offsets)} episodes"
        )

    metrics: list[dict[str, Any]] = []
    episodes_per_take = 100
    for batch_start in range(episode_start, episode_end, episodes_per_take):
        batch_end = min(batch_start + episodes_per_take, episode_end)
        row_start = int(offsets[batch_start])
        row_end = int(ends[batch_end - 1])
        table = dataset.take(
            list(range(row_start, row_end)), columns=list(SCAN_COLUMNS)
        )
        arrays = {
            name: _as_numpy(table.column(name).combine_chunks())
            for name in SCAN_COLUMNS
        }
        for local_ep in range(batch_start, batch_end):
            start = int(offsets[local_ep] - row_start)
            end = start + int(lengths[local_ep])
            episode = {name: value[start:end] for name, value in arrays.items()}
            summary = _summarize_episode(episode)
            if summary["episode_idx"] != int(episode_ids[local_ep]):
                raise RuntimeError("episode index changed during ranged read")
            metrics.append(summary)
        if (batch_end - episode_start) % 1000 == 0 or batch_end == episode_end:
            print(
                f"Scanned episodes [{episode_start:,}, {batch_end:,})",
                flush=True,
            )
    return metrics


def _quantiles(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(a)),
        "p10": float(np.quantile(a, 0.10)),
        "median": float(np.median(a)),
        "p90": float(np.quantile(a, 0.90)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
    }


def aggregate_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    success_counts = Counter(m["success_event_count"] for m in metrics)
    rows = Counter(m["rows"] for m in metrics)
    success_eps = [m for m in metrics if m["success_event_count"] > 0]
    fields = (
        "success_event_count",
        "success_frame_fraction",
        "first_success_step",
        "minimum_position_error_m",
        "cube_path_length_m",
        "effector_path_length_m",
        "cube_max_height_m",
        "mean_action_norm",
        "action_total_variation",
        "action_saturation_fraction",
        "gripper_contact_fraction",
    )
    quantiles = {}
    for field in fields:
        values = [
            float(m[field])
            for m in metrics
            if m[field] is not None and math.isfinite(float(m[field]))
        ]
        quantiles[field] = _quantiles(values)

    return {
        "episodes": len(metrics),
        "rows": int(sum(m["rows"] for m in metrics)),
        "episode_length_counts": {str(k): v for k, v in sorted(rows.items())},
        "episodes_with_recorded_success": len(success_eps),
        "episodes_without_recorded_success": len(metrics) - len(success_eps),
        "episode_success_coverage": len(success_eps) / len(metrics),
        "success_pulse_count_distribution": {
            str(k): v for k, v in sorted(success_counts.items())
        },
        "valid_action_rows": int(sum(m["valid_action_rows"] for m in metrics)),
        "invalid_action_rows": int(sum(m["invalid_action_rows"] for m in metrics)),
        "truncated_at_last_row": sum(m["truncated_last"] for m in metrics),
        "terminated_at_last_row": sum(m["terminated_last"] for m in metrics),
        "quantiles": quantiles,
    }


def choose_representatives(
    metrics: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    successful = [m for m in metrics if m["success_event_count"] > 0]
    failed = [m for m in metrics if m["success_event_count"] == 0]

    def medoid(pool: list[dict[str, Any]], fields: tuple[str, ...]):
        matrix = np.asarray([[float(m[f]) for f in fields] for m in pool])
        center = np.median(matrix, axis=0)
        scale = np.maximum(np.quantile(matrix, 0.75, axis=0) - np.quantile(matrix, 0.25, axis=0), 1e-9)
        index = int(np.argmin(np.linalg.norm((matrix - center) / scale, axis=1)))
        return pool[index]

    typical = medoid(
        successful,
        (
            "first_success_step",
            "success_event_count",
            "cube_path_length_m",
            "action_total_variation",
        ),
    )

    fast_pool = sorted(
        successful,
        key=lambda m: (
            m["first_success_step"],
            -m["success_event_count"],
            m["action_total_variation"],
        ),
    )
    # Avoid a pathological single minimum by choosing near the 2nd percentile.
    fast = fast_pool[min(len(fast_pool) - 1, max(0, len(fast_pool) // 50))]

    late_threshold = np.quantile(
        [m["first_success_step"] for m in successful], 0.9
    )
    slow_pool = [m for m in successful if m["first_success_step"] >= late_threshold]
    slow = max(
        slow_pool,
        key=lambda m: (m["action_total_variation"], m["first_success_step"]),
    )

    if failed:
        failed_near_median = medoid(
            failed,
            (
                "minimum_position_error_m",
                "cube_path_length_m",
                "action_total_variation",
            ),
        )
    else:
        # If every episode has a pulse, use the least productive episode.
        failed_near_median = min(
            metrics,
            key=lambda m: (
                m["success_event_count"],
                -m["first_success_step"],
                -m["minimum_position_error_m"],
            ),
        )

    representatives = {
        "fast_clean_success": fast,
        "typical_success": typical,
        "slow_recovery_success": slow,
    }
    representatives[
        "no_recorded_success" if failed else "fewest_success_events"
    ] = failed_near_median
    return representatives


def load_episode(dataset: Any, episode_idx: int, rows_per_episode: int) -> dict[str, Any]:
    start = episode_idx * rows_per_episode
    indices = list(range(start, start + rows_per_episode))
    table = dataset.take(indices, columns=list(VIDEO_COLUMNS))
    result = {}
    for name in VIDEO_COLUMNS:
        column = table.column(name).combine_chunks()
        result[name] = column.to_pylist() if name == "pixels" else _as_numpy(column)
    return result


def load_episode_from_dataset_server(
    episode_idx: int,
    rows_per_episode: int,
) -> dict[str, Any]:
    """Load one early episode through HF's cached dataset-viewer assets.

    The public viewer currently exposes a processed prefix of the Lance
    table.  It is much more request-efficient for JPEGs than reading the
    remote binary column through anonymous per-range resolver calls.
    """
    import httpx

    start = episode_idx * rows_per_episode
    rows = []
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        for offset in range(start, start + rows_per_episode, 100):
            length = min(100, start + rows_per_episode - offset)
            response = client.get(
                "https://datasets-server.huggingface.co/rows",
                params={
                    "dataset": "galilai-group/ogb_cube_single",
                    "config": "default",
                    "split": "train",
                    "offset": offset,
                    "length": length,
                },
            )
            response.raise_for_status()
            rows.extend(item["row"] for item in response.json()["rows"])

    if len(rows) != rows_per_episode:
        raise RuntimeError(
            f"dataset server returned {len(rows)} rows, expected {rows_per_episode}"
        )
    if any(int(row["episode_idx"]) != episode_idx for row in rows):
        raise RuntimeError("dataset server response crossed an episode boundary")

    urls = [row["pixels"]["src"] for row in rows]

    def download(url: str) -> bytes:
        response = httpx.get(url, timeout=60.0, follow_redirects=True)
        response.raise_for_status()
        return response.content

    with ThreadPoolExecutor(max_workers=16) as pool:
        pixels = list(pool.map(download, urls))

    result = {"pixels": pixels}
    for name in SCAN_COLUMNS:
        values = [row[name] for row in rows]
        result[name] = np.asarray(values)
    return result


def _xy_to_canvas(
    xy: np.ndarray,
    box: tuple[int, int, int, int],
) -> tuple[int, int]:
    x0, y0, width, height = box
    x = x0 + (float(xy[0]) - 0.25) / (0.60 - 0.25) * width
    y = y0 + height - (float(xy[1]) + 0.35) / 0.70 * height
    return int(x), int(y)


def _font(size: int):
    from PIL import ImageFont

    candidates = (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def _draw_polyline(draw: Any, points: np.ndarray, box: tuple[int, int, int, int], fill: str, width: int):
    if len(points) < 2:
        return
    draw.line([_xy_to_canvas(p[:2], box) for p in points], fill=fill, width=width)


def render_video(
    episode: dict[str, Any],
    summary: dict[str, Any],
    label: str,
    output_path: Path,
    fps: int,
) -> None:
    from PIL import Image, ImageDraw
    import imageio_ffmpeg

    cube = episode["privileged_block_0_pos"].astype(float)
    target = episode["privileged_target_block_pos"].astype(float)
    effector = episode["proprio_effector_pos"].astype(float)
    raw_action = episode["action"].astype(float)
    valid_action = np.isfinite(raw_action).all(axis=1)
    action = np.nan_to_num(raw_action, nan=0.0)
    success = np.nan_to_num(episode["success"].astype(float).reshape(-1), nan=0.0) > 0.5
    success_event = success & ~np.concatenate([[False], success[:-1]])
    contact = np.nan_to_num(
        episode["proprio_gripper_contact"].astype(float).reshape(-1), nan=0.0
    )
    opening = episode["proprio_gripper_opening"].astype(float).reshape(-1)
    step = episode["step_idx"].astype(int)
    pos_error = np.linalg.norm(cube - target, axis=1)

    width, height = 1280, 720
    image_box = (28, 78, 584, 584)
    plot_box = (680, 105, 548, 278)
    fonts = {size: _font(size) for size in (16, 18, 22, 28)}
    target_change_steps = np.flatnonzero(
        np.linalg.norm(np.diff(target, axis=0), axis=1) > 1e-5
    ) + 1

    writer = imageio_ffmpeg.write_frames(
        str(output_path),
        (width, height),
        fps=fps,
        codec="libx264",
        quality=7,
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    try:
        for i, blob in enumerate(episode["pixels"]):
            frame = Image.new("RGB", (width, height), "#0c111b")
            draw = ImageDraw.Draw(frame)
            source = Image.open(io.BytesIO(blob)).convert("RGB")
            source = source.resize((image_box[2], image_box[3]), Image.Resampling.LANCZOS)
            frame.paste(source, image_box[:2])

            draw.text((28, 22), "Cube training trajectory", font=fonts[28], fill="#f5f7fa")
            draw.text(
                (390, 27),
                f"{label.replace('_', ' ')}  |  episode {summary['episode_idx']}",
                font=fonts[18],
                fill="#9cc9ff",
            )
            draw.text(
                (680, 67),
                "Privileged top-down state (not model input)",
                font=fonts[18],
                fill="#d7deea",
            )

            x0, y0, pw, ph = plot_box
            draw.rectangle((x0, y0, x0 + pw, y0 + ph), outline="#334155", width=2)
            for gx in np.linspace(0.25, 0.60, 8):
                px, _ = _xy_to_canvas(np.array([gx, -0.35]), plot_box)
                draw.line((px, y0, px, y0 + ph), fill="#1f2937", width=1)
            for gy in np.linspace(-0.35, 0.35, 8):
                _, py = _xy_to_canvas(np.array([0.25, gy]), plot_box)
                draw.line((x0, py, x0 + pw, py), fill="#1f2937", width=1)
            _draw_polyline(draw, cube, plot_box, "#4b6b88", 2)
            _draw_polyline(draw, cube[: i + 1], plot_box, "#ff5f6d", 4)
            _draw_polyline(draw, effector[: i + 1], plot_box, "#53c7f0", 2)
            tx, ty = _xy_to_canvas(target[i, :2], plot_box)
            cx, cy = _xy_to_canvas(cube[i, :2], plot_box)
            ex, ey = _xy_to_canvas(effector[i, :2], plot_box)
            draw.ellipse((tx - 11, ty - 11, tx + 11, ty + 11), outline="#ffd166", width=4)
            draw.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), fill="#ff5f6d")
            draw.ellipse((ex - 5, ey - 5, ex + 5, ey + 5), fill="#53c7f0")
            draw.text((x0 + 10, y0 + 8), "cube", font=fonts[16], fill="#ff7b86")
            draw.text((x0 + 72, y0 + 8), "target", font=fonts[16], fill="#ffd166")
            draw.text((x0 + 148, y0 + 8), "end-effector", font=fonts[16], fill="#53c7f0")

            info_y = 404
            draw.text(
                (680, info_y),
                f"step {step[i]:3d}/200   time {step[i] * 0.05:5.2f}s   "
                f"position error {pos_error[i] * 100:5.1f} cm",
                font=fonts[18],
                fill="#f5f7fa",
            )
            draw.text(
                (680, info_y + 28),
                f"cube xyz [{cube[i,0]:+.3f}, {cube[i,1]:+.3f}, {cube[i,2]:+.3f}] m   "
                f"target [{target[i,0]:+.3f}, {target[i,1]:+.3f}, {target[i,2]:+.3f}] m",
                font=fonts[16],
                fill="#cbd5e1",
            )
            draw.text(
                (680, info_y + 54),
                f"gripper opening {opening[i]:.3f}   contact {int(contact[i] > 0.5)}   "
                f"success {int(success[i])}   action valid {int(valid_action[i])}",
                font=fonts[16],
                fill="#7dd3fc" if not success[i] else "#86efac",
            )

            action_y = 500
            draw.text(
                (680, action_y - 30),
                "Normalized action  (physical scale: 5 cm xyz, 0.3 rad yaw)",
                font=fonts[18],
                fill="#d7deea",
            )
            for j, name in enumerate(ACTION_NAMES):
                yy = action_y + j * 32
                draw.text((680, yy), f"{name:>4}", font=fonts[16], fill="#cbd5e1")
                center = 812
                draw.line((center - 105, yy + 10, center + 105, yy + 10), fill="#475569", width=2)
                end = center + int(np.clip(action[i, j], -1, 1) * 105)
                color = "#fb7185" if action[i, j] < 0 else "#38bdf8"
                draw.rectangle((min(center, end), yy + 4, max(center, end), yy + 16), fill=color)
                physical = action[i, j] * ACTION_SCALES[j]
                unit = "m" if j < 3 else ("rad" if j == 3 else "")
                draw.text(
                    (930, yy),
                    f"{action[i,j]:+6.3f}  ({physical:+7.4f} {unit})",
                    font=fonts[16],
                    fill="#f5f7fa",
                )

            timeline_x0, timeline_x1, timeline_y = 1015, 1218, 684
            draw.line((timeline_x0, timeline_y, timeline_x1, timeline_y), fill="#475569", width=4)
            progress_x = timeline_x0 + int(i / (len(step) - 1) * (timeline_x1 - timeline_x0))
            draw.line((timeline_x0, timeline_y, progress_x, timeline_y), fill="#60a5fa", width=5)
            for s in np.flatnonzero(success_event):
                sx = timeline_x0 + int(s / (len(step) - 1) * (timeline_x1 - timeline_x0))
                draw.ellipse((sx - 4, timeline_y - 4, sx + 4, timeline_y + 4), fill="#86efac")
            for s in target_change_steps:
                sx = timeline_x0 + int(s / (len(step) - 1) * (timeline_x1 - timeline_x0))
                draw.line((sx, timeline_y - 8, sx, timeline_y + 8), fill="#ffd166", width=2)
            draw.text((1015, 650), "success / target-change timeline", font=fonts[16], fill="#cbd5e1")

            writer.send(np.asarray(frame, dtype=np.uint8))
    finally:
        writer.close()


def make_contact_sheet(
    dataset: Any,
    representatives: dict[str, dict[str, Any]],
    rows_per_episode: int,
    output_path: Path,
) -> None:
    from PIL import Image, ImageDraw

    font_title = _font(28)
    font_body = _font(18)
    canvas = Image.new("RGB", (1000, 1050), "#0c111b")
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 22), "Representative Cube training trajectories", font=font_title, fill="#f5f7fa")
    for index, (label, summary) in enumerate(representatives.items()):
        episode = load_episode(dataset, summary["episode_idx"], rows_per_episode)
        success_steps = summary["success_event_steps"]
        key_steps = [0, success_steps[0] if success_steps else 100, 200]
        y = 80 + index * 235
        draw.text(
            (30, y),
            f"{label.replace('_', ' ')} | episode {summary['episode_idx']} | "
            f"success events {summary['success_event_count']} | first {summary['first_success_step']}",
            font=font_body,
            fill="#9cc9ff",
        )
        for j, s in enumerate(key_steps):
            image = Image.open(io.BytesIO(episode["pixels"][s])).convert("RGB")
            image = image.resize((210, 210), Image.Resampling.LANCZOS)
            x = 30 + j * 230
            canvas.paste(image, (x, y + 28))
            draw.text((x + 8, y + 198), f"step {s}", font=_font(16), fill="#ffffff")
        draw.text(
            (735, y + 50),
            f"min error\n{summary['minimum_position_error_m'] * 100:.1f} cm\n\n"
            f"cube path\n{summary['cube_path_length_m']:.2f} m\n\n"
            f"action TV\n{summary['action_total_variation']:.1f}",
            font=font_body,
            fill="#d7deea",
            spacing=6,
        )
    canvas.save(output_path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default=DEFAULT_URI)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--scan-start", type=int)
    parser.add_argument("--scan-end", type=int)
    parser.add_argument("--scan-output", type=Path)
    parser.add_argument("--render-episode", type=int)
    parser.add_argument("--render-label")
    parser.add_argument("--dataset-server", action="store_true")
    return parser.parse_args()


def main() -> None:
    import lance

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.render_episode is not None:
        if args.render_label is None:
            raise ValueError("--render-label is required with --render-episode")
        summary = None
        metrics_path = args.output_dir / "cube_episode_metrics.jsonl"
        with metrics_path.open() as file:
            for line in file:
                candidate = json.loads(line)
                if candidate["episode_idx"] == args.render_episode:
                    summary = candidate
                    break
        if summary is None:
            raise ValueError(f"episode {args.render_episode} not found in metrics")
        episode = (
            load_episode_from_dataset_server(
                args.render_episode, summary["rows"]
            )
            if args.dataset_server
            else load_episode(
                lance.dataset(args.uri), args.render_episode, summary["rows"]
            )
        )
        video_path = (
            args.output_dir
            / f"{args.render_label}_episode_{args.render_episode}.mp4"
        )
        render_video(
            episode,
            summary,
            args.render_label,
            video_path,
            args.fps,
        )
        print(f"Wrote {video_path}", flush=True)
        return

    dataset = lance.dataset(args.uri)
    row_count = dataset.count_rows()

    if args.scan_start is not None or args.scan_end is not None:
        if args.scan_start is None or args.scan_end is None or args.scan_output is None:
            raise ValueError(
                "--scan-start, --scan-end, and --scan-output are required together"
            )
        metrics = scan_episode_metrics(dataset, args.scan_start, args.scan_end)
        args.scan_output.parent.mkdir(parents=True, exist_ok=True)
        with args.scan_output.open("w") as file:
            for row in metrics:
                file.write(json.dumps(row) + "\n")
        print(f"Wrote {len(metrics)} rows to {args.scan_output}", flush=True)
        return

    last_episode = dataset.take(
        [row_count - 1], columns=["episode_idx"]
    ).column("episode_idx")[0].as_py()
    episode_count = int(last_episode) + 1
    shard_dir = args.output_dir / "metric_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = []
    for start in range(0, episode_count, 1000):
        end = min(start + 1000, episode_count)
        shard_path = shard_dir / f"episodes_{start:05d}_{end:05d}.jsonl"
        shard_paths.append(shard_path)
        complete = False
        if shard_path.exists():
            with shard_path.open() as file:
                complete = sum(1 for _ in file) == end - start
        if complete:
            continue
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--uri",
                args.uri,
                "--output-dir",
                str(args.output_dir),
                "--scan-start",
                str(start),
                "--scan-end",
                str(end),
                "--scan-output",
                str(shard_path),
            ],
            check=True,
        )

    metrics = []
    for shard_path in shard_paths:
        with shard_path.open() as file:
            metrics.extend(json.loads(line) for line in file if line.strip())
    aggregate = aggregate_metrics(metrics)
    representatives = choose_representatives(metrics)
    rows_per_episode = next(iter(Counter(m["rows"] for m in metrics)))

    # This pair is pinned in the canonical HDF5 CLEAR manifest.  Matching its
    # row identity and position delta is a compact equivalence anchor.
    anchor = dataset.take(
        [87645, 87670],
        columns=["episode_idx", "step_idx", "privileged_block_0_pos"],
    ).to_pydict()
    anchor_pos = np.asarray(anchor["privileged_block_0_pos"], dtype=float)
    anchor_distance = float(np.linalg.norm(anchor_pos[1] - anchor_pos[0]))

    audit = {
        "source": {
            "canonical_hdf5_repo": "quentinll/lewm-cube",
            "canonical_hdf5_revision": "02a19a67a0dc8c9d6215f89c19e0a597691e152a",
            "canonical_hdf5_name": "cube_single_expert.h5",
            "canonical_metadata_sha256": "c0d8515b4f187d792f986e8894618d32e605c6857f629b4f7c7a32f85ded8007",
            "public_lance_uri": args.uri,
            "public_lance_version": dataset.version,
            "public_lance_rows": row_count,
            "identity_anchor": {
                "rows": [87645, 87670],
                "episode_idx": anchor["episode_idx"],
                "step_idx": anchor["step_idx"],
                "position_distance_m": anchor_distance,
                "canonical_manifest_position_distance_m": 0.06859241639835252,
                "absolute_difference_m": abs(anchor_distance - 0.06859241639835252),
            },
        },
        "schema": {field.name: str(field.type) for field in dataset.schema},
        "aggregate": aggregate,
        "representatives": representatives,
        "action_semantics": {
            name: {"normalized_range": [-1.0, 1.0], "physical_scale": float(scale)}
            for name, scale in zip(ACTION_NAMES, ACTION_SCALES)
        },
    }
    audit_path = args.output_dir / "cube_training_data_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    metrics_path = args.output_dir / "cube_episode_metrics.jsonl"
    with metrics_path.open("w") as file:
        for row in metrics:
            file.write(json.dumps(row) + "\n")

    if not args.skip_videos:
        for label, summary in representatives.items():
            episode = load_episode(dataset, summary["episode_idx"], rows_per_episode)
            render_video(
                episode,
                summary,
                label,
                args.output_dir / f"{label}_episode_{summary['episode_idx']}.mp4",
                args.fps,
            )
        make_contact_sheet(
            dataset,
            representatives,
            rows_per_episode,
            args.output_dir / "representative_trajectories_contact_sheet.png",
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
    manifest_path = args.output_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps({"files": files}, indent=2) + "\n")
    print(json.dumps({"audit": str(audit_path), "representatives": representatives}, indent=2))


if __name__ == "__main__":
    main()
