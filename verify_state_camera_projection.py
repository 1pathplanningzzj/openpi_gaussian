#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from PIL import Image


DEFAULT_CAMERA_POS = np.array([0.5386131746834771, 0.0, 0.7903500240372423], dtype=np.float32)
# stored in repo as [w, x, y, z]
DEFAULT_CAMERA_QUAT_WXYZ = np.array(
    [0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349], dtype=np.float32
)
DEFAULT_FX = 221.7025
DEFAULT_FY = 221.7025
DEFAULT_CX = 128.0
DEFAULT_CY = 128.0
DEFAULT_DEPTH_W = 256.0
DEFAULT_DEPTH_H = 256.0


def decode_image(cell) -> np.ndarray:
    image_bytes = cell["bytes"] if isinstance(cell, dict) else cell
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.asarray(image)


# same convention as action_coord_transform.py / verify_action_camera_alignment.py
# input quat is [x, y, z, w]
def quat_xyzw_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = quat
    return np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
    ], dtype=np.float32)


def build_viewmatrix(cam_pos: np.ndarray, cam_quat_xyzw: np.ndarray) -> np.ndarray:
    R = quat_xyzw_to_rotation_matrix(cam_quat_xyzw)
    viewmatrix = np.eye(4, dtype=np.float32)
    viewmatrix[:3, :3] = R
    viewmatrix[:3, 3] = -R @ cam_pos
    return viewmatrix


def world_to_camera(point_world: np.ndarray, viewmatrix: np.ndarray) -> np.ndarray:
    point_h = np.concatenate([point_world.astype(np.float32), np.array([1.0], dtype=np.float32)], axis=0)
    point_cam_h = viewmatrix @ point_h
    return point_cam_h[:3]


def project_camera_to_image(point_cam: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> tuple[float, float]:
    x, y, z = point_cam.tolist()
    z = max(z, 1e-6)
    u = fx * x / z + cx
    v = fy * y / z + cy
    return float(u), float(v)


def clamp_for_display(u: float, v: float, width: int, height: int, margin: float = 32.0) -> tuple[float, float]:
    u_disp = min(max(u, -margin), width - 1 + margin)
    v_disp = min(max(v, -margin), height - 1 + margin)
    return float(u_disp), float(v_disp)


def image_to_feature_uv(u: float, v: float, width: int, height: int, depth_w: float, depth_h: float) -> tuple[float, float]:
    x_feat = ((u + 0.5) / depth_w) * width - 0.5
    y_feat = ((v + 0.5) / depth_h) * height - 0.5
    return float(x_feat), float(y_feat)


def build_heatmap(x_feat: float, y_feat: float, height: int, width: int, valid: bool) -> np.ndarray:
    yy = np.arange(height, dtype=np.float32)[:, None]
    xx = np.arange(width, dtype=np.float32)[None, :]
    sigma = max(1.0, 0.06 * float(max(height, width)))
    dist2 = (xx - x_feat) ** 2 + (yy - y_feat) ** 2
    heatmap = np.exp(-0.5 * dist2 / (sigma**2)).astype(np.float32)
    if not valid:
        heatmap[...] = 0.0
    return heatmap


def draw_cross(ax, u: float, v: float, color: str, label: str) -> None:
    ax.scatter([u], [v], c=color, s=70, marker="x", linewidths=2, label=label)
    ax.text(u + 4, v + 4, label, color=color, fontsize=8, weight="bold")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify LIBERO state[:3] world->camera->image projection")
    parser.add_argument("--parquet-path", type=str, default="/data/zijianzhang/LIBERA/data/chunk-000/episode_000000.parquet")
    parser.add_argument("--output-dir", type=str, default="/home/zijianzhang/openpi/debug_state_projection")
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--feature-size", type=int, default=32)
    parser.add_argument("--rotate-180", action="store_true", help="Rotate image by 180 degrees before overlaying")
    args = parser.parse_args()

    parquet_path = Path(args.parquet_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(parquet_path, columns=["image", "state", "frame_index", "timestamp"])
    data = table.to_pydict()
    num_rows = len(data["state"])
    if num_rows == 0:
        raise ValueError(f"No rows found in {parquet_path}")

    sample_indices = np.linspace(0, num_rows - 1, min(args.num_samples, num_rows), dtype=int)

    cam_pos = DEFAULT_CAMERA_POS
    cam_quat_xyzw = np.array(
        [
            DEFAULT_CAMERA_QUAT_WXYZ[1],
            DEFAULT_CAMERA_QUAT_WXYZ[2],
            DEFAULT_CAMERA_QUAT_WXYZ[3],
            DEFAULT_CAMERA_QUAT_WXYZ[0],
        ],
        dtype=np.float32,
    )
    viewmatrix = build_viewmatrix(cam_pos, cam_quat_xyzw)

    rows = len(sample_indices)
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    print(f"Using parquet: {parquet_path}")
    print(f"Output dir: {output_dir}")
    print(f"Camera position: {cam_pos.tolist()}")
    print(f"Camera quat [xyzw]: {cam_quat_xyzw.tolist()}")

    for row_idx, sample_idx in enumerate(sample_indices.tolist()):
        state = np.asarray(data["state"][sample_idx], dtype=np.float32)
        eef_world = state[:3]
        image = decode_image(data["image"][sample_idx])
        if args.rotate_180:
            image = np.ascontiguousarray(image[::-1, ::-1])
        h, w = image.shape[:2]

        eef_cam = world_to_camera(eef_world, viewmatrix)
        u, v = project_camera_to_image(eef_cam, DEFAULT_FX, DEFAULT_FY, DEFAULT_CX, DEFAULT_CY)
        x_feat, y_feat = image_to_feature_uv(u, v, args.feature_size, args.feature_size, DEFAULT_DEPTH_W, DEFAULT_DEPTH_H)

        direct_cam = eef_world.copy()
        u_direct, v_direct = project_camera_to_image(direct_cam, DEFAULT_FX, DEFAULT_FY, DEFAULT_CX, DEFAULT_CY)
        x_feat_direct, y_feat_direct = image_to_feature_uv(
            u_direct, v_direct, args.feature_size, args.feature_size, DEFAULT_DEPTH_W, DEFAULT_DEPTH_H
        )

        valid_z = bool(eef_cam[2] > 1e-6)
        in_bounds = bool(0.0 <= u <= (w - 1) and 0.0 <= v <= (h - 1))
        feat_in_bounds = bool(0.0 <= x_feat <= (args.feature_size - 1) and 0.0 <= y_feat <= (args.feature_size - 1))
        valid = valid_z and feat_in_bounds
        heatmap = build_heatmap(x_feat, y_feat, args.feature_size, args.feature_size, valid)

        direct_valid_z = bool(direct_cam[2] > 1e-6)
        direct_in_bounds = bool(0.0 <= u_direct <= (w - 1) and 0.0 <= v_direct <= (h - 1))
        direct_feat_in_bounds = bool(
            0.0 <= x_feat_direct <= (args.feature_size - 1) and 0.0 <= y_feat_direct <= (args.feature_size - 1)
        )

        heat_peak = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
        peak_y, peak_x = int(heat_peak[0]), int(heat_peak[1])

        ax_img = axes[row_idx, 0]
        ax_heat = axes[row_idx, 1]
        ax_text = axes[row_idx, 2]

        ax_img.imshow(image)
        u_disp, v_disp = clamp_for_display(u, v, w, h)
        u_direct_disp, v_direct_disp = clamp_for_display(u_direct, v_direct, w, h)
        draw_cross(ax_img, u_disp, v_disp, "red", "w2c")
        draw_cross(ax_img, u_direct_disp, v_direct_disp, "cyan", "direct")
        ax_img.set_xlim(0, w)
        ax_img.set_ylim(h, 0)
        ax_img.set_title(f"sample {sample_idx} overlay")
        ax_img.axis("off")

        ax_heat.imshow(heatmap, cmap="magma", vmin=0.0, vmax=max(1e-6, float(heatmap.max())))
        ax_heat.scatter([x_feat], [y_feat], c="cyan", s=50, marker="x", linewidths=2)
        ax_heat.scatter([peak_x], [peak_y], c="lime", s=30, marker="o")
        ax_heat.set_title(f"heatmap {args.feature_size}x{args.feature_size}")
        ax_heat.set_xlim(0, args.feature_size)
        ax_heat.set_ylim(args.feature_size, 0)

        ax_text.axis("off")
        lines = [
            f"frame_index: {data['frame_index'][sample_idx]}",
            f"timestamp: {data['timestamp'][sample_idx]}",
            f"state[:3] world: {np.round(eef_world, 4).tolist()}",
            f"w2c eef_cam: {np.round(eef_cam, 4).tolist()}",
            f"w2c uv: ({u:.2f}, {v:.2f})",
            f"w2c feat_uv: ({x_feat:.2f}, {y_feat:.2f})",
            f"w2c valid_z: {valid_z}",
            f"w2c img_in_bounds: {in_bounds}",
            f"w2c feat_in_bounds: {feat_in_bounds}",
            f"direct uv: ({u_direct:.2f}, {v_direct:.2f})",
            f"direct feat_uv: ({x_feat_direct:.2f}, {y_feat_direct:.2f})",
            f"direct valid_z: {direct_valid_z}",
            f"direct img_in_bounds: {direct_in_bounds}",
            f"direct feat_in_bounds: {direct_feat_in_bounds}",
            f"heat_peak: ({peak_x}, {peak_y})",
        ]
        ax_text.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace")

        print(
            f"sample={sample_idx} frame={data['frame_index'][sample_idx]} "
            f"world={np.round(eef_world, 4).tolist()} cam={np.round(eef_cam, 4).tolist()} "
            f"w2c_uv=({u:.2f},{v:.2f}) w2c_feat=({x_feat:.2f},{y_feat:.2f}) "
            f"w2c_valid_z={valid_z} w2c_img_in_bounds={in_bounds} w2c_feat_in_bounds={feat_in_bounds} "
            f"direct_uv=({u_direct:.2f},{v_direct:.2f}) direct_feat=({x_feat_direct:.2f},{y_feat_direct:.2f}) "
            f"direct_valid_z={direct_valid_z} direct_img_in_bounds={direct_in_bounds} direct_feat_in_bounds={direct_feat_in_bounds}"
        )

    plt.tight_layout()
    out_path = output_dir / f"{parquet_path.stem}_projection_debug.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization to: {out_path}")


if __name__ == "__main__":
    main()
