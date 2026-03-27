#!/usr/bin/env python3
"""Compare Farneback vs RAFT optical flow on LIBERO parquet episodes."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large


CAPTION_HEIGHT = 24
PANEL_BG = (18, 18, 18)
TEXT_COLOR = (240, 240, 240)


def decode_image(cell) -> np.ndarray:
    image_bytes = cell["bytes"] if isinstance(cell, dict) else cell
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.asarray(image)


def decode_depth(depth_bytes, depth_shape, depth_dtype) -> np.ndarray:
    dtype = np.dtype(depth_dtype)
    return np.frombuffer(depth_bytes, dtype=dtype).reshape(depth_shape).astype(np.float32)


def maybe_rotate_180(arr: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return arr
    return np.ascontiguousarray(arr[::-1, ::-1])


def load_camera_payload(camera_params_path: Path) -> dict:
    with open(camera_params_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_camera_entry(payload: dict, episode_task_index: int | None, camera_name: str) -> dict:
    if "entries" in payload and episode_task_index is not None:
        for entry in payload["entries"]:
            if entry.get("task_id") == episode_task_index:
                cameras = entry.get("cameras", {})
                if camera_name in cameras and "error" not in cameras[camera_name]:
                    return cameras[camera_name]
    alias = {"agentview": "agent", "robot0_eye_in_hand": "wrist"}.get(camera_name, camera_name)
    if alias in payload:
        return payload[alias]
    cameras = payload.get("cameras", {})
    if camera_name in cameras:
        return cameras[camera_name]
    raise KeyError(f"Camera parameters for {camera_name} not found")


def infer_depth_mode(depth_map: np.ndarray) -> str:
    finite = depth_map[np.isfinite(depth_map)]
    if finite.size == 0:
        return "metric"
    if finite.min() >= 0.0 and finite.max() <= 1.0:
        return "mujoco_normalized"
    return "metric"


def depth_to_metric(depth_map: np.ndarray, depth_mode: str, camera_entry: dict | None) -> np.ndarray:
    if depth_mode == "metric":
        return depth_map.astype(np.float32)
    if depth_mode == "mujoco_normalized":
        if camera_entry is None:
            raise ValueError("camera_entry required for mujoco_normalized depth conversion")
        near = float(camera_entry["near"])
        far = float(camera_entry["far"])
        clipped = np.clip(depth_map, 0.0, 1.0)
        return near / (1.0 - clipped * (1.0 - near / far))
    raise ValueError(f"Unsupported depth mode: {depth_mode}")


def load_episode_payload(
    parquet_path: Path,
    image_column: str,
    depth_column: str,
    rotate_180: bool,
    max_frames: int | None,
) -> dict:
    table = pq.read_table(parquet_path, columns=[image_column, depth_column, "depth_shape", "depth_dtype", "task_index"])
    data = table.to_pydict()

    images = [maybe_rotate_180(decode_image(cell), rotate_180) for cell in data[image_column]]
    depth_shape = tuple(data["depth_shape"][0])
    depth_dtype = data["depth_dtype"][0]
    depths = [maybe_rotate_180(decode_depth(cell, depth_shape, depth_dtype), rotate_180) for cell in data[depth_column]]

    if max_frames is not None:
        images = images[:max_frames]
        depths = depths[:max_frames]

    task_values = data.get("task_index", [])
    task_index = int(task_values[0]) if task_values else None
    return {
        "images": images,
        "depths": depths,
        "depth_shape": depth_shape,
        "depth_dtype": depth_dtype,
        "task_index": task_index,
    }


def compute_farneback_flow(image_t: np.ndarray, image_t1: np.ndarray) -> np.ndarray:
    gray_t = cv2.cvtColor(image_t, cv2.COLOR_RGB2GRAY)
    gray_t1 = cv2.cvtColor(image_t1, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        gray_t,
        gray_t1,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    return flow.astype(np.float32)


class RAFTBackend:
    def __init__(self, device: str) -> None:
        self.device = torch.device(device)
        self.weights = Raft_Large_Weights.DEFAULT
        self.transforms = self.weights.transforms()
        self.model = raft_large(weights=self.weights, progress=True).to(self.device).eval()

    @torch.inference_mode()
    def __call__(self, image_t: np.ndarray, image_t1: np.ndarray) -> np.ndarray:
        tensor_t = torch.from_numpy(np.ascontiguousarray(image_t).copy()).permute(2, 0, 1).unsqueeze(0)
        tensor_t1 = torch.from_numpy(np.ascontiguousarray(image_t1).copy()).permute(2, 0, 1).unsqueeze(0)
        tensor_t, tensor_t1 = self.transforms(tensor_t, tensor_t1)
        flow_predictions = self.model(tensor_t.to(self.device), tensor_t1.to(self.device))
        return flow_predictions[-1][0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)


def make_sampling_plan(num_frames: int, stride: int, start_frame: int, max_pairs: int | None) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    if num_frames < stride + 1:
        return pairs
    last_start = num_frames - stride - 1
    for frame_idx in range(start_frame, last_start + 1, stride):
        pairs.append((frame_idx, frame_idx + stride))
        if max_pairs is not None and len(pairs) >= max_pairs:
            break
    return pairs


def flow_to_color(flow: np.ndarray) -> np.ndarray:
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    scale = float(np.percentile(magnitude, 95))
    if scale <= 1e-6:
        scale = 1.0
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = np.mod(angle / 2.0, 180).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(magnitude / scale * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def warp_image(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32), indexing="xy")
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    return cv2.remap(
        image,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def bilinear_sample_scalar(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    height, width = image.shape
    x0 = np.floor(xs).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(ys).astype(np.int32)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, width - 1)
    x1 = np.clip(x1, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)
    y1 = np.clip(y1, 0, height - 1)

    wa = (x1 - xs) * (y1 - ys)
    wb = (x1 - xs) * (ys - y0)
    wc = (xs - x0) * (y1 - ys)
    wd = (xs - x0) * (ys - y0)

    return (
        wa * image[y0, x0]
        + wb * image[y1, x0]
        + wc * image[y0, x1]
        + wd * image[y1, x1]
    ).astype(np.float32)


def backproject_camera(uv_x: np.ndarray, uv_y: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    x = (uv_x - cx) * depth / fx
    y = (uv_y - cy) * depth / fy
    z = depth
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def build_valid_mask(
    depth_t: np.ndarray,
    depth_t1_sampled: np.ndarray,
    warped_x: np.ndarray,
    warped_y: np.ndarray,
    min_depth: float,
    max_depth: float | None,
) -> np.ndarray:
    height, width = depth_t.shape
    in_bounds = (warped_x >= 0.0) & (warped_x <= (width - 1)) & (warped_y >= 0.0) & (warped_y <= (height - 1))
    valid = in_bounds & np.isfinite(depth_t) & np.isfinite(depth_t1_sampled) & (depth_t > min_depth) & (depth_t1_sampled > min_depth)
    if max_depth is not None:
        valid &= depth_t < max_depth
        valid &= depth_t1_sampled < max_depth
    return valid


def compute_diff_map(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    diff = np.mean(np.abs(prediction.astype(np.float32) - target.astype(np.float32)), axis=-1)
    return diff, float(diff.mean())


def scalar_to_color(values: np.ndarray, valid_mask: np.ndarray | None = None, percentile: float = 95.0) -> np.ndarray:
    valid = np.isfinite(values)
    if valid_mask is not None:
        valid &= valid_mask
    if not np.any(valid):
        return np.zeros((*values.shape, 3), dtype=np.uint8)
    scale = float(np.percentile(values[valid], percentile))
    if scale <= 1e-6:
        scale = 1.0
    normalized = np.zeros_like(values, dtype=np.uint8)
    normalized[valid] = np.clip(values[valid] / scale * 255.0, 0.0, 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    colored[~valid] = 0
    return colored


def signed_to_color(values: np.ndarray, valid_mask: np.ndarray | None = None, percentile: float = 95.0) -> np.ndarray:
    valid = np.isfinite(values)
    if valid_mask is not None:
        valid &= valid_mask
    colored = np.zeros((*values.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return colored

    scale = float(np.percentile(np.abs(values[valid]), percentile))
    if scale <= 1e-6:
        scale = 1.0
    normalized = np.clip(values / scale, -1.0, 1.0)
    positive = valid & (normalized > 0)
    negative = valid & (normalized < 0)
    colored[..., 0][positive] = np.clip(normalized[positive] * 255.0, 0.0, 255.0).astype(np.uint8)
    colored[..., 1][positive] = np.clip(normalized[positive] * 96.0, 0.0, 255.0).astype(np.uint8)
    colored[..., 2][negative] = np.clip((-normalized[negative]) * 255.0, 0.0, 255.0).astype(np.uint8)
    colored[..., 1][negative] = np.clip((-normalized[negative]) * 96.0, 0.0, 255.0).astype(np.uint8)
    return colored


def add_caption(image: np.ndarray, caption: str) -> np.ndarray:
    pil = Image.new("RGB", (image.shape[1], image.shape[0] + CAPTION_HEIGHT), PANEL_BG)
    pil.paste(Image.fromarray(image), (0, CAPTION_HEIGHT))
    draw = ImageDraw.Draw(pil)
    draw.text((6, 4), caption, fill=TEXT_COLOR, font=ImageFont.load_default())
    return np.asarray(pil)


def stack_row(images: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(images, axis=1)


def stack_grid(rows: list[list[np.ndarray]]) -> np.ndarray:
    return np.concatenate([stack_row(row) for row in rows], axis=0)


def summarize_backend(
    flow: np.ndarray,
    image_t: np.ndarray,
    image_t1: np.ndarray,
    depth_t: np.ndarray,
    depth_t1: np.ndarray,
    intrinsics: np.ndarray,
    min_depth: float,
    max_depth: float | None,
) -> dict[str, np.ndarray | float]:
    warped = warp_image(image_t, flow)
    diff_map, warp_diff = compute_diff_map(warped, image_t1)
    magnitude = np.linalg.norm(flow, axis=-1)

    height, width = depth_t.shape
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32), indexing="xy")
    warped_x = grid_x + flow[..., 0]
    warped_y = grid_y + flow[..., 1]
    sampled_depth_t1 = bilinear_sample_scalar(depth_t1, warped_x, warped_y)
    valid_mask = build_valid_mask(depth_t, sampled_depth_t1, warped_x, warped_y, min_depth, max_depth)

    points_t = backproject_camera(grid_x, grid_y, depth_t, intrinsics)
    points_t1 = backproject_camera(warped_x, warped_y, sampled_depth_t1, intrinsics)
    flow_3d = (points_t1 - points_t).astype(np.float32)
    flow_3d[~valid_mask] = 0.0
    flow_3d_norm = np.linalg.norm(flow_3d, axis=-1)
    flow_3d_z = flow_3d[..., 2]

    valid_ratio = float(valid_mask.mean())
    if np.any(valid_mask):
        flow_3d_mean = float(flow_3d_norm[valid_mask].mean())
        flow_3d_p95 = float(np.percentile(flow_3d_norm[valid_mask], 95))
        flow_3d_z_mean = float(flow_3d_z[valid_mask].mean())
        flow_3d_z_abs_p95 = float(np.percentile(np.abs(flow_3d_z[valid_mask]), 95))
    else:
        flow_3d_mean = 0.0
        flow_3d_p95 = 0.0
        flow_3d_z_mean = 0.0
        flow_3d_z_abs_p95 = 0.0

    return {
        "warped": warped,
        "diff_map": diff_map,
        "flow2d_mean": float(magnitude.mean()),
        "flow2d_p95": float(np.percentile(magnitude, 95)),
        "warp_diff": warp_diff,
        "valid_mask": valid_mask,
        "flow_3d_norm": flow_3d_norm,
        "flow_3d_z": flow_3d_z,
        "flow_3d_mean": flow_3d_mean,
        "flow_3d_p95": flow_3d_p95,
        "flow_3d_z_mean": flow_3d_z_mean,
        "flow_3d_z_abs_p95": flow_3d_z_abs_p95,
        "valid_ratio": valid_ratio,
    }


def render_pair_panel(
    image_t: np.ndarray,
    image_t1: np.ndarray,
    depth_t: np.ndarray,
    depth_t1: np.ndarray,
    farneback_flow: np.ndarray,
    raft_flow: np.ndarray,
    intrinsics: np.ndarray,
    min_depth: float,
    max_depth: float | None,
) -> tuple[np.ndarray, dict[str, float]]:
    raw_diff_map, raw_diff = compute_diff_map(image_t, image_t1)
    depth_valid_t = np.isfinite(depth_t) & (depth_t > min_depth)
    depth_valid_t1 = np.isfinite(depth_t1) & (depth_t1 > min_depth)
    if max_depth is not None:
        depth_valid_t &= depth_t < max_depth
        depth_valid_t1 &= depth_t1 < max_depth

    farneback_stats = summarize_backend(farneback_flow, image_t, image_t1, depth_t, depth_t1, intrinsics, min_depth, max_depth)
    raft_stats = summarize_backend(raft_flow, image_t, image_t1, depth_t, depth_t1, intrinsics, min_depth, max_depth)

    panel = stack_grid(
        [
            [
                add_caption(image_t, "frame_t"),
                add_caption(image_t1, "frame_t1"),
                add_caption(scalar_to_color(raw_diff_map), f"raw diff mean={raw_diff:.3f}"),
                add_caption(scalar_to_color(depth_t, depth_valid_t), "depth_t"),
                add_caption(scalar_to_color(depth_t1, depth_valid_t1), "depth_t1"),
            ],
            [
                add_caption(flow_to_color(farneback_flow), f"Farneback flow p95={farneback_stats['flow2d_p95']:.3f}"),
                add_caption(farneback_stats["warped"], f"Farneback warp mean={farneback_stats['warp_diff']:.3f}"),
                add_caption(scalar_to_color(farneback_stats["diff_map"]), "Farneback diff map"),
                add_caption(
                    scalar_to_color(farneback_stats["flow_3d_norm"], farneback_stats["valid_mask"]),
                    f"Farneback |flow3d| p95={farneback_stats['flow_3d_p95']:.3f}",
                ),
                add_caption(
                    signed_to_color(farneback_stats["flow_3d_z"], farneback_stats["valid_mask"]),
                    f"Farneback flow3d_z abs95={farneback_stats['flow_3d_z_abs_p95']:.3f}",
                ),
            ],
            [
                add_caption(flow_to_color(raft_flow), f"RAFT flow p95={raft_stats['flow2d_p95']:.3f}"),
                add_caption(raft_stats["warped"], f"RAFT warp mean={raft_stats['warp_diff']:.3f}"),
                add_caption(scalar_to_color(raft_stats["diff_map"]), "RAFT diff map"),
                add_caption(
                    scalar_to_color(raft_stats["flow_3d_norm"], raft_stats["valid_mask"]),
                    f"RAFT |flow3d| p95={raft_stats['flow_3d_p95']:.3f}",
                ),
                add_caption(
                    signed_to_color(raft_stats["flow_3d_z"], raft_stats["valid_mask"]),
                    f"RAFT flow3d_z abs95={raft_stats['flow_3d_z_abs_p95']:.3f}",
                ),
            ],
        ]
    )

    metrics = {
        "raw_diff": raw_diff,
        "farneback_flow2d_mean": float(farneback_stats["flow2d_mean"]),
        "farneback_flow2d_p95": float(farneback_stats["flow2d_p95"]),
        "farneback_warp_diff": float(farneback_stats["warp_diff"]),
        "farneback_flow3d_mean": float(farneback_stats["flow_3d_mean"]),
        "farneback_flow3d_p95": float(farneback_stats["flow_3d_p95"]),
        "farneback_flow3d_z_mean": float(farneback_stats["flow_3d_z_mean"]),
        "farneback_valid_ratio": float(farneback_stats["valid_ratio"]),
        "raft_flow2d_mean": float(raft_stats["flow2d_mean"]),
        "raft_flow2d_p95": float(raft_stats["flow2d_p95"]),
        "raft_warp_diff": float(raft_stats["warp_diff"]),
        "raft_flow3d_mean": float(raft_stats["flow_3d_mean"]),
        "raft_flow3d_p95": float(raft_stats["flow_3d_p95"]),
        "raft_flow3d_z_mean": float(raft_stats["flow_3d_z_mean"]),
        "raft_valid_ratio": float(raft_stats["valid_ratio"]),
    }
    return panel, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Farneback and RAFT flow on LIBERO parquet frames")
    parser.add_argument("--parquet-path", type=str, required=True, help="Path to one parquet episode")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for comparison PNGs")
    parser.add_argument("--image-column", type=str, default="image", choices=["image", "wrist_image"])
    parser.add_argument("--depth-column", type=str, default=None, help="Depth column name; defaults to match image column")
    parser.add_argument("--camera-name", type=str, default=None, help="Camera name; defaults to match image column")
    parser.add_argument("--camera-params", type=str, default=None, help="Path to extracted camera params JSON")
    parser.add_argument("--depth-mode", type=str, choices=["metric", "mujoco_normalized", "auto"], default="metric")
    parser.add_argument("--min-depth", type=float, default=0.01)
    parser.add_argument("--max-depth", type=float, default=None)
    parser.add_argument("--stride", type=int, default=1, help="Frame gap between compared images")
    parser.add_argument("--start-frame", type=int, default=0, help="First source frame index")
    parser.add_argument("--max-pairs", type=int, default=4, help="Maximum number of visualized pairs")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on loaded frames")
    parser.add_argument("--rotate-180", action="store_true", help="Rotate images and depths by 180 degrees before comparison")
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. cuda or cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parquet_path = Path(args.parquet_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    depth_column = args.depth_column
    if depth_column is None:
        depth_column = {"image": "depth", "wrist_image": "wrist_depth"}[args.image_column]

    camera_name = args.camera_name
    if camera_name is None:
        camera_name = {"image": "agentview", "wrist_image": "robot0_eye_in_hand"}[args.image_column]

    camera_params = args.camera_params
    if camera_params is None:
        default_camera_params = Path("/tmp/libero_camera_params_test.json")
        if default_camera_params.exists():
            camera_params = str(default_camera_params)
        else:
            raise ValueError("--camera-params is required for 3D flow visualization")

    payload = load_episode_payload(
        parquet_path=parquet_path,
        image_column=args.image_column,
        depth_column=depth_column,
        rotate_180=args.rotate_180,
        max_frames=args.max_frames,
    )
    images = payload["images"]
    depths = payload["depths"]
    task_index = payload["task_index"]

    if len(images) < 2:
        raise ValueError(f"Need at least 2 frames, got {len(images)}")

    camera_payload = load_camera_payload(Path(camera_params))
    camera_entry = get_camera_entry(camera_payload, task_index, camera_name)
    intrinsics = np.asarray(camera_entry["intrinsics"], dtype=np.float32)

    auto_mode = args.depth_mode
    if auto_mode == "auto":
        auto_mode = infer_depth_mode(depths[0])
    depths = [depth_to_metric(depth, auto_mode, camera_entry) for depth in depths]

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    raft_backend = RAFTBackend(device=device)
    frame_pairs = make_sampling_plan(len(images), args.stride, args.start_frame, args.max_pairs)
    if not frame_pairs:
        raise ValueError(
            f"No frame pairs available for stride={args.stride}, start_frame={args.start_frame}, num_frames={len(images)}"
        )

    summary_lines = [
        f"parquet_path={parquet_path}",
        f"image_column={args.image_column}",
        f"depth_column={depth_column}",
        f"camera_name={camera_name}",
        f"camera_params={camera_params}",
        f"rotate_180={bool(args.rotate_180)}",
        f"num_loaded_frames={len(images)}",
        f"task_index={task_index}",
        f"depth_mode={auto_mode}",
        f"device={device}",
        "",
    ]

    for source_idx, target_idx in frame_pairs:
        image_t = images[source_idx]
        image_t1 = images[target_idx]
        depth_t = depths[source_idx]
        depth_t1 = depths[target_idx]

        farneback_flow = compute_farneback_flow(image_t, image_t1)
        raft_flow = raft_backend(image_t, image_t1)
        panel, metrics = render_pair_panel(
            image_t=image_t,
            image_t1=image_t1,
            depth_t=depth_t,
            depth_t1=depth_t1,
            farneback_flow=farneback_flow,
            raft_flow=raft_flow,
            intrinsics=intrinsics,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        )

        pair_name = f"pair_{source_idx:03d}_to_{target_idx:03d}"
        Image.fromarray(panel).save(output_dir / f"{pair_name}.png")
        summary_lines.append(
            f"{pair_name}: raw_diff={metrics['raw_diff']:.6f}, "
            f"farneback_mean={metrics['farneback_flow2d_mean']:.6f}, "
            f"farneback_p95={metrics['farneback_flow2d_p95']:.6f}, "
            f"farneback_warp_diff={metrics['farneback_warp_diff']:.6f}, "
            f"farneback_flow3d_mean={metrics['farneback_flow3d_mean']:.6f}, "
            f"farneback_flow3d_p95={metrics['farneback_flow3d_p95']:.6f}, "
            f"farneback_flow3d_z_mean={metrics['farneback_flow3d_z_mean']:.6f}, "
            f"farneback_valid_ratio={metrics['farneback_valid_ratio']:.6f}, "
            f"raft_mean={metrics['raft_flow2d_mean']:.6f}, "
            f"raft_p95={metrics['raft_flow2d_p95']:.6f}, "
            f"raft_warp_diff={metrics['raft_warp_diff']:.6f}, "
            f"raft_flow3d_mean={metrics['raft_flow3d_mean']:.6f}, "
            f"raft_flow3d_p95={metrics['raft_flow3d_p95']:.6f}, "
            f"raft_flow3d_z_mean={metrics['raft_flow3d_z_mean']:.6f}, "
            f"raft_valid_ratio={metrics['raft_valid_ratio']:.6f}"
        )
        print(summary_lines[-1])

    (output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Saved comparison visualizations to {output_dir}")


if __name__ == "__main__":
    main()
