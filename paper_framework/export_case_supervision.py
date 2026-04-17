#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pyarrow.parquet as pq
from matplotlib import colormaps
from PIL import Image

from src.openpi.training.depth_transform import LoadFlowTransform


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RGB/depth/flow images for a paper case folder.")
    parser.add_argument("case_dir", type=Path, help="Case directory containing manifest.json")
    parser.add_argument(
        "--flow-root",
        type=Path,
        default=Path("/data/zijianzhang/LIBERA/flow_sidecars_raft"),
        help="Root directory of the LIBERO flow sidecars",
    )
    parser.add_argument(
        "--future-horizon",
        type=int,
        default=5,
        help="Number of future flow targets to export",
    )
    return parser.parse_args()


def _load_manifest(case_dir: Path) -> dict:
    manifest_path = case_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _load_rows(parquet_path: str | Path, frame_indices: list[int]) -> dict[int, dict]:
    table = pq.read_table(
        parquet_path,
        columns=[
            "image",
            "wrist_image",
            "depth",
            "wrist_depth",
            "depth_shape",
            "depth_dtype",
            "frame_index",
        ],
    )
    rows = {}
    wanted = set(int(idx) for idx in frame_indices)
    for row in table.to_pylist():
        frame_index = int(row["frame_index"])
        if frame_index in wanted:
            rows[frame_index] = row
    missing = sorted(wanted.difference(rows))
    if missing:
        raise KeyError(f"Missing requested frame indices in parquet: {missing}")
    return rows


def _decode_png(image_struct: dict) -> Image.Image:
    return Image.open(io.BytesIO(image_struct["bytes"])).convert("RGB")


def _depth_shape(row: dict) -> tuple[int, int]:
    shape = row.get("depth_shape")
    if shape is None:
        return (256, 256)
    if isinstance(shape, np.ndarray):
        shape = shape.tolist()
    return tuple(int(v) for v in shape)


def _depth_dtype(row: dict) -> np.dtype:
    dtype_name = row.get("depth_dtype") or "float32"
    return np.dtype(str(dtype_name))


def _decode_depth(depth_blob: bytes, row: dict) -> np.ndarray:
    shape = _depth_shape(row)
    dtype = _depth_dtype(row)
    depth = np.frombuffer(depth_blob, dtype=dtype).reshape(shape)
    return depth.astype(np.float32, copy=False)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _compute_depth_range(depth_maps: list[np.ndarray]) -> tuple[float, float]:
    finite_values = [
        depth[np.isfinite(depth) & (depth > 0.0)]
        for depth in depth_maps
    ]
    finite_values = [values for values in finite_values if values.size > 0]
    if not finite_values:
        return 0.0, 1.0
    merged = np.concatenate(finite_values)
    vmin = float(np.percentile(merged, 1.0))
    vmax = float(np.percentile(merged, 99.0))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _save_depth_png(depth: np.ndarray, out_path: Path, *, vmin: float, vmax: float) -> None:
    norm = np.clip((depth - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    rgba = colormaps["viridis"](norm)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    invalid = ~np.isfinite(depth)
    if invalid.any():
        rgb[invalid] = 0
    Image.fromarray(rgb).save(out_path)


def _save_flow_norm_png(flow: np.ndarray, valid_mask: np.ndarray, out_path: Path) -> None:
    magnitude = np.linalg.norm(flow, axis=-1)
    valid_values = magnitude[valid_mask]
    vmax = float(np.percentile(valid_values, 99.0)) if valid_values.size > 0 else 1.0
    vmax = max(vmax, 1e-6)
    norm = np.clip(magnitude / vmax, 0.0, 1.0)
    rgba = colormaps["turbo"](norm)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    rgb[~valid_mask] = 0
    Image.fromarray(rgb).save(out_path)


def _save_flow_xyz_png(flow: np.ndarray, valid_mask: np.ndarray, out_path: Path) -> None:
    valid_values = np.abs(flow[valid_mask])
    scale = float(np.percentile(valid_values, 99.0)) if valid_values.size > 0 else 1.0
    scale = max(scale, 1e-6)
    rgb = np.clip((flow / (2.0 * scale)) + 0.5, 0.0, 1.0)
    rgb = (rgb * 255.0).astype(np.uint8)
    rgb[~valid_mask] = 0
    Image.fromarray(rgb).save(out_path)


def _export_rgb_and_depth(case_dir: Path, manifest: dict, rows: dict[int, dict]) -> None:
    labels = list(manifest["labels"])
    frame_indices = [int(idx) for idx in manifest["frame_indices"]]

    agent_depths = []
    wrist_depths = []
    for frame_index in frame_indices:
        row = rows[frame_index]
        agent_depths.append(_decode_depth(row["depth"], row))
        wrist_depths.append(_decode_depth(row["wrist_depth"], row))

    agent_vmin, agent_vmax = _compute_depth_range(agent_depths)
    wrist_vmin, wrist_vmax = _compute_depth_range(wrist_depths)

    agent_dir = case_dir / "agent"
    wrist_dir = case_dir / "wrist"
    agent_depth_dir = case_dir / "agent_depth"
    wrist_depth_dir = case_dir / "wrist_depth"
    agent_depth_raw_dir = case_dir / "agent_depth_raw"
    wrist_depth_raw_dir = case_dir / "wrist_depth_raw"
    for path in [agent_dir, wrist_dir, agent_depth_dir, wrist_depth_dir, agent_depth_raw_dir, wrist_depth_raw_dir]:
        _ensure_dir(path)

    for label, frame_index in zip(labels, frame_indices, strict=True):
        row = rows[frame_index]
        _decode_png(row["image"]).save(agent_dir / f"{label}_frame_{frame_index:03d}.png")
        _decode_png(row["wrist_image"]).save(wrist_dir / f"{label}_frame_{frame_index:03d}.png")

        agent_depth = _decode_depth(row["depth"], row)
        wrist_depth = _decode_depth(row["wrist_depth"], row)
        np.save(agent_depth_raw_dir / f"{label}_frame_{frame_index:03d}.npy", agent_depth)
        np.save(wrist_depth_raw_dir / f"{label}_frame_{frame_index:03d}.npy", wrist_depth)
        _save_depth_png(
            agent_depth,
            agent_depth_dir / f"{label}_frame_{frame_index:03d}.png",
            vmin=agent_vmin,
            vmax=agent_vmax,
        )
        _save_depth_png(
            wrist_depth,
            wrist_depth_dir / f"{label}_frame_{frame_index:03d}.png",
            vmin=wrist_vmin,
            vmax=wrist_vmax,
        )


def _export_flow(case_dir: Path, manifest: dict, flow_root: Path, future_horizon: int) -> None:
    flow_loader = LoadFlowTransform(str(flow_root), future_horizon=future_horizon)
    episode_index = int(manifest["episode_index"])
    frame_index = int(manifest["current_frame"])
    episode_data = flow_loader._load_episode(flow_loader._sidecar_path(episode_index))
    flow_targets, mask_targets = flow_loader._compose_anchor_flow_targets(
        episode_data["flow_2d"],
        episode_data["flow_3d"],
        episode_data["valid_mask"],
        frame_index,
    )

    flow_norm_dir = case_dir / "flow_3d_norm"
    flow_xyz_dir = case_dir / "flow_3d_xyz"
    flow_raw_dir = case_dir / "flow_3d_raw"
    flow_mask_dir = case_dir / "flow_3d_mask"
    for path in [flow_norm_dir, flow_xyz_dir, flow_raw_dir, flow_mask_dir]:
        _ensure_dir(path)

    labels = [f"tplus{offset}" for offset in range(1, min(future_horizon, flow_targets.shape[0]) + 1)]
    for horizon_idx, label in enumerate(labels):
        flow = flow_targets[horizon_idx]
        valid_mask = mask_targets[horizon_idx]
        np.save(flow_raw_dir / f"{label}_flow3d.npy", flow)
        np.save(flow_mask_dir / f"{label}_valid_mask.npy", valid_mask)
        _save_flow_norm_png(flow, valid_mask, flow_norm_dir / f"{label}_flow3d_norm.png")
        _save_flow_xyz_png(flow, valid_mask, flow_xyz_dir / f"{label}_flow3d_xyz.png")


def main() -> None:
    args = _parse_args()
    case_dir = args.case_dir.resolve()
    manifest = _load_manifest(case_dir)
    rows = _load_rows(manifest["parquet_path"], manifest["frame_indices"])
    _export_rgb_and_depth(case_dir, manifest, rows)
    _export_flow(case_dir, manifest, args.flow_root, args.future_horizon)
    print(
        json.dumps(
            {
                "case_dir": str(case_dir),
                "frame_count": len(manifest["frame_indices"]),
                "future_horizon": int(args.future_horizon),
                "exports": [
                    "agent/*.png",
                    "wrist/*.png",
                    "agent_depth/*.png",
                    "wrist_depth/*.png",
                    "flow_3d_norm/*.png",
                    "flow_3d_xyz/*.png",
                    "agent_depth_raw/*.npy",
                    "wrist_depth_raw/*.npy",
                    "flow_3d_raw/*.npy",
                    "flow_3d_mask/*.npy",
                ],
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
