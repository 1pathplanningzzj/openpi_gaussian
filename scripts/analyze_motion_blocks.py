#!/usr/bin/env python3
"""Analyze LIBERO flow sidecars to infer motion-block structure.

Outputs:
- candidate motion-block visualizations
- translation vs single-rigid residual comparisons
- summary recommendation for Gaussian motion parameterization
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw


def decode_image(cell) -> np.ndarray:
    if isinstance(cell, dict):
        image_bytes = cell["bytes"]
    else:
        image_bytes = cell
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.asarray(image)


def decode_depth(depth_bytes: bytes, depth_shape: tuple[int, int], depth_dtype: str) -> np.ndarray:
    dtype = np.dtype(depth_dtype)
    return np.frombuffer(depth_bytes, dtype=dtype).reshape(depth_shape).astype(np.float32)


def backproject_camera(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    h, w = depth.shape
    grid_y, grid_x = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    x = (grid_x - cx) * depth / fx
    y = (grid_y - cy) * depth / fy
    z = depth
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def connected_components(mask: np.ndarray, min_size: int) -> list[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            q = deque([(y, x)])
            visited[y, x] = True
            coords: list[tuple[int, int]] = []
            while q:
                cy, cx = q.popleft()
                coords.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))
            if len(coords) < min_size:
                continue
            comp = np.zeros_like(mask, dtype=bool)
            ys, xs = zip(*coords)
            comp[np.array(ys), np.array(xs)] = True
            components.append(comp)
    return components


def fit_translation(points0: np.ndarray, points1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    translation = (points1 - points0).mean(axis=0)
    pred = points0 + translation
    residual = np.linalg.norm(pred - points1, axis=1)
    return translation, residual


def fit_rigid(points0: np.ndarray, points1: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c0 = points0.mean(axis=0)
    c1 = points1.mean(axis=0)
    q0 = points0 - c0
    q1 = points1 - c1
    cov = q0.T @ q1
    u, _, vt = np.linalg.svd(cov)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = c1 - r @ c0
    pred = (points0 @ r.T) + t
    residual = np.linalg.norm(pred - points1, axis=1)
    return r, t, residual


def speed_to_heatmap(speed: np.ndarray) -> np.ndarray:
    vmax = np.quantile(speed[np.isfinite(speed)], 0.99) if np.isfinite(speed).any() else 1.0
    vmax = max(float(vmax), 1e-6)
    scaled = np.clip(speed / vmax, 0.0, 1.0)
    r = (255 * scaled).astype(np.uint8)
    g = (255 * np.clip(1.0 - np.abs(scaled - 0.5) * 2.0, 0.0, 1.0)).astype(np.uint8)
    b = (255 * (1.0 - scaled)).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


@dataclass
class BlockMetric:
    episode_index: int
    task_index: int
    frame_index: int
    block_id: int
    pixel_count: int
    mean_depth: float
    mean_flow_mag: float
    translation_residual_mean: float
    translation_residual_median: float
    rigid_residual_mean: float
    rigid_residual_median: float
    rigid_improvement_pct: float
    best_model: str


@dataclass
class FrameMetric:
    episode_index: int
    task_index: int
    frame_index: int
    candidate_blocks: int
    largest_block_pixels: int
    mean_valid_ratio: float
    mean_flow_mag: float


class MotionBlockAnalyzer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(args.output_dir) / f"motion_blocks_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.overlay_dir = self.output_dir / "overlays"
        self.overlay_dir.mkdir(parents=True, exist_ok=True)
        self.block_metrics: list[BlockMetric] = []
        self.frame_metrics: list[FrameMetric] = []

    def run(self) -> None:
        sidecars = sorted(Path(self.args.sidecar_root).glob("data/chunk-*/episode_*.npz"))
        if not sidecars:
            raise FileNotFoundError(f"No sidecars found under {self.args.sidecar_root}")
        if self.args.max_episodes > 0:
            sidecars = sidecars[: self.args.max_episodes]
        for sidecar_path in sidecars:
            self._analyze_episode(sidecar_path)
        self._write_outputs()

    def _analyze_episode(self, sidecar_path: Path) -> None:
        with np.load(sidecar_path, allow_pickle=False) as payload:
            flow_3d = payload["flow_3d"]
            valid_mask = payload["valid_mask"]
            intrinsics = payload["intrinsics"].astype(np.float32)
            source_parquet = Path(str(payload["source_parquet"]))
            episode_index = int(payload["episode_index"])
            task_index = int(payload["task_index"])
        table = pq.read_table(source_parquet, columns=["image", "depth", "depth_shape", "depth_dtype", "frame_index"])
        data = table.to_pydict()
        depth_shape = tuple(data["depth_shape"][0])
        depth_dtype = data["depth_dtype"][0]
        max_frames = min(flow_3d.shape[0], len(data["image"]) - 1)
        frame_ids = np.linspace(0, max_frames - 1, num=min(self.args.max_frames_per_episode, max_frames), dtype=int)
        for t in frame_ids:
            image = decode_image(data["image"][t])
            depth = decode_depth(data["depth"][t], depth_shape, depth_dtype)
            self._analyze_frame(
                image=image,
                depth=depth,
                flow_3d=flow_3d[t],
                valid_mask=valid_mask[t],
                intrinsics=intrinsics,
                episode_index=episode_index,
                task_index=task_index,
                frame_index=int(data["frame_index"][t]),
            )

    def _analyze_frame(
        self,
        *,
        image: np.ndarray,
        depth: np.ndarray,
        flow_3d: np.ndarray,
        valid_mask: np.ndarray,
        intrinsics: np.ndarray,
        episode_index: int,
        task_index: int,
        frame_index: int,
    ) -> None:
        points0 = backproject_camera(depth, intrinsics)
        points1 = points0 + flow_3d
        flow_mag = np.linalg.norm(flow_3d, axis=-1)
        mask = (
            valid_mask.astype(bool)
            & np.isfinite(depth)
            & (depth >= self.args.min_depth)
            & (depth <= self.args.max_depth if self.args.max_depth > 0 else True)
            & np.isfinite(flow_3d).all(axis=-1)
            & (flow_mag >= self.args.flow_mag_threshold)
        )
        components = connected_components(mask, self.args.min_cluster_size)
        largest = max((int(comp.sum()) for comp in components), default=0)
        mean_valid_ratio = float(mask.mean())
        mean_flow_mag = float(flow_mag[mask].mean()) if mask.any() else 0.0
        self.frame_metrics.append(
            FrameMetric(
                episode_index=episode_index,
                task_index=task_index,
                frame_index=frame_index,
                candidate_blocks=len(components),
                largest_block_pixels=largest,
                mean_valid_ratio=mean_valid_ratio,
                mean_flow_mag=mean_flow_mag,
            )
        )
        if self.args.save_overlays:
            self._save_overlay(image, depth, flow_mag, components, episode_index, task_index, frame_index)
        for block_id, comp in enumerate(sorted(components, key=lambda m: int(m.sum()), reverse=True)[: self.args.max_clusters]):
            ys, xs = np.where(comp)
            p0 = points0[ys, xs]
            p1 = points1[ys, xs]
            _, trans_res = fit_translation(p0, p1)
            _, _, rigid_res = fit_rigid(p0, p1)
            trans_mean = float(trans_res.mean())
            rigid_mean = float(rigid_res.mean())
            improvement = 100.0 * max(0.0, trans_mean - rigid_mean) / max(trans_mean, 1e-6)
            best_model = "single_rigid" if rigid_mean < trans_mean else "translation"
            self.block_metrics.append(
                BlockMetric(
                    episode_index=episode_index,
                    task_index=task_index,
                    frame_index=frame_index,
                    block_id=block_id,
                    pixel_count=int(comp.sum()),
                    mean_depth=float(depth[comp].mean()),
                    mean_flow_mag=float(flow_mag[comp].mean()),
                    translation_residual_mean=trans_mean,
                    translation_residual_median=float(np.median(trans_res)),
                    rigid_residual_mean=rigid_mean,
                    rigid_residual_median=float(np.median(rigid_res)),
                    rigid_improvement_pct=float(improvement),
                    best_model=best_model,
                )
            )

    def _save_overlay(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        flow_mag: np.ndarray,
        components: list[np.ndarray],
        episode_index: int,
        task_index: int,
        frame_index: int,
    ) -> None:
        heat = speed_to_heatmap(flow_mag)
        overlay = image.copy()
        alpha = 0.45
        finite = np.isfinite(flow_mag)
        overlay[finite] = (overlay[finite] * (1 - alpha) + heat[finite] * alpha).astype(np.uint8)
        pil = Image.fromarray(overlay)
        draw = ImageDraw.Draw(pil)
        palette = [(0, 255, 0), (255, 200, 0), (0, 255, 255), (255, 0, 255), (255, 0, 0)]
        for block_id, comp in enumerate(sorted(components, key=lambda m: int(m.sum()), reverse=True)[: self.args.max_clusters]):
            ys, xs = np.where(comp)
            y0, y1 = int(ys.min()), int(ys.max())
            x0, x1 = int(xs.min()), int(xs.max())
            color = palette[block_id % len(palette)]
            draw.rectangle((x0, y0, x1, y1), outline=color, width=2)
            draw.text((x0 + 2, y0 + 2), f"{block_id}:{int(comp.sum())}", fill=color)
        out = self.overlay_dir / f"overlay_motion_blocks_ep{episode_index:06d}_task{task_index}_f{frame_index:03d}.png"
        pil.save(out)

        depth_img = np.clip(depth, self.args.min_depth, np.quantile(depth[np.isfinite(depth)], 0.98) if np.isfinite(depth).any() else 1.0)
        depth_norm = (depth_img - depth_img.min()) / max(float(depth_img.max() - depth_img.min()), 1e-6)
        depth_rgb = np.stack([(255 * depth_norm).astype(np.uint8)] * 3, axis=-1)
        depth_pil = Image.fromarray(depth_rgb)
        depth_pil.save(self.overlay_dir / f"overlay_flow_depth_ep{episode_index:06d}_task{task_index}_f{frame_index:03d}.png")

    def _write_outputs(self) -> None:
        block_csv = self.output_dir / "motion_block_metrics.csv"
        with block_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(self.block_metrics[0]).keys()) if self.block_metrics else list(BlockMetric.__dataclass_fields__.keys()))
            writer.writeheader()
            for row in self.block_metrics:
                writer.writerow(asdict(row))

        frame_csv = self.output_dir / "frame_metrics.csv"
        with frame_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(self.frame_metrics[0]).keys()) if self.frame_metrics else list(FrameMetric.__dataclass_fields__.keys()))
            writer.writeheader()
            for row in self.frame_metrics:
                writer.writerow(asdict(row))

        summary = self._build_summary()
        (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        (self.output_dir / "recommendation.md").write_text(self._build_recommendation(summary), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"Saved outputs to {self.output_dir}")

    def _build_summary(self) -> dict:
        trans = np.array([m.translation_residual_mean for m in self.block_metrics], dtype=np.float32)
        rigid = np.array([m.rigid_residual_mean for m in self.block_metrics], dtype=np.float32)
        improvements = np.array([m.rigid_improvement_pct for m in self.block_metrics], dtype=np.float32)
        best_counts: dict[str, int] = {}
        for metric in self.block_metrics:
            best_counts[metric.best_model] = best_counts.get(metric.best_model, 0) + 1
        return {
            "output_dir": str(self.output_dir),
            "frames_analyzed": len(self.frame_metrics),
            "blocks_analyzed": len(self.block_metrics),
            "translation_residual_mean": float(trans.mean()) if trans.size else None,
            "rigid_residual_mean": float(rigid.mean()) if rigid.size else None,
            "rigid_improvement_mean_pct": float(improvements.mean()) if improvements.size else None,
            "best_model_counts": best_counts,
        }

    def _build_recommendation(self, summary: dict) -> str:
        rigid_better = summary["best_model_counts"].get("single_rigid", 0)
        total = max(summary["blocks_analyzed"], 1)
        frac = rigid_better / total
        if frac >= 0.7:
            recommendation = "Use at least single-rigid motion as the primary Gaussian motion function; dense translation-only looks too weak."
        elif frac >= 0.4:
            recommendation = "Single-rigid often helps; a part-aware SE(3) + residual parameterization is a strong next step."
        else:
            recommendation = "Translation-only remains competitive on this subset; collect broader evidence before replacing dense deltas."
        lines = [
            "# Recommendation",
            "",
            f"- Blocks analyzed: {summary['blocks_analyzed']}",
            f"- Mean translation residual: {summary['translation_residual_mean']}",
            f"- Mean rigid residual: {summary['rigid_residual_mean']}",
            f"- Mean rigid improvement (%): {summary['rigid_improvement_mean_pct']}",
            f"- Best-model counts: {summary['best_model_counts']}",
            "",
            recommendation,
            "",
            "Future direction: if rigid consistently wins on larger blocks, extend this script with multi-rigid clustering and then consider an articulation + residual Gaussian motion model.",
        ]
        return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze motion blocks from LIBERO flow sidecars")
    parser.add_argument("--sidecar-root", type=str, required=True, help="Root directory containing flow sidecars")
    parser.add_argument("--output-dir", type=str, default="/tmp", help="Directory to write analysis outputs")
    parser.add_argument("--max-episodes", type=int, default=5)
    parser.add_argument("--max-frames-per-episode", type=int, default=4)
    parser.add_argument("--flow-mag-threshold", type=float, default=0.08)
    parser.add_argument("--min-depth", type=float, default=0.0)
    parser.add_argument("--max-depth", type=float, default=2.0)
    parser.add_argument("--min-cluster-size", type=int, default=200)
    parser.add_argument("--max-clusters", type=int, default=4)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--save-overlays", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.sample_seed)
    MotionBlockAnalyzer(args).run()


if __name__ == "__main__":
    main()
