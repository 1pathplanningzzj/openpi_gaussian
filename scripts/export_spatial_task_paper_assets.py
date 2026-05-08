#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("HF_HOME", "/tmp/huggingface")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/huggingface/datasets")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter
import torch

import openpi.models.model as model_lib
import openpi.training.config as training_config
import openpi.training.data_loader as data_loader
import visualize_world_model_predictions as viz


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export one representative world-model visualization case for each of the 10 LIBERO spatial tasks. "
            "Each task folder contains 6 images: current GT RGB/depth, current reconstruction RGB/pred depth, "
            "and ghosted future RGB/depth overlays for t+1..t+5."
        )
    )
    parser.add_argument("--config-name", default="pi05_libero")
    parser.add_argument(
        "--checkpoint-dir",
        default="/data/zijianzhang/train_ckpts/pi05_libero/gaussian_world_model_exp0413",
    )
    parser.add_argument("--checkpoint-step", type=int, default=39000)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "data_subsets" / "libero_tasks_0_9_with_depth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "paper_framework" / "gaussian_world_model_exp0413_step39000_spatial10_tasks",
    )
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--frame-index", type=int, default=None)
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--write-directly-to-output-dir", action="store_true")
    parser.add_argument("--future-count", type=int, default=5)
    parser.add_argument("--min-history-frames", type=int, default=10)
    parser.add_argument("--thumbnail-size", type=int, default=64)
    parser.add_argument("--evolution-style", choices=["default", "deep"], default="default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _safe_slug(text: str, *, max_len: int = 56) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    slug = "_".join(part for part in slug.split("_") if part)
    return slug[:max_len].rstrip("_") or "task"


def _select_representative_cases(
    dataset_root: Path,
    *,
    future_count: int,
    min_history_frames: int,
    thumbnail_size: int,
) -> list[dict]:
    tasks = _read_jsonl(dataset_root / "meta" / "tasks.jsonl")
    episodes = _read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    task_name_to_index = {record["task"]: int(record["task_index"]) for record in tasks}
    episode_start_index = 0
    task_to_candidate: dict[int, dict] = {}

    for episode in episodes:
        episode_index = int(episode["episode_index"])
        task_name = episode["tasks"][0]
        task_index = task_name_to_index[task_name]
        episode_length = int(episode["length"])
        max_frame = episode_length - future_count - 1
        if max_frame < min_history_frames:
            episode_start_index += episode_length
            continue

        target_frame = min(max_frame, max(min_history_frames, int(round(episode_length * 0.65))))
        candidate = {
            "task_index": task_index,
            "task_name": task_name,
            "episode_index": episode_index,
            "frame_index": int(target_frame),
            "global_index": int(episode_start_index + target_frame),
            "selection_score": float(episode_length),
            "episode_length": episode_length,
        }
        best = task_to_candidate.get(task_index)
        if best is None or candidate["episode_length"] > best["episode_length"]:
            task_to_candidate[task_index] = candidate

        episode_start_index += episode_length

    selected = [task_to_candidate[int(task["task_index"])] for task in tasks]
    return selected


def _select_explicit_case(args: argparse.Namespace) -> list[dict]:
    if (args.episode_index is None) != (args.frame_index is None):
        raise ValueError("`--episode-index` and `--frame-index` must be provided together.")
    if args.episode_index is None:
        return []
    task_name = args.task_name or f"episode_{args.episode_index}_frame_{args.frame_index}"
    global_index = _lookup_global_index_from_metadata(args.dataset_root.resolve(), int(args.episode_index), int(args.frame_index))
    return [
        {
            "task_index": int(args.task_index) if args.task_index is not None else -1,
            "task_name": task_name,
            "episode_index": int(args.episode_index),
            "frame_index": int(args.frame_index),
            "global_index": global_index,
            "selection_strategy": "explicit_episode_frame",
            "selection_score": float(args.frame_index),
            "episode_length": None,
        }
    ]


def _lookup_global_index_from_metadata(dataset_root: Path, episode_index: int, frame_index: int) -> int | None:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        return None

    offset = 0
    for record in _read_jsonl(episodes_path):
        current_episode = int(record["episode_index"])
        length = int(record["length"])
        if current_episode == episode_index:
            if frame_index < 0 or frame_index >= length:
                raise ValueError(
                    f"frame_index={frame_index} is out of bounds for episode_index={episode_index} with length={length}"
                )
            return offset + frame_index
        offset += length

    raise ValueError(f"episode_index={episode_index} not found in {episodes_path}")


def _future_palette() -> np.ndarray:
    return np.asarray(
        [
            [244, 86, 72],
            [252, 153, 70],
            [247, 206, 70],
            [118, 214, 109],
            [79, 189, 247],
        ],
        dtype=np.float32,
    )


def _normalize_change_map(change_map: np.ndarray, *, low_pct: float = 68.0, high_pct: float = 99.2) -> np.ndarray:
    finite = np.isfinite(change_map)
    if not finite.any():
        return np.zeros_like(change_map, dtype=np.float32)

    values = change_map[finite].astype(np.float32, copy=False)
    low = float(np.percentile(values, low_pct))
    high = float(np.percentile(values, high_pct))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(values.mean())
        high = float(values.max()) + 1e-6
    norm = np.clip((change_map.astype(np.float32) - low) / max(high - low, 1e-6), 0.0, 1.0)
    norm[~finite] = 0.0
    return norm


def _blur_map(array: np.ndarray, radius: float) -> np.ndarray:
    image = Image.fromarray(np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8))
    image = image.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(image, dtype=np.float32) / 255.0


def _get_evolution_style_config(style: str) -> dict[str, float]:
    if style == "deep":
        return {
            "rgb_low_pct": 74.0,
            "rgb_high_pct": 99.6,
            "rgb_motion_threshold": 0.12,
            "rgb_motion_power": 0.74,
            "rgb_overlay_mix_base": 0.52,
            "rgb_overlay_mix_step": 0.09,
            "rgb_halo_base": 0.14,
            "rgb_halo_step": 0.07,
            "rgb_alpha_base": 0.05,
            "rgb_alpha_motion_base": 0.34,
            "rgb_alpha_step": 0.06,
            "rgb_alpha_max": 0.82,
            "depth_low_pct": 66.0,
            "depth_high_pct": 99.0,
            "depth_motion_threshold": 0.08,
            "depth_motion_power": 0.72,
            "depth_vis_mix": 0.18,
            "depth_halo_base": 0.18,
            "depth_halo_step": 0.07,
            "depth_alpha_base": 0.04,
            "depth_alpha_motion_base": 0.42,
            "depth_alpha_step": 0.06,
            "depth_alpha_max": 0.84,
        }
    return {
        "rgb_low_pct": 78.0,
        "rgb_high_pct": 99.6,
        "rgb_motion_threshold": 0.18,
        "rgb_motion_power": 0.82,
        "rgb_overlay_mix_base": 0.40,
        "rgb_overlay_mix_step": 0.08,
        "rgb_halo_base": 0.08,
        "rgb_halo_step": 0.05,
        "rgb_alpha_base": 0.03,
        "rgb_alpha_motion_base": 0.24,
        "rgb_alpha_step": 0.05,
        "rgb_alpha_max": 0.72,
        "depth_low_pct": 72.0,
        "depth_high_pct": 99.0,
        "depth_motion_threshold": 0.12,
        "depth_motion_power": 0.78,
        "depth_vis_mix": 0.30,
        "depth_halo_base": 0.12,
        "depth_halo_step": 0.05,
        "depth_alpha_base": 0.02,
        "depth_alpha_motion_base": 0.34,
        "depth_alpha_step": 0.05,
        "depth_alpha_max": 0.78,
    }


def _build_rgb_evolution_overlay(
    base_rgb: np.ndarray,
    overlay_rgb_list: list[np.ndarray],
    *,
    style: str = "default",
) -> Image.Image:
    if not overlay_rgb_list:
        return Image.fromarray(base_rgb.astype(np.uint8))

    config = _get_evolution_style_config(style)
    palette = _future_palette()
    canvas = base_rgb.astype(np.float32).copy()
    anchor_rgb = base_rgb.astype(np.float32)
    base_rgb_float = base_rgb.astype(np.float32)

    for idx, overlay_rgb in enumerate(overlay_rgb_list):
        overlay_float = overlay_rgb.astype(np.float32)
        color = palette[min(idx, len(palette) - 1)]

        diff_to_prev = np.abs(overlay_float - anchor_rgb).mean(axis=-1)
        diff_to_base = np.abs(overlay_float - base_rgb_float).mean(axis=-1)
        motion = _normalize_change_map(
            np.maximum(diff_to_base, 0.7 * diff_to_prev),
            low_pct=config["rgb_low_pct"],
            high_pct=config["rgb_high_pct"],
        )
        motion = np.maximum(motion, _blur_map(motion, radius=1.2 + 0.25 * idx))
        motion = np.clip((motion - config["rgb_motion_threshold"]) / (1.0 - config["rgb_motion_threshold"]), 0.0, 1.0)
        motion = motion ** config["rgb_motion_power"]

        overlay_mix = config["rgb_overlay_mix_base"] + config["rgb_overlay_mix_step"] * idx
        tinted = overlay_float * (1.0 - overlay_mix) + color[None, None, :] * overlay_mix
        tinted_img = Image.fromarray(np.clip(tinted, 0.0, 255.0).astype(np.uint8))
        tinted_img = tinted_img.filter(ImageFilter.GaussianBlur(radius=0.8 + 0.25 * idx))
        tinted = np.asarray(tinted_img, dtype=np.float32)

        halo = _blur_map(motion, radius=1.8 + 0.25 * idx)[..., None]
        halo_color = color[None, None, :] * (config["rgb_halo_base"] + config["rgb_halo_step"] * idx)
        canvas = np.clip(canvas + halo * halo_color, 0.0, 255.0)

        alpha_map = np.clip(
            config["rgb_alpha_base"] + (config["rgb_alpha_motion_base"] + config["rgb_alpha_step"] * idx) * motion,
            0.0,
            config["rgb_alpha_max"],
        )[..., None]
        canvas = canvas * (1.0 - alpha_map) + tinted * alpha_map
        anchor_rgb = overlay_float

    return Image.fromarray(np.clip(canvas, 0.0, 255.0).astype(np.uint8))


def _build_depth_evolution_overlay(
    base_depth_vis: np.ndarray,
    base_depth: np.ndarray,
    overlay_depth_list: list[np.ndarray],
    *,
    vmin: float,
    vmax: float,
    style: str = "default",
) -> Image.Image:
    if not overlay_depth_list:
        return Image.fromarray(base_depth_vis.astype(np.uint8))

    config = _get_evolution_style_config(style)
    palette = _future_palette()
    canvas = base_depth_vis.astype(np.float32).copy()
    base_depth_float = base_depth.astype(np.float32)
    anchor_depth = base_depth_float

    for idx, overlay_depth in enumerate(overlay_depth_list):
        overlay_depth_float = overlay_depth.astype(np.float32)
        overlay_depth_vis = _colorize_depth(overlay_depth_float, vmin=vmin, vmax=vmax).astype(np.float32)
        color = palette[min(idx, len(palette) - 1)]

        diff_to_base = np.abs(overlay_depth_float - base_depth_float)
        diff_to_prev = np.abs(overlay_depth_float - anchor_depth)
        motion = _normalize_change_map(
            np.maximum(diff_to_base, 0.9 * diff_to_prev),
            low_pct=config["depth_low_pct"],
            high_pct=config["depth_high_pct"],
        )
        motion = np.maximum(motion, _blur_map(motion, radius=1.4 + 0.3 * idx))
        motion = np.clip(
            (motion - config["depth_motion_threshold"]) / (1.0 - config["depth_motion_threshold"]),
            0.0,
            1.0,
        )
        motion = motion ** config["depth_motion_power"]

        highlight = overlay_depth_vis * config["depth_vis_mix"] + color[None, None, :] * (1.0 - config["depth_vis_mix"])
        highlight_img = Image.fromarray(np.clip(highlight, 0.0, 255.0).astype(np.uint8))
        highlight_img = highlight_img.filter(ImageFilter.GaussianBlur(radius=1.0 + 0.2 * idx))
        highlight = np.asarray(highlight_img, dtype=np.float32)

        halo = _blur_map(motion, radius=2.0 + 0.3 * idx)[..., None]
        canvas = np.clip(
            canvas + halo * color[None, None, :] * (config["depth_halo_base"] + config["depth_halo_step"] * idx),
            0.0,
            255.0,
        )

        alpha_map = np.clip(
            config["depth_alpha_base"] + (config["depth_alpha_motion_base"] + config["depth_alpha_step"] * idx) * motion,
            0.0,
            config["depth_alpha_max"],
        )[..., None]
        canvas = canvas * (1.0 - alpha_map) + highlight * alpha_map
        anchor_depth = overlay_depth_float

    return Image.fromarray(np.clip(canvas, 0.0, 255.0).astype(np.uint8))


def _make_depth_visuals(frames: list[dict]) -> tuple[float, float]:
    depth_vmin, depth_vmax, _ = viz._collect_depth_stats(frames)
    return depth_vmin, depth_vmax


def _colorize_depth(depth: np.ndarray, *, vmin: float, vmax: float) -> np.ndarray:
    return viz._colorize_map(
        depth,
        vmin=vmin,
        vmax=vmax,
        cmap_name="viridis",
        invalid_mask=~np.isfinite(depth),
    )


def _resize_hw3(array: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(array.astype(np.uint8))
    image = image.resize((target_hw[1], target_hw[0]), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def _colorize_flow_xyz(flow_xyz: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    flow = flow_xyz.astype(np.float32, copy=False)
    if valid_mask is None:
        valid_mask = np.isfinite(flow).all(axis=-1)
    else:
        valid_mask = valid_mask & np.isfinite(flow).all(axis=-1)

    valid_values = np.abs(flow[valid_mask])
    scale = float(np.percentile(valid_values, 99.0)) if valid_values.size > 0 else 1.0
    scale = max(scale, 1e-6)
    rgb = np.clip((flow / (2.0 * scale)) + 0.5, 0.0, 1.0)
    rgb = (rgb * 255.0).astype(np.uint8)
    rgb[~valid_mask] = 0
    return rgb


def _extract_pred_flow_xyz(gaussian_params: dict, target_hw: tuple[int, int]) -> np.ndarray | None:
    raw_delta_xyz = gaussian_params.get("raw_delta_xyz")
    if raw_delta_xyz is None:
        return None
    delta = raw_delta_xyz[0].detach().float().cpu().numpy()
    if delta.ndim != 2 or delta.shape[-1] != 3:
        return None
    num_points = delta.shape[0]
    grid_size = int(round(num_points ** 0.5))
    if grid_size * grid_size != num_points:
        return None
    delta = delta.reshape(grid_size, grid_size, 3)
    rgb = _colorize_flow_xyz(delta)
    if rgb.shape[:2] != target_hw:
        rgb = _resize_hw3(rgb, target_hw)
    return rgb


def _extract_gt_flow_xyz(future_target, target_hw: tuple[int, int]) -> np.ndarray | None:
    flow = getattr(future_target, "flow_3d", None)
    mask = getattr(future_target, "flow_valid_mask", None)
    if flow is None:
        return None
    flow_np = flow[0].detach().float().cpu().numpy()
    if flow_np.ndim != 3 or flow_np.shape[-1] != 3:
        return None
    mask_np = None
    if mask is not None:
        mask_np = mask[0].detach().cpu().numpy().astype(bool)
    rgb = _colorize_flow_xyz(flow_np, mask_np)
    if rgb.shape[:2] != target_hw:
        rgb = _resize_hw3(rgb, target_hw)
    return rgb


def _build_flow_overlay_sequence(
    base_rgb: np.ndarray,
    flow_xyz_list: list[np.ndarray],
    *,
    style: str = "default",
) -> Image.Image | None:
    valid_flow_rgbs = [flow for flow in flow_xyz_list if flow is not None]
    if not valid_flow_rgbs:
        return None
    return _build_rgb_evolution_overlay(base_rgb, valid_flow_rgbs, style=style)


def _time_labels(max_future_count: int = 5) -> list[str]:
    return ["t", *[f"t+{idx}" for idx in range(1, max_future_count + 1)]]


def _blank_rgb(target_hw: tuple[int, int], color: tuple[int, int, int] = (245, 245, 245)) -> np.ndarray:
    target_h, target_w = target_hw
    return np.full((target_h, target_w, 3), color, dtype=np.uint8)


def _ensure_rgb_array(image: np.ndarray, target_hw: tuple[int, int] | None = None) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3:
        raise ValueError(f"Expected image array with 2 or 3 dims, got {array.shape}")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] > 3:
        array = array[..., :3]
    array = np.clip(array, 0, 255).astype(np.uint8)
    if target_hw is not None and array.shape[:2] != target_hw:
        array = _resize_hw3(array, target_hw)
    return array


def _save_time_strip(path: Path, images: list[np.ndarray], labels: list[str]) -> None:
    if len(images) != len(labels):
        raise ValueError(f"Expected {len(labels)} sequence images, got {len(images)}")
    arrays = [_ensure_rgb_array(image) for image in images]
    total_width = sum(array.shape[1] for array in arrays)
    max_height = max(array.shape[0] for array in arrays)
    canvas = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))
    x = 0
    for array in arrays:
        canvas.paste(Image.fromarray(array), (x, 0))
        x += array.shape[1]
    canvas.save(path)


def _tensor_to_scalar_map(value, target_hw: tuple[int, int]) -> np.ndarray | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        array = value.detach().float().cpu().numpy()
    else:
        array = np.asarray(value, dtype=np.float32)
    array = np.squeeze(array)
    if array.ndim != 2:
        return None
    array = array.astype(np.float32, copy=False)
    if array.shape != target_hw:
        array = viz._resize_scalar_map(array, target_hw)
    return array


def _colorize_signed_maps(maps: list[np.ndarray | None], target_hw: tuple[int, int]) -> list[np.ndarray]:
    values = []
    for delta_map in maps:
        if delta_map is None:
            continue
        finite = np.isfinite(delta_map)
        if finite.any():
            values.append(np.abs(delta_map[finite]).reshape(-1))
    vmax = float(np.percentile(np.concatenate(values), 99.0)) if values else 1.0
    vmax = max(vmax, 1e-6)
    colored = []
    for delta_map in maps:
        if delta_map is None:
            colored.append(_blank_rgb(target_hw))
            continue
        colored.append(
            viz._colorize_map(
                delta_map,
                vmin=-vmax,
                vmax=vmax,
                cmap_name="coolwarm",
                invalid_mask=~np.isfinite(delta_map),
            )
        )
    return colored


def _build_depth_sequence(
    maps: list[np.ndarray | None],
    *,
    target_hw: tuple[int, int],
    vmin: float,
    vmax: float,
) -> list[np.ndarray]:
    images = []
    for depth_map in maps:
        if depth_map is None:
            images.append(_blank_rgb(target_hw))
        else:
            images.append(_colorize_depth(depth_map, vmin=vmin, vmax=vmax))
    return images


def _save_case_assets(
    case_dir: Path,
    case: dict,
    frames: list[dict],
    export: dict | None = None,
    *,
    evolution_style: str = "default",
) -> dict[str, str]:
    depth_vmin, depth_vmax = _make_depth_visuals(frames)
    current_frame = frames[0]
    future_frames = [frame for frame in frames[1:] if frame["offset"] > 0][:5]
    time_labels = ["t", *[frame["label"] for frame in future_frames]]

    current_gt_depth_vis = _colorize_depth(current_frame["gt_depth"], vmin=depth_vmin, vmax=depth_vmax)
    current_pred_depth_vis = _colorize_depth(current_frame["pred_depth"], vmin=depth_vmin, vmax=depth_vmax)
    future_rgb_ghost = _build_rgb_evolution_overlay(
        current_frame["pred_rgb"],
        [frame["pred_rgb"] for frame in future_frames],
        style=evolution_style,
    )
    future_depth_ghost = _build_depth_evolution_overlay(
        current_pred_depth_vis,
        current_frame["pred_depth"],
        [frame["pred_depth"] for frame in future_frames],
        vmin=depth_vmin,
        vmax=depth_vmax,
        style=evolution_style,
    )

    assets = {
        "current_gt_rgb": case_dir / "01_current_gt_rgb.png",
        "current_gt_depth": case_dir / "02_current_gt_depth.png",
        "current_pred_rgb": case_dir / "03_current_reconstruction_rgb.png",
        "current_pred_depth": case_dir / "04_current_pred_depth.png",
        "future_pred_rgb": case_dir / "05_future_evolution_rgb.png",
        "future_pred_depth": case_dir / "06_future_evolution_depth.png",
    }
    extra_images: dict[str, str] = {}
    pred_flow_xyz_seq: list[np.ndarray | None] = []
    gt_flow_xyz_seq: list[np.ndarray | None] = []

    rgb_hw = current_frame["pred_rgb"].shape[:2]
    depth_hw = current_frame["gt_depth"].shape

    if export is not None and export.get("future_gaussian_params_seq") and export.get("future_targets"):
        pred_flow_xyz_seq = [
            _extract_pred_flow_xyz(gaussian_params, rgb_hw)
            for gaussian_params in export["future_gaussian_params_seq"][: len(future_frames)]
        ]
        gt_flow_xyz_seq = [
            _extract_gt_flow_xyz(future_target, rgb_hw)
            for future_target in export["future_targets"][: len(future_frames)]
        ]
        pred_flow_xyz = pred_flow_xyz_seq[0] if pred_flow_xyz_seq else None
        gt_flow_xyz = gt_flow_xyz_seq[0] if gt_flow_xyz_seq else None
        pred_flow_overlay = _build_flow_overlay_sequence(current_frame["pred_rgb"], pred_flow_xyz_seq, style=evolution_style)
        gt_flow_overlay = _build_flow_overlay_sequence(current_frame["gt_rgb"], gt_flow_xyz_seq, style=evolution_style)
        if pred_flow_xyz is not None:
            pred_flow_path = case_dir / "07_future_flow_xyz_pred_tplus1.png"
            Image.fromarray(pred_flow_xyz).save(pred_flow_path)
            extra_images["future_flow_xyz_pred_tplus1"] = str(pred_flow_path)
        if gt_flow_xyz is not None:
            gt_flow_path = case_dir / "08_future_flow_xyz_gt_tplus1.png"
            Image.fromarray(gt_flow_xyz).save(gt_flow_path)
            extra_images["future_flow_xyz_gt_tplus1"] = str(gt_flow_path)
        if pred_flow_overlay is not None:
            pred_flow_overlay_path = case_dir / "09_future_flow_xyz_pred_overlay.png"
            pred_flow_overlay.save(pred_flow_overlay_path)
            extra_images["future_flow_xyz_pred_overlay"] = str(pred_flow_overlay_path)
        if gt_flow_overlay is not None:
            gt_flow_overlay_path = case_dir / "10_future_flow_xyz_gt_overlay.png"
            gt_flow_overlay.save(gt_flow_overlay_path)
            extra_images["future_flow_xyz_gt_overlay"] = str(gt_flow_overlay_path)

    Image.fromarray(current_frame["gt_rgb"]).save(assets["current_gt_rgb"])
    Image.fromarray(current_gt_depth_vis).save(assets["current_gt_depth"])
    Image.fromarray(current_frame["pred_rgb"]).save(assets["current_pred_rgb"])
    Image.fromarray(current_pred_depth_vis).save(assets["current_pred_depth"])
    future_rgb_ghost.save(assets["future_pred_rgb"])
    future_depth_ghost.save(assets["future_pred_depth"])

    sequence_assets = {
        "sequence_gt_rgb_agent": case_dir / "11_sequence_gt_rgb_agent.png",
        "sequence_gt_depth": case_dir / "12_sequence_gt_depth.png",
        "sequence_rendered_rgb": case_dir / "13_sequence_rendered_rgb.png",
        "sequence_rendered_depth": case_dir / "14_sequence_rendered_depth.png",
        "sequence_aux_depth": case_dir / "15_sequence_aux_depth.png",
        "sequence_incremental_depth": case_dir / "16_sequence_incremental_depth.png",
        "sequence_pred_flow_xyz": case_dir / "17_sequence_pred_flow_xyz.png",
        "sequence_gt_flow_xyz": case_dir / "18_sequence_gt_flow_xyz.png",
    }

    gt_rgb_seq = [_ensure_rgb_array(current_frame["gt_rgb"], rgb_hw)] + [
        _ensure_rgb_array(frame["gt_rgb"], rgb_hw) for frame in future_frames
    ]
    rendered_rgb_seq = [_ensure_rgb_array(current_frame["pred_rgb"], rgb_hw)] + [
        _ensure_rgb_array(frame["pred_rgb"], rgb_hw) for frame in future_frames
    ]
    gt_depth_seq = _build_depth_sequence(
        [current_frame["gt_depth"], *[frame["gt_depth"] for frame in future_frames]],
        target_hw=depth_hw,
        vmin=depth_vmin,
        vmax=depth_vmax,
    )
    rendered_depth_maps = [
        _tensor_to_scalar_map(current_frame["pred_depth"], depth_hw),
        *[_tensor_to_scalar_map(frame["pred_depth"], depth_hw) for frame in future_frames],
    ]
    rendered_depth_seq = _build_depth_sequence(
        rendered_depth_maps,
        target_hw=depth_hw,
        vmin=depth_vmin,
        vmax=depth_vmax,
    )

    aux_depth_maps: list[np.ndarray | None] = [rendered_depth_maps[0]]
    aux_depth_seq = export.get("future_depth_aux_seq", []) if export is not None else []
    for aux_depth in aux_depth_seq[: len(future_frames)]:
        aux_depth_maps.append(_tensor_to_scalar_map(aux_depth, depth_hw))
    while len(aux_depth_maps) < len(time_labels):
        aux_depth_maps.append(None)
    aux_depth_images = _build_depth_sequence(
        aux_depth_maps,
        target_hw=depth_hw,
        vmin=depth_vmin,
        vmax=depth_vmax,
    )

    base_pred_depth = rendered_depth_maps[0]
    future_gaussian_params_seq = export.get("future_gaussian_params_seq", []) if export is not None else []
    incremental_depth_maps: list[np.ndarray | None] = [np.zeros(depth_hw, dtype=np.float32)]
    for idx, pred_depth in enumerate(rendered_depth_maps[1:]):
        delta_depth = None
        if idx < len(future_gaussian_params_seq):
            delta_depth = _tensor_to_scalar_map(future_gaussian_params_seq[idx].get("depth_delta_map"), depth_hw)
        if delta_depth is None and pred_depth is not None and base_pred_depth is not None:
            delta_depth = pred_depth - base_pred_depth
        incremental_depth_maps.append(delta_depth)
    incremental_depth_images = _colorize_signed_maps(incremental_depth_maps, depth_hw)

    pred_flow_seq = [_blank_rgb(rgb_hw)] + [
        _ensure_rgb_array(flow, rgb_hw) if flow is not None else _blank_rgb(rgb_hw)
        for flow in pred_flow_xyz_seq[: len(future_frames)]
    ]
    gt_flow_seq = [_blank_rgb(rgb_hw)] + [
        _ensure_rgb_array(flow, rgb_hw) if flow is not None else _blank_rgb(rgb_hw)
        for flow in gt_flow_xyz_seq[: len(future_frames)]
    ]
    while len(pred_flow_seq) < len(time_labels):
        pred_flow_seq.append(_blank_rgb(rgb_hw))
    while len(gt_flow_seq) < len(time_labels):
        gt_flow_seq.append(_blank_rgb(rgb_hw))

    _save_time_strip(sequence_assets["sequence_gt_rgb_agent"], gt_rgb_seq, time_labels)
    _save_time_strip(sequence_assets["sequence_gt_depth"], gt_depth_seq, time_labels)
    _save_time_strip(sequence_assets["sequence_rendered_rgb"], rendered_rgb_seq, time_labels)
    _save_time_strip(sequence_assets["sequence_rendered_depth"], rendered_depth_seq, time_labels)
    _save_time_strip(sequence_assets["sequence_aux_depth"], aux_depth_images, time_labels)
    _save_time_strip(sequence_assets["sequence_incremental_depth"], incremental_depth_images, time_labels)
    _save_time_strip(sequence_assets["sequence_pred_flow_xyz"], pred_flow_seq, time_labels)
    _save_time_strip(sequence_assets["sequence_gt_flow_xyz"], gt_flow_seq, time_labels)

    metadata = {
        "task_index": int(case["task_index"]),
        "task_name": case["task_name"],
        "episode_index": int(case["episode_index"]),
        "frame_index": int(case["frame_index"]),
        "global_index": int(case["global_index"]),
        "selection_strategy": case.get("selection_strategy", "longest_episode_at_65_percent_progress"),
        "selection_score": float(case["selection_score"]),
        "evolution_style": evolution_style,
        "future_overlay_palette_rgb": ["#f45648", "#fc9946", "#f7ce46", "#76d66d", "#4fbdf7"],
        "depth_range": [float(depth_vmin), float(depth_vmax)],
        "time_labels": time_labels,
        "visualization_schema": "0508_aux_incremental_flow_sequence_v1",
        "aux_depth_source": "export.future_depth_aux_seq",
        "incremental_depth_source": "rendered_depth_minus_current_rendered_depth",
        "images": (
            {name: str(path) for name, path in assets.items()}
            | extra_images
            | {name: str(path) for name, path in sequence_assets.items()}
        ),
    }
    with (case_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    return metadata


def _build_index(output_dir: Path, manifest_cases: list[dict]) -> None:
    tiles = []
    for case in manifest_cases:
        preview = Image.open(case["images"]["future_pred_rgb"]).convert("RGB")
        title_h = 48
        tile = Image.new("RGB", (preview.width, preview.height + title_h), color=(255, 255, 255))
        tile.paste(preview, (0, title_h))
        draw = ImageDraw.Draw(tile)
        draw.text((8, 6), f"task {case['task_index']}", fill=(0, 0, 0))
        draw.text((8, 24), case["task_name"], fill=(70, 70, 70))
        tiles.append(tile)

    if not tiles:
        return

    columns = 2
    rows = (len(tiles) + columns - 1) // columns
    tile_w = max(tile.width for tile in tiles)
    tile_h = max(tile.height for tile in tiles)
    gap = 16
    canvas = Image.new(
        "RGB",
        (columns * tile_w + (columns - 1) * gap, rows * tile_h + (rows - 1) * gap),
        color=(255, 255, 255),
    )

    for idx, tile in enumerate(tiles):
        row = idx // columns
        col = idx % columns
        x = col * (tile_w + gap)
        y = row * (tile_h + gap)
        canvas.paste(tile, (x, y))

    canvas.save(output_dir / "index.png")


def _load_transformed_dataset(config):
    data_config = config.data.create(config.assets_dirs, config.model)
    raw_dataset = data_loader.create_torch_dataset(
        data_config,
        config.model.action_horizon,
        config.model,
        use_single_frame_mode=getattr(config.model, "use_single_frame_mode", False),
    )
    transformed_dataset = data_loader.transform_dataset(raw_dataset, data_config)
    return transformed_dataset


def _override_data_roots(config, dataset_root: Path) -> object:
    subset_mapping_path = dataset_root / "meta" / "subset_mapping.json"
    source_flow_root = None
    if subset_mapping_path.exists():
        subset_mapping = json.loads(subset_mapping_path.read_text(encoding="utf-8"))
        source_flow_root = subset_mapping.get("source_flow_root")

    base_config = config.data.base_config or training_config.DataConfig()
    updated_base = dataclasses.replace(
        base_config,
        dataset_root=str(dataset_root),
        flow_root=source_flow_root or base_config.flow_root,
    )
    return dataclasses.replace(config, data=dataclasses.replace(config.data, base_config=updated_base))


def _load_sample_from_global_index(dataset, global_index: int):
    sample = dataset[int(global_index)]
    batch = data_loader._collate_fn([sample])
    batch = _to_torch_tensors(batch)
    observation = model_lib.Observation.from_dict(batch)
    actions = batch["actions"]
    return observation, actions


def _to_torch_tensors(obj):
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    if isinstance(obj, dict):
        return {key: _to_torch_tensors(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_torch_tensors(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_to_torch_tensors(value) for value in obj)
    return obj


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)

    if device.type != "cuda":
        raise RuntimeError("This export requires CUDA because Gaussian rendering is GPU-only.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_cases = _select_explicit_case(args)
    if selected_cases:
        print("Using explicitly requested rollout case...", flush=True)
        for case in selected_cases:
            print(
                f"  task {case['task_index']}: ep={case['episode_index']} frame={case['frame_index']}",
                flush=True,
            )
    else:
        print("Selecting one representative case for each spatial task...", flush=True)
        selected_cases = _select_representative_cases(
            args.dataset_root.resolve(),
            future_count=args.future_count,
            min_history_frames=args.min_history_frames,
            thumbnail_size=args.thumbnail_size,
        )
        for case in selected_cases:
            print(
                f"  task {case['task_index']}: ep={case['episode_index']} frame={case['frame_index']} "
                f"length={case['episode_length']}",
                flush=True,
            )

    checkpoint_path = viz._resolve_checkpoint_path(Path(args.checkpoint_dir), args.checkpoint_step)
    config = training_config.get_config(args.config_name)
    if getattr(config.model, "use_lpips", False):
        config = dataclasses.replace(config, model=dataclasses.replace(config.model, use_lpips=False))
    config = _override_data_roots(config, args.dataset_root.resolve())

    print(f"Loading checkpoint from {checkpoint_path} ...", flush=True)
    model = config.model.load_pytorch(config, str(checkpoint_path))
    model = model.to(device)
    model.eval()
    model.set_world_model_image_fusion(True)

    needs_dataset_lookup = any(case["global_index"] is not None for case in selected_cases)
    dataset = None
    if needs_dataset_lookup:
        print("Building transformed dataset once for direct sample lookup...", flush=True)
        dataset = _load_transformed_dataset(config)

    manifest = {
        "config_name": args.config_name,
        "checkpoint_path": str(checkpoint_path),
        "dataset_root": str(args.dataset_root.resolve()),
        "device": str(device),
        "future_count": int(args.future_count),
        "cases": [],
    }

    for case in selected_cases:
        if args.write_directly_to_output_dir and len(selected_cases) == 1:
            case_dir = output_dir
            folder_name = output_dir.name
        else:
            folder_name = f"task_{case['task_index']:02d}_{_safe_slug(case['task_name'])}"
            case_dir = output_dir / folder_name
            case_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"Exporting task {case['task_index']} -> {folder_name} "
            f"(episode {case['episode_index']}, frame {case['frame_index']})",
            flush=True,
        )

        if case["global_index"] is None:
            observation_cpu, actions_cpu, dataset_index = viz._load_specific_sample(
                config,
                episode_index=int(case["episode_index"]),
                frame_index=int(case["frame_index"]),
            )
            case["global_index"] = int(dataset_index)
        else:
            if dataset is None:
                raise RuntimeError("Dataset lookup requested, but transformed dataset was not initialized.")
            observation_cpu, actions_cpu = _load_sample_from_global_index(dataset, case["global_index"])
        observation = viz._to_device(observation_cpu, device)
        actions = torch.as_tensor(actions_cpu, device=device)

        with torch.inference_mode():
            export = model.export_future_rollout_gaussians(observation, actions=actions)
            frames = viz._prepare_frames(model, export, device)

        metadata = _save_case_assets(case_dir, case, frames, export, evolution_style=args.evolution_style)
        manifest["cases"].append(metadata)
        torch.cuda.empty_cache()

    if not (args.write_directly_to_output_dir and len(selected_cases) == 1):
        _build_index(output_dir, manifest["cases"])

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
