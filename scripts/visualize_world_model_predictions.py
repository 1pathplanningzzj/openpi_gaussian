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
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

import openpi.training.config as training_config
import openpi.training.data_loader as data_loader
import openpi.models.model as model_lib


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize current-frame reconstruction and future-frame predictions for the world model."
    )
    parser.add_argument("--config-name", default="pi05_libero")
    parser.add_argument(
        "--checkpoint-dir",
        default="/data/zijianzhang/train_ckpts/pi05_libero/gaussian_world_model_exp0413",
    )
    parser.add_argument("--checkpoint-step", type=int, default=39000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "paper_framework" / "gaussian_world_model_exp0413_step39000_viz",
    )
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--random-pool-size", type=int, default=96)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--frame-index", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _to_device(obj, device: torch.device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_device(v, device) for v in obj)
    if hasattr(obj, "replace") and hasattr(obj, "__dict__"):
        updates = {}
        for key, value in vars(obj).items():
            updates[key] = None if value is None else _to_device(value, device)
        return obj.replace(**updates)
    if hasattr(obj, "__dict__"):
        new_obj = type(obj).__new__(type(obj))
        for key, value in vars(obj).items():
            object.__setattr__(new_obj, key, None if value is None else _to_device(value, device))
        return new_obj
    return obj


def _slice_sample(obj, sample_idx: int):
    if torch.is_tensor(obj):
        return obj[sample_idx : sample_idx + 1]
    if isinstance(obj, dict):
        return {k: _slice_sample(v, sample_idx) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_slice_sample(v, sample_idx) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_slice_sample(v, sample_idx) for v in obj)
    if hasattr(obj, "replace") and hasattr(obj, "__dict__"):
        updates = {}
        for key, value in vars(obj).items():
            updates[key] = None if value is None else _slice_sample(value, sample_idx)
        return obj.replace(**updates)
    if hasattr(obj, "__dict__"):
        new_obj = type(obj).__new__(type(obj))
        for key, value in vars(obj).items():
            object.__setattr__(new_obj, key, None if value is None else _slice_sample(value, sample_idx))
        return new_obj
    return obj


def _cast_floating_tensors(obj, dtype: torch.dtype):
    if torch.is_tensor(obj):
        return obj.to(dtype=dtype) if torch.is_floating_point(obj) else obj
    if isinstance(obj, dict):
        return {k: _cast_floating_tensors(v, dtype) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cast_floating_tensors(v, dtype) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_cast_floating_tensors(v, dtype) for v in obj)
    return obj


def _resolve_checkpoint_path(checkpoint_dir: Path, step: int | None) -> Path:
    if step is not None:
        candidate = checkpoint_dir / str(step) / "model.safetensors"
        if not candidate.exists():
            raise FileNotFoundError(f"Checkpoint not found: {candidate}")
        return candidate
    step_dirs = sorted((p for p in checkpoint_dir.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))
    if not step_dirs:
        raise FileNotFoundError(f"No checkpoint steps found under {checkpoint_dir}")
    return step_dirs[-1] / "model.safetensors"


def _collect_random_samples(loader, count: int, *, max_pool_size: int, seed: int):
    pool = []
    global_idx = 0
    target_pool_size = max(count, max_pool_size)
    for observation, actions in loader:
        batch_size = actions.shape[0]
        batch_start = global_idx
        for local_idx in range(batch_size):
            sample_idx = batch_start + local_idx
            pool.append((_slice_sample(observation, local_idx), actions[local_idx : local_idx + 1].clone(), sample_idx))
            if len(pool) >= target_pool_size:
                break
        global_idx += batch_size
        if len(pool) >= target_pool_size:
            break

    if not pool:
        return []

    rng = np.random.default_rng(seed)
    pick_count = min(count, len(pool))
    picked = sorted(rng.choice(len(pool), size=pick_count, replace=False).tolist())
    return [pool[idx] for idx in picked]


def _find_dataset_index(raw_dataset, *, episode_index: int, frame_index: int) -> int:
    for dataset_index in range(len(raw_dataset)):
        sample = raw_dataset[dataset_index]
        sample_episode = int(sample["episode_index"])
        sample_frame = int(sample["frame_index"])
        if sample_episode == episode_index and sample_frame == frame_index:
            return dataset_index
    raise ValueError(
        f"Could not find sample for episode_index={episode_index}, frame_index={frame_index} in dataset of size {len(raw_dataset)}"
    )


def _load_specific_sample(
    config,
    *,
    episode_index: int,
    frame_index: int,
):
    data_config = config.data.create(config.assets_dirs, config.model)
    raw_dataset = data_loader.create_torch_dataset(
        data_config,
        config.model.action_horizon,
        config.model,
        use_single_frame_mode=getattr(config.model, "use_single_frame_mode", False),
    )
    dataset_index = _find_dataset_index(raw_dataset, episode_index=episode_index, frame_index=frame_index)
    transformed_dataset = data_loader.transform_dataset(raw_dataset, data_config)
    sample = transformed_dataset[dataset_index]
    batch = data_loader._collate_fn([sample])
    observation = model_lib.Observation.from_dict(batch)
    actions = batch["actions"]
    return observation, actions, dataset_index


def _ensure_hwc_image(image: torch.Tensor) -> np.ndarray:
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"Expected image tensor [C,H,W], got {tuple(image.shape)}")
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    elif image.shape[0] > 3:
        image = image[:3]
    image = image.detach().float().cpu().clamp(0.0, 1.0)
    return (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def _resize_rgb(image: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    if image.shape[:2] == (target_h, target_w):
        return image
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    tensor = F.interpolate(tensor.unsqueeze(0), size=target_hw, mode="bilinear", align_corners=False).squeeze(0)
    return (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def _resize_scalar_map(array: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    if array.shape == (target_h, target_w):
        return array.astype(np.float32, copy=False)
    tensor = torch.from_numpy(array).float().unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=target_hw, mode="bilinear", align_corners=False)
    return tensor.squeeze(0).squeeze(0).numpy().astype(np.float32)


def _extract_agent_rgb(observation) -> np.ndarray:
    if observation is None or not hasattr(observation, "images"):
        raise ValueError("Observation does not contain images")
    for key, value in observation.images.items():
        key_lower = key.lower()
        if (
            key == "image"
            or "agent" in key_lower
            or "high" in key_lower
            or "cam_high" in key_lower
            or key_lower.startswith("base")
            or "agentview" in key_lower
        ):
            image = value
            if image.ndim == 5:
                image = image[:, -1]
            if image.shape[1] != 3 and image.shape[-1] == 3:
                image = image.permute(0, 3, 1, 2)
            return _ensure_hwc_image((image + 1.0) / 2.0)
    raise KeyError(f"Could not find agent image key in {list(observation.images.keys())}")


def _extract_depth(observation) -> np.ndarray:
    if observation is None or not hasattr(observation, "depth") or observation.depth is None:
        raise ValueError("Observation does not contain depth")
    depth = observation.depth
    if depth.ndim == 5:
        depth = depth[:, -1]
    if depth.ndim == 4:
        depth = depth[0, 0]
    elif depth.ndim == 3:
        depth = depth[0]
    return depth.detach().float().cpu().numpy().astype(np.float32)


def _extract_prompt(observation) -> str | None:
    for key in ("prompt", "task_name", "language_instruction"):
        value = getattr(observation, key, None)
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)) and value and isinstance(value[0], str):
            return value[0]
    return None


def _render_agent_rgb(model, gaussian_params: dict, device: torch.device, target_hw: tuple[int, int]) -> np.ndarray:
    render_params = _cast_floating_tensors(gaussian_params, torch.float32)
    renders = model.render_gaussian_views_for_export(
        render_params,
        device=device,
        batch_size=1,
        agent_view=True,
        target_hw=target_hw,
    )
    return _resize_rgb(_ensure_hwc_image(renders["agent"]), target_hw)


def _colorize_map(array: np.ndarray, *, vmin: float, vmax: float, cmap_name: str, invalid_mask: np.ndarray | None = None) -> np.ndarray:
    denom = max(vmax - vmin, 1e-6)
    norm = np.clip((array - vmin) / denom, 0.0, 1.0)
    rgba = colormaps[cmap_name](norm)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    if invalid_mask is not None:
        rgb[invalid_mask] = 0
    return rgb


def _collect_depth_stats(frames: list[dict]) -> tuple[float, float, float]:
    depth_values = []
    depth_diffs = []
    for frame in frames:
        valid_gt = np.isfinite(frame["gt_depth"]) & (frame["gt_depth"] > 0.0)
        valid_pred = np.isfinite(frame["pred_depth"]) & (frame["pred_depth"] > 0.0)
        if valid_gt.any():
            depth_values.append(frame["gt_depth"][valid_gt])
        if valid_pred.any():
            depth_values.append(frame["pred_depth"][valid_pred])
        valid_both = valid_gt & valid_pred
        if valid_both.any():
            depth_diffs.append(np.abs(frame["pred_depth"] - frame["gt_depth"])[valid_both])
    if depth_values:
        merged = np.concatenate(depth_values)
        depth_vmin = float(np.percentile(merged, 1.0))
        depth_vmax = float(np.percentile(merged, 99.0))
    else:
        depth_vmin, depth_vmax = 0.0, 1.0
    if not np.isfinite(depth_vmin) or not np.isfinite(depth_vmax) or depth_vmax <= depth_vmin:
        depth_vmax = depth_vmin + 1.0
    if depth_diffs:
        diff_vmax = float(np.percentile(np.concatenate(depth_diffs), 99.0))
    else:
        diff_vmax = 1.0
    diff_vmax = max(diff_vmax, 1e-6)
    return depth_vmin, depth_vmax, diff_vmax


def _collect_rgb_diff_scale(frames: list[dict]) -> float:
    diffs = []
    for frame in frames:
        rgb_diff = np.abs(frame["pred_rgb"].astype(np.float32) - frame["gt_rgb"].astype(np.float32)).mean(axis=-1)
        diffs.append(rgb_diff.reshape(-1))
    if not diffs:
        return 1.0
    vmax = float(np.percentile(np.concatenate(diffs), 99.0))
    return max(vmax, 1.0)


def _make_panel(image: np.ndarray, title: str) -> Image.Image:
    title_h = 24
    panel = Image.new("RGB", (image.shape[1], image.shape[0] + title_h), color=(255, 255, 255))
    panel.paste(Image.fromarray(image), (0, title_h))
    draw = ImageDraw.Draw(panel)
    draw.text((6, 5), title, fill=(0, 0, 0))
    return panel


def _make_text_panel(width: int, height: int, lines: list[str]) -> Image.Image:
    panel = Image.new("RGB", (width, height), color=(250, 250, 250))
    draw = ImageDraw.Draw(panel)
    y = 8
    for line in lines:
        draw.text((8, y), line, fill=(0, 0, 0))
        y += 18
    return panel


def _stack_row(images: list[Image.Image]) -> Image.Image:
    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    row = Image.new("RGB", (width, height), color=(255, 255, 255))
    x = 0
    for image in images:
        row.paste(image, (x, 0))
        x += image.width
    return row


def _stack_column(images: list[Image.Image], gap: int = 8) -> Image.Image:
    width = max(image.width for image in images)
    height = sum(image.height for image in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    y = 0
    for image in images:
        canvas.paste(image, (0, y))
        y += image.height + gap
    return canvas


def _prepare_frames(model, export: dict, device: torch.device) -> list[dict]:
    frame_entries = []

    current_obs = export["current_observation"]
    current_rgb = _extract_agent_rgb(current_obs)
    current_depth = _extract_depth(current_obs)
    current_pred_rgb = _render_agent_rgb(model, export["base_gaussian_params"], device, current_rgb.shape[:2])
    current_pred_depth = export["base_gaussian_params"]["depth_map"][0, 0].detach().float().cpu().numpy()
    current_pred_depth = _resize_scalar_map(current_pred_depth, current_depth.shape)
    frame_entries.append(
        {
            "label": "current",
            "title": "Current Reconstruction",
            "offset": 0,
            "gt_rgb": current_rgb,
            "pred_rgb": current_pred_rgb,
            "gt_depth": current_depth,
            "pred_depth": current_pred_depth,
        }
    )

    for offset, future_target, gaussian_params in zip(
        export["future_offsets"], export["future_targets"], export["future_gaussian_params_seq"], strict=True
    ):
        gt_rgb = _extract_agent_rgb(future_target)
        gt_depth = _extract_depth(future_target)
        pred_rgb = _render_agent_rgb(model, gaussian_params, device, gt_rgb.shape[:2])
        pred_depth = gaussian_params["depth_map"][0, 0].detach().float().cpu().numpy()
        pred_depth = _resize_scalar_map(pred_depth, gt_depth.shape)
        frame_entries.append(
            {
                "label": f"t+{offset}",
                "title": f"Future Prediction t+{offset}",
                "offset": int(offset),
                "gt_rgb": gt_rgb,
                "pred_rgb": pred_rgb,
                "gt_depth": gt_depth,
                "pred_depth": pred_depth,
            }
        )

    return frame_entries


def _save_raw_arrays(sample_dir: Path, frame: dict) -> None:
    frame_dir = sample_dir / frame["label"].replace("+", "plus")
    frame_dir.mkdir(parents=True, exist_ok=True)
    np.save(frame_dir / "gt_depth.npy", frame["gt_depth"])
    np.save(frame_dir / "pred_depth.npy", frame["pred_depth"])
    Image.fromarray(frame["gt_rgb"]).save(frame_dir / "gt_rgb.png")
    Image.fromarray(frame["pred_rgb"]).save(frame_dir / "pred_rgb.png")


def _alpha_schedule(count: int) -> list[float]:
    if count <= 0:
        return []
    if count == 1:
        return [0.55]
    return np.linspace(0.55, 0.15, count).tolist()


def _blend_future_predictions(
    base_rgb: np.ndarray,
    future_frames: list[dict],
    *,
    max_future_frames: int = 5,
) -> np.ndarray:
    if not future_frames:
        return base_rgb

    selected_frames = future_frames[:max_future_frames]
    blended = base_rgb.astype(np.float32) / 255.0
    for alpha, frame in zip(_alpha_schedule(len(selected_frames)), selected_frames, strict=True):
        overlay = frame["pred_rgb"].astype(np.float32) / 255.0
        blended = overlay * alpha + blended * (1.0 - alpha)
    return np.clip(blended * 255.0, 0.0, 255.0).round().astype(np.uint8)


def _save_paper_assets(sample_dir: Path, frames: list[dict]) -> dict[str, str]:
    paper_dir = sample_dir / "paper_assets"
    paper_dir.mkdir(parents=True, exist_ok=True)

    current_frame = frames[0]
    future_frames = [frame for frame in frames[1:] if frame["offset"] > 0]
    future_ghost = _blend_future_predictions(current_frame["gt_rgb"], future_frames, max_future_frames=5)

    gt_path = paper_dir / "gt.png"
    current_recon_path = paper_dir / "current_reconstruction.png"
    future_ghost_path = paper_dir / "future_prediction_ghosted.png"
    Image.fromarray(current_frame["gt_rgb"]).save(gt_path)
    Image.fromarray(current_frame["pred_rgb"]).save(current_recon_path)
    Image.fromarray(future_ghost).save(future_ghost_path)

    future_frame_paths = {}
    for frame in future_frames[:5]:
        future_path = paper_dir / f"future_prediction_{frame['label'].replace('+', 'plus')}.png"
        Image.fromarray(frame["pred_rgb"]).save(future_path)
        future_frame_paths[frame["label"]] = str(future_path)

    return {
        "paper_dir": str(paper_dir),
        "gt": str(gt_path),
        "current_reconstruction": str(current_recon_path),
        "future_prediction_ghosted": str(future_ghost_path),
        "future_frames": future_frame_paths,
    }


def _build_overview_image(sample_idx: int, prompt: str | None, frames: list[dict]) -> tuple[Image.Image, dict]:
    depth_vmin, depth_vmax, depth_diff_vmax = _collect_depth_stats(frames)
    rgb_diff_vmax = _collect_rgb_diff_scale(frames)

    rows = []
    metrics = {"sample_index": int(sample_idx), "prompt": prompt, "frames": []}
    for frame in frames:
        rgb_diff = np.abs(frame["pred_rgb"].astype(np.float32) - frame["gt_rgb"].astype(np.float32)).mean(axis=-1)
        depth_valid = (
            np.isfinite(frame["gt_depth"])
            & np.isfinite(frame["pred_depth"])
            & (frame["gt_depth"] > 0.0)
            & (frame["pred_depth"] > 0.0)
        )
        depth_diff = np.abs(frame["pred_depth"] - frame["gt_depth"])
        depth_diff_masked = depth_diff.copy()
        depth_diff_masked[~depth_valid] = 0.0

        rgb_mae = float(np.mean(np.abs(frame["pred_rgb"].astype(np.float32) - frame["gt_rgb"].astype(np.float32))))
        depth_mae = float(depth_diff[depth_valid].mean()) if depth_valid.any() else float("nan")

        gt_depth_vis = _colorize_map(
            frame["gt_depth"],
            vmin=depth_vmin,
            vmax=depth_vmax,
            cmap_name="viridis",
            invalid_mask=~np.isfinite(frame["gt_depth"]),
        )
        pred_depth_vis = _colorize_map(
            frame["pred_depth"],
            vmin=depth_vmin,
            vmax=depth_vmax,
            cmap_name="viridis",
            invalid_mask=~np.isfinite(frame["pred_depth"]),
        )
        rgb_diff_vis = _colorize_map(rgb_diff, vmin=0.0, vmax=rgb_diff_vmax, cmap_name="magma")
        depth_diff_vis = _colorize_map(depth_diff_masked, vmin=0.0, vmax=depth_diff_vmax, cmap_name="magma", invalid_mask=~depth_valid)

        text_lines = [
            frame["title"],
            f"rgb_mae={rgb_mae:.2f}",
            f"depth_mae={depth_mae:.4f}" if np.isfinite(depth_mae) else "depth_mae=nan",
        ]
        text_panel = _make_text_panel(width=170, height=frame["gt_rgb"].shape[0] + 24, lines=text_lines)

        row = _stack_row(
            [
                text_panel,
                _make_panel(frame["gt_rgb"], "GT RGB"),
                _make_panel(frame["pred_rgb"], "Pred RGB"),
                _make_panel(rgb_diff_vis, "RGB |Pred-GT|"),
                _make_panel(gt_depth_vis, "GT Depth"),
                _make_panel(pred_depth_vis, "Pred Depth"),
                _make_panel(depth_diff_vis, "Depth |Pred-GT|"),
            ]
        )
        rows.append(row)

        metrics["frames"].append(
            {
                "label": frame["label"],
                "rgb_mae": rgb_mae,
                "depth_mae": depth_mae,
            }
        )

    header_lines = [f"sample_index={sample_idx}"]
    if prompt:
        header_lines.append(f"prompt={prompt}")
    header_lines.append(f"depth_range=[{depth_vmin:.3f}, {depth_vmax:.3f}]")
    header_lines.append(f"rgb_diff_vmax={rgb_diff_vmax:.2f}, depth_diff_vmax={depth_diff_vmax:.4f}")
    header = _make_text_panel(width=rows[0].width, height=90, lines=header_lines)
    overview = _stack_column([header, *rows], gap=10)
    return overview, metrics


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)

    if device.type != "cuda":
        raise RuntimeError(
            "Gaussian RGB rendering currently requires CUDA. Please rerun on a GPU-enabled environment "
            "with `--device cuda`."
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = _resolve_checkpoint_path(Path(args.checkpoint_dir), args.checkpoint_step)

    config = training_config.get_config(args.config_name)
    if getattr(config.model, "use_lpips", False):
        config = dataclasses.replace(config, model=dataclasses.replace(config.model, use_lpips=False))
    model = config.model.load_pytorch(config, str(checkpoint_path))
    model = model.to(device)
    model.eval()
    model.set_world_model_image_fusion(True)

    if (args.episode_index is None) != (args.frame_index is None):
        raise ValueError("`--episode-index` and `--frame-index` must be provided together.")

    if args.episode_index is not None:
        selected_samples = [
            _load_specific_sample(
                config,
                episode_index=int(args.episode_index),
                frame_index=int(args.frame_index),
            )
        ]
    else:
        loader = data_loader.create_data_loader(
            config,
            framework="pytorch",
            shuffle=False,
            num_batches=args.max_batches,
        )
        selected_samples = _collect_random_samples(
            loader,
            args.num_samples,
            max_pool_size=args.random_pool_size,
            seed=args.random_seed,
        )
    if not selected_samples:
        raise RuntimeError("No samples were collected from the dataloader.")

    manifest = {
        "config_name": args.config_name,
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "num_samples": len(selected_samples),
        "random_seed": int(args.random_seed),
        "random_pool_size": int(args.random_pool_size),
        "episode_index": args.episode_index,
        "frame_index": args.frame_index,
        "samples": [],
    }

    overview_paths = []
    for rank, (observation_cpu, actions_cpu, sample_idx) in enumerate(selected_samples):
        sample_dir = output_dir / f"sample_{sample_idx:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        observation = _to_device(observation_cpu, device)
        actions = actions_cpu.to(device)

        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
                export = model.export_future_rollout_gaussians(observation, actions=actions)
            frames = _prepare_frames(model, export, device)

        prompt = _extract_prompt(export.get("preprocessed_observation"))
        for frame in frames:
            _save_raw_arrays(sample_dir, frame)
        paper_assets = _save_paper_assets(sample_dir, frames)

        overview_image, metrics = _build_overview_image(sample_idx, prompt, frames)
        overview_path = sample_dir / "overview.png"
        overview_image.save(overview_path)
        overview_paths.append(overview_path)

        with (sample_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        manifest["samples"].append(
            {
                "rank": rank,
                "sample_index": int(sample_idx),
                "sample_dir": str(sample_dir),
                "overview": str(overview_path),
                "metrics": str(sample_dir / "metrics.json"),
                "paper_assets": paper_assets,
            }
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    if overview_paths:
        overview_images = [Image.open(path).convert("RGB") for path in overview_paths]
        index_image = _stack_column(overview_images, gap=16)
        index_image.save(output_dir / "index.png")

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
