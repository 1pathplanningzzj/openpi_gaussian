import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_project_root = Path(__file__).resolve().parents[1]
_src_path = _project_root / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

import openpi.training.config as training_config
import openpi.training.data_loader as data_loader


def _to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_device(v, device) for v in obj)
    if hasattr(obj, "replace") and hasattr(obj, "__dict__"):
        updates = {key: _to_device(value, device) for key, value in vars(obj).items()}
        return obj.replace(**updates)
    if hasattr(obj, "__dict__"):
        new_obj = type(obj).__new__(type(obj))
        for key, value in vars(obj).items():
            object.__setattr__(new_obj, key, _to_device(value, device))
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
            if value is None:
                updates[key] = None
            else:
                updates[key] = _slice_sample(value, sample_idx)
        return obj.replace(**updates)
    if hasattr(obj, "__dict__"):
        new_obj = type(obj).__new__(type(obj))
        for key, value in vars(obj).items():
            object.__setattr__(new_obj, key, None if value is None else _slice_sample(value, sample_idx))
        return new_obj
    return obj


def _tensor_to_uint8_image(image: torch.Tensor, target_hw=None):
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"Expected image [C,H,W], got {tuple(image.shape)}")
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    elif image.shape[0] > 3:
        image = image[:3]
    image = image.detach().float().cpu().clamp(0.0, 1.0)
    if target_hw is not None and tuple(image.shape[-2:]) != tuple(target_hw):
        image = F.interpolate(image.unsqueeze(0), size=target_hw, mode="bilinear", align_corners=False).squeeze(0)
    return (image.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")


def _extract_agent_image(observation, target_hw=None):
    if observation is None or not hasattr(observation, "images"):
        return None
    for key, value in observation.images.items():
        key_lower = key.lower()
        if key == "image" or "agent" in key_lower or "high" in key_lower or "cam_high" in key_lower or "exterior" in key_lower or "base" in key_lower:
            img = value
            if img.ndim == 5:
                img = img[:, -1]
            if img.shape[1] != 3 and img.shape[-1] == 3:
                img = img.permute(0, 3, 1, 2)
            return _tensor_to_uint8_image((img + 1.0) / 2.0, target_hw=target_hw)
    return None


def _draw_text(img, text: str):
    out = img.copy()
    h = out.shape[0]
    bar_h = min(26, max(20, h // 10))
    out[:bar_h, :, :] = (0.15 * out[:bar_h, :, :] + 0.85 * 255).astype("uint8")
    try:
        from PIL import Image, ImageDraw

        pil = Image.fromarray(out)
        draw = ImageDraw.Draw(pil)
        draw.text((8, 5), text, fill=(0, 0, 0))
        return np.array(pil)
    except Exception:
        return out


def _stack_panels(panels):
    max_h = max(panel.shape[0] for panel in panels)
    normalized = []
    for panel in panels:
        if panel.shape[0] != max_h:
            scale = max_h / panel.shape[0]
            target_w = int(round(panel.shape[1] * scale))
            tensor = torch.from_numpy(panel).permute(2, 0, 1).float() / 255.0
            tensor = F.interpolate(tensor.unsqueeze(0), size=(max_h, target_w), mode="bilinear", align_corners=False).squeeze(0)
            panel = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
        normalized.append(panel)
    return np.concatenate(normalized, axis=1)


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


def _collect_selected_samples(loader, device, start_index: int, count: int):
    selected = []
    global_idx = 0
    end_index = start_index + count
    for observation, actions in loader:
        observation = _to_device(observation, device)
        actions = actions.to(device)
        batch_size = actions.shape[0]
        batch_start = global_idx
        batch_end = global_idx + batch_size
        if batch_end <= start_index:
            global_idx = batch_end
            continue
        for local_idx in range(batch_size):
            sample_idx = batch_start + local_idx
            if sample_idx < start_index:
                continue
            if sample_idx >= end_index:
                return selected
            selected.append((_slice_sample(observation, local_idx), actions[local_idx : local_idx + 1], sample_idx))
        global_idx = batch_end
    return selected


def _render_panels(model, gaussian_params, gt_observation, label_left: str, label_mid: str, orbit_label: str, *, device, orbit_azimuth_deg: float, orbit_elevation_deg: float, orbit_radius_scale: float, target_hw=None):
    renders = model.render_gaussian_views_for_export(
        gaussian_params,
        device=device,
        batch_size=1,
        agent_view=True,
        orbit_azimuth_deg=orbit_azimuth_deg,
        orbit_elevation_deg=orbit_elevation_deg,
        orbit_radius_scale=orbit_radius_scale,
        target_hw=target_hw,
    )
    panel_hw = renders["agent"].shape[-2:]
    gt_img = _extract_agent_image(gt_observation, target_hw=panel_hw)
    agent_pred = _tensor_to_uint8_image(renders["agent"], target_hw=panel_hw)
    orbit_pred = _tensor_to_uint8_image(renders["orbit"], target_hw=panel_hw)
    frame = _stack_panels([
        _draw_text(gt_img, label_left),
        _draw_text(agent_pred, label_mid),
        _draw_text(orbit_pred, orbit_label),
    ])
    return frame, panel_hw


def _render_video_frames(model, export, device, orbit_mode: str, orbit_azimuth_deg: float, orbit_elevation_deg: float, orbit_radius_scale: float, orbit_frames: int):
    frames = []
    context_observations = export["context_observations"]
    context_labels = export["context_labels"]
    future_offsets = export["future_offsets"]
    future_targets = export["future_targets"]
    future_gaussians = export["future_gaussian_params_seq"]

    if orbit_mode == "fixed":
        orbit_schedule = [float(orbit_azimuth_deg)]
    else:
        orbit_schedule = np.linspace(0.0, 360.0, num=max(orbit_frames, 1), endpoint=False).astype(float).tolist()

    panel_hw = None
    for context_idx, context_obs in enumerate(context_observations):
        gaussian_params = export["base_gaussian_params"]
        if orbit_mode == "fixed":
            frame, panel_hw = _render_panels(
                model,
                gaussian_params,
                context_obs,
                f"GT Context {context_labels[context_idx]}",
                "Base Gaussian Agent",
                f"Base Gaussian Orbit az={orbit_azimuth_deg:.0f}",
                device=device,
                orbit_azimuth_deg=orbit_schedule[0],
                orbit_elevation_deg=orbit_elevation_deg,
                orbit_radius_scale=orbit_radius_scale,
                target_hw=panel_hw,
            )
            frames.append(frame)
        else:
            for az in orbit_schedule:
                frame, panel_hw = _render_panels(
                    model,
                    gaussian_params,
                    context_obs,
                    f"GT Context {context_labels[context_idx]}",
                    "Base Gaussian Agent",
                    f"Orbit Spin az={az:.0f}",
                    device=device,
                    orbit_azimuth_deg=az,
                    orbit_elevation_deg=orbit_elevation_deg,
                    orbit_radius_scale=orbit_radius_scale,
                    target_hw=panel_hw,
                )
                frames.append(frame)

    for offset, future_target, gaussian_params in zip(future_offsets, future_targets, future_gaussians, strict=True):
        if orbit_mode == "fixed":
            frame, panel_hw = _render_panels(
                model,
                gaussian_params,
                future_target,
                f"GT Future t+{offset}",
                f"Pred Agent t+{offset}",
                f"Pred Orbit t+{offset}",
                device=device,
                orbit_azimuth_deg=orbit_schedule[0],
                orbit_elevation_deg=orbit_elevation_deg,
                orbit_radius_scale=orbit_radius_scale,
                target_hw=panel_hw,
            )
            frames.append(frame)
        else:
            for az in orbit_schedule:
                frame, panel_hw = _render_panels(
                    model,
                    gaussian_params,
                    future_target,
                    f"GT Future t+{offset}",
                    f"Pred Agent t+{offset}",
                    f"Orbit Spin t+{offset} az={az:.0f}",
                    device=device,
                    orbit_azimuth_deg=az,
                    orbit_elevation_deg=orbit_elevation_deg,
                    orbit_radius_scale=orbit_radius_scale,
                    target_hw=panel_hw,
                )
                frames.append(frame)
    return frames


def main():
    parser = argparse.ArgumentParser(description="Export future Gaussian rollout videos offline")
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--orbit-mode", choices=("fixed", "spin"), default="fixed")
    parser.add_argument("--orbit-frames", type=int, default=24)
    parser.add_argument("--orbit-azimuth", type=float, default=35.0)
    parser.add_argument("--orbit-elevation", type=float, default=28.0)
    parser.add_argument("--orbit-radius-scale", type=float, default=0.7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    config = training_config.get_config(args.config_name)
    loader = data_loader.create_data_loader(
        config,
        framework="pytorch",
        shuffle=False,
        num_batches=args.max_batches,
    )

    checkpoint_path = _resolve_checkpoint_path(Path(args.checkpoint_dir), args.checkpoint_step)
    model = config.model.load_pytorch(config, str(checkpoint_path))
    model = model.to(device)
    model.eval()
    model.set_world_model_image_fusion(True)

    step_name = checkpoint_path.parent.name
    output_dir = Path(args.output_dir) / f"future_rollout_step{step_name}_{args.orbit_mode}_samples_{args.sample_index}_{args.sample_index + args.num_samples - 1}"
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_samples = _collect_selected_samples(loader, device, args.sample_index, args.num_samples)
    if not selected_samples:
        raise IndexError(f"No samples found from sample_index={args.sample_index}")

    for observation, actions, sample_idx in selected_samples:
        with torch.no_grad():
            export = model.export_future_rollout_gaussians(observation, actions=actions)
        frames = _render_video_frames(
            model,
            export,
            device,
            orbit_mode=args.orbit_mode,
            orbit_azimuth_deg=args.orbit_azimuth,
            orbit_elevation_deg=args.orbit_elevation,
            orbit_radius_scale=args.orbit_radius_scale,
            orbit_frames=args.orbit_frames,
        )
        output_path = output_dir / f"future_rollout_step{step_name}_sample{sample_idx}.mp4"
        imageio.mimwrite(output_path, frames, fps=args.fps, quality=8)
        print(output_path)


if __name__ == "__main__":
    main()
