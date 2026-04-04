"""Visualize raw VGGT current-frame depth predictions against GT depth.

This script bypasses the OpenPI world-model depth heads and directly runs the
pretrained VGGT depth head on the packed context frames used by GaussianAdapter.
It is intended as a diagnosis tool for checking whether VGGT itself captures
robot-arm depth before the downstream future/static decoders.
"""

import argparse
import dataclasses
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/openpi_mpl")
os.environ.setdefault("HF_HOME", "/tmp/openpi_hf")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/openpi_hf/datasets")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_path = os.path.join(_project_root, "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

import openpi.training.config as _config
import openpi.training.data_loader as _data
from openpi.models_pytorch.pi0_vggt import GaussianAdapter


def _init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def _context_count(model_config) -> int:
    use_single_frame_mode = bool(getattr(model_config, "use_single_frame_mode", False))
    if use_single_frame_mode:
        return 1
    temporal_context_offsets = getattr(model_config, "temporal_context_offsets", (-10, -5, 0))
    return max(1, len(tuple(temporal_context_offsets)))


def _slice_context_observation(observation, context_count: int):
    def _slice(value):
        if value is None:
            return None
        if value.ndim >= 2 and value.shape[1] >= context_count:
            return value[:, :context_count]
        return value

    images = {key: _slice(value) if value.ndim == 5 else value for key, value in observation.images.items()}
    image_masks = {
        key: _slice(value) if value.ndim == 2 else value for key, value in observation.image_masks.items()
    }
    state = _slice(observation.state) if getattr(observation, "state", None) is not None and observation.state.ndim == 3 else getattr(observation, "state", None)
    tokenized_prompt = (
        _slice(observation.tokenized_prompt)
        if getattr(observation, "tokenized_prompt", None) is not None and observation.tokenized_prompt.ndim == 3
        else getattr(observation, "tokenized_prompt", None)
    )
    tokenized_prompt_mask = (
        _slice(observation.tokenized_prompt_mask)
        if getattr(observation, "tokenized_prompt_mask", None) is not None and observation.tokenized_prompt_mask.ndim == 3
        else getattr(observation, "tokenized_prompt_mask", None)
    )
    depth = _slice(observation.depth) if getattr(observation, "depth", None) is not None and observation.depth.ndim == 5 else getattr(observation, "depth", None)

    return SimpleNamespace(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        depth=depth,
    )


def _extract_agent_image(observation, current_idx: int) -> torch.Tensor:
    for key, value in observation.images.items():
        key_lower = key.lower()
        if "wrist" in key_lower:
            continue
        if key == "image" or any(prefix in key_lower for prefix in ["base", "agent", "sideview"]) or "_image" in key_lower or "_rgb" in key_lower:
            img = value
            if img.ndim == 5:
                img = img[:, current_idx]
            if img.shape[-1] == 3:
                img = img.permute(0, 3, 1, 2)
            return img
    raise KeyError(f"No agent-view image found in keys: {list(observation.images.keys())}")


def _normalize_rgb(rgb_tensor: torch.Tensor) -> torch.Tensor:
    rgb = rgb_tensor.detach().float().cpu()
    if rgb.min() < -0.5:
        rgb = (rgb + 1.0) / 2.0
    elif rgb.max() > 1.5:
        rgb = rgb / 255.0
    return rgb.clamp(0.0, 1.0)


def _save_grid(rgb, gt_depth, pred_depth, output_path: Path, title: str, max_items: int) -> None:
    num_items = min(max_items, rgb.shape[0])
    fig, axes = plt.subplots(num_items, 4, figsize=(16, 4 * num_items))
    if num_items == 1:
        axes = axes[None, :]

    for row in range(num_items):
        rgb_np = rgb[row].permute(1, 2, 0).numpy()
        gt_np = gt_depth[row, 0].numpy()
        pred_np = pred_depth[row, 0].numpy()
        err_np = abs(pred_np - gt_np)

        depth_vmin = float(min(gt_np.min(), pred_np.min()))
        depth_vmax = float(max(gt_np.max(), pred_np.max()))

        axes[row, 0].imshow(rgb_np)
        axes[row, 0].set_title("RGB")
        axes[row, 1].imshow(gt_np, cmap="magma", vmin=depth_vmin, vmax=depth_vmax)
        axes[row, 1].set_title("GT Depth")
        axes[row, 2].imshow(pred_np, cmap="magma", vmin=depth_vmin, vmax=depth_vmax)
        axes[row, 2].set_title("VGGT Depth")
        axes[row, 3].imshow(err_np, cmap="inferno")
        axes[row, 3].set_title("|Pred-GT|")

        for col in range(4):
            axes[row, col].axis("off")

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug raw VGGT current-frame depth predictions.")
    parser.add_argument("config_name", help="Train config name, e.g. pi05_libero")
    parser.add_argument("--batch-index", type=int, default=0, help="Which batch to visualize")
    parser.add_argument("--max-items", type=int, default=2, help="How many samples from the batch to save")
    parser.add_argument("--batch-size", type=int, default=2, help="Temporary batch size override for this debug run")
    parser.add_argument("--num-batches", type=int, default=1, help="Only iterate over this many batches")
    parser.add_argument("--random-samples", action="store_true", help="Randomly choose samples from the selected batch")
    parser.add_argument("--seed", type=int, default=0, help="Seed for random sample selection")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./visualizations/vggt_depth_debug/current_frame_vggt_depth.png"),
        help="Output image path",
    )
    return parser.parse_args()


def main() -> None:
    _init_logging()
    args = parse_args()

    config = _config.get_config(args.config_name)
    config = dataclasses.replace(config, batch_size=args.batch_size, num_workers=0)
    loader = _data.create_data_loader(
        config,
        framework="pytorch",
        shuffle=False,
        num_batches=args.num_batches,
    )

    observation = None
    for batch_idx, batch in enumerate(loader):
        observation, _ = batch
        if batch_idx == args.batch_index:
            break
    if observation is None:
        raise RuntimeError(f"Failed to load batch {args.batch_index}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context_count = _context_count(config.model)
    current_idx = context_count - 1
    logging.info("Using context_count=%d, current_idx=%d", context_count, current_idx)

    context_observation = _slice_context_observation(observation, context_count)
    batch_size = next(iter(context_observation.images.values())).shape[0]

    adapter = GaussianAdapter(
        use_gaussian=bool(getattr(config.model, "use_gaussian", True)),
        action_expert_width=2048,
        num_frames=context_count,
        inference_num_frames=context_count,
        use_single_frame_mode=bool(getattr(config.model, "use_single_frame_mode", False)),
        temporal_context_offsets=tuple(getattr(config.model, "temporal_context_offsets", (-10, -5, 0))),
        unfreeze_encoder=False,
        use_lora=False,
    ).to(device)
    adapter.eval()

    gaussian_inputs = adapter.prepare_inputs(context_observation, device, batch_size, is_training=True)
    if gaussian_inputs is None:
        raise RuntimeError("GaussianAdapter.prepare_inputs returned None")

    with torch.no_grad():
        depth_maps, _, _, _, _, _, _ = adapter.encoder(gaussian_inputs)

    if depth_maps.ndim != 4 and depth_maps.ndim != 5:
        raise ValueError(f"Unexpected VGGT depth shape: {tuple(depth_maps.shape)}")

    if depth_maps.ndim == 4:
        pred_depth = depth_maps[:, current_idx : current_idx + 1]
    else:
        pred_depth = depth_maps[:, current_idx]
        if pred_depth.ndim == 3:
            pred_depth = pred_depth.unsqueeze(1)

    gt_depth = getattr(context_observation, "depth", None)
    if gt_depth is None:
        raise RuntimeError("Observation does not contain GT depth")
    if gt_depth.ndim == 5:
        gt_depth = gt_depth[:, current_idx]
    if gt_depth.ndim == 3:
        gt_depth = gt_depth.unsqueeze(1)

    if pred_depth.shape[-2:] != gt_depth.shape[-2:]:
        pred_depth = torch.nn.functional.interpolate(
            pred_depth.float(),
            size=gt_depth.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    rgb = _extract_agent_image(context_observation, current_idx)
    rgb = _normalize_rgb(rgb)
    gt_depth = gt_depth.detach().float().cpu()
    pred_depth = pred_depth.detach().float().cpu()

    if args.random_samples and rgb.shape[0] > args.max_items:
        generator = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(rgb.shape[0], generator=generator)[: args.max_items]
        rgb = rgb[indices]
        gt_depth = gt_depth[indices]
        pred_depth = pred_depth[indices]

    title = f"VGGT Current-Frame Depth Debug | config={args.config_name} | batch={args.batch_index}"
    _save_grid(rgb, gt_depth, pred_depth, args.output, title, args.max_items)
    logging.info("Saved VGGT depth debug visualization to %s", args.output)


if __name__ == "__main__":
    main()
