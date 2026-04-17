#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import h5py
from matplotlib import colormaps
import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
LIBERO_PACKAGE = REPO_ROOT / "third_party/libero/libero/libero"
LEGACY_ROBOSUITE_PACKAGE = (
    REPO_ROOT / "examples/libero/.venv/lib/python3.8/site-packages/robosuite"
)
LIBERO_ASSET_ROOT = LIBERO_PACKAGE / "assets"


def _prepare_python_imports() -> None:
    alias_root = Path(tempfile.gettempdir()) / "openpi_paper_framework_alias"
    libero_alias_parent = alias_root / "libero"
    libero_alias_parent.mkdir(parents=True, exist_ok=True)

    libero_alias = libero_alias_parent / "libero"
    if libero_alias.exists() or libero_alias.is_symlink():
        libero_alias.unlink()
    os.symlink(LIBERO_PACKAGE, libero_alias)

    robosuite_alias = alias_root / "robosuite"
    if robosuite_alias.exists() or robosuite_alias.is_symlink():
        robosuite_alias.unlink()
    os.symlink(LEGACY_ROBOSUITE_PACKAGE, robosuite_alias)

    sys.path.insert(0, str(alias_root))


_prepare_python_imports()

from libero.libero.envs import TASK_MAPPING  # noqa: E402
import libero.libero.utils.utils as libero_utils  # noqa: E402
from robosuite.utils.camera_utils import get_real_depth_map  # noqa: E402
import robosuite  # noqa: E402


@dataclass
class DemoBundle:
    problem_name: str
    task_name: str
    env_name: str
    env_kwargs: dict
    model_xml: str
    states: np.ndarray
    agent_rgb: np.ndarray
    wrist_rgb: np.ndarray


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export RGB / depth / flow visualizations for a LIBERO demo case folder."
    )
    parser.add_argument("demo_file", type=Path, help="LIBERO demo HDF5 file")
    parser.add_argument("case_dir", type=Path, help="Output case directory")
    parser.add_argument(
        "--demo",
        default="demo_34",
        help="Demo key inside the HDF5 file, e.g. demo_34",
    )
    parser.add_argument(
        "--current-frame",
        type=int,
        default=80,
        help="Current frame index inside the selected demo",
    )
    parser.add_argument(
        "--past-offsets",
        nargs="*",
        type=int,
        default=[10, 5, 0],
        help="Past offsets to export before the current frame",
    )
    parser.add_argument(
        "--future-count",
        type=int,
        default=5,
        help="Number of future frames / flow targets to export",
    )
    return parser.parse_args()


def _safe_label(label: str) -> str:
    return label.replace("-", "minus").replace("+", "plus")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_demo_bundle(demo_file: Path, demo_key: str) -> DemoBundle:
    with h5py.File(demo_file, "r") as handle:
        data_group = handle["data"]
        env_args = json.loads(data_group.attrs["env_args"])
        problem_info = json.loads(data_group.attrs["problem_info"])
        episode = data_group[demo_key]
        return DemoBundle(
            problem_name=problem_info["problem_name"],
            task_name=problem_info["language_instruction"],
            env_name=str(data_group.attrs["env_name"]),
            env_kwargs=dict(env_args["env_kwargs"]),
            model_xml=str(episode.attrs["model_file"]),
            states=episode["states"][()],
            agent_rgb=episode["obs/agentview_rgb"][()],
            wrist_rgb=episode["obs/eye_in_hand_rgb"][()],
        )


def _resolve_bddl_path(demo_file: Path) -> Path:
    task_stem = demo_file.stem.replace("_demo", "")
    return LIBERO_PACKAGE / "bddl_files/libero_spatial" / f"{task_stem}.bddl"


def _postprocess_model_xml_local(xml_str: str) -> str:
    root = ET.fromstring(xml_str)
    robosuite_root = Path(robosuite.__file__).resolve().parent
    for elem in root.findall(".//mesh") + root.findall(".//texture"):
        old_path = elem.get("file")
        if not old_path:
            continue
        parts = old_path.split("/")
        if "robosuite" in parts:
            idx = max(i for i, part in enumerate(parts) if part == "robosuite")
            elem.set("file", str(robosuite_root / Path(*parts[idx + 1 :])))
        elif "assets" in parts:
            idx = max(i for i, part in enumerate(parts) if part == "assets")
            elem.set("file", str(LIBERO_ASSET_ROOT / Path(*parts[idx + 1 :])))
    return ET.tostring(root, encoding="utf8").decode("utf8")


def _build_env(bundle: DemoBundle, bddl_path: Path):
    env_kwargs = dict(bundle.env_kwargs)
    libero_utils.update_env_kwargs(
        env_kwargs,
        bddl_file_name=str(bddl_path),
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        camera_depths=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=128,
        camera_widths=128,
        camera_segmentations=None,
    )
    env = TASK_MAPPING[bundle.problem_name](**env_kwargs)
    env.reset()
    env.reset_from_xml_string(_postprocess_model_xml_local(bundle.model_xml))
    env.sim.reset()
    return env


def _render_depths_for_indices(env, states: np.ndarray, frame_indices: list[int]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    rendered: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for frame_idx in frame_indices:
        env.sim.set_state_from_flattened(states[frame_idx])
        env.sim.forward()
        env._post_process()
        env._update_observables(force=True)
        obs = env._get_observations()
        agent_depth = get_real_depth_map(env.sim, np.asarray(obs["agentview_depth"])[..., 0]).astype(np.float32)
        wrist_depth = get_real_depth_map(env.sim, np.asarray(obs["robot0_eye_in_hand_depth"])[..., 0]).astype(np.float32)
        rendered[frame_idx] = (agent_depth, wrist_depth)
    return rendered


def _compute_depth_range(depth_maps: list[np.ndarray]) -> tuple[float, float]:
    valid = [depth[np.isfinite(depth) & (depth > 0)] for depth in depth_maps]
    valid = [item for item in valid if item.size > 0]
    if not valid:
        return 0.0, 1.0
    merged = np.concatenate(valid)
    vmin = float(np.percentile(merged, 1.0))
    vmax = float(np.percentile(merged, 99.0))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _save_depth_png(depth: np.ndarray, out_path: Path, *, vmin: float, vmax: float) -> None:
    norm = np.clip((depth - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    rgba = colormaps["viridis"](norm)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    rgb[~np.isfinite(depth)] = 0
    Image.fromarray(rgb).save(out_path)


def _write_rgb_frame(image: np.ndarray, out_dir: Path, label: str, frame_idx: int) -> None:
    image_pil = Image.fromarray(image)
    image_pil.save(out_dir / f"{label}_frame_{frame_idx:03d}.png")
    safe_label = _safe_label(label)
    if safe_label != label:
        image_pil.save(out_dir / f"{safe_label}_frame_{frame_idx:03d}.png")


def _build_timeline(
    images: list[np.ndarray],
    labels: list[str],
    title: str,
    out_path: Path,
) -> None:
    tile_h, tile_w = images[0].shape[:2]
    title_h = 24
    label_h = 20
    canvas = Image.new("RGB", (tile_w * len(images), title_h + tile_h + label_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), title, fill=(0, 0, 0))
    for idx, (image, label) in enumerate(zip(images, labels, strict=True)):
        x = idx * tile_w
        canvas.paste(Image.fromarray(image), (x, title_h))
        draw.text((x + 4, title_h + tile_h + 2), label, fill=(0, 0, 0))
    canvas.save(out_path)


def _motion_score(images: np.ndarray, frame_idx: int, horizon: int = 5) -> float:
    if frame_idx + horizon >= len(images):
        return -1.0
    current = images[frame_idx].astype(np.float32)
    future = images[frame_idx + horizon].astype(np.float32)
    return float(np.mean(np.abs(future - current)))


def _build_candidate_overview(demo_file: Path, selected_demo: str, selected_frame: int, out_path: Path) -> None:
    candidates: list[tuple[float, str, int, np.ndarray]] = []
    with h5py.File(demo_file, "r") as handle:
        for demo_key in sorted(handle["data"].keys(), key=lambda item: int(item.split("_")[1])):
            images = handle[f"data/{demo_key}/obs/agentview_rgb"][()]
            best_score = -1.0
            best_frame = 0
            for frame_idx in range(10, max(11, len(images) - 6)):
                score = _motion_score(images, frame_idx)
                if score > best_score:
                    best_score = score
                    best_frame = frame_idx
            candidates.append((best_score, demo_key, best_frame, images[best_frame]))

    chosen: list[tuple[str, int, np.ndarray, bool]] = []
    chosen.append(
        (
            selected_demo,
            selected_frame,
            next(item[3] for item in candidates if item[1] == selected_demo and item[2] == selected_frame)
            if any(item[1] == selected_demo and item[2] == selected_frame for item in candidates)
            else _load_demo_bundle(demo_file, selected_demo).agent_rgb[selected_frame],
            True,
        )
    )
    for _, demo_key, frame_idx, image in sorted(candidates, reverse=True):
        if demo_key == selected_demo and frame_idx == selected_frame:
            continue
        chosen.append((demo_key, frame_idx, image, False))
        if len(chosen) == 6:
            break

    tile_w = chosen[0][2].shape[1]
    tile_h = chosen[0][2].shape[0]
    caption_h = 22
    cols = 3
    rows = 2
    canvas = Image.new("RGB", (cols * tile_w, rows * (tile_h + caption_h)), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for idx, (demo_key, frame_idx, image, is_selected) in enumerate(chosen):
        x = (idx % cols) * tile_w
        y = (idx // cols) * (tile_h + caption_h)
        tile = Image.fromarray(image)
        if is_selected:
            bordered = Image.new("RGB", (tile_w, tile_h), (232, 248, 219))
            bordered.paste(tile, (0, 0))
            tile = bordered
            draw.rectangle((x, y, x + tile_w - 1, y + tile_h - 1), outline=(86, 125, 70), width=3)
        canvas.paste(tile, (x, y))
        label = f"{demo_key} | t{frame_idx}"
        if is_selected:
            label = f"SELECTED {label}"
        draw.text((x + 4, y + tile_h + 2), label, fill=(0, 0, 0))
    canvas.save(out_path)


def _camera_intrinsics(env, camera_name: str, width: int, height: int) -> tuple[float, float, float, float]:
    cam_id = env.sim.model.camera_name2id(camera_name)
    fovy = math.radians(float(env.sim.model.cam_fovy[cam_id]))
    fy = height / (2.0 * math.tan(fovy / 2.0))
    fx = fy
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return fx, fy, cx, cy


def _scene_flow_from_rgbd(
    rgb0: np.ndarray,
    rgb1: np.ndarray,
    depth0: np.ndarray,
    depth1: np.ndarray,
    intrinsics: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    gray0 = cv2.cvtColor(rgb0, cv2.COLOR_RGB2GRAY)
    gray1 = cv2.cvtColor(rgb1, cv2.COLOR_RGB2GRAY)
    flow_2d = cv2.calcOpticalFlowFarneback(
        gray0,
        gray1,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    height, width = depth0.shape
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    target_x = grid_x + flow_2d[..., 0]
    target_y = grid_y + flow_2d[..., 1]

    sampled_depth1 = cv2.remap(depth1, target_x, target_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = (
        np.isfinite(depth0)
        & np.isfinite(sampled_depth1)
        & (depth0 > 1e-6)
        & (sampled_depth1 > 1e-6)
        & (target_x >= 0)
        & (target_x <= width - 1)
        & (target_y >= 0)
        & (target_y <= height - 1)
    )

    fx, fy, cx, cy = intrinsics

    def unproject(x: np.ndarray, y: np.ndarray, depth: np.ndarray) -> np.ndarray:
        x3 = (x - cx) * depth / fx
        y3 = (y - cy) * depth / fy
        return np.stack([x3, y3, depth], axis=-1)

    pts0 = unproject(grid_x, grid_y, depth0)
    pts1 = unproject(target_x, target_y, sampled_depth1)
    flow_3d = pts1 - pts0
    flow_3d[~valid] = 0.0
    return flow_3d.astype(np.float32), valid


def _save_flow_norm_png(flow: np.ndarray, valid_mask: np.ndarray, out_path: Path) -> None:
    magnitude = np.linalg.norm(flow, axis=-1)
    valid_values = magnitude[valid_mask]
    vmax = float(np.percentile(valid_values, 99.0)) if valid_values.size > 0 else 1.0
    vmax = max(vmax, 1e-6)
    norm = np.clip(magnitude / vmax, 0.0, 1.0)
    rgb = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
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


def _write_manifest(
    case_dir: Path,
    *,
    demo_file: Path,
    demo_key: str,
    bundle: DemoBundle,
    current_frame: int,
    past_offsets: list[int],
    future_count: int,
    labels: list[str],
    frame_indices: list[int],
) -> None:
    manifest = {
        "demo_file": str(demo_file),
        "demo_name": demo_key,
        "task_name": bundle.task_name,
        "env_name": bundle.env_name,
        "num_frames": int(bundle.agent_rgb.shape[0]),
        "current_frame": int(current_frame),
        "past_offsets": [int(offset) for offset in past_offsets],
        "future_count": int(future_count),
        "labels": labels,
        "frame_indices": frame_indices,
    }
    (case_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    case_dir = args.case_dir.resolve()
    if case_dir.exists():
        shutil.rmtree(case_dir)
    _ensure_dir(case_dir)

    bundle = _load_demo_bundle(args.demo_file, args.demo)
    current_frame = int(args.current_frame)
    if current_frame < 0 or current_frame >= bundle.agent_rgb.shape[0]:
        raise IndexError(f"Current frame {current_frame} is outside demo length {bundle.agent_rgb.shape[0]}")

    past_offsets = [int(offset) for offset in args.past_offsets]
    if 0 not in past_offsets:
        past_offsets = sorted(past_offsets + [0], reverse=True)
    else:
        past_offsets = sorted(set(past_offsets), reverse=True)

    labels = [f"t-{offset}" for offset in past_offsets if offset > 0] + ["t"] + [f"t+{offset}" for offset in range(1, args.future_count + 1)]
    frame_indices = (
        [current_frame - offset for offset in past_offsets if offset > 0]
        + [current_frame]
        + [current_frame + offset for offset in range(1, args.future_count + 1)]
    )
    if min(frame_indices) < 0 or max(frame_indices) >= bundle.agent_rgb.shape[0]:
        raise IndexError(f"Requested frame window {frame_indices} falls outside demo length {bundle.agent_rgb.shape[0]}")

    agent_dir = case_dir / "agent"
    wrist_dir = case_dir / "wrist"
    agent_depth_dir = case_dir / "agent_depth"
    wrist_depth_dir = case_dir / "wrist_depth"
    agent_depth_raw_dir = case_dir / "agent_depth_raw"
    wrist_depth_raw_dir = case_dir / "wrist_depth_raw"
    flow_norm_dir = case_dir / "flow_3d_norm"
    flow_xyz_dir = case_dir / "flow_3d_xyz"
    flow_raw_dir = case_dir / "flow_3d_raw"
    flow_mask_dir = case_dir / "flow_3d_mask"
    for directory in [
        agent_dir,
        wrist_dir,
        agent_depth_dir,
        wrist_depth_dir,
        agent_depth_raw_dir,
        wrist_depth_raw_dir,
        flow_norm_dir,
        flow_xyz_dir,
        flow_raw_dir,
        flow_mask_dir,
    ]:
        _ensure_dir(directory)

    bddl_path = _resolve_bddl_path(args.demo_file)
    env = _build_env(bundle, bddl_path)
    try:
        depth_maps = _render_depths_for_indices(env, bundle.states, frame_indices)
        intrinsics = _camera_intrinsics(env, "agentview", width=bundle.agent_rgb.shape[2], height=bundle.agent_rgb.shape[1])
    finally:
        env.close()

    agent_vmin, agent_vmax = _compute_depth_range([depth_maps[idx][0] for idx in frame_indices])
    wrist_vmin, wrist_vmax = _compute_depth_range([depth_maps[idx][1] for idx in frame_indices])

    for label, frame_idx in zip(labels, frame_indices, strict=True):
        _write_rgb_frame(bundle.agent_rgb[frame_idx], agent_dir, label, frame_idx)
        _write_rgb_frame(bundle.wrist_rgb[frame_idx], wrist_dir, label, frame_idx)

        agent_depth, wrist_depth = depth_maps[frame_idx]
        np.save(agent_depth_raw_dir / f"{label}_frame_{frame_idx:03d}.npy", agent_depth)
        np.save(wrist_depth_raw_dir / f"{label}_frame_{frame_idx:03d}.npy", wrist_depth)
        _save_depth_png(agent_depth, agent_depth_dir / f"{label}_frame_{frame_idx:03d}.png", vmin=agent_vmin, vmax=agent_vmax)
        _save_depth_png(wrist_depth, wrist_depth_dir / f"{label}_frame_{frame_idx:03d}.png", vmin=wrist_vmin, vmax=wrist_vmax)

    for horizon in range(1, args.future_count + 1):
        future_idx = current_frame + horizon
        flow_3d, valid_mask = _scene_flow_from_rgbd(
            bundle.agent_rgb[current_frame],
            bundle.agent_rgb[future_idx],
            depth_maps[current_frame][0],
            depth_maps[future_idx][0],
            intrinsics,
        )
        prefix = f"tplus{horizon}"
        np.save(flow_raw_dir / f"{prefix}_flow3d.npy", flow_3d)
        np.save(flow_mask_dir / f"{prefix}_valid_mask.npy", valid_mask)
        _save_flow_norm_png(flow_3d, valid_mask, flow_norm_dir / f"{prefix}_flow3d_norm.png")
        _save_flow_xyz_png(flow_3d, valid_mask, flow_xyz_dir / f"{prefix}_flow3d_xyz.png")

    _build_timeline(
        [bundle.agent_rgb[idx] for idx in frame_indices],
        labels,
        f"Agent View | {args.demo} | {bundle.task_name}",
        case_dir / "agent_timeline.png",
    )
    _build_timeline(
        [bundle.wrist_rgb[idx] for idx in frame_indices],
        labels,
        f"Wrist View | {args.demo} | {bundle.task_name}",
        case_dir / "wrist_timeline.png",
    )
    _build_candidate_overview(args.demo_file, args.demo, current_frame, case_dir / "candidate_overview.png")
    _write_manifest(
        case_dir,
        demo_file=args.demo_file,
        demo_key=args.demo,
        bundle=bundle,
        current_frame=current_frame,
        past_offsets=past_offsets,
        future_count=args.future_count,
        labels=labels,
        frame_indices=frame_indices,
    )

    print(
        json.dumps(
            {
                "case_dir": str(case_dir),
                "demo": args.demo,
                "current_frame": current_frame,
                "exports": [
                    "agent/*.png",
                    "wrist/*.png",
                    "agent_depth/*.png",
                    "wrist_depth/*.png",
                    "agent_depth_raw/*.npy",
                    "wrist_depth_raw/*.npy",
                    "flow_3d_norm/*.png",
                    "flow_3d_xyz/*.png",
                    "flow_3d_raw/*.npy",
                    "flow_3d_mask/*.npy",
                    "agent_timeline.png",
                    "wrist_timeline.png",
                    "candidate_overview.png",
                    "manifest.json",
                ],
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
