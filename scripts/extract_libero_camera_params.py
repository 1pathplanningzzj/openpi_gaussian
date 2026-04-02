#!/usr/bin/env python3
"""Extract LIBERO camera parameters from robosuite environments.

This script recreates LIBERO tasks and saves camera intrinsics plus explicit
transform directions for downstream geometry code.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Configure headless rendering before importing MuJoCo / LIBERO.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

BASE_PATH = Path(__file__).resolve().parent.parent
# Prefer the active environment's robosuite installation. Only inject third_party
# LIBERO since this environment does not provide a pip-installed libero package.
sys.path.insert(0, str(BASE_PATH / "third_party" / "libero"))

from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from robosuite.utils.camera_utils import (  # noqa: E402
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
)
from robosuite.utils.transform_utils import pose_inv  # noqa: E402


def _make_env(task, resolution: int, seed: int):
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    try:
        env.seed(seed)
    except (AttributeError, TypeError):
        pass
    return env


def _extract_single_camera(sim, camera_name: str, image_height: int, image_width: int) -> dict:
    cam_id = sim.model.camera_name2id(camera_name)
    intrinsics = get_camera_intrinsic_matrix(
        sim=sim,
        camera_name=camera_name,
        camera_height=image_height,
        camera_width=image_width,
    )
    camera_to_world = get_camera_extrinsic_matrix(sim=sim, camera_name=camera_name)
    world_to_camera = pose_inv(camera_to_world)

    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    fov_y = float(sim.model.cam_fovy[cam_id])
    fov_x = float(2 * np.degrees(np.arctan(image_width / (2 * fx))))

    extent = float(sim.model.stat.extent)
    near = float(sim.model.vis.map.znear * extent)
    far = float(sim.model.vis.map.zfar * extent)

    return {
        "camera_name": camera_name,
        "intrinsics": intrinsics.tolist(),
        "camera_to_world": camera_to_world.tolist(),
        "world_to_camera": world_to_camera.tolist(),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "fov_x_deg": fov_x,
        "fov_y_deg": fov_y,
        "image_height": int(image_height),
        "image_width": int(image_width),
        "near": near,
        "far": far,
        "notes": {
            "get_camera_extrinsic_matrix_returns": "camera pose in world frame (camera_to_world)",
            "world_to_camera": "inverse of camera_to_world",
        },
    }


def _extract_task_entry(task_suite_name: str, task_id: int, resolution: int, seed: int) -> dict:
    task_suite = benchmark.get_benchmark_dict()[task_suite_name]()
    task = task_suite.get_task(task_id)
    init_states = task_suite.get_task_init_states(task_id)

    env = _make_env(task, resolution, seed)
    try:
        env.reset()
        if len(init_states) > 0:
            env.set_init_state(init_states[0])

        cameras = {}
        for camera_name in ["agentview", "robot0_eye_in_hand"]:
            try:
                cameras[camera_name] = _extract_single_camera(env.sim, camera_name, resolution, resolution)
            except Exception as exc:  # pragma: no cover - best effort for missing cameras
                cameras[camera_name] = {"camera_name": camera_name, "error": str(exc)}

        return {
            "benchmark": task_suite_name,
            "task_id": int(task_id),
            "task_name": task.name,
            "language": task.language,
            "problem_folder": task.problem_folder,
            "bddl_file": task.bddl_file,
            "cameras": cameras,
        }
    finally:
        try:
            env.close()
        except Exception:
            pass


def _iter_task_ids(task_suite_name: str, task_id: int | None):
    task_suite = benchmark.get_benchmark_dict()[task_suite_name]()
    if task_id is not None:
        if task_id < 0 or task_id >= task_suite.n_tasks:
            raise ValueError(f"task_id {task_id} out of range for {task_suite_name} ({task_suite.n_tasks} tasks)")
        return [task_id]
    return list(range(task_suite.n_tasks))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract LIBERO camera parameters")
    parser.add_argument("--output", type=str, default="libero_camera_params.json", help="Output JSON path")
    parser.add_argument("--image-height", type=int, default=256, help="Image height in pixels")
    parser.add_argument("--image-width", type=int, default=256, help="Image width in pixels")
    parser.add_argument(
        "--task-suite-name",
        type=str,
        default="libero_10",
        help="Benchmark suite name or 'all' to extract all suites",
    )
    parser.add_argument("--task-id", type=int, default=None, help="Optional task id within the suite")
    parser.add_argument("--seed", type=int, default=0, help="Environment seed")
    args = parser.parse_args()

    if args.image_height != args.image_width:
        raise ValueError("This script currently expects square render resolution.")

    benchmark_dict = benchmark.get_benchmark_dict()
    if args.task_suite_name == "all":
        suite_names = sorted(benchmark_dict.keys())
    else:
        if args.task_suite_name not in benchmark_dict:
            raise ValueError(f"Unknown task suite: {args.task_suite_name}. Available: {sorted(benchmark_dict.keys())}")
        suite_names = [args.task_suite_name]

    entries = []
    for suite_name in suite_names:
        task_ids = _iter_task_ids(suite_name, args.task_id if len(suite_names) == 1 else None)
        for task_id in task_ids:
            print(f"Extracting {suite_name} task {task_id}...")
            entries.append(_extract_task_entry(suite_name, task_id, args.image_height, args.seed))

    payload = {
        "format_version": 2,
        "image_height": int(args.image_height),
        "image_width": int(args.image_width),
        "task_suite_name": args.task_suite_name,
        "entries": entries,
    }

    if entries:
        first_cameras = entries[0].get("cameras", {})
        if "agentview" in first_cameras:
            payload["agent"] = first_cameras["agentview"]
        if "robot0_eye_in_hand" in first_cameras:
            payload["wrist"] = first_cameras["robot0_eye_in_hand"]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved camera parameters to {output_path}")
    print(f"Extracted {len(entries)} task entries")


if __name__ == "__main__":
    main()
