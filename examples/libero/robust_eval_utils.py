from __future__ import annotations

import os
import sys

# CRITICAL: Set rendering backend BEFORE any imports that might initialize OpenGL/GLFW
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"  # Use EGL for GPU-accelerated rendering

# Disable GLFW (prevents X11 initialization issues in headless environments)
os.environ["PYOPENGL_PLATFORM"] = "egl"

import collections
import dataclasses
import hashlib
import logging
import math
import pathlib
import pickle

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class RobustEvalArgs:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8020
    resize_size: int = 224
    replan_steps: int = 5  # Test with 1 for closed-loop control

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    task_id: list[int] | None = None  # Specific task ID(s) to evaluate (None = evaluate all tasks)
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Robustness perturbation parameters
    #################################################################################################################
    perturb_target: str = "obj_of_interest"  # obj_of_interest | all_movable | comma separated names
    initial_perturb: bool = True  # Apply an initial object pose perturbation after loading the init state
    initial_perturb_xy_m: float = 0.02  # Max absolute xy offset in meters for the initial pose perturbation
    initial_perturb_yaw_deg: float = 10.0  # Max absolute yaw offset in degrees for the initial pose perturbation
    mid_perturb: bool = True  # Apply additional object pose perturbation(s) during the rollout
    mid_perturb_xy_m: float = 0.01  # Max absolute xy offset in meters for each mid-rollout pose perturbation
    mid_perturb_yaw_deg: float = 5.0  # Max absolute yaw offset in degrees for each mid-rollout pose perturbation
    mid_perturb_after_steps: int = 40  # First perturbation after this many policy-controlled steps
    mid_perturb_interval: int = 80  # Steps between perturbations after the first one; <=0 means one-shot
    mid_perturb_max_times: int = 1  # Upper bound on the number of mid-rollout perturbations per episode
    light_perturb: bool = False  # Randomize scene lights once per episode
    light_target: str = "all"  # all | comma separated light names
    light_position_jitter_m: float = 0.15  # Max absolute xyz shift per light
    light_direction_jitter_deg: float = 20.0  # Max absolute xyz direction delta before renormalization
    light_ambient_jitter: float = 0.10  # Max additive per-channel ambient delta
    light_diffuse_jitter: float = 0.20  # Max multiplicative diffuse scale delta around 1.0
    light_specular_jitter: float = 0.10  # Max additive per-channel specular delta
    texture_perturb: bool = False  # Randomize target object textures once per episode
    texture_variation: str = "swap"  # swap | tint
    texture_tint_strength: float = 0.35  # Blend factor used for tint perturbations / swap fallback
    initial_force_perturb: bool = False  # Apply an external wrench for a few warmup env steps
    initial_force_xy_n: float = 2.0  # Max absolute xy force in Newtons
    initial_force_z_n: float = 1.0  # Max absolute z force in Newtons
    initial_torque_nm: float = 0.0  # Max absolute xyz torque in N*m
    initial_force_duration_steps: int = 4  # Number of env steps the initial wrench stays active
    mid_force_perturb: bool = False  # Apply external wrench disturbance(s) during rollout
    mid_force_xy_n: float = 1.5  # Max absolute xy force in Newtons for each mid-rollout disturbance
    mid_force_z_n: float = 0.8  # Max absolute z force in Newtons for each mid-rollout disturbance
    mid_torque_nm: float = 0.0  # Max absolute xyz torque in N*m for each mid-rollout disturbance
    mid_force_duration_steps: int = 3  # Number of env steps each mid-rollout wrench stays active
    mid_force_after_steps: int = 40  # First force disturbance after this many policy-controlled steps
    mid_force_interval: int = 80  # Steps between force disturbances after the first one; <=0 means one-shot
    mid_force_max_times: int = 1  # Upper bound on the number of mid-rollout force disturbances per episode
    perturb_seed_offset: int = 100000  # Keeps perturbation randomness deterministic but separate from env seed

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data_411/gaussian_vla_exp411_45000_spatial_test_2/videos_robust"  # Path to save videos
    # Gaussian_vla_exp315_12000_libero_10 这个实际上是goal
    save_videos: bool = True  # Whether to save rollout videos
    seed: int = 7  # Random Seed (for reproducibility)
    debug_log_path: str | None = None  # Path to save debug logs (None = disabled)
    # Match training `temporal_context_offsets=(-10, -5, 0)`: steps before current at control (env) rate (~10Hz).
    # History shorter than max offset pads with the current frame (same as server-side repeat, but real frames when available).
    agent_temporal_step_offsets: tuple[int, int, int] = (10, 5, 0)


@dataclasses.dataclass
class InitPoseArgs:
    host: str = "0.0.0.0"
    port: int = 8020
    resize_size: int = 224
    replan_steps: int = 5
    task_suite_name: str = "libero_spatial"
    task_id: list[int] | None = None
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    video_out_path: str = "data_411/gaussian_vla_exp411_45000_spatial_test_2/videos_robust"
    save_videos: bool = True
    seed: int = 7
    debug_log_path: str | None = None
    agent_temporal_step_offsets: tuple[int, int, int] = (10, 5, 0)
    perturb_target: str = "obj_of_interest"
    initial_perturb_xy_m: float = 0.02
    initial_perturb_yaw_deg: float = 10.0


@dataclasses.dataclass
class LightArgs:
    host: str = "0.0.0.0"
    port: int = 8020
    resize_size: int = 224
    replan_steps: int = 5
    task_suite_name: str = "libero_spatial"
    task_id: list[int] | None = None
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    video_out_path: str = "data_411/gaussian_vla_exp411_45000_spatial_test_2/videos_robust"
    save_videos: bool = True
    seed: int = 7
    debug_log_path: str | None = None
    agent_temporal_step_offsets: tuple[int, int, int] = (10, 5, 0)
    light_target: str = "all"
    light_position_jitter_m: float = 0.15
    light_direction_jitter_deg: float = 20.0
    light_ambient_jitter: float = 0.10
    light_diffuse_jitter: float = 0.20
    light_specular_jitter: float = 0.10


@dataclasses.dataclass
class TextureArgs:
    host: str = "0.0.0.0"
    port: int = 8020
    resize_size: int = 224
    replan_steps: int = 5
    task_suite_name: str = "libero_spatial"
    task_id: list[int] | None = None
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    video_out_path: str = "data_411/gaussian_vla_exp411_45000_spatial_test_2/videos_robust"
    save_videos: bool = True
    seed: int = 7
    debug_log_path: str | None = None
    agent_temporal_step_offsets: tuple[int, int, int] = (10, 5, 0)
    perturb_target: str = "obj_of_interest"
    texture_variation: str = "swap"
    texture_tint_strength: float = 0.35


@dataclasses.dataclass
class InitForceArgs:
    host: str = "0.0.0.0"
    port: int = 8020
    resize_size: int = 224
    replan_steps: int = 5
    task_suite_name: str = "libero_spatial"
    task_id: list[int] | None = None
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    video_out_path: str = "data_411/gaussian_vla_exp411_45000_spatial_test_2/videos_robust"
    save_videos: bool = True
    seed: int = 7
    debug_log_path: str | None = None
    agent_temporal_step_offsets: tuple[int, int, int] = (10, 5, 0)
    perturb_target: str = "obj_of_interest"
    initial_force_xy_n: float = 2.0
    initial_force_z_n: float = 1.0
    initial_torque_nm: float = 0.0
    initial_force_duration_steps: int = 4


@dataclasses.dataclass
class MidForceArgs:
    host: str = "0.0.0.0"
    port: int = 8020
    resize_size: int = 224
    replan_steps: int = 5
    task_suite_name: str = "libero_spatial"
    task_id: list[int] | None = None
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    video_out_path: str = "data_411/gaussian_vla_exp411_45000_spatial_test_2/videos_robust"
    save_videos: bool = True
    seed: int = 7
    debug_log_path: str | None = None
    agent_temporal_step_offsets: tuple[int, int, int] = (10, 5, 0)
    perturb_target: str = "obj_of_interest"
    mid_force_xy_n: float = 1.5
    mid_force_z_n: float = 0.8
    mid_torque_nm: float = 0.0
    mid_force_duration_steps: int = 3
    mid_force_after_steps: int = 40
    mid_force_interval: int = 80
    mid_force_max_times: int = 1


def _common_kwargs(args: object) -> dict[str, object]:
    return {
        "host": args.host,
        "port": args.port,
        "resize_size": args.resize_size,
        "replan_steps": args.replan_steps,
        "task_suite_name": args.task_suite_name,
        "task_id": args.task_id,
        "num_steps_wait": args.num_steps_wait,
        "num_trials_per_task": args.num_trials_per_task,
        "video_out_path": args.video_out_path,
        "save_videos": args.save_videos,
        "seed": args.seed,
        "debug_log_path": args.debug_log_path,
        "agent_temporal_step_offsets": args.agent_temporal_step_offsets,
    }


def build_init_pose_config(args: InitPoseArgs) -> RobustEvalArgs:
    return RobustEvalArgs(
        **_common_kwargs(args),
        perturb_target=args.perturb_target,
        initial_perturb=True,
        initial_perturb_xy_m=args.initial_perturb_xy_m,
        initial_perturb_yaw_deg=args.initial_perturb_yaw_deg,
        mid_perturb=False,
    )


def build_light_config(args: LightArgs) -> RobustEvalArgs:
    return RobustEvalArgs(
        **_common_kwargs(args),
        initial_perturb=False,
        mid_perturb=False,
        light_perturb=True,
        light_target=args.light_target,
        light_position_jitter_m=args.light_position_jitter_m,
        light_direction_jitter_deg=args.light_direction_jitter_deg,
        light_ambient_jitter=args.light_ambient_jitter,
        light_diffuse_jitter=args.light_diffuse_jitter,
        light_specular_jitter=args.light_specular_jitter,
    )


def build_texture_config(args: TextureArgs) -> RobustEvalArgs:
    return RobustEvalArgs(
        **_common_kwargs(args),
        perturb_target=args.perturb_target,
        initial_perturb=False,
        mid_perturb=False,
        texture_perturb=True,
        texture_variation=args.texture_variation,
        texture_tint_strength=args.texture_tint_strength,
    )


def build_init_force_config(args: InitForceArgs) -> RobustEvalArgs:
    return RobustEvalArgs(
        **_common_kwargs(args),
        perturb_target=args.perturb_target,
        initial_perturb=False,
        mid_perturb=False,
        initial_force_perturb=True,
        initial_force_xy_n=args.initial_force_xy_n,
        initial_force_z_n=args.initial_force_z_n,
        initial_torque_nm=args.initial_torque_nm,
        initial_force_duration_steps=args.initial_force_duration_steps,
    )


def build_mid_force_config(args: MidForceArgs) -> RobustEvalArgs:
    return RobustEvalArgs(
        **_common_kwargs(args),
        perturb_target=args.perturb_target,
        initial_perturb=False,
        mid_perturb=False,
        mid_force_perturb=True,
        mid_force_xy_n=args.mid_force_xy_n,
        mid_force_z_n=args.mid_force_z_n,
        mid_torque_nm=args.mid_torque_nm,
        mid_force_duration_steps=args.mid_force_duration_steps,
        mid_force_after_steps=args.mid_force_after_steps,
        mid_force_interval=args.mid_force_interval,
        mid_force_max_times=args.mid_force_max_times,
    )


def _stack_agent_temporal_frames(
    past_frames: collections.deque,
    current_frame: np.ndarray,
    step_offsets: tuple[int, ...],
) -> np.ndarray:
    """Build (T, H, W, C) uint8 stack: one slot per offset in order (oldest context first).

    `step_offsets` are non-negative integers = control steps before the current frame
    (10, 5, 0 corresponds to training t-10, t-5, t). If a requested index is before the
    start of `past_frames + [current]`, use the newest available frame (typically current).
    """
    seq = list(past_frames) + [current_frame]
    l = len(seq)
    out: list[np.ndarray] = []
    for off in step_offsets:
        i = l - 1 - off
        if i < 0:
            out.append(seq[-1])
        else:
            out.append(seq[i])
    return np.stack(out, axis=0)


def _configure_logging(video_out_path: str) -> None:
    data_dir = pathlib.Path(video_out_path).parent
    data_dir.mkdir(parents=True, exist_ok=True)

    log_file = data_dir / "eval.log"
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    logging.info(f"Logging to: {log_file}")


def _quat_multiply_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def _normalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm <= 0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quat / norm


def _yaw_quat_xyzw(yaw_rad: float) -> np.ndarray:
    return np.array([0.0, 0.0, math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0)], dtype=np.float64)


@dataclasses.dataclass
class ActiveForceEvent:
    stage_name: str
    remaining_steps: int
    body_wrenches: dict[int, np.ndarray]


def _get_episode_rng(args: RobustEvalArgs, task_id: int, episode_idx: int, stage_offset: int) -> np.random.Generator:
    seed = args.seed + args.perturb_seed_offset + task_id * 1000 + episode_idx * 10 + stage_offset
    return np.random.default_rng(seed)


def _clip_unit_interval(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 1.0)


def _resolve_light_names(env: OffScreenRenderEnv, light_target: str) -> list[str]:
    available_light_names = [name for name in env.sim.model.light_names if name]
    if light_target == "all":
        return available_light_names
    requested = [name.strip() for name in light_target.split(",") if name.strip()]
    available_set = set(available_light_names)
    return [name for name in requested if name in available_set]


def _resolve_perturb_object_names(env: OffScreenRenderEnv, perturb_target: str) -> list[str]:
    movable_objects = set(env.env.objects_dict.keys())
    if perturb_target == "obj_of_interest":
        return [name for name in env.obj_of_interest if name in movable_objects]
    if perturb_target == "all_movable":
        return sorted(movable_objects)
    names = [name.strip() for name in perturb_target.split(",") if name.strip()]
    return [name for name in names if name in movable_objects]


def _get_object_body_names(env: OffScreenRenderEnv, object_name: str) -> list[str]:
    obj = env.env.objects_dict[object_name]
    body_names = [body_name for body_name in getattr(obj, "bodies", []) if body_name]
    root_body = getattr(obj, "root_body", None)
    if root_body and root_body not in body_names:
        body_names.insert(0, root_body)
    return body_names


def _resolve_object_geom_ids(env: OffScreenRenderEnv, object_names: list[str]) -> list[int]:
    model = env.sim.model
    geom_ids: set[int] = set()
    for object_name in object_names:
        for body_name in _get_object_body_names(env, object_name):
            try:
                body_id = model.body_name2id(body_name)
            except (KeyError, ValueError):
                logging.warning(f"Failed to resolve body {body_name} for object {object_name}; skipping its geoms.")
                continue
            geom_ids.update(np.where(model.geom_bodyid == body_id)[0].tolist())
    return sorted(geom_ids)


def _resolve_force_targets(env: OffScreenRenderEnv, object_names: list[str]) -> list[tuple[str, str, int]]:
    model = env.sim.model
    targets: list[tuple[str, str, int]] = []
    for object_name in object_names:
        body_names = _get_object_body_names(env, object_name)
        if not body_names:
            logging.warning(f"No bodies found for object {object_name}; skipping force perturbation target.")
            continue
        body_name = body_names[0]
        try:
            body_id = model.body_name2id(body_name)
        except (KeyError, ValueError):
            logging.warning(f"Failed to resolve body {body_name} for object {object_name}; skipping force target.")
            continue
        targets.append((object_name, body_name, body_id))
    return targets


def _capture_visual_defaults(env: OffScreenRenderEnv) -> dict[str, np.ndarray]:
    model = env.sim.model
    return {
        "light_pos": np.array(model.light_pos, copy=True),
        "light_dir": np.array(model.light_dir, copy=True),
        "light_diffuse": np.array(model.light_diffuse, copy=True),
        "light_ambient": np.array(model.light_ambient, copy=True),
        "light_specular": np.array(model.light_specular, copy=True),
        "light_active": np.array(model.light_active, copy=True),
        "mat_texid": np.array(model.mat_texid, copy=True),
        "mat_rgba": np.array(model.mat_rgba, copy=True),
        "geom_rgba": np.array(model.geom_rgba, copy=True),
    }


def _restore_visual_defaults(env: OffScreenRenderEnv, visual_defaults: dict[str, np.ndarray]) -> None:
    model = env.sim.model
    model.light_pos[:] = visual_defaults["light_pos"]
    model.light_dir[:] = visual_defaults["light_dir"]
    model.light_diffuse[:] = visual_defaults["light_diffuse"]
    model.light_ambient[:] = visual_defaults["light_ambient"]
    model.light_specular[:] = visual_defaults["light_specular"]
    model.light_active[:] = visual_defaults["light_active"]
    model.mat_texid[:] = visual_defaults["mat_texid"]
    model.mat_rgba[:] = visual_defaults["mat_rgba"]
    model.geom_rgba[:] = visual_defaults["geom_rgba"]
    env.sim.forward()


def _apply_light_perturbation(
    env: OffScreenRenderEnv,
    light_names: list[str],
    rng: np.random.Generator,
    args: RobustEvalArgs,
    stage_name: str,
    env_timestep: int,
) -> list[dict[str, object]]:
    if not light_names:
        logging.warning(f"[{stage_name}] No lights matched light_target; skipping light perturbation.")
        return []

    applied: list[dict[str, object]] = []
    model = env.sim.model
    direction_jitter = math.radians(args.light_direction_jitter_deg)

    for light_name in light_names:
        light_id = model.light_name2id(light_name)
        delta_pos = (
            rng.uniform(low=-args.light_position_jitter_m, high=args.light_position_jitter_m, size=3)
            if args.light_position_jitter_m > 0
            else np.zeros(3, dtype=np.float64)
        )
        delta_dir = (
            rng.uniform(low=-direction_jitter, high=direction_jitter, size=3)
            if direction_jitter > 0
            else np.zeros(3, dtype=np.float64)
        )
        ambient_delta = (
            rng.uniform(low=-args.light_ambient_jitter, high=args.light_ambient_jitter, size=3)
            if args.light_ambient_jitter > 0
            else np.zeros(3, dtype=np.float64)
        )
        diffuse_scale = (
            1.0 + rng.uniform(low=-args.light_diffuse_jitter, high=args.light_diffuse_jitter, size=3)
            if args.light_diffuse_jitter > 0
            else np.ones(3, dtype=np.float64)
        )
        specular_delta = (
            rng.uniform(low=-args.light_specular_jitter, high=args.light_specular_jitter, size=3)
            if args.light_specular_jitter > 0
            else np.zeros(3, dtype=np.float64)
        )

        base_dir = np.array(model.light_dir[light_id], dtype=np.float64, copy=True)
        new_dir = base_dir + delta_dir
        norm = np.linalg.norm(new_dir)
        if norm > 1e-8:
            new_dir /= norm
        else:
            new_dir = base_dir

        model.light_pos[light_id] += delta_pos
        model.light_dir[light_id] = new_dir
        model.light_ambient[light_id] = _clip_unit_interval(model.light_ambient[light_id] + ambient_delta)
        model.light_diffuse[light_id] = _clip_unit_interval(model.light_diffuse[light_id] * diffuse_scale)
        model.light_specular[light_id] = _clip_unit_interval(model.light_specular[light_id] + specular_delta)

        applied.append(
            {
                "stage": stage_name,
                "type": "light",
                "env_timestep": int(env_timestep),
                "light_name": light_name,
                "delta_pos_m": delta_pos.astype(np.float64),
                "delta_dir": delta_dir.astype(np.float64),
                "ambient_delta": ambient_delta.astype(np.float64),
                "diffuse_scale": diffuse_scale.astype(np.float64),
                "specular_delta": specular_delta.astype(np.float64),
            }
        )

    env.sim.forward()
    return applied


def _first_valid_texture_id(mat_texid_row: np.ndarray) -> int:
    valid_ids = [int(tex_id) for tex_id in np.asarray(mat_texid_row).tolist() if int(tex_id) >= 0]
    return valid_ids[0] if valid_ids else -1


def _blend_rgb(base_rgb: np.ndarray, target_rgb: np.ndarray, strength: float) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    return _clip_unit_interval((1.0 - strength) * base_rgb + strength * target_rgb)


def _apply_texture_perturbation(
    env: OffScreenRenderEnv,
    geom_ids: list[int],
    rng: np.random.Generator,
    args: RobustEvalArgs,
    visual_defaults: dict[str, np.ndarray],
    stage_name: str,
    env_timestep: int,
) -> list[dict[str, object]]:
    if not geom_ids:
        logging.warning(f"[{stage_name}] No geoms matched perturb_target; skipping texture perturbation.")
        return []

    applied: list[dict[str, object]] = []
    model = env.sim.model
    texture_pool_by_type: dict[int, list[int]] = {}
    for tex_id in range(model.ntex):
        tex_name = model.tex(tex_id).name
        if not tex_name:
            continue
        texture_pool_by_type.setdefault(int(model.tex_type[tex_id]), []).append(tex_id)

    seen_material_ids: set[int] = set()
    seen_geom_ids: set[int] = set()

    for geom_id in geom_ids:
        geom_name = model.geom_id2name(int(geom_id))
        mat_id = int(model.geom_matid[geom_id])
        tint_rgb = rng.uniform(low=0.0, high=1.0, size=3)

        if mat_id >= 0:
            if mat_id in seen_material_ids:
                continue
            seen_material_ids.add(mat_id)

            base_tex_row = np.array(visual_defaults["mat_texid"][mat_id], copy=True)
            base_tex_id = _first_valid_texture_id(base_tex_row)
            used_swap = False
            new_tex_id = -1

            if args.texture_variation == "swap" and base_tex_id >= 0:
                base_tex_type = int(model.tex_type[base_tex_id])
                candidates = [candidate for candidate in texture_pool_by_type.get(base_tex_type, []) if candidate != base_tex_id]
                if candidates:
                    new_tex_id = int(rng.choice(candidates))
                    new_tex_row = np.array(base_tex_row, copy=True)
                    new_tex_row[new_tex_row >= 0] = new_tex_id
                    model.mat_texid[mat_id] = new_tex_row
                    model.mat_rgba[mat_id] = visual_defaults["mat_rgba"][mat_id]
                    model.mat_rgba[mat_id, :3] = 1.0
                    used_swap = True

            if not used_swap:
                model.mat_rgba[mat_id] = visual_defaults["mat_rgba"][mat_id]
                model.mat_rgba[mat_id, :3] = _blend_rgb(
                    visual_defaults["mat_rgba"][mat_id, :3],
                    tint_rgb,
                    args.texture_tint_strength,
                )

            applied.append(
                {
                    "stage": stage_name,
                    "type": "texture",
                    "env_timestep": int(env_timestep),
                    "geom_name": geom_name,
                    "material_id": mat_id,
                    "base_texture_id": base_tex_id,
                    "new_texture_id": new_tex_id if used_swap else None,
                    "variation": "swap" if used_swap else "tint",
                    "tint_rgb": tint_rgb.astype(np.float64),
                }
            )
        else:
            if geom_id in seen_geom_ids:
                continue
            seen_geom_ids.add(geom_id)
            model.geom_rgba[geom_id] = visual_defaults["geom_rgba"][geom_id]
            model.geom_rgba[geom_id, :3] = _blend_rgb(
                visual_defaults["geom_rgba"][geom_id, :3],
                tint_rgb,
                args.texture_tint_strength,
            )
            applied.append(
                {
                    "stage": stage_name,
                    "type": "texture",
                    "env_timestep": int(env_timestep),
                    "geom_name": geom_name,
                    "material_id": None,
                    "base_texture_id": None,
                    "new_texture_id": None,
                    "variation": "geom_tint",
                    "tint_rgb": tint_rgb.astype(np.float64),
                }
            )

    env.sim.forward()
    return applied


def _apply_object_perturbation(
    env: OffScreenRenderEnv,
    object_names: list[str],
    xy_m: float,
    yaw_deg: float,
    rng: np.random.Generator,
    stage_name: str,
    env_timestep: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    applied: list[dict[str, object]] = []
    if not object_names:
        logging.warning(f"[{stage_name}] No movable objects matched perturb_target; skipping perturbation.")
        return env.regenerate_obs_from_state(env.get_sim_state()), applied

    for obj_name in object_names:
        obj = env.env.objects_dict[obj_name]
        joint_name = obj.joints[-1]
        joint_qpos = np.array(env.sim.data.get_joint_qpos(joint_name), dtype=np.float64, copy=True)
        if joint_qpos.ndim == 0 or joint_qpos.shape[0] < 7:
            logging.warning(
                f"[{stage_name}] Skip object {obj_name}: expected a free joint with 7D qpos, got shape {joint_qpos.shape}."
            )
            continue

        delta_xy = rng.uniform(low=-xy_m, high=xy_m, size=2) if xy_m > 0 else np.zeros(2, dtype=np.float64)
        delta_yaw_deg = float(rng.uniform(low=-yaw_deg, high=yaw_deg)) if yaw_deg > 0 else 0.0

        joint_qpos[0] += delta_xy[0]
        joint_qpos[1] += delta_xy[1]
        joint_qpos[3:7] = _normalize_quat_xyzw(
            _quat_multiply_xyzw(_yaw_quat_xyzw(math.radians(delta_yaw_deg)), joint_qpos[3:7])
        )

        env.sim.data.set_joint_qpos(joint_name, joint_qpos)
        applied.append(
            {
                "stage": stage_name,
                "env_timestep": int(env_timestep),
                "object_name": obj_name,
                "joint_name": joint_name,
                "delta_xy_m": delta_xy.astype(np.float64),
                "delta_yaw_deg": delta_yaw_deg,
            }
        )

    env.sim.forward()
    obs = env.regenerate_obs_from_state(env.get_sim_state())
    return obs, applied


def _should_apply_periodic_perturbation(
    policy_step: int,
    perturb_count: int,
    enabled: bool,
    after_steps: int,
    interval: int,
    max_times: int,
) -> bool:
    if not enabled:
        return False
    if max_times <= 0 or perturb_count >= max_times:
        return False
    if policy_step < after_steps:
        return False
    if interval <= 0:
        return policy_step == after_steps
    return (policy_step - after_steps) % interval == 0


def _sample_external_wrench(
    rng: np.random.Generator,
    force_xy_n: float,
    force_z_n: float,
    torque_nm: float,
) -> np.ndarray:
    return np.array(
        [
            float(rng.uniform(low=-force_xy_n, high=force_xy_n)) if force_xy_n > 0 else 0.0,
            float(rng.uniform(low=-force_xy_n, high=force_xy_n)) if force_xy_n > 0 else 0.0,
            float(rng.uniform(low=-force_z_n, high=force_z_n)) if force_z_n > 0 else 0.0,
            float(rng.uniform(low=-torque_nm, high=torque_nm)) if torque_nm > 0 else 0.0,
            float(rng.uniform(low=-torque_nm, high=torque_nm)) if torque_nm > 0 else 0.0,
            float(rng.uniform(low=-torque_nm, high=torque_nm)) if torque_nm > 0 else 0.0,
        ],
        dtype=np.float64,
    )


def _create_force_event(
    env: OffScreenRenderEnv,
    force_targets: list[tuple[str, str, int]],
    rng: np.random.Generator,
    force_xy_n: float,
    force_z_n: float,
    torque_nm: float,
    duration_steps: int,
    stage_name: str,
    env_timestep: int,
    policy_step: int | None = None,
) -> tuple[ActiveForceEvent | None, list[dict[str, object]]]:
    if duration_steps <= 0:
        logging.warning(f"[{stage_name}] Force duration <= 0; skipping force perturbation.")
        return None, []
    if not force_targets:
        logging.warning(f"[{stage_name}] No bodies matched perturb_target; skipping force perturbation.")
        return None, []

    body_wrenches: dict[int, np.ndarray] = {}
    applied: list[dict[str, object]] = []

    for object_name, body_name, body_id in force_targets:
        wrench = _sample_external_wrench(rng, force_xy_n=force_xy_n, force_z_n=force_z_n, torque_nm=torque_nm)
        if np.allclose(wrench, 0.0):
            continue
        body_wrenches[body_id] = wrench
        applied.append(
            {
                "stage": stage_name,
                "type": "force",
                "env_timestep": int(env_timestep),
                "policy_step": None if policy_step is None else int(policy_step),
                "object_name": object_name,
                "body_name": body_name,
                "body_id": int(body_id),
                "duration_steps": int(duration_steps),
                "wrench": wrench.astype(np.float64),
            }
        )

    if not body_wrenches:
        logging.warning(f"[{stage_name}] Sampled zero wrench for all bodies; skipping force perturbation.")
        return None, []

    return ActiveForceEvent(stage_name=stage_name, remaining_steps=duration_steps, body_wrenches=body_wrenches), applied


def _apply_active_force_events(env: OffScreenRenderEnv, active_force_events: list[ActiveForceEvent]) -> None:
    env.sim.data.xfrc_applied[:] = 0.0
    for event in active_force_events:
        for body_id, wrench in event.body_wrenches.items():
            env.sim.data.xfrc_applied[body_id] += wrench


def _advance_active_force_events(active_force_events: list[ActiveForceEvent]) -> list[ActiveForceEvent]:
    still_active: list[ActiveForceEvent] = []
    for event in active_force_events:
        event.remaining_steps -= 1
        if event.remaining_steps > 0:
            still_active.append(event)
    return still_active


def eval_libero(args: RobustEvalArgs) -> None:
    _configure_logging(args.video_out_path)

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")
    logging.info(
        "Robust eval config: target=%s, initial=(enabled=%s, xy=%.4fm, yaw=%.1fdeg), "
        "mid_pose=(enabled=%s, xy=%.4fm, yaw=%.1fdeg, after=%d, interval=%d, max_times=%d), "
        "lights=(enabled=%s, target=%s), textures=(enabled=%s, variation=%s), "
        "initial_force=(enabled=%s, xy=%.2fN, z=%.2fN, torque=%.2fNm, duration=%d), "
        "mid_force=(enabled=%s, xy=%.2fN, z=%.2fN, torque=%.2fNm, duration=%d, after=%d, interval=%d, max_times=%d)",
        args.perturb_target,
        args.initial_perturb,
        args.initial_perturb_xy_m,
        args.initial_perturb_yaw_deg,
        args.mid_perturb,
        args.mid_perturb_xy_m,
        args.mid_perturb_yaw_deg,
        args.mid_perturb_after_steps,
        args.mid_perturb_interval,
        args.mid_perturb_max_times,
        args.light_perturb,
        args.light_target,
        args.texture_perturb,
        args.texture_variation,
        args.initial_force_perturb,
        args.initial_force_xy_n,
        args.initial_force_z_n,
        args.initial_torque_nm,
        args.initial_force_duration_steps,
        args.mid_force_perturb,
        args.mid_force_xy_n,
        args.mid_force_z_n,
        args.mid_torque_nm,
        args.mid_force_duration_steps,
        args.mid_force_after_steps,
        args.mid_force_interval,
        args.mid_force_max_times,
    )

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Setup debug logging directory
    debug_log_dir = None
    if args.debug_log_path:
        debug_log_dir = pathlib.Path(args.debug_log_path)
        debug_log_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Debug logging enabled, saving to: {debug_log_dir}")

    # Start evaluation
    total_episodes, total_successes = 0, 0

    # Determine which tasks to evaluate
    if args.task_id is not None:
        task_ids = args.task_id if isinstance(args.task_id, list) else [args.task_id]
        logging.info(f"Evaluating tasks: {task_ids}")
    else:
        task_ids = range(num_tasks_in_suite)
        logging.info(f"Evaluating all {num_tasks_in_suite} tasks")

    for task_id in tqdm.tqdm(task_ids):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        perturb_object_names = _resolve_perturb_object_names(env, args.perturb_target)
        light_names = _resolve_light_names(env, args.light_target)
        texture_geom_ids = _resolve_object_geom_ids(env, perturb_object_names)
        force_targets = _resolve_force_targets(env, perturb_object_names)
        visual_defaults = _capture_visual_defaults(env)
        logging.info(f"Perturbation candidates for task {task_id}: {perturb_object_names}")
        logging.info(
            "Task %d perturbation resources: lights=%s, texture_geoms=%d, force_targets=%s",
            task_id,
            light_names,
            len(texture_geom_ids),
            [body_name for _, body_name, _ in force_targets],
        )

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            _restore_visual_defaults(env, visual_defaults)
            env.sim.data.xfrc_applied[:] = 0.0
            action_plan = collections.deque()
            hist_len = max(args.agent_temporal_step_offsets)
            agent_img_history: collections.deque = collections.deque(maxlen=hist_len)

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])
            perturbation_records: list[dict[str, object]] = []
            mid_perturb_count = 0
            mid_force_count = 0
            active_force_events: list[ActiveForceEvent] = []

            if args.initial_perturb:
                obs, applied = _apply_object_perturbation(
                    env=env,
                    object_names=perturb_object_names,
                    xy_m=args.initial_perturb_xy_m,
                    yaw_deg=args.initial_perturb_yaw_deg,
                    rng=_get_episode_rng(args, task_id, episode_idx, stage_offset=0),
                    stage_name="initial",
                    env_timestep=0,
                )
                perturbation_records.extend(applied)
                if applied:
                    logging.info(f"Applied initial perturbation to {[event['object_name'] for event in applied]}")

            visual_updates = False
            if args.light_perturb:
                applied = _apply_light_perturbation(
                    env=env,
                    light_names=light_names,
                    rng=_get_episode_rng(args, task_id, episode_idx, stage_offset=100),
                    args=args,
                    stage_name="light_initial",
                    env_timestep=0,
                )
                perturbation_records.extend(applied)
                if applied:
                    visual_updates = True
                    logging.info(f"Applied light perturbation to {[event['light_name'] for event in applied]}")

            if args.texture_perturb:
                applied = _apply_texture_perturbation(
                    env=env,
                    geom_ids=texture_geom_ids,
                    rng=_get_episode_rng(args, task_id, episode_idx, stage_offset=200),
                    args=args,
                    visual_defaults=visual_defaults,
                    stage_name="texture_initial",
                    env_timestep=0,
                )
                perturbation_records.extend(applied)
                if applied:
                    visual_updates = True
                    logging.info("Applied texture perturbation to %d geom/material targets.", len(applied))

            if visual_updates:
                obs = env.regenerate_obs_from_state(env.get_sim_state())

            if args.initial_force_perturb:
                force_event, applied = _create_force_event(
                    env=env,
                    force_targets=force_targets,
                    rng=_get_episode_rng(args, task_id, episode_idx, stage_offset=300),
                    force_xy_n=args.initial_force_xy_n,
                    force_z_n=args.initial_force_z_n,
                    torque_nm=args.initial_torque_nm,
                    duration_steps=args.initial_force_duration_steps,
                    stage_name="initial_force",
                    env_timestep=0,
                )
                perturbation_records.extend(applied)
                if force_event is not None:
                    active_force_events.append(force_event)
                    logging.info("Scheduled initial force perturbation for %s", [event["object_name"] for event in applied])

            # Setup
            t = 0
            replay_images = []
            debug_records = []  # per-step debug records for this episode
            done = False

            logging.info(f"Starting episode {task_episodes + 1}...")
            logging.info(
                f"Simulator warmup: {args.num_steps_wait} env steps for objects to settle "
                "(dummy actions only; no policy calls yet)."
            )
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        _apply_active_force_events(env, active_force_events)
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        active_force_events = _advance_active_force_events(active_force_events)
                        t += 1
                        continue

                    if t == args.num_steps_wait:
                        logging.info(f"Warmup finished at env timestep {t}. Starting policy-controlled rollout.")

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    agent_img_stack = _stack_agent_temporal_frames(
                        agent_img_history, img, args.agent_temporal_step_offsets
                    )

                    # Save preprocessed image for replay video
                    if args.save_videos:
                        replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": agent_img_stack,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        logging.debug(
                            f"Querying policy server for a new action chunk (env timestep {t}). "
                            "First call can take a long time; if this line stays for many minutes, "
                            "check the policy server process and GPU in another terminal."
                        )
                        infer_result = client.infer(element)
                        action_chunk = infer_result["actions"]
                        logging.debug(
                            f"Received action chunk of length {len(action_chunk)} "
                            f"(using {args.replan_steps} steps per replan)."
                        )
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                        # Debug logging: record input and output
                        if debug_log_dir:
                            debug_records.append({
                                "timestep": t,
                                "state": element["observation/state"].copy(),
                                "image": img.copy(),
                                "agent_image_stack": agent_img_stack.copy(),
                                "wrist_image": wrist_img.copy(),
                                "image_hash": hashlib.md5(img.tobytes()).hexdigest(),
                                "agent_image_stack_hash": hashlib.md5(agent_img_stack.tobytes()).hexdigest(),
                                "wrist_image_hash": hashlib.md5(wrist_img.tobytes()).hexdigest(),
                                "prompt": element["prompt"],
                                "action_chunk": [a.tolist() if hasattr(a, 'tolist') else a for a in action_chunk],
                            })

                    action = action_plan.popleft()
                    policy_step = t - args.num_steps_wait + 1

                    if _should_apply_periodic_perturbation(
                        policy_step=policy_step,
                        perturb_count=mid_force_count,
                        enabled=args.mid_force_perturb,
                        after_steps=args.mid_force_after_steps,
                        interval=args.mid_force_interval,
                        max_times=args.mid_force_max_times,
                    ):
                        force_event, applied = _create_force_event(
                            env=env,
                            force_targets=force_targets,
                            rng=_get_episode_rng(
                                args,
                                task_id,
                                episode_idx,
                                stage_offset=400 + mid_force_count,
                            ),
                            force_xy_n=args.mid_force_xy_n,
                            force_z_n=args.mid_force_z_n,
                            torque_nm=args.mid_torque_nm,
                            duration_steps=args.mid_force_duration_steps,
                            stage_name=f"mid_force_{mid_force_count + 1}",
                            env_timestep=t,
                            policy_step=policy_step,
                        )
                        perturbation_records.extend(applied)
                        if force_event is not None:
                            active_force_events.append(force_event)
                            mid_force_count += 1
                            action_plan.clear()
                            logging.info(
                                "Applied mid force perturbation #%d at rollout step %d to %s; cleared current action chunk.",
                                mid_force_count,
                                policy_step,
                                [event["object_name"] for event in applied],
                            )

                    # Execute action in environment
                    _apply_active_force_events(env, active_force_events)
                    obs, reward, done, info = env.step(action.tolist())
                    active_force_events = _advance_active_force_events(active_force_events)
                    agent_img_history.append(img)
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break

                    if _should_apply_periodic_perturbation(
                        policy_step=policy_step,
                        perturb_count=mid_perturb_count,
                        enabled=args.mid_perturb,
                        after_steps=args.mid_perturb_after_steps,
                        interval=args.mid_perturb_interval,
                        max_times=args.mid_perturb_max_times,
                    ):
                        obs, applied = _apply_object_perturbation(
                            env=env,
                            object_names=perturb_object_names,
                            xy_m=args.mid_perturb_xy_m,
                            yaw_deg=args.mid_perturb_yaw_deg,
                            rng=_get_episode_rng(
                                args,
                                task_id,
                                episode_idx,
                                stage_offset=mid_perturb_count + 1,
                            ),
                            stage_name=f"mid_{mid_perturb_count + 1}",
                            env_timestep=t + 1,
                        )
                        perturbation_records.extend(applied)
                        if applied:
                            mid_perturb_count += 1
                            action_plan.clear()
                            logging.info(
                                "Applied mid perturbation #%d at rollout step %d to %s; cleared current action chunk.",
                                mid_perturb_count,
                                policy_step,
                                [event["object_name"] for event in applied],
                            )
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            env.sim.data.xfrc_applied[:] = 0.0
            task_episodes += 1
            total_episodes += 1

            # Save debug records for this episode
            if debug_log_dir and debug_records:
                pkl_path = debug_log_dir / f"task_{task_id}_ep_{episode_idx}.pkl"
                with open(pkl_path, "wb") as f:
                    pickle.dump({
                        "task_id": task_id,
                        "episode_idx": episode_idx,
                        "task_description": str(task_description),
                        "success": done,
                        "total_steps": t,
                        "perturbations": perturbation_records,
                        "records": debug_records,
                    }, f)

            # Save a replay video of the episode
            if args.save_videos:
                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def eval_init_pose(args: InitPoseArgs) -> None:
    eval_libero(build_init_pose_config(args))


def eval_light(args: LightArgs) -> None:
    eval_libero(build_light_config(args))


def eval_texture(args: TextureArgs) -> None:
    eval_libero(build_texture_config(args))


def eval_init_force(args: InitForceArgs) -> None:
    eval_libero(build_init_force_config(args))


def eval_mid_force(args: MidForceArgs) -> None:
    eval_libero(build_mid_force_config(args))


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)

    # Set seed - handle both old gym API (seed method) and new API (reset with seed)
    try:
        env.seed(seed)  # Old gym API
    except (TypeError, AttributeError):
        # New gym/gymnasium API - seed is passed to reset()
        pass  # Will be handled in reset() call

    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
