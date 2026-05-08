#!/usr/bin/env python3
"""Sanity-check RoboCasa action semantics against next-state EEF motion.

This script helps answer questions like:
- Does `action[0]` have the same sign as next-step EEF x motion?
- Are the first 3 action dims aligned with `robot0_base_to_eef_pos` deltas?
- Are gripper actions correlated with `robot0_gripper_qpos` changes?
- Do saved training images have the expected orientation?

Example:
  /home/zijianzhang/openpi/uv_venv/bin/python scripts/check_robocasa_action_consistency.py \
      --dataset-root /data/zijianzhang/robocasa-H50-legacy-v4 \
      --task-family PnPCounterToCab \
      --max-episodes 50 \
      --save-sample-images /tmp/robocasa_samples
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


MAIN_IMAGE_KEY = "observation.images.robot0_agentview_left_image"
WRIST_IMAGE_KEY = "observation.images.robot0_eye_in_hand_image"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
TASK_KEY = "task_index"
TIMESTAMP_KEY = "timestamp"


def map_family(text: str) -> str:
    if text == "close the cabinet doors":
        return "CloseDoubleDoor"
    if text in {"close the cabinet door", "close the microwave door"}:
        return "CloseSingleDoor"
    if text in {"close the left drawer", "close the right drawer"}:
        return "CloseDrawer"
    if text == "open the cabinet doors":
        return "OpenDoubleDoor"
    if text in {"open the cabinet door", "open the microwave door"}:
        return "OpenSingleDoor"
    if text in {"open the left drawer", "open the right drawer"}:
        return "OpenDrawer"
    if text == "press the button on the coffee machine to serve coffee":
        return "CoffeePressButton"
    if text == "pick the mug from under the coffee machine dispenser and place it on the counter":
        return "CoffeeServeMug"
    if text == "pick the mug from the counter and place it under the coffee machine dispenser":
        return "CoffeeSetupMug"
    if text == "press the start button on the microwave":
        return "TurnOnMicrowave"
    if text == "press the stop button on the microwave":
        return "TurnOffMicrowave"
    if text == "turn on the sink faucet":
        return "TurnOnSinkFaucet"
    if text == "turn off the sink faucet":
        return "TurnOffSinkFaucet"
    if text.startswith("turn the sink spout"):
        return "TurnSinkSpout"
    if text.startswith("turn on the ") and "burner of the stove" in text:
        return "TurnOnStove"
    if text.startswith("turn off the ") and "burner of the stove" in text:
        return "TurnOffStove"
    if text.startswith("pick the "):
        if " from the cabinet and place it on the counter" in text:
            return "PnPCabToCounter"
        if " from the counter and place it in the cabinet" in text:
            return "PnPCounterToCab"
        if " from the counter and place it in the sink" in text:
            return "PnPCounterToSink"
        if " from the sink and place it on the plate located on the counter" in text:
            return "PnPSinkToCounter"
        if " from the counter and place it in the microwave" in text:
            return "PnPCounterToMicrowave"
        if " from the microwave and place it on plate located on the counter" in text:
            return "PnPMicrowaveToCounter"
        if " from the plate and place it in the pan" in text:
            return "PnPCounterToStove"
        if " from the pan and place it " in text and (" on the plate" in text or " in the bowl" in text):
            return "PnPStoveToCounter"
    return "UNKNOWN"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_task_mapping(dataset_root: Path) -> dict[int, str]:
    parquet_path = dataset_root / "meta" / "tasks.parquet"
    jsonl_path = dataset_root / "meta" / "tasks.jsonl"
    if parquet_path.exists():
        tasks_df = pd.read_parquet(parquet_path)
        return {int(task_index): str(text) for text, task_index in zip(tasks_df.index, tasks_df["task_index"], strict=True)}
    if jsonl_path.exists():
        mapping = {}
        with jsonl_path.open() as f:
            for line in f:
                record = json.loads(line)
                mapping[int(record["task_index"])] = str(record["task"])
        return mapping
    raise FileNotFoundError(f"Could not find {parquet_path} or {jsonl_path}")


def _iter_episode_paths(dataset_root: Path) -> list[Path]:
    return sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet"))


def _to_array_series(series: pd.Series) -> np.ndarray:
    return np.stack([np.asarray(v, dtype=np.float32) for v in series.tolist()], axis=0)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std < 1e-12 or y_std < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _sign_agreement(x: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = (np.abs(x) > eps) & (np.abs(y) > eps)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.sign(x[mask]) == np.sign(y[mask])))


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    denom = float(np.dot(x, x))
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(x, y) / denom)


def _parse_image(image) -> np.ndarray:
    if isinstance(image, dict):
        if "bytes" in image:
            return np.array(Image.open(io.BytesIO(image["bytes"])))
        raise ValueError(f"Unexpected image dict keys: {list(image)}")
    arr = np.asarray(image)
    if arr.dtype.kind == "f":
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def _save_sample_images(df: pd.DataFrame, out_dir: Path, episode_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    row = df.iloc[0]
    for key, suffix in [(MAIN_IMAGE_KEY, "agentview"), (WRIST_IMAGE_KEY, "wrist")]:
        image = _parse_image(row[key])
        Image.fromarray(image).save(out_dir / f"{episode_name}_{suffix}.png")


def _episode_summary(actions: np.ndarray, states: np.ndarray) -> dict[str, np.ndarray | float]:
    if len(actions) < 2 or len(states) < 2:
        raise ValueError("Episode must have at least 2 frames.")

    action_t = actions[:-1]
    eef_delta = states[1:, 0:3] - states[:-1, 0:3]
    grip_delta = states[1:, 7:9] - states[:-1, 7:9]

    corr = np.full((7, 3), np.nan, dtype=np.float64)
    slope = np.full((7, 3), np.nan, dtype=np.float64)
    sign = np.full((3,), np.nan, dtype=np.float64)
    for action_idx in range(7):
        for axis in range(3):
            corr[action_idx, axis] = _safe_corr(action_t[:, action_idx], eef_delta[:, axis])
            slope[action_idx, axis] = _linear_slope(action_t[:, action_idx], eef_delta[:, axis])
    for axis in range(3):
        sign[axis] = _sign_agreement(action_t[:, axis], eef_delta[:, axis])

    return {
        "corr": corr,
        "slope": slope,
        "sign": sign,
        "mean_action": np.mean(action_t[:, :7], axis=0),
        "mean_eef_delta": np.mean(eef_delta, axis=0),
        "net_eef_delta": states[-1, 0:3] - states[0, 0:3],
        "grip_corr": _safe_corr(action_t[:, 6], grip_delta[:, 0] - grip_delta[:, 1]),
    }


def _print_matrix(name: str, mat: np.ndarray, row_labels: list[str], col_labels: list[str]) -> None:
    print(name)
    header = " " * 14 + " ".join(f"{label:>12s}" for label in col_labels)
    print(header)
    for row_label, row in zip(row_labels, mat, strict=True):
        values = " ".join(f"{value:12.4f}" if np.isfinite(value) else f"{'nan':>12s}" for value in row)
        print(f"{row_label:>14s} {values}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check RoboCasa action/state consistency.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--task-family", type=str, default="PnPCounterToCab")
    parser.add_argument("--max-episodes", type=int, default=20)
    parser.add_argument("--print-steps", type=int, default=5)
    parser.add_argument("--save-sample-images", type=Path, default=None)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    info = _read_json(dataset_root / "meta" / "info.json")
    task_text_by_index = _load_task_mapping(dataset_root)
    family_by_task = {idx: map_family(text) for idx, text in task_text_by_index.items()}

    print(f"dataset_root: {dataset_root}")
    print(f"robot_type: {info['robot_type']}")
    print(f"state_shape: {info['features'][STATE_KEY]['shape']}")
    print(f"action_shape: {info['features'][ACTION_KEY]['shape']}")
    print(f"task_family filter: {args.task_family}")
    print()

    selected = []
    for episode_path in _iter_episode_paths(dataset_root):
        df = pd.read_parquet(episode_path, columns=[TASK_KEY])
        task_index = int(df[TASK_KEY].iloc[0])
        family = family_by_task.get(task_index, "UNKNOWN")
        if family == args.task_family:
            selected.append((episode_path, task_index))
        if len(selected) >= args.max_episodes:
            break

    if not selected:
        raise ValueError(f"No episodes found for task family {args.task_family!r}")

    print(f"selected_episodes: {len(selected)}")
    print()

    all_actions = []
    all_eef_delta = []
    sample_row_labels = [f"a{i}" for i in range(7)]
    axis_labels = ["dx", "dy", "dz"]

    for episode_idx, (episode_path, task_index) in enumerate(selected):
        columns = [STATE_KEY, ACTION_KEY, TASK_KEY, TIMESTAMP_KEY]
        if args.save_sample_images is not None and episode_idx < 3:
            columns.extend([MAIN_IMAGE_KEY, WRIST_IMAGE_KEY])
        df = pd.read_parquet(episode_path, columns=columns)

        states = _to_array_series(df[STATE_KEY])
        actions = _to_array_series(df[ACTION_KEY])
        task_text = task_text_by_index.get(task_index, "UNKNOWN")
        summary = _episode_summary(actions, states)

        print(f"episode: {episode_path.name}")
        print(f"  task_text: {task_text}")
        print(f"  family: {family_by_task.get(task_index, 'UNKNOWN')}")
        print(f"  net_eef_delta: {np.round(summary['net_eef_delta'], 4)}")
        print(f"  mean_eef_delta_per_step: {np.round(summary['mean_eef_delta'], 5)}")
        print(f"  mean_action[:7]: {np.round(summary['mean_action'], 4)}")
        print(f"  sign_agreement(a0..a2 vs dx..dz): {np.round(summary['sign'], 4)}")
        print(f"  gripper_corr(action6, dqpos0-dqpos1): {summary['grip_corr']:.4f}")

        _print_matrix("  corr(action_i, next_eef_delta)", summary["corr"], sample_row_labels, axis_labels)
        _print_matrix("  slope(next_eef_delta ~ action_i)", summary["slope"], sample_row_labels, axis_labels)

        print("  first_steps:")
        limit = min(args.print_steps, len(actions) - 1)
        for step in range(limit):
            next_delta = states[step + 1, 0:3] - states[step, 0:3]
            print(
                f"    step {step:02d} action[:7]={np.round(actions[step, :7], 4)} "
                f"next_eef_delta={np.round(next_delta, 5)}"
            )
        print()

        if args.save_sample_images is not None and episode_idx < 3:
            _save_sample_images(df, args.save_sample_images, episode_path.stem)

        all_actions.append(actions[:-1, :7])
        all_eef_delta.append(states[1:, 0:3] - states[:-1, 0:3])

    all_actions_arr = np.concatenate(all_actions, axis=0)
    all_eef_delta_arr = np.concatenate(all_eef_delta, axis=0)
    global_corr = np.full((7, 3), np.nan, dtype=np.float64)
    global_slope = np.full((7, 3), np.nan, dtype=np.float64)
    for action_idx in range(7):
        for axis in range(3):
            global_corr[action_idx, axis] = _safe_corr(all_actions_arr[:, action_idx], all_eef_delta_arr[:, axis])
            global_slope[action_idx, axis] = _linear_slope(all_actions_arr[:, action_idx], all_eef_delta_arr[:, axis])

    print("=== Aggregate Summary ===")
    print(f"total_pairs: {len(all_actions_arr)}")
    print(
        "aggregate_sign_agreement(a0..a2 vs dx..dz):",
        np.round([_sign_agreement(all_actions_arr[:, i], all_eef_delta_arr[:, i]) for i in range(3)], 4),
    )
    _print_matrix("aggregate corr(action_i, next_eef_delta)", global_corr, sample_row_labels, axis_labels)
    _print_matrix("aggregate slope(next_eef_delta ~ action_i)", global_slope, sample_row_labels, axis_labels)
    if args.save_sample_images is not None:
        print(f"saved_sample_images: {args.save_sample_images.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except ImportError as exc:
        if "pyarrow" in str(exc) or "fastparquet" in str(exc):
            raise SystemExit(
                "Parquet support is missing. Run this script in an env with `pyarrow`, "
                "for example `/home/zijianzhang/openpi/uv_venv/bin/python`."
            ) from exc
        raise
