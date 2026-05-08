#!/usr/bin/env python3
"""Offline train-set imitation check for RoboCasa checkpoints.

This script loads a trained checkpoint locally, replays observations from a
LeRobot-format RoboCasa dataset, and compares the model's predicted actions
against the recorded dataset actions without doing environment rollout.

Typical usage:
  /home/zijianzhang/openpi/uv_venv/bin/python scripts/check_robocasa_trainset_imitation.py \
      --config-name pi05_robocasa_overfit \
      --checkpoint-dir /home/yuqingjiang/openpi_shared/train_ckpts/pi05_robocasa_overfit/pnpcab_overfit_noinit/2000 \
      --dataset-root /home/yuqingjiang/openpi_shared/data/robocasa-H50-overfit-pnpcab \
      --prompt-mode exact \
      --device cuda:0
"""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import io
import numpy as np
import openpi
from PIL import Image
import pandas as pd
import torch

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


MAIN_IMAGE_KEY = "observation.images.robot0_agentview_left_image"
WRIST_IMAGE_KEY = "observation.images.robot0_eye_in_hand_image"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
TASK_KEY = "task_index"


FAMILY_PROMPTS = {
    "PnPCounterToCab": "pick and place from counter to cabinet",
    "PnPCounterToSink": "pick and place from counter to sink",
    "PnPMicrowaveToCounter": "pick and place from microwave to counter",
    "PnPStoveToCounter": "pick and place from stove to counter",
    "OpenSingleDoor": "open cabinet or microwave door",
    "CloseDrawer": "close drawer",
    "TurnOnMicrowave": "turn on microwave",
    "TurnOnSinkFaucet": "turn on sink faucet",
    "TurnOnStove": "turn on stove",
    "ArrangeVegetables": "arrange vegetables on a cutting board",
    "MicrowaveThawing": "place frozen food in microwave for thawing",
    "RestockPantry": "restock cans in pantry",
    "PreSoakPan": "prepare pan for washing",
    "PrepareCoffee": "make coffee",
}


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


@dataclass
class SampleResult:
    episode: str
    step: int
    prompt: str
    task_text: str
    first_l2_12: float
    first_l2_7: float
    first_l1_12: float
    first_max_abs_12: float
    chunk_l2_12: float
    pred_abs_max: float
    gt_abs_max: float


def _load_task_mapping(dataset_root: Path) -> dict[int, str]:
    parquet_path = dataset_root / "meta" / "tasks.parquet"
    jsonl_path = dataset_root / "meta" / "tasks.jsonl"
    if parquet_path.exists():
        tasks_df = pd.read_parquet(parquet_path)
        return {
            int(task_index): str(text)
            for text, task_index in zip(tasks_df.index, tasks_df["task_index"], strict=True)
        }
    if jsonl_path.exists():
        mapping: dict[int, str] = {}
        with jsonl_path.open() as f:
            for line in f:
                record = json.loads(line)
                mapping[int(record["task_index"])] = str(record["task"])
        return mapping
    raise FileNotFoundError(f"Could not find {parquet_path} or {jsonl_path}")


def _iter_episode_paths(dataset_root: Path) -> list[Path]:
    return sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet"))


def _resolve_device(device_arg: str | None) -> str:
    if device_arg:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_prompt(task_text: str, prompt_mode: str, prompt_text: str | None) -> str:
    if prompt_mode == "exact":
        return task_text
    if prompt_mode == "family":
        family = map_family(task_text)
        return FAMILY_PROMPTS.get(family, task_text)
    if prompt_mode == "fixed":
        if prompt_text is None:
            raise ValueError("--prompt-text is required when --prompt-mode=fixed")
        return prompt_text
    raise ValueError(f"Unsupported prompt mode: {prompt_mode}")


def _stack_actions(series: pd.Series) -> np.ndarray:
    return np.stack([np.asarray(v, dtype=np.float32) for v in series.tolist()], axis=0)


def _parse_image(img_data):
    """万能图像解析器：无论拿到什么格式，都转成标准 uint8 numpy 数组"""
    if isinstance(img_data, dict) and "bytes" in img_data:
        img = Image.open(io.BytesIO(img_data["bytes"]))
    elif isinstance(img_data, bytes):
        img = Image.open(io.BytesIO(img_data))
    else:
        img = img_data  # 假设已经是 PIL Image 或 numpy 数组
    return np.asarray(img, dtype=np.uint8)

def _make_request(row, prompt):
    return {
        # 使用万能解析器处理双视角图像
        "observation/image": _parse_image(row["observation.images.robot0_agentview_left_image"]),
        "observation/wrist_image": _parse_image(row["observation.images.robot0_eye_in_hand_image"]),
        # 确保 state 也是纯净的 float32 数组
        "observation/state": np.asarray(row["observation.state"], dtype=np.float32),
        "prompt": prompt
    }


def _summarize_metric(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.9)),
        "max": float(np.max(arr)),
    }


def _print_runtime_sanity(policy, checkpoint_dir: Path) -> None:
    metadata_path = checkpoint_dir / "metadata.pt"
    print("runtime_sanity:")
    print(f"  openpi_package: {Path(inspect.getfile(openpi)).resolve()}")
    print(f"  policy_config_module: {Path(inspect.getfile(_policy_config)).resolve()}")
    print(f"  metadata_exists: {metadata_path.exists()} ({metadata_path})")

    model = getattr(policy, "_model", None)
    model_config = getattr(model, "config", None)
    if model_config is not None:
        print(
            "  loaded_model_config: "
            f"action_dim={getattr(model_config, 'action_dim', None)} "
            f"action_horizon={getattr(model_config, 'action_horizon', None)} "
            f"max_token_len={getattr(model_config, 'max_token_len', None)} "
            f"action_loss_dim={getattr(model_config, 'action_loss_dim', None)} "
            f"mask_inactive_action_dims={getattr(model_config, 'mask_inactive_action_dims', None)} "
            f"first_action_loss_weight={getattr(model_config, 'first_action_loss_weight', None)}"
        )
    else:
        print("  loaded_model_config: unavailable")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline train-set imitation check for RoboCasa checkpoints.")
    parser.add_argument("--config-name", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None, help="PyTorch device, e.g. cpu / cuda / cuda:0")
    parser.add_argument("--prompt-mode", type=str, default="exact", choices=("exact", "family", "fixed"))
    parser.add_argument("--prompt-text", type=str, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-steps-per-episode", type=int, default=None)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument(
        "--compare-horizon",
        type=int,
        default=1,
        help="Compare up to this many actions from the predicted chunk against future dataset actions.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.compare_horizon <= 0:
        raise ValueError("--compare-horizon must be positive")

    dataset_root = args.dataset_root.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve()
    device = _resolve_device(args.device)
    task_text_by_index = _load_task_mapping(dataset_root)
    episode_paths = _iter_episode_paths(dataset_root)
    if args.max_episodes is not None:
        episode_paths = episode_paths[: args.max_episodes]
    if not episode_paths:
        raise ValueError(f"No episode parquet files found under {dataset_root / 'data'}")

    print(f"config_name: {args.config_name}")
    print(f"checkpoint_dir: {checkpoint_dir}")
    print(f"dataset_root: {dataset_root}")
    print(f"device: {device}")
    print(f"prompt_mode: {args.prompt_mode}")
    print(f"episodes_to_check: {len(episode_paths)}")
    print()

    train_config = _config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        pytorch_device=device,
    )
    _print_runtime_sanity(policy, checkpoint_dir)

    results: list[SampleResult] = []

    with torch.inference_mode():
        for episode_idx, episode_path in enumerate(episode_paths, start=1):
            df = pd.read_parquet(
                episode_path,
                columns=[MAIN_IMAGE_KEY, WRIST_IMAGE_KEY, STATE_KEY, ACTION_KEY, TASK_KEY],
            )
            if len(df) == 0:
                continue

            task_index = int(df[TASK_KEY].iloc[0])
            task_text = task_text_by_index.get(task_index, "UNKNOWN")
            prompt = _resolve_prompt(task_text, args.prompt_mode, args.prompt_text)
            gt_actions = _stack_actions(df[ACTION_KEY])

            step_indices = list(range(0, len(df), args.stride))
            if args.max_steps_per_episode is not None:
                step_indices = step_indices[: args.max_steps_per_episode]

            print(
                f"[Episode {episode_idx}/{len(episode_paths)}] {episode_path.name} | "
                f"task={task_text!r} | prompt={prompt!r} | sampled_steps={len(step_indices)}"
            )

            episode_results: list[SampleResult] = []
            for step in step_indices:
                row = df.iloc[step]
                pred_chunk = np.asarray(policy.infer(_make_request(row, prompt))["actions"], dtype=np.float32)
                if pred_chunk.ndim != 2:
                    raise ValueError(f"Expected policy action chunk [T, D], got shape {pred_chunk.shape}")

                compare_horizon = min(args.compare_horizon, len(pred_chunk), len(gt_actions) - step)
                pred_first = pred_chunk[0, :12]
                gt_first = gt_actions[step, :12]
                pred_compare = pred_chunk[:compare_horizon, :12]
                gt_compare = gt_actions[step : step + compare_horizon, :12]

                sample = SampleResult(
                    episode=episode_path.name,
                    step=step,
                    prompt=prompt,
                    task_text=task_text,
                    first_l2_12=float(np.linalg.norm(pred_first - gt_first)),
                    first_l2_7=float(np.linalg.norm(pred_first[:7] - gt_first[:7])),
                    first_l1_12=float(np.mean(np.abs(pred_first - gt_first))),
                    first_max_abs_12=float(np.max(np.abs(pred_first - gt_first))),
                    chunk_l2_12=float(np.sqrt(np.mean((pred_compare - gt_compare) ** 2))),
                    pred_abs_max=float(np.max(np.abs(pred_first))),
                    gt_abs_max=float(np.max(np.abs(gt_first))),
                )
                results.append(sample)
                episode_results.append(sample)

            if episode_results:
                l2s = [item.first_l2_12 for item in episode_results]
                chunk_l2s = [item.chunk_l2_12 for item in episode_results]
                print(
                    f"  first_l2_12 mean={np.mean(l2s):.4f} median={np.median(l2s):.4f} max={np.max(l2s):.4f} | "
                    f"chunk_rmse_12 mean={np.mean(chunk_l2s):.4f}"
                )
            print()

    if not results:
        raise ValueError("No samples were evaluated.")

    first_l2_12 = [item.first_l2_12 for item in results]
    first_l2_7 = [item.first_l2_7 for item in results]
    first_l1_12 = [item.first_l1_12 for item in results]
    first_max_abs_12 = [item.first_max_abs_12 for item in results]
    chunk_l2_12 = [item.chunk_l2_12 for item in results]
    pred_abs_max = [item.pred_abs_max for item in results]
    gt_abs_max = [item.gt_abs_max for item in results]

    summary = {
        "config_name": args.config_name,
        "checkpoint_dir": str(checkpoint_dir),
        "dataset_root": str(dataset_root),
        "device": device,
        "prompt_mode": args.prompt_mode,
        "prompt_text": args.prompt_text,
        "num_episodes": len(episode_paths),
        "num_samples": len(results),
        "metrics": {
            "first_l2_12": _summarize_metric(first_l2_12),
            "first_l2_7": _summarize_metric(first_l2_7),
            "first_l1_12": _summarize_metric(first_l1_12),
            "first_max_abs_12": _summarize_metric(first_max_abs_12),
            "chunk_l2_12": _summarize_metric(chunk_l2_12),
            "pred_abs_max": _summarize_metric(pred_abs_max),
            "gt_abs_max": _summarize_metric(gt_abs_max),
        },
        "worst_samples_by_first_l2_12": [
            asdict(item) for item in sorted(results, key=lambda item: item.first_l2_12, reverse=True)[:10]
        ],
    }

    print("=== Aggregate Summary ===")
    print(f"num_samples: {summary['num_samples']}")
    for key, metrics in summary["metrics"].items():
        print(
            f"{key}: mean={metrics['mean']:.4f} median={metrics['median']:.4f} "
            f"p90={metrics['p90']:.4f} max={metrics['max']:.4f}"
        )

    if args.output_json is not None:
        output_path = args.output_json.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print()
        print(f"saved_summary: {output_path}")


if __name__ == "__main__":
    main()
