#!/usr/bin/env python3
"""Analyze a LeRobot-formatted RoboCasa dataset and map language tasks to atomic families.

Example:
  ./uv_venv/bin/python scripts/analyze_robocasa_h50.py --dataset-root /data/zijianzhang/robocasa-H50
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import pandas as pd


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

    raise ValueError(f"Unmapped RoboCasa task: {text}")


def load_task_mapping(dataset_root: Path) -> dict[int, str]:
    tasks_df = pd.read_parquet(dataset_root / "meta" / "tasks.parquet")
    return {int(task_index): text for text, task_index in zip(tasks_df.index, tasks_df["task_index"])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a LeRobot-formatted RoboCasa dataset.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    task_text_by_index = load_task_mapping(dataset_root)
    family_by_task_index = {task_index: map_family(text) for task_index, text in task_text_by_index.items()}

    data_df = pd.read_parquet(
        dataset_root / "data" / "chunk-000" / "file-000.parquet",
        columns=["episode_index", "task_index"],
    )
    episode_task_df = data_df.groupby("episode_index", as_index=False)["task_index"].first()

    task_count_by_family = collections.Counter(family_by_task_index.values())
    episode_count_by_family = collections.Counter(
        family_by_task_index[int(task_index)] for task_index in episode_task_df["task_index"].tolist()
    )

    print(f"dataset_root: {dataset_root}")
    print(f"num_language_tasks: {len(task_text_by_index)}")
    print(f"num_atomic_families: {len(task_count_by_family)}")
    print(f"num_episodes: {len(episode_task_df)}")
    print()
    print("Atomic family summary:")
    print("family\tlanguage_tasks\tepisodes")
    for family in sorted(task_count_by_family):
        print(f"{family}\t{task_count_by_family[family]}\t{episode_count_by_family[family]}")


if __name__ == "__main__":
    main()
