#!/usr/bin/env python3
"""Create a task-family-filtered RoboCasa subset with remapped episode/task ids.

The legacy LeRobot loader expects each row's ``episode_index`` and ``task_index`` to
match the subset metadata. A plain symlinked subset keeps the original parquet ids and
breaks small-subset training, so this script rewrites a tiny filtered copy.

Example:
  /home/zijianzhang/openpi/uv_venv/bin/python scripts/create_robocasa_task_subset.py \
      --source-dataset-root /data/zijianzhang/robocasa-H50-legacy-v4 \
      --output-dataset-root /home/yuqingjiang/openpi_shared/data/robocasa-H50-overfit-pnpcab \
      --task-family PnPCounterToCab \
      --max-episodes 8 \
      --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path


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


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _rewrite_episode_parquet(
    src_parquet: Path,
    dst_parquet: Path,
    *,
    new_episode_index: int,
    task_index_map: dict[int, int],
    global_frame_offset: int,
) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(src_parquet)
    num_rows = table.num_rows
    names = table.column_names

    if "episode_index" not in names or "task_index" not in names:
        raise KeyError(f"Expected episode_index and task_index in {src_parquet}")

    column_index = {name: i for i, name in enumerate(names)}

    episode_type = table.schema.field("episode_index").type
    task_type = table.schema.field("task_index").type
    table = table.set_column(
        column_index["episode_index"],
        "episode_index",
        pa.array([new_episode_index] * num_rows, type=episode_type),
    )

    old_task_values = table["task_index"].to_pylist()
    try:
        new_task_values = [task_index_map[int(value)] for value in old_task_values]
    except KeyError as exc:
        raise KeyError(f"Found task_index {exc.args[0]} in {src_parquet} that is missing from subset task map") from exc
    table = table.set_column(
        column_index["task_index"],
        "task_index",
        pa.array(new_task_values, type=task_type),
    )

    if "index" in column_index:
        index_type = table.schema.field("index").type
        table = table.set_column(
            column_index["index"],
            "index",
            pa.array(range(global_frame_offset, global_frame_offset + num_rows), type=index_type),
        )

    dst_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst_parquet)
    return num_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a task-family-filtered RoboCasa subset.")
    parser.add_argument("--source-dataset-root", type=Path, required=True)
    parser.add_argument("--output-dataset-root", type=Path, required=True)
    parser.add_argument("--task-family", type=str, required=True)
    parser.add_argument("--max-episodes", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_root = args.source_dataset_root.resolve()
    out_root = args.output_dataset_root.resolve()

    meta_dir = source_root / "meta"
    info = _load_json(meta_dir / "info.json")
    tasks = _load_jsonl(meta_dir / "tasks.jsonl")
    episodes = _load_jsonl(meta_dir / "episodes.jsonl")

    family_by_task_name = {row["task"]: map_family(str(row["task"])) for row in tasks}
    selected_task_rows = [row for row in tasks if family_by_task_name[row["task"]] == args.task_family]
    if not selected_task_rows:
        raise ValueError(f"No tasks found for task family {args.task_family!r}")

    remapped_task_rows: list[dict] = []
    task_index_map: dict[int, int] = {}
    selected_task_names: set[str] = set()
    for new_task_index, row in enumerate(selected_task_rows):
        original_task_index = int(row["task_index"])
        task_index_map[original_task_index] = new_task_index
        selected_task_names.add(row["task"])
        remapped_task_rows.append({"task_index": new_task_index, "task": row["task"]})

    selected_episodes = [row for row in episodes if any(task in selected_task_names for task in row.get("tasks", []))]
    selected_episodes = selected_episodes[: args.max_episodes]
    if not selected_episodes:
        raise ValueError(f"No episodes found for task family {args.task_family!r}")

    if out_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output dataset root already exists: {out_root}")
        shutil.rmtree(out_root)

    (out_root / "meta").mkdir(parents=True, exist_ok=True)
    (out_root / "data").mkdir(parents=True, exist_ok=True)

    subset_episode_rows: list[dict] = []
    episode_mapping_rows: list[dict] = []
    total_frames = 0
    global_frame_offset = 0

    for new_episode_index, episode_row in enumerate(selected_episodes):
        original_episode_index = int(episode_row["episode_index"])
        old_chunk = f"chunk-{original_episode_index // args.chunk_size:03d}"
        new_chunk = f"chunk-{new_episode_index // args.chunk_size:03d}"
        src_parquet = source_root / "data" / old_chunk / f"episode_{original_episode_index:06d}.parquet"
        dst_parquet = out_root / "data" / new_chunk / f"episode_{new_episode_index:06d}.parquet"
        if not src_parquet.exists():
            raise FileNotFoundError(f"Missing source episode parquet: {src_parquet}")

        length = _rewrite_episode_parquet(
            src_parquet,
            dst_parquet,
            new_episode_index=new_episode_index,
            task_index_map=task_index_map,
            global_frame_offset=global_frame_offset,
        )
        total_frames += length
        global_frame_offset += length

        subset_episode_rows.append(
            {
                "episode_index": new_episode_index,
                "tasks": episode_row.get("tasks", []),
                "length": length,
            }
        )
        episode_mapping_rows.append(
            {
                "new_episode_index": new_episode_index,
                "original_episode_index": original_episode_index,
                "tasks": episode_row.get("tasks", []),
                "length": length,
            }
        )

    subset_info = dict(info)
    subset_info["total_episodes"] = len(selected_episodes)
    subset_info["total_frames"] = total_frames
    subset_info["total_tasks"] = len(remapped_task_rows)
    subset_info["total_chunks"] = max(1, math.ceil(len(selected_episodes) / args.chunk_size))
    subset_info["chunks_size"] = args.chunk_size
    subset_info["splits"] = {"train": f"0:{len(selected_episodes)}"}
    subset_info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"

    (out_root / "meta" / "info.json").write_text(
        json.dumps(subset_info, ensure_ascii=True, indent=4) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(out_root / "meta" / "tasks.jsonl", remapped_task_rows)
    _write_jsonl(out_root / "meta" / "episodes.jsonl", subset_episode_rows)

    stats_src = meta_dir / "stats.json"
    if stats_src.exists():
        shutil.copy2(stats_src, out_root / "meta" / "stats.json")

    _write_json(
        out_root / "meta" / "subset_mapping.json",
        {
            "source_dataset_root": str(source_root),
            "task_family": args.task_family,
            "num_selected_tasks": len(remapped_task_rows),
            "num_selected_episodes": len(selected_episodes),
            "total_selected_frames": total_frames,
            "episode_mappings": episode_mapping_rows,
            "task_index_mapping": task_index_map,
        },
    )

    print(
        json.dumps(
            {
                "output_dataset_root": str(out_root),
                "task_family": args.task_family,
                "selected_task_count": len(remapped_task_rows),
                "selected_episode_count": len(selected_episodes),
                "selected_frame_count": total_frames,
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
