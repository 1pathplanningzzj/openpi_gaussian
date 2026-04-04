#!/usr/bin/env python3
"""Create task-filtered LIBERO dataset/flow subsets via symlinks.

This script rebuilds a LeRobot-style dataset root with contiguous episode indices so
the existing training pipeline can consume it without loader changes. Large parquet
and npz payloads are not copied; instead, the subset roots contain symlinks back to
the source files.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path


def _parse_task_indices(value: str) -> list[int]:
    if "," in value:
        items = [item.strip() for item in value.split(",") if item.strip()]
        return [int(item) for item in items]
    if "-" in value:
        start_str, end_str = value.split("-", 1)
        start = int(start_str)
        end = int(end_str)
        if end < start:
            raise ValueError(f"Invalid range: {value}")
        return list(range(start, end + 1))
    return [int(value)]


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _safe_unlink(path: Path) -> None:
    if path.is_symlink() or path.exists():
        path.unlink()


def _symlink_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    _safe_unlink(dst)
    dst.symlink_to(src)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a task-filtered LIBERO subset with matching flow sidecars")
    parser.add_argument("--source-dataset-root", type=Path, required=True)
    parser.add_argument("--source-flow-root", type=Path, required=True)
    parser.add_argument("--output-dataset-root", type=Path, required=True)
    parser.add_argument("--output-flow-root", type=Path, required=True)
    parser.add_argument(
        "--task-indices",
        type=str,
        required=True,
        help="Task indices to keep. Examples: 0-9 or 0,1,2,3",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size to encode in the rebuilt info.json and file layout.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output roots before rebuilding them.",
    )
    args = parser.parse_args()

    task_indices = sorted(set(_parse_task_indices(args.task_indices)))
    if not task_indices:
        raise ValueError("No task indices selected")

    dataset_root = args.source_dataset_root.resolve()
    flow_root = args.source_flow_root.resolve()
    out_dataset_root = args.output_dataset_root.resolve()
    out_flow_root = args.output_flow_root.resolve()

    dataset_meta_dir = dataset_root / "meta"
    source_info = json.loads((dataset_meta_dir / "info.json").read_text(encoding="utf-8"))
    source_tasks = _load_jsonl(dataset_meta_dir / "tasks.jsonl")
    source_episodes = _load_jsonl(dataset_meta_dir / "episodes.jsonl")

    task_name_by_index = {int(row["task_index"]): row["task"] for row in source_tasks}
    selected_task_rows = [row for row in source_tasks if int(row["task_index"]) in task_indices]
    selected_task_names = {row["task"] for row in selected_task_rows}
    if len(selected_task_rows) != len(task_indices):
        missing = sorted(set(task_indices) - {int(row["task_index"]) for row in selected_task_rows})
        raise ValueError(f"Missing task definitions for task_index values: {missing}")

    selected_episodes = [
        row for row in source_episodes
        if any(task_name in selected_task_names for task_name in row.get("tasks", []))
    ]
    if not selected_episodes:
        raise ValueError(f"No episodes matched task indices {task_indices}")

    if out_dataset_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output dataset root already exists: {out_dataset_root}")
        shutil.rmtree(out_dataset_root)
    if out_flow_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output flow root already exists: {out_flow_root}")
        shutil.rmtree(out_flow_root)

    (out_dataset_root / "meta").mkdir(parents=True, exist_ok=True)
    (out_flow_root / "data").mkdir(parents=True, exist_ok=True)

    episode_mapping_rows: list[dict] = []
    subset_episode_rows: list[dict] = []
    total_frames = 0

    for new_episode_index, episode_row in enumerate(selected_episodes):
        original_episode_index = int(episode_row["episode_index"])
        length = int(episode_row.get("length", 0))
        total_frames += length

        old_chunk = f"chunk-{original_episode_index // args.chunk_size:03d}"
        new_chunk = f"chunk-{new_episode_index // args.chunk_size:03d}"

        src_parquet = dataset_root / "data" / old_chunk / f"episode_{original_episode_index:06d}.parquet"
        dst_parquet = out_dataset_root / "data" / new_chunk / f"episode_{new_episode_index:06d}.parquet"
        _symlink_file(src_parquet, dst_parquet)

        src_sidecar = flow_root / "data" / old_chunk / f"episode_{original_episode_index:06d}.npz"
        if not src_sidecar.exists():
            raise FileNotFoundError(f"Missing flow sidecar for episode {original_episode_index}: {src_sidecar}")
        dst_sidecar = out_flow_root / "data" / new_chunk / f"episode_{new_episode_index:06d}.npz"
        _symlink_file(src_sidecar, dst_sidecar)

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

    subset_info = dict(source_info)
    subset_info["total_episodes"] = len(selected_episodes)
    subset_info["total_frames"] = total_frames
    subset_info["total_tasks"] = len(selected_task_rows)
    subset_info["total_chunks"] = max(1, math.ceil(len(selected_episodes) / args.chunk_size))
    subset_info["chunks_size"] = args.chunk_size
    subset_info["splits"] = {"train": f"0:{len(selected_episodes)}"}
    subset_info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"

    (out_dataset_root / "meta" / "info.json").write_text(
        json.dumps(subset_info, ensure_ascii=True, indent=4) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(out_dataset_root / "meta" / "tasks.jsonl", selected_task_rows)
    _write_jsonl(out_dataset_root / "meta" / "episodes.jsonl", subset_episode_rows)

    stats_src = dataset_meta_dir / "stats.json"
    if stats_src.exists():
        shutil.copy2(stats_src, out_dataset_root / "meta" / "stats.json")

    mapping_payload = {
        "source_dataset_root": str(dataset_root),
        "source_flow_root": str(flow_root),
        "selected_task_indices": task_indices,
        "selected_task_names": [task_name_by_index[idx] for idx in task_indices],
        "num_selected_episodes": len(selected_episodes),
        "total_selected_frames": total_frames,
        "episode_mappings": episode_mapping_rows,
    }
    (out_dataset_root / "meta" / "subset_mapping.json").write_text(
        json.dumps(mapping_payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_flow_root / "subset_mapping.json").write_text(
        json.dumps(mapping_payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(
        {
            "output_dataset_root": str(out_dataset_root),
            "output_flow_root": str(out_flow_root),
            "selected_task_indices": task_indices,
            "selected_task_count": len(selected_task_rows),
            "selected_episode_count": len(selected_episodes),
            "selected_frame_count": total_frames,
        },
        ensure_ascii=True,
        indent=2,
    ))


if __name__ == "__main__":
    main()
