from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, ImageDraw


DEFAULT_DATASET_ROOT = Path("/home/zijianzhang/openpi/data_subsets/libero_tasks_0_9_with_depth")
DEFAULT_OUTPUT_DIR = Path("/home/zijianzhang/openpi/paper_framework")
DEFAULT_PAST_OFFSETS = (10, 5, 0)
DEFAULT_FUTURE_COUNT = 5


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _decode_rgb(payload) -> Image.Image:
    if isinstance(payload, dict):
        if "bytes" not in payload:
            raise ValueError(f"Unsupported image payload dict keys: {sorted(payload.keys())}")
        payload = payload["bytes"]
    return Image.open(io.BytesIO(payload)).convert("RGB")


def _mean_brightness(image: Image.Image) -> float:
    grayscale = image.convert("L").resize((32, 32))
    pixels = list(grayscale.getdata())
    return float(sum(pixels) / len(pixels))


def _find_episode_path(dataset_root: Path, episode_index: int) -> Path:
    chunk = f"chunk-{episode_index // 1000:03d}"
    path = dataset_root / "data" / chunk / f"episode_{episode_index:06d}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Episode parquet not found: {path}")
    return path


def _choose_current_index(num_frames: int, past_offsets: tuple[int, ...], future_count: int, requested: int | None) -> int:
    min_index = max(past_offsets)
    max_index = num_frames - 1 - future_count
    if max_index < min_index:
        raise ValueError(
            f"Episode too short for requested offsets/future horizon: "
            f"num_frames={num_frames}, past_offsets={past_offsets}, future_count={future_count}"
        )
    if requested is None:
        return (min_index + max_index) // 2
    if requested < min_index or requested > max_index:
        raise ValueError(
            f"Requested current frame {requested} is out of valid range [{min_index}, {max_index}] "
            f"for num_frames={num_frames}"
        )
    return requested


def _tile_with_labels(images: list[Image.Image], labels: list[str], title: str) -> Image.Image:
    if len(images) != len(labels):
        raise ValueError("images and labels must have the same length")
    if not images:
        raise ValueError("images must be non-empty")

    width, height = images[0].size
    caption_h = 28
    title_h = 36
    canvas = Image.new("RGB", (width * len(images), height + caption_h + title_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), title, fill=(0, 0, 0))

    for idx, (image, label) in enumerate(zip(images, labels, strict=True)):
        x0 = idx * width
        canvas.paste(image, (x0, title_h))
        draw.rectangle((x0, title_h + height, x0 + width, title_h + height + caption_h), fill=(245, 245, 245))
        draw.text((x0 + 10, title_h + height + 6), label, fill=(0, 0, 0))

    return canvas


def _load_task_mappings(dataset_root: Path) -> tuple[dict[int, str], dict[str, int]]:
    task_rows = _load_jsonl(dataset_root / "meta" / "tasks.jsonl")
    task_name_by_index = {int(row["task_index"]): row["task"] for row in task_rows}
    task_index_by_name = {name: index for index, name in task_name_by_index.items()}
    return task_name_by_index, task_index_by_name


def _select_brightest_scene(
    dataset_root: Path,
    task_index: int,
    past_offsets: tuple[int, ...],
    future_count: int,
    sample_stride: int,
) -> tuple[int, int, float]:
    if sample_stride <= 0:
        raise ValueError(f"sample_stride must be positive, got {sample_stride}")

    task_name_by_index, _ = _load_task_mappings(dataset_root)
    if task_index not in task_name_by_index:
        raise ValueError(f"Unknown task_index={task_index}")

    target_task = task_name_by_index[task_index]
    episode_rows = _load_jsonl(dataset_root / "meta" / "episodes.jsonl")
    best_score = float("-inf")
    best_episode_index: int | None = None
    best_current_frame: int | None = None

    for row in episode_rows:
        if not row.get("tasks") or row["tasks"][0] != target_task:
            continue

        episode_index = int(row["episode_index"])
        parquet_path = _find_episode_path(dataset_root, episode_index)
        table = pq.read_table(parquet_path, columns=["image"])
        image_payloads = table["image"].to_pylist()
        num_frames = len(image_payloads)
        min_index = max(past_offsets)
        max_index = num_frames - 1 - future_count
        if max_index < min_index:
            continue

        for current_frame in range(min_index, max_index + 1, sample_stride):
            brightness = _mean_brightness(_decode_rgb(image_payloads[current_frame]))
            if brightness > best_score:
                best_score = brightness
                best_episode_index = episode_index
                best_current_frame = current_frame

    if best_episode_index is None or best_current_frame is None:
        raise ValueError(f"No valid scene found for task_index={task_index}")

    return best_episode_index, best_current_frame, best_score


def export_frames(
    dataset_root: Path,
    output_dir: Path,
    episode_index: int,
    current_frame: int | None,
    past_offsets: tuple[int, ...],
    future_count: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    task_name_by_index, _ = _load_task_mappings(dataset_root)

    parquet_path = _find_episode_path(dataset_root, episode_index)
    table = pq.read_table(parquet_path, columns=["image", "wrist_image", "task_index", "episode_index", "frame_index"])
    payload = table.to_pydict()

    num_frames = len(payload["image"])
    chosen_current = _choose_current_index(num_frames, past_offsets, future_count, current_frame)
    labels = [f"t-{off}" if off > 0 else "t" for off in past_offsets] + [f"t+{i}" for i in range(1, future_count + 1)]
    frame_indices = [chosen_current - off for off in past_offsets] + [
        chosen_current + i for i in range(1, future_count + 1)
    ]

    agent_images = [_decode_rgb(payload["image"][idx]) for idx in frame_indices]
    wrist_images = [_decode_rgb(payload["wrist_image"][idx]) for idx in frame_indices]

    agent_dir = output_dir / "agent"
    wrist_dir = output_dir / "wrist"
    agent_dir.mkdir(exist_ok=True)
    wrist_dir.mkdir(exist_ok=True)

    for image, label, frame_idx in zip(agent_images, labels, frame_indices, strict=True):
        image.save(agent_dir / f"{label.replace('+', 'plus').replace('-', 'minus')}_frame_{frame_idx:03d}.png")
    for image, label, frame_idx in zip(wrist_images, labels, frame_indices, strict=True):
        image.save(wrist_dir / f"{label.replace('+', 'plus').replace('-', 'minus')}_frame_{frame_idx:03d}.png")

    task_index = int(payload["task_index"][0])
    task_name = task_name_by_index.get(task_index, f"task_{task_index}")
    title_prefix = f"episode {episode_index:06d} | task {task_index}: {task_name}"

    agent_strip = _tile_with_labels(agent_images, labels, f"Agent View | {title_prefix}")
    wrist_strip = _tile_with_labels(wrist_images, labels, f"Wrist View | {title_prefix}")
    agent_strip.save(output_dir / "agent_timeline.png")
    wrist_strip.save(output_dir / "wrist_timeline.png")

    manifest = {
        "dataset_root": str(dataset_root),
        "parquet_path": str(parquet_path),
        "episode_index": episode_index,
        "task_index": task_index,
        "task_name": task_name,
        "num_frames": num_frames,
        "current_frame": chosen_current,
        "current_frame_brightness": _mean_brightness(agent_images[len(past_offsets) - 1]),
        "past_offsets": list(past_offsets),
        "future_count": future_count,
        "labels": labels,
        "frame_indices": frame_indices,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LIBERO context/future frames for paper figures.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--current-frame", type=int, default=None)
    parser.add_argument("--past-offsets", type=int, nargs="+", default=list(DEFAULT_PAST_OFFSETS))
    parser.add_argument("--future-count", type=int, default=DEFAULT_FUTURE_COUNT)
    parser.add_argument("--selection-mode", choices=("middle", "brightest"), default="middle")
    parser.add_argument("--sample-stride", type=int, default=5)
    args = parser.parse_args()

    episode_index = args.episode_index
    current_frame = args.current_frame

    if args.selection_mode == "brightest":
        if args.task_index is None:
            raise ValueError("--selection-mode brightest requires --task-index")
        episode_index, current_frame, brightness = _select_brightest_scene(
            dataset_root=args.dataset_root,
            task_index=int(args.task_index),
            past_offsets=tuple(int(x) for x in args.past_offsets),
            future_count=int(args.future_count),
            sample_stride=int(args.sample_stride),
        )
        print(
            json.dumps(
                {
                    "selection_mode": "brightest",
                    "task_index": int(args.task_index),
                    "episode_index": episode_index,
                    "current_frame": current_frame,
                    "brightness": brightness,
                },
                ensure_ascii=False,
            )
        )
    elif episode_index is None:
        episode_index = 0

    manifest = export_frames(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        episode_index=int(episode_index),
        current_frame=current_frame,
        past_offsets=tuple(int(x) for x in args.past_offsets),
        future_count=int(args.future_count),
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
