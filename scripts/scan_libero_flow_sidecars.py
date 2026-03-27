#!/usr/bin/env python3
"""Scan LIBERO flow sidecars and generate rerun command lists.

Checks three anomaly types against expected sidecars:
- missing sidecar
- 0-byte sidecar
- np.load failure
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


@dataclass
class Anomaly:
    category: str
    source_parquet: str
    expected_sidecar: str
    reason: str
    size_bytes: int


def resolve_parquet_paths(dataset_root: Path) -> list[Path]:
    data_dir = dataset_root / "data"
    return sorted(data_dir.glob("chunk-*/episode_*.parquet"))


def expected_sidecar_path(dataset_root: Path, sidecar_root: Path, parquet_path: Path) -> Path:
    rel = parquet_path.relative_to(dataset_root).with_suffix(".npz")
    return sidecar_root / rel


def check_npz_load(npz_path: Path) -> tuple[bool, str]:
    try:
        with np.load(npz_path, allow_pickle=False) as payload:
            _ = payload.files
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def build_rerun_command(
    python_exe: str,
    generator_script: str,
    backend: str,
    device: str,
    dataset_root: str,
    parquet_path: str,
    camera_params: str,
    output_dir: str,
    depth_mode: str,
    camera_name: str,
    rotate_180: bool,
    extra_args: str,
) -> str:
    cmd = [
        "PYTHONUNBUFFERED=1",
        shlex.quote(python_exe),
        shlex.quote(generator_script),
        "--backend",
        shlex.quote(backend),
        "--device",
        shlex.quote(device),
        "--dataset-root",
        shlex.quote(dataset_root),
        "--parquet-path",
        shlex.quote(parquet_path),
        "--camera-params",
        shlex.quote(camera_params),
        "--output-dir",
        shlex.quote(output_dir),
        "--depth-mode",
        shlex.quote(depth_mode),
        "--camera-name",
        shlex.quote(camera_name),
    ]
    if rotate_180:
        cmd.append("--rotate-180")
    if extra_args.strip():
        cmd.append(extra_args.strip())
    return " ".join(cmd)


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan LIBERO flow sidecars and generate rerun lists")
    parser.add_argument("--dataset-root", type=str, required=True, help="Root of LIBERO dataset")
    parser.add_argument("--sidecar-root", type=str, required=True, help="Root of sidecar output directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to store scan reports")
    parser.add_argument("--camera-params", type=str, required=True, help="Camera params JSON path")
    parser.add_argument(
        "--generator-script",
        type=str,
        default="/home/zijianzhang/openpi/scripts/generate_libero_pseudo_scene_flow.py",
        help="Path to sidecar generator script",
    )
    parser.add_argument("--python-exe", type=str, default=sys.executable, help="Python executable in rerun commands")
    parser.add_argument("--backend", type=str, default="raft", choices=["raft", "farneback"])
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--depth-mode", type=str, default="metric", choices=["metric", "mujoco_normalized", "auto"])
    parser.add_argument("--camera-name", type=str, default="agentview")
    parser.add_argument("--rotate-180", action="store_true")
    parser.add_argument("--extra-args", type=str, default="", help="Extra args appended to every rerun command")
    parser.add_argument("--progress-every", type=int, default=500, help="Print progress every N episodes")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    sidecar_root = Path(args.sidecar_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else sidecar_root / "scan_reports" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_paths = resolve_parquet_paths(dataset_root)
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet episodes found under {dataset_root / 'data'}")

    anomalies: list[Anomaly] = []
    total = len(parquet_paths)

    for idx, parquet_path in enumerate(parquet_paths, start=1):
        expected = expected_sidecar_path(dataset_root, sidecar_root, parquet_path)

        if not expected.exists():
            anomalies.append(
                Anomaly(
                    category="missing",
                    source_parquet=str(parquet_path),
                    expected_sidecar=str(expected),
                    reason="sidecar file not found",
                    size_bytes=-1,
                )
            )
        else:
            size_bytes = expected.stat().st_size
            if size_bytes == 0:
                anomalies.append(
                    Anomaly(
                        category="zero_byte",
                        source_parquet=str(parquet_path),
                        expected_sidecar=str(expected),
                        reason="sidecar file size is 0",
                        size_bytes=0,
                    )
                )
            else:
                ok, reason = check_npz_load(expected)
                if not ok:
                    anomalies.append(
                        Anomaly(
                            category="np_load_failed",
                            source_parquet=str(parquet_path),
                            expected_sidecar=str(expected),
                            reason=reason,
                            size_bytes=size_bytes,
                        )
                    )

        if args.progress_every > 0 and (idx % args.progress_every == 0 or idx == total):
            print(f"Scanned {idx}/{total} episodes...")

    by_category: dict[str, list[Anomaly]] = {
        "missing": [],
        "zero_byte": [],
        "np_load_failed": [],
    }
    for item in anomalies:
        by_category[item.category].append(item)

    all_bad_parquets = sorted({item.source_parquet for item in anomalies})
    rerun_commands = [
        build_rerun_command(
            python_exe=args.python_exe,
            generator_script=args.generator_script,
            backend=args.backend,
            device=args.device,
            dataset_root=str(dataset_root),
            parquet_path=parquet_path,
            camera_params=args.camera_params,
            output_dir=str(sidecar_root),
            depth_mode=args.depth_mode,
            camera_name=args.camera_name,
            rotate_180=args.rotate_180,
            extra_args=args.extra_args,
        )
        for parquet_path in all_bad_parquets
    ]

    csv_path = output_dir / "bad_sidecars.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["category", "source_parquet", "expected_sidecar", "reason", "size_bytes"],
        )
        writer.writeheader()
        for item in anomalies:
            writer.writerow(asdict(item))

    write_lines(output_dir / "missing_parquets.txt", [x.source_parquet for x in by_category["missing"]])
    write_lines(output_dir / "zero_byte_parquets.txt", [x.source_parquet for x in by_category["zero_byte"]])
    write_lines(output_dir / "np_load_failed_parquets.txt", [x.source_parquet for x in by_category["np_load_failed"]])
    write_lines(output_dir / "rerun_parquets.txt", all_bad_parquets)
    write_lines(output_dir / "rerun_commands.txt", rerun_commands)
    write_lines(output_dir / "rerun_commands.sh", ["#!/usr/bin/env bash", "set -euo pipefail", ""] + rerun_commands)

    summary = {
        "dataset_root": str(dataset_root),
        "sidecar_root": str(sidecar_root),
        "total_parquets": total,
        "bad_total": len(anomalies),
        "bad_unique_parquets": len(all_bad_parquets),
        "counts": {k: len(v) for k, v in by_category.items()},
        "report_dir": str(output_dir),
        "artifacts": {
            "csv": str(csv_path),
            "missing": str(output_dir / "missing_parquets.txt"),
            "zero_byte": str(output_dir / "zero_byte_parquets.txt"),
            "np_load_failed": str(output_dir / "np_load_failed_parquets.txt"),
            "rerun_parquets": str(output_dir / "rerun_parquets.txt"),
            "rerun_commands_txt": str(output_dir / "rerun_commands.txt"),
            "rerun_commands_sh": str(output_dir / "rerun_commands.sh"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
