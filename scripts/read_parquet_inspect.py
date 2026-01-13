#!/usr/bin/env python3
"""Inspect a LeRobot/Libero episode Parquet file.

Usage:
  python scripts/read_parquet_inspect.py /path/to/episode_000000.parquet --num 3 --save-images

This prints schema, columns, dtypes and a few sample rows. If images are found, saves first ones to
`val_image/` (created if missing).

python scripts/read_parquet_inspect.py /home/zijianzhang/.cache/huggingface/lerobot/physical-intelligence/libero/data/chunk-000/episode_000000.parquet --num 5  --save-images --out-dir val_image

"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image


def try_save_image(obj, out_path: Path):
    """Try to convert common image-like objects to a PNG and save them.

    Handles:
    - raw bytes representing PNG/JPEG
    - numpy arrays (HWC or CHW)
    - lists/tuples of ints
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if obj is None:
        return False

    # Handle struct/dict (often found in LeRobot/Parquet images)
    if isinstance(obj, dict):
        # 1. 'bytes' field
        if 'bytes' in obj:
             return try_save_image(obj['bytes'], out_path)
        # 2. 'path' field (if it's an absolute path)
        if 'path' in obj and isinstance(obj['path'], str):
             if os.path.isfile(obj['path']):
                 try:
                     with open(obj['path'], 'rb') as f:
                         return try_save_image(f.read(), out_path)
                 except Exception:
                     pass

    # Raw bytes (jpeg/png)
    if isinstance(obj, (bytes, bytearray)):
        try:
            img = Image.open(io.BytesIO(obj)).convert("RGB")
            img.save(out_path, format="PNG")
            return True
        except Exception:
            return False

    # Numpy array
    if isinstance(obj, (list, tuple)):
        try:
            arr = np.asarray(obj)
        except Exception:
            return False
    elif hasattr(obj, "dtype") and hasattr(obj, "shape"):
        arr = obj
    else:
        return False

    # normalize shape HWC or CHW
    if arr.ndim == 3:
        if arr.shape[0] in (1, 3):
            # CHW -> HWC
            arr = np.transpose(arr, (1, 2, 0))
        # ensure uint8
        if arr.dtype != np.uint8:
            try:
                arr = (255 * (arr - arr.min()) / (arr.max() - arr.min())).astype(np.uint8)
            except Exception:
                arr = arr.astype(np.uint8)
        try:
            Image.fromarray(arr).save(out_path, format="PNG")
            return True
        except Exception:
            return False

    return False


def inspect_parquet(path: Path, num: int = 3, save_images: bool = False, out_dir: Path | None = None):
    print(f"Parquet: {path}")
    print("--- Arrow schema (pyarrow):")
    table = pq.read_table(str(path))
    print(table.schema)

    print("\n--- Pandas reading (first rows):")
    # pandas will read nested/complex columns as object dtype
    df = pd.read_parquet(path)
    print(f"Rows: {len(df)}")
    print("Columns and dtypes:")
    print(df.dtypes)

    print("\n--- Sample rows")
    sample = df.head(num)
    for idx, row in sample.iterrows():
        print(f"\n--- Row {idx} ---")
        for col in df.columns:
            val = row[col]
            ty = type(val)
            summary = None

            # summarize common types
            if isinstance(val, (bytes, bytearray)):
                summary = f"bytes, {len(val)} bytes"
            elif isinstance(val, (list, tuple, np.ndarray)):
                try:
                    arr = np.asarray(val)
                    summary = f"ndarray shape={arr.shape}, dtype={arr.dtype}"
                except Exception:
                    summary = f"list/tuple, len={len(val)}"
            elif isinstance(val, dict):
                summary = f"dict, keys={list(val.keys())}"
            else:
                summary = repr(val)

            print(f"{col} ({ty.__name__}): {summary}")

            if save_images and out_dir is not None:
                # heuristic column names commonly used for images
                if any(k in col.lower() for k in ("image", "img", "rgb")):
                    out_file = out_dir / f"row{idx}_{col}.png"
                    ok = try_save_image(val, out_file)
                    print(f"    -> saved image: {out_file} ({ok})")

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", type=str, help="Path to the Parquet file (episode)")
    parser.add_argument("--num", type=int, default=3, help="Number of rows to print")
    parser.add_argument("--save-images", action="store_true", help="Save detected image columns to disk")
    parser.add_argument("--out-dir", type=str, default="val_image", help="Directory to save images")
    args = parser.parse_args()

    path = Path(args.parquet)
    if not path.exists():
        raise FileNotFoundError(path)

    out_dir = Path(args.out_dir) if args.save_images else None
    inspect_parquet(path, num=args.num, save_images=args.save_images, out_dir=out_dir)


if __name__ == "__main__":
    main()
