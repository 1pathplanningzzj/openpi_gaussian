"""Compare debug logs from two evaluation runs to find sources of instability.

Usage:
    python examples/libero/compare_debug_logs.py --run1 debug_logs_run1 --run2 debug_logs_run2
    python examples/libero/compare_debug_logs.py --run1 debug_logs_run1 --run2 debug_logs_run2 --visualize
"""

import argparse
import pathlib
import pickle

import imageio
import matplotlib.pyplot as plt
import numpy as np


ACTION_LABELS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


def load_run(path: pathlib.Path) -> dict:
    """Load all pkl files from a run directory, keyed by (task_id, episode_idx)."""
    data = {}
    for f in sorted(path.glob("task_*_ep_*.pkl")):
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        data[(d["task_id"], d["episode_idx"])] = d
    return data


def compare_records(rec1: dict, rec2: dict) -> dict:
    """Compare two step records and return differences."""
    diffs = {}
    if rec1["image_hash"] != rec2["image_hash"]:
        diffs["image_hash"] = (rec1["image_hash"], rec2["image_hash"])
    if rec1["wrist_image_hash"] != rec2["wrist_image_hash"]:
        diffs["wrist_image_hash"] = (rec1["wrist_image_hash"], rec2["wrist_image_hash"])

    state_diff = np.max(np.abs(rec1["state"] - rec2["state"]))
    if state_diff > 1e-8:
        diffs["state_max_diff"] = float(state_diff)

    a1 = np.array(rec1["action_chunk"])
    a2 = np.array(rec2["action_chunk"])
    if a1.shape == a2.shape:
        action_diff = np.max(np.abs(a1 - a2))
        if action_diff > 1e-8:
            diffs["action_max_diff"] = float(action_diff)
    else:
        diffs["action_shape_mismatch"] = (a1.shape, a2.shape)
    return diffs
