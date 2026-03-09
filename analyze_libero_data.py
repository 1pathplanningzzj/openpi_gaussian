#!/usr/bin/env python3
"""
Deep dive into LIBERO data format to understand coordinate systems.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_libero_data():
    """Analyze LIBERO dataset structure in detail."""

    data_dir = "/data/zijianzhang/LIBERA/data"
    parquet_path = Path(data_dir) / "chunk-000" / "episode_000000.parquet"

    df = pd.read_parquet(parquet_path)

    print("="*80)
    print("LIBERO Dataset Analysis")
    print("="*80)

    # 1. Actions analysis
    actions = np.stack(df['actions'].values)
    print("\n1. ACTIONS (7D):")
    print(f"   Shape: {actions.shape}")
    print(f"   Format: [eef_pos(3), axis_angle(3), gripper(1)]")
    print(f"\n   Statistics per dimension:")
    for i in range(7):
        dim_name = ["X", "Y", "Z", "RotX", "RotY", "RotZ", "Gripper"][i]
        print(f"   [{i}] {dim_name:8s}: min={actions[:, i].min():8.4f}, max={actions[:, i].max():8.4f}, "
              f"mean={actions[:, i].mean():8.4f}, std={actions[:, i].std():8.4f}")

    # Check if actions are normalized
    print(f"\n   Analysis:")
    if np.abs(actions[:, :3].mean()) < 0.2 and actions[:, :3].std() < 0.5:
        print(f"   ⚠ Actions appear to be NORMALIZED or RELATIVE (mean≈0, small std)")
    else:
        print(f"   ✓ Actions appear to be ABSOLUTE world positions")

    # 2. States analysis
    states = np.stack(df['state'].values)
    print(f"\n2. STATES (8D):")
    print(f"   Shape: {states.shape}")
    print(f"   Expected format: [gripper_qpos(2), eef_pos(3), eef_quat(3)]")
    print(f"\n   Statistics per dimension:")
    for i in range(8):
        print(f"   [{i}]: min={states[:, i].min():8.4f}, max={states[:, i].max():8.4f}, "
              f"mean={states[:, i].mean():8.4f}, std={states[:, i].std():8.4f}")

    # Try to identify which dims are eef_pos
    print(f"\n   Interpretation attempts:")

    # V1: [gripper(2), eef_pos(3), eef_quat(3)]
    eef_pos_v1 = states[:, 2:5]
    print(f"\n   V1: eef_pos = state[2:5]")
    print(f"       Range: X=[{eef_pos_v1[:, 0].min():.3f}, {eef_pos_v1[:, 0].max():.3f}], "
          f"Y=[{eef_pos_v1[:, 1].min():.3f}, {eef_pos_v1[:, 1].max():.3f}], "
          f"Z=[{eef_pos_v1[:, 2].min():.3f}, {eef_pos_v1[:, 2].max():.3f}]")
    if eef_pos_v1[:, 2].min() < 0:
        print(f"       ⚠ Z has negative values - unlikely for height above ground")
    else:
        print(f"       ✓ Z is positive - plausible")

    # V2: Maybe state format is different
    eef_pos_v2 = states[:, :3]
    print(f"\n   V2: eef_pos = state[0:3]")
    print(f"       Range: X=[{eef_pos_v2[:, 0].min():.3f}, {eef_pos_v2[:, 0].max():.3f}], "
          f"Y=[{eef_pos_v2[:, 1].min():.3f}, {eef_pos_v2[:, 1].max():.3f}], "
          f"Z=[{eef_pos_v2[:, 2].min():.3f}, {eef_pos_v2[:, 2].max():.3f}]")
    if eef_pos_v2[:, 2].min() > 0.3:
        print(f"       ✓ Z > 0.3 - plausible height")

    # 3. Compare actions vs states
    print(f"\n3. ACTIONS vs STATES comparison:")
    print(f"   Actions[:3] range: {actions[:, :3].min():.3f} to {actions[:, :3].max():.3f}")
    print(f"   States[2:5] range: {eef_pos_v1.min():.3f} to {eef_pos_v1.max():.3f}")
    print(f"   States[0:3] range: {eef_pos_v2.min():.3f} to {eef_pos_v2.max():.3f}")

    # Correlation check
    from scipy.stats import pearsonr
    print(f"\n   Correlation (Actions[:3] vs States[2:5]):")
    for i in range(3):
        corr, _ = pearsonr(actions[:, i], eef_pos_v1[:, i])
        print(f"       Dim {i}: r={corr:.4f}")

    print(f"\n   Correlation (Actions[:3] vs States[0:3]):")
    for i in range(3):
        corr, _ = pearsonr(actions[:, i], eef_pos_v2[:, i])
        print(f"       Dim {i}: r={corr:.4f}")

    # 4. Recommendation
    print(f"\n4. RECOMMENDATION:")
    if np.abs(actions[:, :3].mean()) < 0.2:
        print(f"   ✓ Use STATE for end-effector position (actions are normalized/relative)")
        if eef_pos_v1[:, 2].min() >= 0:
            print(f"   ✓ Use state[2:5] as eef_pos (Z is positive)")
        else:
            print(f"   ✓ Use state[0:3] as eef_pos (Z is positive)")
    else:
        print(f"   ✓ Actions contain absolute positions, can use directly")

    print("="*80)

if __name__ == "__main__":
    analyze_libero_data()
