#!/usr/bin/env python3
"""
Visualize LIBERO dataset: camera images + action trajectories
Check if actions are incremental (delta) or absolute positions
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add openpi to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def load_libero_episode(data_dir, chunk_idx=0, episode_idx=0):
    """Load a single episode from LIBERO dataset (parquet format)"""
    import pandas as pd

    # Load parquet file
    parquet_path = Path(data_dir) / f"chunk-{chunk_idx:03d}" / f"episode_{episode_idx:06d}.parquet"
    print(f"Loading episode from: {parquet_path}")

    if not parquet_path.exists():
        raise FileNotFoundError(f"Episode file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    print(f"Loaded dataframe with {len(df)} rows and columns: {df.columns.tolist()}")

    return df

def visualize_action_trajectory(df, save_path="libero_action_viz.png"):
    """Visualize action trajectory and check if incremental"""

    # Extract data from dataframe
    # Check available columns
    print(f"\nAvailable columns: {df.columns.tolist()}")

    # Extract actions - try different possible column names
    if 'action' in df.columns:
        actions = np.stack(df['action'].values)  # [T, 7]
    elif 'actions' in df.columns:
        actions = np.stack(df['actions'].values)
    else:
        raise ValueError(f"No action column found. Available: {df.columns.tolist()}")

    # Extract images - try different possible column names
    image_col = None
    for col in ['observation.images.cam_high', 'cam_high', 'agentview_image', 'image']:
        if col in df.columns:
            image_col = col
            break

    if image_col:
        images = df[image_col].values
        print(f"Using image column: {image_col}")
    else:
        images = []
        print(f"No image column found. Available: {df.columns.tolist()}")

    T = actions.shape[0]
    print(f"\nEpisode info:")
    print(f"  Length: {T} timesteps")
    print(f"  Action shape: {actions.shape}")
    print(f"  Action dim: {actions.shape[1]} (6D pose + 1D gripper)")
    if len(images) > 0:
        print(f"  Image shape: {images[0].shape if hasattr(images[0], 'shape') else 'N/A'}")

    # Analyze action statistics
    print(f"\nAction statistics:")
    print(f"  Min: {actions.min(axis=0)}")
    print(f"  Max: {actions.max(axis=0)}")
    print(f"  Mean: {actions.mean(axis=0)}")
    print(f"  Std: {actions.std(axis=0)}")

    # Check if incremental by computing cumulative sum
    actions_cumsum = np.cumsum(actions, axis=0)

    # Create visualization
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(4, 5, hspace=0.4, wspace=0.3)

    # Row 1: Sample images (5 frames)
    sample_indices = np.linspace(0, T-1, 5, dtype=int)
    for i, idx in enumerate(sample_indices):
        ax = fig.add_subplot(gs[0, i])
        if len(images) > idx:
            try:
                img = images[idx]
                # Handle different image formats
                if isinstance(img, bytes):
                    # Decode bytes to image
                    from PIL import Image
                    import io
                    img = Image.open(io.BytesIO(img))
                    img = np.array(img)
                elif hasattr(img, 'convert'):
                    # PIL Image
                    img = np.array(img.convert('RGB'))
                elif isinstance(img, np.ndarray):
                    # Already numpy array
                    pass
                else:
                    # Unknown format, skip
                    ax.text(0.5, 0.5, f'Image format:\n{type(img)}',
                           ha='center', va='center', fontsize=8)
                    ax.axis('off')
                    continue

                ax.imshow(img)
            except Exception as e:
                ax.text(0.5, 0.5, f'Error:\n{str(e)[:30]}',
                       ha='center', va='center', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No image', ha='center', va='center')
        ax.set_title(f"t={idx}", fontsize=10)
        ax.axis('off')

    # Row 2: Raw actions (XYZ position)
    ax1 = fig.add_subplot(gs[1, :3])
    ax1.plot(actions[:, 0], label='X', alpha=0.7)
    ax1.plot(actions[:, 1], label='Y', alpha=0.7)
    ax1.plot(actions[:, 2], label='Z', alpha=0.7)
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Action Value')
    ax1.set_title('Raw Actions: XYZ (Position)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Row 2: Raw actions (Rotation + Gripper)
    ax2 = fig.add_subplot(gs[1, 3:])
    ax2.plot(actions[:, 3], label='Rot X', alpha=0.7)
    ax2.plot(actions[:, 4], label='Rot Y', alpha=0.7)
    ax2.plot(actions[:, 5], label='Rot Z', alpha=0.7)
    ax2.plot(actions[:, 6], label='Gripper', alpha=0.7, linewidth=2)
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Action Value')
    ax2.set_title('Raw Actions: Rotation + Gripper')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Row 3: Cumulative sum (if incremental, this should look like smooth trajectory)
    ax3 = fig.add_subplot(gs[2, :3])
    ax3.plot(actions_cumsum[:, 0], label='Cumsum X', alpha=0.7)
    ax3.plot(actions_cumsum[:, 1], label='Cumsum Y', alpha=0.7)
    ax3.plot(actions_cumsum[:, 2], label='Cumsum Z', alpha=0.7)
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Cumulative Sum')
    ax3.set_title('Cumulative Sum: XYZ (If incremental, should be smooth)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Row 3: Action deltas (difference between consecutive actions)
    ax4 = fig.add_subplot(gs[2, 3:])
    action_deltas = np.diff(actions, axis=0)
    ax4.plot(action_deltas[:, 0], label='ΔX', alpha=0.7)
    ax4.plot(action_deltas[:, 1], label='ΔY', alpha=0.7)
    ax4.plot(action_deltas[:, 2], label='ΔZ', alpha=0.7)
    ax4.set_xlabel('Timestep')
    ax4.set_ylabel('Action Delta')
    ax4.set_title('Action Deltas: Δ(t+1) - Δ(t)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Row 4: 3D trajectory visualization
    ax5 = fig.add_subplot(gs[3, :2], projection='3d')
    # Plot raw actions as 3D trajectory
    ax5.plot(actions[:, 0], actions[:, 1], actions[:, 2],
             'b-', alpha=0.6, linewidth=2, label='Raw Actions')
    ax5.scatter(actions[0, 0], actions[0, 1], actions[0, 2],
                c='green', s=100, marker='o', label='Start')
    ax5.scatter(actions[-1, 0], actions[-1, 1], actions[-1, 2],
                c='red', s=100, marker='x', label='End')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_zlabel('Z')
    ax5.set_title('3D Trajectory: Raw Actions')
    ax5.legend()

    # Row 4: 3D trajectory (cumulative sum)
    ax6 = fig.add_subplot(gs[3, 2:4], projection='3d')
    ax6.plot(actions_cumsum[:, 0], actions_cumsum[:, 1], actions_cumsum[:, 2],
             'r-', alpha=0.6, linewidth=2, label='Cumsum')
    ax6.scatter(actions_cumsum[0, 0], actions_cumsum[0, 1], actions_cumsum[0, 2],
                c='green', s=100, marker='o', label='Start')
    ax6.scatter(actions_cumsum[-1, 0], actions_cumsum[-1, 1], actions_cumsum[-1, 2],
                c='red', s=100, marker='x', label='End')
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    ax6.set_zlabel('Z')
    ax6.set_title('3D Trajectory: Cumulative Sum')
    ax6.legend()

    # Row 4: Action magnitude over time
    ax7 = fig.add_subplot(gs[3, 4])
    action_magnitude = np.linalg.norm(actions[:, :3], axis=1)
    ax7.plot(action_magnitude, 'purple', linewidth=2)
    ax7.set_xlabel('Timestep')
    ax7.set_ylabel('||action||')
    ax7.set_title('Action Magnitude')
    ax7.grid(True, alpha=0.3)

    # Analysis text
    analysis_text = f"""
    Analysis:
    - If actions are INCREMENTAL (delta):
      * Raw actions should be small values around 0
      * Cumsum should show smooth trajectory
      * Action deltas should be noisy

    - If actions are ABSOLUTE (positions):
      * Raw actions should show smooth trajectory
      * Cumsum will diverge/accumulate
      * Action deltas should be small

    Observations:
    - Action range: [{actions.min():.3f}, {actions.max():.3f}]
    - Action std: {actions.std():.4f}
    - Cumsum range: [{actions_cumsum.min():.3f}, {actions_cumsum.max():.3f}]
    """

    fig.text(0.02, 0.02, analysis_text, fontsize=9, family='monospace',
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    plt.close()

    # Determine if incremental
    raw_smoothness = np.std(np.diff(actions[:, :3], axis=0))
    cumsum_smoothness = np.std(np.diff(actions_cumsum[:, :3], axis=0))

    print(f"\nSmoothness analysis:")
    print(f"  Raw action smoothness (std of deltas): {raw_smoothness:.6f}")
    print(f"  Cumsum smoothness (std of deltas): {cumsum_smoothness:.6f}")

    if raw_smoothness < cumsum_smoothness:
        print(f"\n✓ Actions appear to be ABSOLUTE positions (raw is smoother)")
    else:
        print(f"\n✓ Actions appear to be INCREMENTAL deltas (cumsum is smoother)")

def main():
    # LIBERO dataset path
    data_dir = "/data/zijianzhang/LIBERA/data"

    try:
        df = load_libero_episode(data_dir, chunk_idx=0, episode_idx=0)
        visualize_action_trajectory(df, save_path="/home/zijianzhang/openpi/visualizations/libero_action_analysis.png")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
