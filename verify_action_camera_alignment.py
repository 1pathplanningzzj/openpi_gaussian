#!/usr/bin/env python3
"""
Verify Action/State alignment with Base Camera View

This script:
1. Loads LIBERO episode data (images + actions/state)
2. Projects end-effector position to camera view
3. Visualizes the projection on images to verify alignment
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from PIL import Image
import io


def quat_to_rotation_matrix(quat):
    """
    Convert quaternion to rotation matrix.

    Args:
        quat: [4] - quaternion [x, y, z, w] (LIBERO format)

    Returns:
        R: [3, 3] - rotation matrix
    """
    x, y, z, w = quat

    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)],
    ])

    return R


def build_viewmatrix(cam_pos, cam_quat):
    """
    Build view matrix from camera pose.

    Args:
        cam_pos: [3] - camera position in world frame
        cam_quat: [4] - camera quaternion [x, y, z, w]

    Returns:
        viewmatrix: [4, 4] - transforms world → camera
    """
    R = quat_to_rotation_matrix(cam_quat)

    # View matrix: [R | -R @ t]
    viewmatrix = np.eye(4)
    viewmatrix[:3, :3] = R
    viewmatrix[:3, 3] = -R @ cam_pos

    return viewmatrix


def project_3d_to_2d(points_3d_world, viewmatrix, intrinsics):
    """
    Project 3D points in world frame to 2D image coordinates.

    Args:
        points_3d_world: [N, 3] - 3D points in world frame
        viewmatrix: [4, 4] - world → camera transform
        intrinsics: dict with fx, fy, cx, cy

    Returns:
        points_2d: [N, 2] - 2D image coordinates (u, v)
        depths: [N] - depth values (z in camera frame)
    """
    N = points_3d_world.shape[0]

    # Transform to camera frame
    points_homo = np.concatenate([points_3d_world, np.ones((N, 1))], axis=1)  # [N, 4]
    points_cam_homo = points_homo @ viewmatrix.T  # [N, 4]
    points_cam = points_cam_homo[:, :3]  # [N, 3]

    # Project to 2D
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']

    x_cam, y_cam, z_cam = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]

    # Avoid division by zero
    z_cam = np.maximum(z_cam, 1e-6)

    u = fx * (x_cam / z_cam) + cx
    v = fy * (y_cam / z_cam) + cy

    points_2d = np.stack([u, v], axis=1)
    depths = z_cam

    return points_2d, depths


def visualize_multi_frame_trajectory(df, save_path="action_camera_multi_frame.png", num_frames=6):
    """
    Visualize multiple frames with action trajectory overlay.
    Shows how action deltas correspond to actual visual motion.
    """
    # Extract data
    if 'actions' in df.columns:
        actions = np.stack(df['actions'].values)  # [T, 7]
    else:
        raise ValueError("No actions column found")

    # Extract images
    image_col = None
    for col in ['image', 'agentview_image', 'cam_high']:
        if col in df.columns:
            image_col = col
            break

    if not image_col:
        raise ValueError("No image column found")

    images = df[image_col].values
    T = actions.shape[0]

    # LIBERO camera parameters (canonical_agentview from libero_coffee_table_manipulation.py)
    camera_pos = np.array([0.5386131746834771, 0.0, 0.7903500240372423])
    camera_quat = np.array([0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349])  # [w, x, y, z]
    camera_quat_xyzw = np.array([camera_quat[1], camera_quat[2], camera_quat[3], camera_quat[0]])
    viewmatrix = build_viewmatrix(camera_pos, camera_quat_xyzw)

    intrinsics = {
        'fx': 221.7025,
        'fy': 221.7025,
        'cx': 128.0,
        'cy': 128.0,
    }

    # Compute action deltas
    action_deltas = np.diff(actions, axis=0)  # [T-1, 7]

    # Select frames to visualize (evenly spaced)
    frame_indices = np.linspace(0, T-1, num_frames, dtype=int)

    # Create subplot grid
    rows = 2
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(18, 12))
    axes = axes.flatten()

    print(f"\nVisualizing {num_frames} frames with action trajectories...")

    R_cam = viewmatrix[:3, :3]
    avg_depth = 0.5
    arrow_scale = 100.0

    for idx, frame_idx in enumerate(frame_indices):
        ax = axes[idx]

        try:
            # Load image
            img_data = images[frame_idx]

            if isinstance(img_data, dict):
                if 'bytes' in img_data:
                    img = Image.open(io.BytesIO(img_data['bytes']))
                    img = np.array(img)
                elif 'path' in img_data:
                    img = Image.open(img_data['path'])
                    img = np.array(img)
                else:
                    raise ValueError(f"Unknown dict format: {img_data.keys()}")
            elif isinstance(img_data, bytes):
                img = Image.open(io.BytesIO(img_data))
                img = np.array(img)
            elif isinstance(img_data, Image.Image):
                img = np.array(img_data.convert('RGB'))
            elif isinstance(img_data, np.ndarray):
                img = img_data
            else:
                img = np.array(img_data)

            # Ensure uint8 format
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

            ax.imshow(img)
            img_h, img_w = img.shape[:2]
            ax.set_xlim(0, img_w)
            ax.set_ylim(img_h, 0)

            # Compute trajectory from this frame forward (next 10 steps)
            trajectory_u = [img_w // 2]
            trajectory_v = [img_h // 2]

            num_future_steps = min(10, T - frame_idx - 1)

            for step in range(num_future_steps):
                delta_idx = frame_idx + step
                if delta_idx >= len(action_deltas):
                    break

                action_delta = action_deltas[delta_idx]
                delta_xyz = action_delta[:3]

                # Transform to camera frame
                delta_cam = R_cam @ delta_xyz

                # Project to 2D
                # NOTE: Flip X direction to match image coordinate system
                du = -intrinsics['fx'] * delta_cam[0] / avg_depth  # Flip sign
                dv = intrinsics['fy'] * delta_cam[1] / avg_depth

                # Scale and accumulate
                du_vis = du * arrow_scale
                dv_vis = dv * arrow_scale

                trajectory_u.append(trajectory_u[-1] + du_vis)
                trajectory_v.append(trajectory_v[-1] + dv_vis)

            # Plot trajectory
            trajectory_u = np.array(trajectory_u)
            trajectory_v = np.array(trajectory_v)

            if len(trajectory_u) > 1:
                # Draw trajectory line (make it thicker)
                for i in range(len(trajectory_u) - 1):
                    color_ratio = i / max(1, len(trajectory_u) - 1)
                    color = plt.cm.coolwarm(color_ratio)
                    ax.plot(trajectory_u[i:i+2], trajectory_v[i:i+2],
                           color=color, linewidth=4, alpha=0.9, zorder=5)

                # Draw arrow at the end (make it bigger and more visible)
                if len(trajectory_u) >= 2:
                    du = trajectory_u[-1] - trajectory_u[-2]
                    dv = trajectory_v[-1] - trajectory_v[-2]
                    # Make arrow much larger
                    ax.arrow(trajectory_u[-2], trajectory_v[-2], du, dv,
                            head_width=15, head_length=15, fc='red', ec='red',
                            linewidth=3, alpha=1.0, zorder=10)

                # Mark start (make it bigger)
                ax.plot(trajectory_u[0], trajectory_v[0], 'go', markersize=15,
                       markeredgewidth=3, markerfacecolor='lime', zorder=10)

            ax.set_title(f'Frame {frame_idx}/{T-1} (t={frame_idx/20:.2f}s)',
                        fontsize=12, weight='bold')
            ax.axis('off')

        except Exception as e:
            ax.text(0.5, 0.5, f'Error loading frame {frame_idx}:\n{str(e)}',
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.axis('off')

    plt.suptitle('Multi-Frame Action Trajectory Visualization\n(Green dot = start, Red arrow = predicted motion direction)',
                fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nMulti-frame visualization saved to: {save_path}")
    plt.close()


def visualize_projection(df, save_path="action_camera_alignment.png"):
    """
    Visualize action/state projection on camera images.
    Test coordinate transformation by comparing action deltas with visual motion.
    """
    # Extract data
    if 'actions' in df.columns:
        actions = np.stack(df['actions'].values)  # [T, 7]
    else:
        raise ValueError("No actions column found")

    if 'state' in df.columns:
        states = np.stack(df['state'].values)  # [T, state_dim]
    else:
        states = None

    # Extract images
    image_col = None
    for col in ['image', 'agentview_image', 'cam_high']:
        if col in df.columns:
            image_col = col
            break

    if not image_col:
        raise ValueError("No image column found")

    images = df[image_col].values

    T = actions.shape[0]
    print(f"\nDataset info:")
    print(f"  Episodes: {T} timesteps")
    print(f"  Actions shape: {actions.shape}")
    if states is not None:
        print(f"  States shape: {states.shape}")

    # Analyze action statistics
    print(f"\nAction statistics (normalized commands):")
    for i in range(min(7, actions.shape[1])):
        print(f"  Dim {i}: mean={actions[:, i].mean():.4f}, std={actions[:, i].std():.4f}, "
              f"min={actions[:, i].min():.4f}, max={actions[:, i].max():.4f}")

    # LIBERO camera parameters (canonical_agentview from libero_coffee_table_manipulation.py)
    camera_pos = np.array([0.5386131746834771, 0.0, 0.7903500240372423])
    camera_quat = np.array([0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349])  # [w, x, y, z]

    # Camera intrinsics (LIBERO default: 256x256)
    intrinsics = {
        'fx': 221.7025,
        'fy': 221.7025,
        'cx': 128.0,
        'cy': 128.0,
    }

    # Build view matrix - need to convert quaternion format
    # quat_to_rotation_matrix expects [x, y, z, w]
    camera_quat_xyzw = np.array([camera_quat[1], camera_quat[2], camera_quat[3], camera_quat[0]])
    viewmatrix = build_viewmatrix(camera_pos, camera_quat_xyzw)

    print(f"\nCamera parameters:")
    print(f"  Position: {camera_pos}")
    print(f"  Quaternion (wxyz): {camera_quat}")
    print(f"  Intrinsics: fx={intrinsics['fx']}, fy={intrinsics['fy']}, cx={intrinsics['cx']}, cy={intrinsics['cy']}")

    # Compute action deltas (consecutive differences)
    action_deltas = np.diff(actions, axis=0)  # [T-1, 7]
    print(f"\nAction deltas (consecutive differences):")
    print(f"  Shape: {action_deltas.shape}")
    print(f"  Translation deltas (dims 0-2):")
    for i in range(3):
        print(f"    Dim {i}: mean={action_deltas[:, i].mean():.6f}, std={action_deltas[:, i].std():.6f}")

    # Test: Do action deltas correlate with visual motion?
    # We'll compute optical flow-like motion and compare with action direction
    print(f"\n{'='*60}")
    print(f"Testing Action-Visual Motion Correlation:")
    print(f"{'='*60}")

    # Visualize: Show all action trajectory on first frame
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # Visualize: Show all action trajectory on first frame
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # Load first frame
    try:
        img_data = images[0]

        # Handle different image formats
        if isinstance(img_data, dict):
            if 'bytes' in img_data:
                img = Image.open(io.BytesIO(img_data['bytes']))
                img = np.array(img)
            elif 'path' in img_data:
                img = Image.open(img_data['path'])
                img = np.array(img)
            else:
                raise ValueError(f"Unknown dict format: {img_data.keys()}")
        elif isinstance(img_data, bytes):
            img = Image.open(io.BytesIO(img_data))
            img = np.array(img)
        elif isinstance(img_data, Image.Image):
            img = np.array(img_data.convert('RGB'))
        elif isinstance(img_data, np.ndarray):
            img = img_data
        else:
            img = np.array(img_data)

        # Ensure uint8 format
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

        # Display image
        ax.imshow(img)
        img_h, img_w = img.shape[:2]

        # Set axis limits to match image size
        ax.set_xlim(0, img_w)
        ax.set_ylim(img_h, 0)  # Invert y-axis for image coordinates

        print(f"\nImage loaded successfully: shape={img.shape}, dtype={img.dtype}")
        print(f"  Image range: [{img.min()}, {img.max()}]")

        # Compute cumulative action trajectory
        # Start from image center
        center_u, center_v = img_w // 2, img_h // 2

        # Transform all action deltas to camera frame and accumulate
        trajectory_u = [center_u]
        trajectory_v = [center_v]

        R_cam = viewmatrix[:3, :3]
        avg_depth = 0.5
        arrow_scale = 100.0  # Scale for visualization

        for delta_idx in range(len(action_deltas)):
            action_delta = action_deltas[delta_idx]
            delta_xyz = action_delta[:3]

            # Transform to camera frame
            delta_cam = R_cam @ delta_xyz

            # Project to 2D
            du = intrinsics['fx'] * delta_cam[0] / avg_depth
            dv = intrinsics['fy'] * delta_cam[1] / avg_depth

            # Scale and accumulate
            du_vis = du * arrow_scale
            dv_vis = dv * arrow_scale

            trajectory_u.append(trajectory_u[-1] + du_vis)
            trajectory_v.append(trajectory_v[-1] + dv_vis)

        # Plot trajectory as connected line
        trajectory_u = np.array(trajectory_u)
        trajectory_v = np.array(trajectory_v)

        # Draw trajectory line with color gradient (time progression)
        for i in range(len(trajectory_u) - 1):
            # Color from blue (start) to red (end)
            color_ratio = i / (len(trajectory_u) - 1)
            color = plt.cm.coolwarm(color_ratio)

            ax.plot(trajectory_u[i:i+2], trajectory_v[i:i+2],
                   color=color, linewidth=2, alpha=0.7)

            # Add arrow every 10 steps
            if i % 10 == 0:
                du = trajectory_u[i+1] - trajectory_u[i]
                dv = trajectory_v[i+1] - trajectory_v[i]
                ax.arrow(trajectory_u[i], trajectory_v[i], du, dv,
                        head_width=8, head_length=8, fc=color, ec=color,
                        linewidth=1.5, alpha=0.8)

        # Mark start and end
        ax.plot(trajectory_u[0], trajectory_v[0], 'go', markersize=15,
               markeredgewidth=3, markerfacecolor='lime', label='Start')
        ax.plot(trajectory_u[-1], trajectory_v[-1], 'ro', markersize=15,
               markeredgewidth=3, markerfacecolor='red', label='End')

        # Add colorbar to show time progression
        sm = plt.cm.ScalarMappable(cmap='coolwarm',
                                   norm=plt.Normalize(vmin=0, vmax=T-1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Timestep', rotation=270, labelpad=20)

        ax.legend(loc='upper right', fontsize=12)
        ax.set_title('Action Trajectory Visualization (Transformed to Camera Frame)',
                    fontsize=14, weight='bold')
        # Don't turn off axis - keep it to ensure image is visible
        ax.set_aspect('equal')

    except Exception as e:
        ax.text(0.5, 0.5, f'Error loading image:\n{str(e)}',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)

    # Compute correlation statistics
    # Check if action deltas have consistent directions
    action_directions = action_deltas[:, :3] / (np.linalg.norm(action_deltas[:, :3], axis=1, keepdims=True) + 1e-8)
    direction_consistency = np.mean([
        np.dot(action_directions[i], action_directions[i+1])
        for i in range(len(action_directions)-1)
    ])

    print(f"\nAction Direction Consistency: {direction_consistency:.3f}")
    print(f"  (1.0 = perfectly consistent, 0.0 = random, -1.0 = alternating)")

    # Add summary text
    summary_text = f"""
Action-Camera Coordinate Transformation Test

Camera: LIBERO agentview/cam_high
- Position: {camera_pos}
- Quaternion (wxyz): {camera_quat}

Action Statistics:
- Total timesteps: {T}
- Translation deltas (mean): [{action_deltas[:, 0].mean():.4f}, {action_deltas[:, 1].mean():.4f}, {action_deltas[:, 2].mean():.4f}]
- Translation deltas (std): [{action_deltas[:, 0].std():.4f}, {action_deltas[:, 1].std():.4f}, {action_deltas[:, 2].std():.4f}]
- Direction consistency: {direction_consistency:.3f}

Visualization:
- Trajectory shows cumulative action deltas transformed to camera frame
- Color: Blue (start) → Red (end)
- Green circle: Start position (image center)
- Red circle: End position
- Arrows: Direction every 10 timesteps
    """

    fig.text(0.02, 0.02, summary_text, fontsize=10, family='monospace',
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    plt.close()

    # Return statistics
    return {
        'action_delta_mean': action_deltas[:, :3].mean(axis=0),
        'action_delta_std': action_deltas[:, :3].std(axis=0),
        'direction_consistency': direction_consistency,
    }


def main():
    data_dir = "/data/zijianzhang/LIBERA/data"

    try:
        # Load episode
        parquet_path = Path(data_dir) / "chunk-000" / "episode_000000.parquet"
        print(f"Loading episode from: {parquet_path}")
        df = pd.read_parquet(parquet_path)

        # Visualize multi-frame trajectory
        visualize_multi_frame_trajectory(
            df,
            save_path="/home/zijianzhang/openpi/visualizations/action_camera_multi_frame.png",
            num_frames=6
        )

        # Visualize full trajectory on single frame
        stats = visualize_projection(df, save_path="/home/zijianzhang/openpi/visualizations/action_camera_alignment.png")

        print(f"\n{'='*60}")
        print(f"Action-Camera Transformation Test Results:")
        print(f"{'='*60}")
        print(f"  Action delta mean: {stats['action_delta_mean']}")
        print(f"  Action delta std: {stats['action_delta_std']}")
        print(f"  Direction consistency: {stats['direction_consistency']:.3f}")
        print(f"{'='*60}")

        if stats['direction_consistency'] > 0.5:
            print("\n✓ Actions show consistent directions - transformation may be meaningful")
        else:
            print("\n⚠ Actions show inconsistent directions - may be noisy or normalized")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
