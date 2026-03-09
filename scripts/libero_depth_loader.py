"""
Data loader utilities for LIBERO dataset with depth maps.
"""

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import io
import torch


def load_episode_with_depth(parquet_path):
    """
    Load a LIBERO episode with depth maps.

    Args:
        parquet_path: Path to parquet file

    Returns:
        dict with keys:
            - image: List of PIL Images
            - wrist_image: List of PIL Images
            - depth: List of [H, W] numpy arrays
            - wrist_depth: List of [H, W] numpy arrays
            - state: List of state vectors
            - actions: List of action vectors
            - ... (other metadata)
    """
    # Load parquet
    table = pq.read_table(parquet_path)
    episode = table.to_pydict()

    # Decode images
    images = [Image.open(io.BytesIO(img_bytes)).convert('RGB')
              for img_bytes in episode['image']]
    wrist_images = [Image.open(io.BytesIO(img_bytes)).convert('RGB')
                   for img_bytes in episode['wrist_image']]

    # Decode depth maps (if available)
    depth_maps = None
    wrist_depth_maps = None

    if 'depth' in episode:
        depth_shape = episode['depth_shape'][0]  # (H, W)
        depth_dtype = episode['depth_dtype'][0]  # 'float32'

        depth_maps = [
            np.frombuffer(depth_bytes, dtype=np.float32).reshape(depth_shape)
            for depth_bytes in episode['depth']
        ]

        wrist_depth_maps = [
            np.frombuffer(depth_bytes, dtype=np.float32).reshape(depth_shape)
            for depth_bytes in episode['wrist_depth']
        ]

    return {
        'image': images,
        'wrist_image': wrist_images,
        'depth': depth_maps,
        'wrist_depth': wrist_depth_maps,
        'state': episode['state'],
        'actions': episode['actions'],
        'timestamp': episode['timestamp'],
        'frame_index': episode['frame_index'],
        'episode_index': episode['episode_index'],
        'task_index': episode['task_index'],
    }


def prepare_batch_with_depth(episodes, frame_indices, device='cuda'):
    """
    Prepare a batch of frames with depth for training.

    Args:
        episodes: List of episode dicts (from load_episode_with_depth)
        frame_indices: List of frame indices to extract
        device: torch device

    Returns:
        dict with tensors:
            - images: [B, 3, H, W]
            - depth: [B, 1, H, W]
            - actions: [B, action_dim]
            - ...
    """
    batch_images = []
    batch_depth = []
    batch_actions = []
    batch_states = []

    for episode, frame_idx in zip(episodes, frame_indices):
        # Image
        image = episode['image'][frame_idx]
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)  # [3, H, W]
        batch_images.append(image_tensor)

        # Depth
        if episode['depth'] is not None:
            depth = episode['depth'][frame_idx]
            depth_tensor = torch.from_numpy(depth).unsqueeze(0)  # [1, H, W]
            batch_depth.append(depth_tensor)

        # Action
        action = episode['actions'][frame_idx]
        action_tensor = torch.from_numpy(np.array(action, dtype=np.float32))
        batch_actions.append(action_tensor)

        # State
        state = episode['state'][frame_idx]
        state_tensor = torch.from_numpy(np.array(state, dtype=np.float32))
        batch_states.append(state_tensor)

    # Stack into batches
    batch = {
        'images': torch.stack(batch_images).to(device),  # [B, 3, H, W]
        'actions': torch.stack(batch_actions).to(device),  # [B, action_dim]
        'states': torch.stack(batch_states).to(device),  # [B, state_dim]
    }

    if batch_depth:
        batch['depth'] = torch.stack(batch_depth).to(device)  # [B, 1, H, W]

    return batch


# Example usage
if __name__ == '__main__':
    # Load episode
    episode = load_episode_with_depth('/data/zijianzhang/LIBERA/data_with_depth/chunk-000/episode_000000.parquet')

    print(f"Episode loaded:")
    print(f"  - Frames: {len(episode['image'])}")
    print(f"  - Image size: {episode['image'][0].size}")
    print(f"  - Has depth: {episode['depth'] is not None}")

    if episode['depth'] is not None:
        print(f"  - Depth shape: {episode['depth'][0].shape}")
        print(f"  - Depth range: [{episode['depth'][0].min():.3f}, {episode['depth'][0].max():.3f}]")

    # Prepare a batch
    batch = prepare_batch_with_depth([episode, episode], [0, 1])
    print(f"\nBatch prepared:")
    print(f"  - images: {batch['images'].shape}")
    print(f"  - depth: {batch['depth'].shape}")
    print(f"  - actions: {batch['actions'].shape}")
