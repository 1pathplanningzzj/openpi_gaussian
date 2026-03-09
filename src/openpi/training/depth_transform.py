"""
Transform to load depth data from parquet files with depth annotations.
"""
import io
import numpy as np
from PIL import Image
import torch


class LoadDepthTransform:
    """
    Transform that loads depth data from parquet files.

    Expects the dataset to have 'depth' and 'wrist_depth' columns containing
    serialized float32 depth maps of shape (256, 256).
    """

    def __init__(self, use_depth: bool = True, depth_key: str = "depth"):
        """
        Args:
            use_depth: Whether to load depth data
            depth_key: Key name for the main camera depth in the parquet file
        """
        self.use_depth = use_depth
        self.depth_key = depth_key
        self.wrist_depth_key = "wrist_depth"

    def __call__(self, sample: dict) -> dict:
        """
        Load depth data from the sample if available.

        Args:
            sample: Dictionary containing the data sample

        Returns:
            Modified sample with depth data added
        """
        import logging

        if not self.use_depth:
            return sample

        # Try to load main camera depth (4 frames to match image temporal dimension)
        if self.depth_key in sample:
            depth_bytes = sample[self.depth_key]

            if isinstance(depth_bytes, (list, tuple)):
                # Multiple frames (expected: 4 frames [t-2, t-1, t, t+1])
                depth_frames = []
                for depth_data in depth_bytes:
                    if isinstance(depth_data, bytes):
                        depth_array = np.frombuffer(depth_data, dtype=np.float32)
                        depth_map = depth_array.reshape(256, 256)
                    else:
                        depth_map = np.array(depth_data, dtype=np.float32)
                    depth_frames.append(depth_map)

                # Stack frames: [T, H, W] -> [T, 1, H, W] (add channel dimension)
                depth_tensor = torch.from_numpy(np.stack([f.copy() for f in depth_frames], axis=0))
                depth_tensor = depth_tensor.unsqueeze(1)  # [T, H, W] -> [T, 1, H, W]
            else:
                # Single frame (fallback)
                if isinstance(depth_bytes, bytes):
                    depth_array = np.frombuffer(depth_bytes, dtype=np.float32)
                    depth_map = depth_array.reshape(256, 256)
                else:
                    depth_map = np.array(depth_bytes, dtype=np.float32)

                # [H, W] -> [1, 1, H, W]
                depth_tensor = torch.from_numpy(depth_map.copy()).unsqueeze(0).unsqueeze(0)

            sample["observation/depth"] = depth_tensor

        # Try to load wrist camera depth (optional, 4 frames to match image temporal dimension)
        if self.wrist_depth_key in sample:
            wrist_depth_bytes = sample[self.wrist_depth_key]
            if isinstance(wrist_depth_bytes, (list, tuple)):
                # Multiple frames (expected: 4 frames [t-2, t-1, t, t+1])
                depth_frames = []
                for depth_data in wrist_depth_bytes:
                    if isinstance(depth_data, bytes):
                        depth_array = np.frombuffer(depth_data, dtype=np.float32)
                        depth_map = depth_array.reshape(256, 256)
                    else:
                        depth_map = np.array(depth_data, dtype=np.float32)
                    depth_frames.append(depth_map)

                # Stack frames: [T, H, W] -> [T, 1, H, W]
                wrist_depth_tensor = torch.from_numpy(np.stack([f.copy() for f in depth_frames], axis=0))
                wrist_depth_tensor = wrist_depth_tensor.unsqueeze(1)
            else:
                # Single frame (fallback)
                if isinstance(wrist_depth_bytes, bytes):
                    depth_array = np.frombuffer(wrist_depth_bytes, dtype=np.float32)
                    depth_map = depth_array.reshape(256, 256)
                else:
                    depth_map = np.array(wrist_depth_bytes, dtype=np.float32)

                wrist_depth_tensor = torch.from_numpy(depth_map.copy()).unsqueeze(0).unsqueeze(0)

            sample["observation/wrist_depth"] = wrist_depth_tensor

        return sample
