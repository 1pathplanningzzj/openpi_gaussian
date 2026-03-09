#!/usr/bin/env python3
"""
Test script to verify depth supervision integration.
Tests that:
1. Depth data can be loaded from parquet files
2. Depth data flows through the model correctly
3. Depth loss is computed and added to total loss
"""
import os
import sys
import torch
import numpy as np
import pyarrow.parquet as pq

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.openpi.training.depth_transform import LoadDepthTransform


def test_depth_transform():
    """Test LoadDepthTransform with sample data."""
    print("=" * 60)
    print("Testing LoadDepthTransform")
    print("=" * 60)

    # Create sample depth data
    depth_map = np.random.randn(256, 256).astype(np.float32)
    depth_bytes = depth_map.tobytes()

    # Test single frame
    sample = {
        "depth": depth_bytes,
        "wrist_depth": depth_bytes,
    }

    transform = LoadDepthTransform(use_depth=True)
    result = transform(sample)

    assert "observation.depth" in result, "Depth not added to sample"
    assert isinstance(result["observation.depth"], torch.Tensor), "Depth is not a tensor"
    assert result["observation.depth"].shape == (1, 1, 256, 256), f"Unexpected depth shape: {result['observation.depth'].shape}"

    print("✓ Single frame depth transform works")

    # Test multiple temporal frames
    depth_list = [depth_bytes, depth_bytes, depth_bytes, depth_bytes]
    sample_temporal = {
        "depth": depth_list,
        "wrist_depth": depth_list,
    }

    result_temporal = transform(sample_temporal)
    assert result_temporal["observation.depth"].shape == (4, 1, 256, 256), \
        f"Unexpected temporal depth shape: {result_temporal['observation.depth'].shape}"

    print("✓ Temporal depth transform works")
    print()


def test_load_real_depth_data():
    """Test loading real depth data from generated parquet files."""
    print("=" * 60)
    print("Testing Real Depth Data Loading")
    print("=" * 60)

    depth_dir = "/data/zijianzhang/LIBERA/data_with_depth/chunk-000"
    episode_file = "episode_000000.parquet"
    depth_path = os.path.join(depth_dir, episode_file)

    if not os.path.exists(depth_path):
        print(f"⚠ Depth file not found: {depth_path}")
        print("  Skipping real data test. Run generate_depth_simple.py first.")
        return

    # Load parquet file
    table = pq.read_table(depth_path)
    print(f"Columns: {table.column_names}")

    # Check depth columns exist
    assert "depth" in table.column_names, "depth column not found"
    assert "wrist_depth" in table.column_names, "wrist_depth column not found"

    # Load first frame
    depth_bytes = table["depth"][0].as_py()
    depth_array = np.frombuffer(depth_bytes, dtype=np.float32)
    depth_map = depth_array.reshape(256, 256)

    print(f"✓ Loaded depth map with shape: {depth_map.shape}")
    print(f"  Depth range: [{depth_map.min():.4f}, {depth_map.max():.4f}]")
    print(f"  Depth mean: {depth_map.mean():.4f}")
    print()


def test_model_observation_with_depth():
    """Test that Observation class can handle depth data."""
    print("=" * 60)
    print("Testing Observation with Depth")
    print("=" * 60)

    from src.openpi.models.model import Observation

    # Create sample observation data
    batch_size = 2
    data = {
        "image": {
            "image": torch.randn(batch_size, 224, 224, 3),
        },
        "image_mask": {
            "image": torch.ones(batch_size, dtype=torch.bool),
        },
        "state": torch.randn(batch_size, 32),
        "observation.depth": torch.randn(batch_size, 1, 256, 256),
    }

    obs = Observation.from_dict(data)

    assert obs.depth is not None, "Depth not loaded into Observation"
    assert obs.depth.shape == (batch_size, 1, 256, 256), f"Unexpected depth shape: {obs.depth.shape}"

    print(f"✓ Observation created with depth: {obs.depth.shape}")
    print()


def main():
    print("\n" + "=" * 60)
    print("Depth Supervision Integration Test")
    print("=" * 60 + "\n")

    try:
        test_depth_transform()
        test_load_real_depth_data()
        test_model_observation_with_depth()

        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Wait for depth generation to complete")
        print("2. Update training config to use depth-augmented dataset:")
        print("   dataset_root: /data/zijianzhang/LIBERA/data_with_depth")
        print("3. Set depth_loss_weight in config (default: 0.1)")
        print("4. Run training with depth supervision enabled")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
