#!/usr/bin/env python3
"""
验证生成的深度图与原始图像的对应关系
Verify correspondence between generated depth maps and original images
"""
import os
import io
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

def load_episode_data(original_path, depth_path):
    """加载原始episode和带深度的episode"""
    original_table = pq.read_table(original_path)
    depth_table = pq.read_table(depth_path)

    original_episode = {
        'image': original_table['image'].to_pylist(),
        'wrist_image': original_table['wrist_image'].to_pylist(),
    }

    depth_episode = {
        'image': depth_table['image'].to_pylist(),
        'wrist_image': depth_table['wrist_image'].to_pylist(),
        'depth': depth_table['depth'].to_pylist(),
        'wrist_depth': depth_table['wrist_depth'].to_pylist(),
    }

    return original_episode, depth_episode

def bytes_to_image(image_data):
    """将bytes或dict格式的图像数据转换为PIL Image"""
    if isinstance(image_data, dict):
        image_bytes = image_data['bytes']
    else:
        image_bytes = image_data
    return Image.open(io.BytesIO(image_bytes))

def bytes_to_depth(depth_bytes):
    """将bytes格式的深度数据转换为numpy数组"""
    depth_array = np.frombuffer(depth_bytes, dtype=np.float32)
    # 假设深度图是256x256
    return depth_array.reshape(256, 256)

def visualize_correspondence(image, depth, title, save_path):
    """可视化图像和深度图的对应关系"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原始图像
    axes[0].imshow(image)
    axes[0].set_title(f'{title} - Original Image')
    axes[0].axis('off')

    # 深度图
    depth_vis = axes[1].imshow(depth, cmap='turbo')
    axes[1].set_title(f'{title} - Depth Map')
    axes[1].axis('off')
    plt.colorbar(depth_vis, ax=axes[1], fraction=0.046, pad=0.04)

    # 叠加显示
    axes[2].imshow(image)
    overlay = axes[2].imshow(depth, cmap='turbo', alpha=0.5)
    axes[2].set_title(f'{title} - Overlay')
    axes[2].axis('off')
    plt.colorbar(overlay, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved visualization to {save_path}")

def check_depth_statistics(depth, camera_name):
    """检查深度图的统计信息"""
    print(f"\n{camera_name} Depth Statistics:")
    print(f"  Shape: {depth.shape}")
    print(f"  Min: {depth.min():.4f}")
    print(f"  Max: {depth.max():.4f}")
    print(f"  Mean: {depth.mean():.4f}")
    print(f"  Std: {depth.std():.4f}")
    print(f"  Has NaN: {np.isnan(depth).any()}")
    print(f"  Has Inf: {np.isinf(depth).any()}")

def main():
    # 配置路径
    original_dir = "/data/zijianzhang/LIBERA/data/chunk-000"
    depth_dir = "/data/zijianzhang/LIBERA/data_with_depth/chunk-000"
    output_dir = "/home/zijianzhang/openpi/visualizations/depth_verification"

    os.makedirs(output_dir, exist_ok=True)

    # 选择第一个episode进行验证
    episode_file = "episode_000000.parquet"
    original_path = os.path.join(original_dir, episode_file)
    depth_path = os.path.join(depth_dir, episode_file)

    if not os.path.exists(depth_path):
        print(f"Error: Depth file not found at {depth_path}")
        print("Please wait for depth generation to complete.")
        return

    print(f"Loading episode from:")
    print(f"  Original: {original_path}")
    print(f"  With depth: {depth_path}")

    # 加载数据
    original_episode, depth_episode = load_episode_data(original_path, depth_path)

    num_frames = len(original_episode['image'])
    print(f"\nTotal frames: {num_frames}")

    # 验证几个关键帧：开始、中间、结束
    test_frames = [0, num_frames // 2, num_frames - 1]

    for frame_idx in test_frames:
        print(f"\n{'='*60}")
        print(f"Verifying Frame {frame_idx}")
        print(f"{'='*60}")

        # 主相机
        original_img = bytes_to_image(original_episode['image'][frame_idx])
        depth_img = bytes_to_image(depth_episode['image'][frame_idx])
        depth_map = bytes_to_depth(depth_episode['depth'][frame_idx])

        # 检查图像是否一致
        original_array = np.array(original_img)
        depth_array = np.array(depth_img)

        if np.array_equal(original_array, depth_array):
            print("✓ Main camera images match perfectly")
        else:
            print("✗ WARNING: Main camera images do NOT match!")
            print(f"  Original shape: {original_array.shape}")
            print(f"  Depth version shape: {depth_array.shape}")
            print(f"  Max difference: {np.abs(original_array - depth_array).max()}")

        check_depth_statistics(depth_map, "Main Camera")

        # 可视化主相机
        save_path = os.path.join(output_dir, f"frame_{frame_idx:04d}_main.png")
        visualize_correspondence(original_img, depth_map, f"Frame {frame_idx} - Main Camera", save_path)

        # 手腕相机
        original_wrist = bytes_to_image(original_episode['wrist_image'][frame_idx])
        depth_wrist_img = bytes_to_image(depth_episode['wrist_image'][frame_idx])
        depth_wrist_map = bytes_to_depth(depth_episode['wrist_depth'][frame_idx])

        # 检查图像是否一致
        original_wrist_array = np.array(original_wrist)
        depth_wrist_array = np.array(depth_wrist_img)

        if np.array_equal(original_wrist_array, depth_wrist_array):
            print("✓ Wrist camera images match perfectly")
        else:
            print("✗ WARNING: Wrist camera images do NOT match!")
            print(f"  Original shape: {original_wrist_array.shape}")
            print(f"  Depth version shape: {depth_wrist_array.shape}")
            print(f"  Max difference: {np.abs(original_wrist_array - depth_wrist_array).max()}")

        check_depth_statistics(depth_wrist_map, "Wrist Camera")

        # 可视化手腕相机
        save_path = os.path.join(output_dir, f"frame_{frame_idx:04d}_wrist.png")
        visualize_correspondence(original_wrist, depth_wrist_map, f"Frame {frame_idx} - Wrist Camera", save_path)

    print(f"\n{'='*60}")
    print("Verification Complete!")
    print(f"{'='*60}")
    print(f"Visualizations saved to: {output_dir}")
    print("\nPlease check the generated images to verify:")
    print("  1. Original images are preserved correctly")
    print("  2. Depth maps show reasonable depth values")
    print("  3. Overlay shows depth aligns with image content")
    print("  4. Closer objects have smaller depth values (warmer colors)")

if __name__ == "__main__":
    main()
