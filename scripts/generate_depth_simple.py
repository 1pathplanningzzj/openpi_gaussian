#!/usr/bin/env python3
"""
Simple script to generate depth maps for LIBERO dataset using Depth Anything V2.
Uses HuggingFace transformers for easy setup.

Usage:
    python scripts/generate_depth_simple.py
"""

from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import pyarrow.parquet as pq
import pyarrow as pa
from PIL import Image
import io


def load_depth_model():
    """Load Depth Anything V2 from HuggingFace."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    import os

    print("Loading Depth Anything V2 from HuggingFace...")

    # Use Depth Anything V2 Small (fastest)
    model_name = "depth-anything/Depth-Anything-V2-Small-hf"

    # Try using HuggingFace mirror for China
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    try:
        print(f"Attempting to load from HuggingFace (using mirror)...")
        processor = AutoImageProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            resume_download=True,
        )
        model = AutoModelForDepthEstimation.from_pretrained(
            model_name,
            trust_remote_code=True,
            resume_download=True,
        )
    except Exception as e:
        print(f"Failed to load from HuggingFace: {e}")
        print("\nTrying alternative: Using local Depth Anything V2 implementation...")

        # Fallback: Use depth-anything-v2 package directly
        try:
            from depth_anything_v2.dpt import DepthAnythingV2

            # Model config for small model
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            }

            model = DepthAnythingV2(**model_configs['vits'])

            # Try to load weights from local path or download
            checkpoint_path = '/tmp/depth_anything_v2_vits.pth'
            if not os.path.exists(checkpoint_path):
                print("Downloading model weights...")
                import urllib.request
                url = 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth'
                urllib.request.urlretrieve(url, checkpoint_path)

            state_dict = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(state_dict)

            processor = None  # Will use custom preprocessing
            print("Loaded Depth Anything V2 using direct implementation")

        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            print("\nPlease install depth-anything-v2:")
            print("  pip install depth-anything-v2")
            raise RuntimeError("Could not load Depth Anything V2 model.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()

    print(f"Model loaded on {device}")
    return processor, model, device


@torch.no_grad()
def predict_depth(processor, model, device, image, target_size=(256, 256)):
    """
    Predict depth for a PIL Image.

    Args:
        processor: HuggingFace image processor (can be None for custom preprocessing)
        model: Depth model
        device: torch device
        image: PIL Image
        target_size: (H, W) output size

    Returns:
        depth: [H, W] numpy array
    """
    if processor is not None:
        # Use HuggingFace processor
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Predict
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    else:
        # Custom preprocessing for depth-anything-v2
        import torchvision.transforms as transforms

        # Resize and normalize
        transform = transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image_tensor = transform(image).unsqueeze(0).to(device)

        # Predict
        predicted_depth = model(image_tensor)

    # Interpolate to target size
    predicted_depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1) if predicted_depth.ndim == 3 else predicted_depth,
        size=target_size,
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    # Convert to numpy
    depth = predicted_depth.cpu().numpy()

    return depth


def process_episode(pq_file, processor, model, device, output_dir):
    """
    Process one episode and add depth maps.

    Args:
        pq_file: Path to parquet file
        processor: Image processor
        model: Depth model
        device: torch device
        output_dir: Output directory
    """
    # Load episode
    table = pq.read_table(pq_file)
    episode = table.to_pydict()

    num_frames = len(episode['image'])

    # Process each frame
    depth_maps = []
    wrist_depth_maps = []

    for i in tqdm(range(num_frames), desc=f"Processing {pq_file.name}", leave=False):
        # Main camera depth
        image_data = episode['image'][i]
        # Handle both dict format {'bytes': ..., 'path': ...} and direct bytes
        if isinstance(image_data, dict):
            image_bytes = image_data['bytes']
        else:
            image_bytes = image_data
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        depth = predict_depth(processor, model, device, image, target_size=(256, 256))
        depth_maps.append(depth.astype(np.float32))

        # Wrist camera depth
        wrist_data = episode['wrist_image'][i]
        if isinstance(wrist_data, dict):
            wrist_bytes = wrist_data['bytes']
        else:
            wrist_bytes = wrist_data
        wrist_image = Image.open(io.BytesIO(wrist_bytes)).convert('RGB')
        wrist_depth = predict_depth(processor, model, device, wrist_image, target_size=(256, 256))
        wrist_depth_maps.append(wrist_depth.astype(np.float32))

    # Serialize depth arrays to bytes
    depth_serialized = [d.tobytes() for d in depth_maps]
    wrist_depth_serialized = [d.tobytes() for d in wrist_depth_maps]

    # Add to episode
    episode['depth'] = depth_serialized
    episode['wrist_depth'] = wrist_depth_serialized
    episode['depth_shape'] = [(256, 256)] * num_frames
    episode['depth_dtype'] = ['float32'] * num_frames

    # Save
    output_file = output_dir / pq_file.name
    table = pa.table(episode)
    pq.write_table(table, output_file)


def main():
    # Configuration
    data_dir = Path('/data/zijianzhang/LIBERA/data')
    output_dir = Path('/data/zijianzhang/LIBERA/data_with_depth')
    chunks = ['chunk-000', 'chunk-001']

    # Load model
    processor, model, device = load_depth_model()

    # Process each chunk
    for chunk_name in chunks:
        print(f"\n{'='*60}")
        print(f"Processing {chunk_name}")
        print(f"{'='*60}")

        chunk_dir = data_dir / chunk_name
        output_chunk_dir = output_dir / chunk_name
        output_chunk_dir.mkdir(parents=True, exist_ok=True)

        # Get all parquet files
        parquet_files = sorted(list(chunk_dir.glob("episode_*.parquet")))
        print(f"Found {len(parquet_files)} episodes")

        # Process each episode
        for pq_file in tqdm(parquet_files, desc=f"Episodes in {chunk_name}"):
            process_episode(pq_file, processor, model, device, output_chunk_dir)

    print("\n" + "="*60)
    print("✅ All chunks processed successfully!")
    print(f"Output saved to: {output_dir}")
    print("="*60)

    # Print usage instructions
    print("\nTo use the depth data in your dataloader:")
    print("```python")
    print("# Load episode")
    print("table = pq.read_table('episode_000000.parquet')")
    print("episode = table.to_pydict()")
    print("")
    print("# Deserialize depth")
    print("depth_bytes = episode['depth'][0]")
    print("depth = np.frombuffer(depth_bytes, dtype=np.float32).reshape(256, 256)")
    print("```")


if __name__ == '__main__':
    main()
