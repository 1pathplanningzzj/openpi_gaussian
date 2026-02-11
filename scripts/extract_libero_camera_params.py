#!/usr/bin/env python3
"""Extract camera parameters from LIBERO robosuite environment.

This script creates a LIBERO environment and extracts camera intrinsics and extrinsics
for both agent and wrist cameras. The parameters are saved to a JSON file that can be
loaded during training.

Usage:
    python scripts/extract_libero_camera_params.py --output camera_params.json
"""

import argparse
import json
import numpy as np
import sys
from pathlib import Path

# Add third_party to path
base_path = Path(__file__).parent.parent
sys.path.insert(0, str(base_path / "third_party" / "robosuite"))
sys.path.insert(0, str(base_path / "third_party" / "libero"))

try:
    import robosuite
    from robosuite.utils.camera_utils import (
        get_camera_intrinsic_matrix,
        get_camera_extrinsic_matrix,
    )
except ImportError as e:
    print(f"Error importing robosuite: {e}")
    print("Please ensure robosuite is installed and in PYTHONPATH")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from libero.libero.benchmark import get_benchmark_dict
    from libero.libero.envs.env_wrapper import ControlEnv
except ImportError as e:
    print(f"Error importing libero: {e}")
    print("Please ensure libero is installed and in PYTHONPATH")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def extract_camera_params(env, camera_name, image_height=256, image_width=256):
    """Extract camera intrinsics and extrinsics from robosuite environment.
    
    Args:
        env: robosuite environment instance
        camera_name: Name of the camera (e.g., "agentview", "robot0_eye_in_hand")
        image_height: Image height in pixels
        image_width: Image width in pixels
    
    Returns:
        dict: Camera parameters including intrinsics and extrinsics
    """
    sim = env.sim
    
    # Get intrinsics
    intrinsics = get_camera_intrinsic_matrix(
        sim=sim,
        camera_name=camera_name,
        camera_height=image_height,
        camera_width=image_width
    )
    
    # Get extrinsics (4x4 matrix: world to camera)
    extrinsics = get_camera_extrinsic_matrix(
        sim=sim,
        camera_name=camera_name
    )
    
    # Extract FOV from intrinsics
    # f = fx = fy (assuming square pixels)
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    
    # Calculate FOV from focal length
    fov_x = 2 * np.arctan(image_width / (2 * fx)) * 180 / np.pi
    fov_y = 2 * np.arctan(image_height / (2 * fy)) * 180 / np.pi
    
    return {
        "intrinsics": intrinsics.tolist(),  # 3x3 matrix
        "extrinsics": extrinsics.tolist(),  # 4x4 matrix (world to camera)
        "fov_x_deg": float(fov_x),
        "fov_y_deg": float(fov_y),
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "image_height": image_height,
        "image_width": image_width,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract LIBERO camera parameters")
    parser.add_argument(
        "--output",
        type=str,
        default="libero_camera_params.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=256,
        help="Image height in pixels (default: 256)"
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=256,
        help="Image width in pixels (default: 256)"
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="libero_10",
        help="LIBERO task name (default: libero_10)"
    )
    args = parser.parse_args()
    
    print("Creating LIBERO environment...")
    
    # Get a LIBERO task to create environment
    # We'll use the first available task
    try:
        benchmark_dict = get_benchmark_dict()
        if args.task_name not in benchmark_dict:
            print(f"Warning: Task '{args.task_name}' not found. Using first available task.")
            task_name = list(benchmark_dict.keys())[0]
        else:
            task_name = args.task_name
        
        task_spec = benchmark_dict[task_name][0]  # Get first task in the benchmark
        
        # Get BDDL file path
        bddl_file = task_spec.get_bddl_file()
        
        # Create environment using ControlEnv (same as LIBERO uses)
        env = ControlEnv(
            bddl_file_name=bddl_file,
            has_renderer=False,
            has_offscreen_renderer=True,
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=args.image_height,
            camera_widths=args.image_width,
        )
        
        print(f"Environment created successfully!")
        print(f"Task: {task_name}")
        print(f"Camera names: {env.camera_names}")
        
        # Extract camera parameters
        camera_params = {}
        
        # Agent camera (agentview)
        if "agentview" in env.camera_names:
            print("\nExtracting agentview camera parameters...")
            agent_params = extract_camera_params(
                env, "agentview", args.image_height, args.image_width
            )
            camera_params["agent"] = agent_params
            print(f"  FOV: {agent_params['fov_x_deg']:.2f}° x {agent_params['fov_y_deg']:.2f}°")
            print(f"  Focal length: fx={agent_params['fx']:.2f}, fy={agent_params['fy']:.2f}")
            print(f"  Principal point: cx={agent_params['cx']:.2f}, cy={agent_params['cy']:.2f}")
        
        # Wrist camera (robot0_eye_in_hand)
        if "robot0_eye_in_hand" in env.camera_names:
            print("\nExtracting robot0_eye_in_hand camera parameters...")
            wrist_params = extract_camera_params(
                env, "robot0_eye_in_hand", args.image_height, args.image_width
            )
            camera_params["wrist"] = wrist_params
            print(f"  FOV: {wrist_params['fov_x_deg']:.2f}° x {wrist_params['fov_y_deg']:.2f}°")
            print(f"  Focal length: fx={wrist_params['fx']:.2f}, fy={wrist_params['fy']:.2f}")
            print(f"  Principal point: cx={wrist_params['cx']:.2f}, cy={wrist_params['cy']:.2f}")
        
        # Save to JSON
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(camera_params, f, indent=2)
        
        print(f"\nCamera parameters saved to: {output_path}")
        print("\nSummary:")
        print(f"  Agent camera FOV: {camera_params['agent']['fov_x_deg']:.2f}°")
        print(f"  Wrist camera FOV: {camera_params['wrist']['fov_x_deg']:.2f}°")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
