#!/usr/bin/env python3
"""Create LIBERO camera parameters file based on known robosuite/LIBERO configuration.

Since LIBERO uses fixed camera configurations in robosuite, we can create the camera
parameters file based on the standard robosuite camera settings.

LIBERO cameras:
- agentview: Fixed position and orientation
- robot0_eye_in_hand: Wrist camera on robot

Default robosuite camera settings:
- FOV: Typically 60 degrees (fovy)
- Image size: 256x256 (or 128x128 in some configs)
"""

import argparse
import json
import math
import numpy as np
from pathlib import Path


def create_camera_params(image_height=256, image_width=256, fovy_deg=60.0):
    """Create camera parameters based on standard robosuite/LIBERO settings.
    
    Args:
        image_height: Image height in pixels
        image_width: Image width in pixels
        fovy_deg: Vertical field of view in degrees (default 60 for robosuite)
    
    Returns:
        dict: Camera parameters for agent and wrist cameras
    """
    # Convert FOV to radians
    fovy_rad = math.radians(fovy_deg)
    fovx_rad = fovy_rad * (image_width / image_height)  # Aspect ratio correction
    
    # Calculate focal length from FOV
    # f = (H/2) / tan(fovy/2)
    fy = (image_height / 2.0) / math.tan(fovy_rad / 2.0)
    fx = (image_width / 2.0) / math.tan(fovx_rad / 2.0)
    
    # Principal point (usually at image center)
    cx = image_width / 2.0
    cy = image_height / 2.0
    
    # Intrinsics matrix
    intrinsics = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    # For LIBERO, both cameras use the same intrinsics (same FOV and resolution)
    # Extrinsics are different but we'll use identity for now (can be updated if needed)
    # The actual extrinsics depend on the camera position in the scene
    
    # Default extrinsics (identity = camera at origin, looking down -Z)
    # This is a placeholder - actual extrinsics would come from robosuite
    extrinsics = np.eye(4)
    
    camera_params = {
        "agent": {
            "intrinsics": intrinsics.tolist(),
            "extrinsics": extrinsics.tolist(),
            "fov_x_deg": math.degrees(fovx_rad),
            "fov_y_deg": fovy_deg,
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "image_height": image_height,
            "image_width": image_width,
            "note": "Extrinsics are placeholder (identity). Actual values depend on camera position in scene."
        },
        "wrist": {
            "intrinsics": intrinsics.tolist(),
            "extrinsics": extrinsics.tolist(),
            "fov_x_deg": math.degrees(fovx_rad),
            "fov_y_deg": fovy_deg,
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "image_height": image_height,
            "image_width": image_width,
            "note": "Extrinsics are placeholder (identity). Actual values depend on camera position on robot."
        }
    }
    
    return camera_params


def main():
    parser = argparse.ArgumentParser(description="Create LIBERO camera parameters file")
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
        "--fovy",
        type=float,
        default=60.0,
        help="Vertical field of view in degrees (default: 60.0, standard robosuite)"
    )
    args = parser.parse_args()
    
    print("Creating LIBERO camera parameters...")
    print(f"  Image size: {args.image_width}x{args.image_height}")
    print(f"  FOV: {args.fovy}° (vertical)")
    
    camera_params = create_camera_params(
        image_height=args.image_height,
        image_width=args.image_width,
        fovy_deg=args.fovy
    )
    
    # Print summary
    print("\nCamera Parameters:")
    for view_name, params in camera_params.items():
        print(f"\n{view_name.upper()} Camera:")
        print(f"  FOV: {params['fov_x_deg']:.2f}° x {params['fov_y_deg']:.2f}°")
        print(f"  Focal length: fx={params['fx']:.2f}, fy={params['fy']:.2f}")
        print(f"  Principal point: cx={params['cx']:.2f}, cy={params['cy']:.2f}")
        print(f"  Note: {params['note']}")
    
    # Save to JSON
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(camera_params, f, indent=2)
    
    print(f"\nCamera parameters saved to: {output_path}")
    print("\nNote: Extrinsics are set to identity (placeholder).")
    print("      For accurate rendering, extrinsics should be extracted from robosuite environment.")


if __name__ == "__main__":
    main()
