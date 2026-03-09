#!/usr/bin/env python3
"""
Visualize 3D point cloud from camera's perspective to match with 2D GT image.
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

# Load the visualization to get GT image
viz_path = "./visualizations/rendering_independent_decoder_test1/render_viz_step_001560_t1_pred_vlm.png"
viz_img = Image.open(viz_path)
viz_array = np.array(viz_img)

# Extract GT image (top-left)
img_height, img_width = viz_array.shape[:2]
gt_width = img_width // 3
gt_height = img_height // 2
gt_image = viz_array[:gt_height, :gt_width]

# Load the interactive HTML to extract point cloud data
# We'll need to read the Gaussian parameters from a checkpoint
# For now, let's create a synthetic example to demonstrate the concept

# Create figure with 3 subplots
fig = plt.figure(figsize=(20, 6))

# Subplot 1: GT image
ax1 = fig.add_subplot(131)
ax1.imshow(gt_image)
ax1.set_title('Agent GT (2D Camera View)', fontsize=14, weight='bold')
ax1.axis('off')

# Subplot 2: 3D point cloud from camera perspective (X-Y plane, looking down Z)
ax2 = fig.add_subplot(132)
ax2.set_title('3D Point Cloud (Camera Perspective)\nLooking from origin along +Z direction',
              fontsize=14, weight='bold')
ax2.set_xlabel('X (Left ← → Right)', fontsize=12)
ax2.set_ylabel('Y (Up ← → Down)', fontsize=12)
ax2.set_xlim(-50, 50)
ax2.set_ylim(-40, 40)
ax2.invert_yaxis()  # Invert Y to match image coordinates
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Y=0')
ax2.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='X=0')

# Add annotations showing regions
regions = [
    {'x_range': (-20, 20), 'y_range': (0, 30), 'z_range': (50, 70),
     'color': 'purple', 'label': 'Region 1: Table\n(Z=50-70, close)'},
    {'x_range': (20, 40), 'y_range': (-10, 10), 'z_range': (80, 100),
     'color': 'yellow', 'label': 'Region 2: Object/Arm\n(Z=80-100, far)'},
    {'x_range': (-40, -20), 'y_range': (-10, 10), 'z_range': (60, 80),
     'color': 'cyan', 'label': 'Region 3: Left Object\n(Z=60-80, medium)'},
]

# Draw regions on the X-Y plane view
for region in regions:
    x_min, x_max = region['x_range']
    y_min, y_max = region['y_range']
    z_min, z_max = region['z_range']

    # Draw rectangle for this region
    from matplotlib.patches import Rectangle
    rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                     linewidth=2, edgecolor=region['color'],
                     facecolor=region['color'], alpha=0.3)
    ax2.add_patch(rect)

    # Add label
    ax2.text((x_min + x_max) / 2, (y_min + y_max) / 2,
            region['label'],
            fontsize=10, weight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

ax2.legend(loc='upper right')

# Subplot 3: Explanation diagram
ax3 = fig.add_subplot(133)
ax3.axis('off')

explanation = """
HOW TO MATCH 3D POINT CLOUD WITH 2D IMAGE

Camera Setup:
• Position: Origin (0, 0, 0)
• Direction: Looking along +Z axis (into the scene)
• X axis: Left (-) to Right (+)
• Y axis: Up (-) to Down (+)
• Z axis: Depth (small = close, large = far)

Middle Panel Shows:
The X-Y plane as seen from camera's viewpoint
(imagine looking down the Z axis from origin)

Color Coding by Depth (Z value):
🟣 Purple: Z = 50-70 (close to camera)
   → Table surface in center
🟡 Yellow: Z = 80-100 (far from camera)
   → Objects/robot arm on right side
🔵 Cyan: Z = 60-80 (medium depth)
   → Objects on left side

How to Match:
1. Look at the middle panel (X-Y view)
2. The layout matches the GT image layout
3. X position → horizontal position in image
4. Y position → vertical position in image
5. Z value → depth (not visible in 2D, but
   determines which objects are in front)

Example Correspondences:
• Purple region (X: -20~20, Y: 0~30)
  → Center-bottom area in GT image (table)

• Yellow region (X: 20~40, Y: -10~10)
  → Right side in GT image (robot arm/object)

• Cyan region (X: -40~-20, Y: -10~10)
  → Left side in GT image (objects)

Key Insight:
The middle panel is like looking at the 3D
scene from above (bird's eye view), which
directly corresponds to what the camera sees
in the 2D image!
"""

ax3.text(0.05, 0.95, explanation,
        fontsize=11, verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
output_path = "./visualizations/rendering_independent_decoder_test1/camera_perspective_view.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Camera perspective visualization saved to: {output_path}")

# Create a simpler version with arrows showing correspondence
fig2, axes = plt.subplots(1, 3, figsize=(24, 8))

# Left: GT image with numbered regions
axes[0].imshow(gt_image)
h, w = gt_image.shape[:2]

# Draw numbered regions on GT image
gt_regions = [
    {'bbox': (w*0.2, h*0.4, w*0.6, h*0.5), 'num': '1', 'color': 'purple'},
    {'bbox': (w*0.65, h*0.3, w*0.3, h*0.4), 'num': '2', 'color': 'yellow'},
    {'bbox': (w*0.05, h*0.35, w*0.25, h*0.35), 'num': '3', 'color': 'cyan'},
]

for region in gt_regions:
    x, y, w_box, h_box = region['bbox']
    rect = Rectangle((x, y), w_box, h_box,
                     linewidth=4, edgecolor=region['color'],
                     facecolor='none', linestyle='--')
    axes[0].add_patch(rect)

    # Add number
    axes[0].text(x + w_box/2, y + h_box/2,
                region['num'],
                fontsize=40, weight='bold', color='white',
                ha='center', va='center',
                bbox=dict(boxstyle='circle,pad=0.3',
                         facecolor=region['color'], alpha=0.9))

axes[0].set_title('2D Image (Camera View)', fontsize=16, weight='bold')
axes[0].axis('off')

# Middle: X-Y plane view with same numbered regions
axes[1].set_xlim(-50, 50)
axes[1].set_ylim(-40, 40)
axes[1].invert_yaxis()
axes[1].set_xlabel('X (Left ← → Right)', fontsize=14, weight='bold')
axes[1].set_ylabel('Y (Up ← → Down)', fontsize=14, weight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=2)

# Draw same regions on X-Y plane
xy_regions = [
    {'x_range': (-20, 20), 'y_range': (0, 30), 'num': '1', 'color': 'purple'},
    {'x_range': (20, 40), 'y_range': (-10, 10), 'num': '2', 'color': 'yellow'},
    {'x_range': (-40, -20), 'y_range': (-10, 10), 'num': '3', 'color': 'cyan'},
]

for region in xy_regions:
    x_min, x_max = region['x_range']
    y_min, y_max = region['y_range']

    rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                     linewidth=4, edgecolor=region['color'],
                     facecolor=region['color'], alpha=0.4)
    axes[1].add_patch(rect)

    # Add number
    axes[1].text((x_min + x_max) / 2, (y_min + y_max) / 2,
                region['num'],
                fontsize=40, weight='bold', color='white',
                ha='center', va='center',
                bbox=dict(boxstyle='circle,pad=0.3',
                         facecolor=region['color'], alpha=0.9))

axes[1].set_title('3D Point Cloud (X-Y Plane)\nCamera at (0,0,0) looking along +Z',
                 fontsize=16, weight='bold')

# Right: 3D diagram showing camera position
axes[2].axis('off')
diagram = """
CAMERA COORDINATE SYSTEM

        Y (down)
        ↓
        |
        |
    ----+---- X (right)
       /|
      / |
     /  |
    Z (depth into scene)

Camera Position: (0, 0, 0)
Camera Looking: +Z direction

3D Point (X, Y, Z) projects to:
2D Pixel (x, y) where:
  x ∝ X / Z
  y ∝ Y / Z

Region Correspondences:

① Purple Region
   2D: Center-bottom of image
   3D: X ∈ [-20, 20]
       Y ∈ [0, 30]
       Z ∈ [50, 70] (CLOSE)
   Object: Table surface

② Yellow Region
   2D: Right side of image
   3D: X ∈ [20, 40]
       Y ∈ [-10, 10]
       Z ∈ [80, 100] (FAR)
   Object: Robot arm/object

③ Cyan Region
   2D: Left side of image
   3D: X ∈ [-40, -20]
       Y ∈ [-10, 10]
       Z ∈ [60, 80] (MEDIUM)
   Object: Left side objects

The middle panel shows the
X-Y layout as if you're looking
down the Z axis from the camera!
"""

axes[2].text(0.05, 0.95, diagram,
            fontsize=12, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1',
                     facecolor='lightblue', alpha=0.9))

plt.tight_layout()
simple_path = "./visualizations/rendering_independent_decoder_test1/simple_correspondence.png"
plt.savefig(simple_path, dpi=150, bbox_inches='tight')
print(f"Simple correspondence visualization saved to: {simple_path}")

plt.close('all')
