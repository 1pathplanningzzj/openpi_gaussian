# zijian
# date 2026.01.26
# Description: Gaussian Renderer Module for World Model Supervision
# Purpose: Integrate AD-FFgsStudio's GaussianRasterizer for rendering loss computation

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

# Add AD-FFgsStudio to python path
_root_path = Path(__file__).resolve().parents[3]
_diff_gauss_path = _root_path / "third_party" / "AD-FFgsStudio" / "diff-gaussian-rasterization"
if str(_diff_gauss_path) not in sys.path:
    sys.path.append(str(_diff_gauss_path))

try:
    from diff_gaussian_rasterization import (
        GaussianRasterizer as _GaussianRasterizer,
        GaussianRasterizationSettings
    )
    RASTERIZER_AVAILABLE = True
except ImportError:
    print("Warning: GaussianRasterizer not available. Rendering loss will be disabled.")
    RASTERIZER_AVAILABLE = False


def convert_sigma_to_scale_rotation(sigma_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert 6D covariance parameters to scales and rotations.
    转换协方差矩阵 
    Args:
        sigma_params: [B, N, 6] - Upper triangle of covariance matrix
                      [s11, s12, s13, s22, s23, s33]

    Returns:
        scales: [B, N, 3] - 3D scales
        rotations: [B, N, 4] - Quaternion rotations (w, x, y, z)
    """
    B, N, _ = sigma_params.shape
    device = sigma_params.device

    # Build 3x3 covariance matrix (symmetric)
    cov = torch.zeros(B, N, 3, 3, device=device, dtype=sigma_params.dtype)
    cov[..., 0, 0] = sigma_params[..., 0]  # s11
    cov[..., 0, 1] = sigma_params[..., 1]  # s12
    cov[..., 0, 2] = sigma_params[..., 2]  # s13
    cov[..., 1, 0] = sigma_params[..., 1]  # s21 = s12
    cov[..., 1, 1] = sigma_params[..., 3]  # s22
    cov[..., 1, 2] = sigma_params[..., 4]  # s23
    cov[..., 2, 0] = sigma_params[..., 2]  # s31 = s13
    cov[..., 2, 1] = sigma_params[..., 4]  # s32 = s23
    cov[..., 2, 2] = sigma_params[..., 5]  # s33

    # Eigenvalue decomposition: Σ = R S S^T R^T
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)

    # Scales = sqrt(eigenvalues), clamped to avoid numerical issues
    scales = torch.sqrt(torch.clamp(eigenvalues, min=1e-6))  # [B, N, 3]

    # Convert rotation matrix to quaternion
    rotations = rotation_matrix_to_quaternion(eigenvectors)  # [B, N, 4]

    return scales, rotations


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to quaternions.

    Args:
        R: [B, N, 3, 3] - Rotation matrices

    Returns:
        q: [B, N, 4] - Quaternions (w, x, y, z)
    """
    input_shape = R.shape
    if len(input_shape) > 3:
        # Flatten batch and N dimensions for safe processing
        R = R.reshape(-1, 3, 3)
    
    B_flat, _, _ = R.shape
    device = R.device

    # Extract rotation matrix elements
    r00, r01, r02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    r10, r11, r12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    r20, r21, r22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]

    # Compute quaternion components
    trace = r00 + r11 + r22

    q = torch.zeros(B_flat, 4, device=device, dtype=R.dtype)

    # Case 1: trace > 0
    mask1 = trace > 0
    if mask1.any():
        s = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * w
        q[mask1, 0] = 0.25 * s
        q[mask1, 1] = (r21[mask1] - r12[mask1]) / s
        q[mask1, 2] = (r02[mask1] - r20[mask1]) / s
        q[mask1, 3] = (r10[mask1] - r01[mask1]) / s

    # Case 2: r00 is the largest diagonal element
    mask2 = (~mask1) & (r00 > r11) & (r00 > r22)
    if mask2.any():
        s = torch.sqrt(1.0 + r00[mask2] - r11[mask2] - r22[mask2]) * 2  # s = 4 * x
        q[mask2, 0] = (r21[mask2] - r12[mask2]) / s
        q[mask2, 1] = 0.25 * s
        q[mask2, 2] = (r01[mask2] + r10[mask2]) / s
        q[mask2, 3] = (r02[mask2] + r20[mask2]) / s

    # Case 3: r11 is the largest diagonal element
    mask3 = (~mask1) & (~mask2) & (r11 > r22)
    if mask3.any():
        s = torch.sqrt(1.0 + r11[mask3] - r00[mask3] - r22[mask3]) * 2  # s = 4 * y
        q[mask3, 0] = (r02[mask3] - r20[mask3]) / s
        q[mask3, 1] = (r01[mask3] + r10[mask3]) / s
        q[mask3, 2] = 0.25 * s
        q[mask3, 3] = (r12[mask3] + r21[mask3]) / s

    # Case 4: r22 is the largest diagonal element
    mask4 = (~mask1) & (~mask2) & (~mask3)
    if mask4.any():
        s = torch.sqrt(1.0 + r22[mask4] - r00[mask4] - r11[mask4]) * 2  # s = 4 * z
        q[mask4, 0] = (r10[mask4] - r01[mask4]) / s
        q[mask4, 1] = (r02[mask4] + r20[mask4]) / s
        q[mask4, 2] = (r12[mask4] + r21[mask4]) / s
        q[mask4, 3] = 0.25 * s

    # Normalize quaternion
    q = F.normalize(q, dim=-1)

    # Reshape back to original shape [B, N, 4]
    if len(input_shape) > 3:
        q = q.reshape(*input_shape[:-2], 4)

    return q


def project_to_2d(
    xyz: torch.Tensor,
    camera_params: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """
    Project 3D points to 2D pixel coordinates.

    Args:
        xyz: [B, N, 3] - 3D points in world coordinates
        camera_params: Dictionary containing camera parameters

    Returns:
        pixels_2d: [B, N, 2] - Pixel coordinates (u, v)
    """
    B, N, _ = xyz.shape
    device = xyz.device

    # Transform to camera coordinates
    viewmatrix = camera_params["viewmatrix"]  # [B, 4, 4]

    # Homogeneous coordinates
    xyz_homo = torch.cat([
        xyz,
        torch.ones(B, N, 1, device=device, dtype=xyz.dtype)
    ], dim=-1)  # [B, N, 4]

    # Transform to camera space
    # viewmatrix is [B, 4, 4]. We want (XYZ_homo @ View.T) per batch.
    # xyz_homo: [B, N, 4]
    # viewmatrix.transpose(-1, -2): [B, 4, 4] (Row-major camera world-to-cam?)
    # Usually xyz_cam = xyz_world @ M^T if M is 4x4 multiplying column vectors.
    # Here M is [B, 4, 4]. We need M^T for "right multiplication".
    # torch.matmul handles batch dimensions correctly.
    # [B, N, 4] x [B, 4, 4] -> [B, N, 4]
    xyz_cam = torch.matmul(xyz_homo, viewmatrix.transpose(-1, -2))  # [B, N, 4]

    # Project to image plane
    intrinsics = camera_params["intrinsics"]  # [B, 3, 3] or [3, 3]
    
    # Handle batched intrinsics
    if intrinsics.ndim == 3:
        # [B, 3, 3]
        fx = intrinsics[:, 0, 0].unsqueeze(1) # [B, 1]
        fy = intrinsics[:, 1, 1].unsqueeze(1)
        cx = intrinsics[:, 0, 2].unsqueeze(1)
        cy = intrinsics[:, 1, 2].unsqueeze(1)
    else:
        # [3, 3]
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Perspective projection
    x_cam = xyz_cam[..., 0]
    y_cam = xyz_cam[..., 1]
    z_cam = xyz_cam[..., 2].clamp(min=1e-6)  # Avoid division by zero

    u = fx * (x_cam / z_cam) + cx
    v = fy * (y_cam / z_cam) + cy

    # AD-FFgsStudio's diff-gaussian-rasterization returns [N, 3] gradients for means2D
    # So we must provide [N, 3] inputs. The 3rd component is likely depth or ignored.
    # We'll pass z_cam just in case.
    pixels_2d = torch.stack([u, v, z_cam], dim=-1)  # [B, N, 3]

    return pixels_2d


class GaussianRenderer(nn.Module):
    """
    Wrapper for AD-FFgsStudio's GaussianRasterizer.
    Renders 3D Gaussians to 2D images for supervision.
    """

    def __init__(self, image_size: int = 224, sh_degree: int = 3, scale_factor: float = 1.0):
        super().__init__()
        self.image_size = image_size
        self.sh_degree = sh_degree
        self.scale_factor = scale_factor  # Scale multiplier to adjust Gaussian sizes

        if not RASTERIZER_AVAILABLE:
            raise ImportError(
                "GaussianRasterizer not available. "
                "Please compile diff-gaussian-rasterization."
            )

    def forward(
        self,
        gaussian_params: Dict[str, torch.Tensor],
        camera_params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Render 3D Gaussians from a specific camera viewpoint.
        """
        B, N, _ = gaussian_params["xyz"].shape
        device = gaussian_params["xyz"].device

        # Convert covariance parameters to scales and rotations
        scales, rotations = convert_sigma_to_scale_rotation(
            gaussian_params["sigma"]
        )
        
        # Apply scale factor to adjust Gaussian sizes (for debugging blurriness)
        scales = scales * self.scale_factor
        
        # Debug: Print scale statistics (only for first batch, every 40 steps would be too verbose here)
        # You can enable this by checking step number in the calling code

        # Create screenspace points tensor for gradient computation
        # Following AD-FFgsStudio convention: use zeros_like with requires_grad
        # The rasterizer will compute screen-space positions internally
        screenspace_points = torch.zeros_like(
            gaussian_params["xyz"],
            dtype=gaussian_params["xyz"].dtype,
            device=device,
            requires_grad=True
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Prepare for batch processing
        rendered_images = []

        for b in range(B):
            # Create rasterization settings for this batch element
            # Note: diff-gaussian-rasterization expects Tensors on GPU, not numpy arrays
            # Also, it expects Transposed matrices typically? 
            # If using Identity, it doesn't matter. If using real cameras, be careful.
            # AD-FFgsStudio usage passes CUDA tensors directly.
            # date 2026.01.30 zijianzhang notes 
            raster_settings = GaussianRasterizationSettings(
                image_height=self.image_size,
                image_width=self.image_size,
                tanfovx=camera_params["tanfovx"].item() if isinstance(camera_params["tanfovx"], torch.Tensor) else camera_params["tanfovx"],
                tanfovy=camera_params["tanfovy"].item() if isinstance(camera_params["tanfovy"], torch.Tensor) else camera_params["tanfovy"],
                bg=torch.zeros(3, device=device),
                scale_modifier=1.0,
                # IMPORTANT: diff-gaussian-rasterization expects Transposed matrices (Column-Major)
                viewmatrix=camera_params["viewmatrix"][b].transpose(0, 1), 
                projmatrix=camera_params["projmatrix"][b].transpose(0, 1),
                sh_degree=self.sh_degree,
                campos=camera_params["campos"][b], # Keep as Tensor
                prefiltered=False,
                debug=False
            )

            rasterizer = _GaussianRasterizer(raster_settings)

            # Reshape SH from [N, 48] to [N, 16, 3] to match diff-gaussian-rasterization expectation
            # Degree 3 -> (3+1)^2 = 16 coeffs. 3 RGB channels.
            shs_val = gaussian_params["sh"][b]
            # Calculate num_coeffs dynamically based on sh_degree
            num_coeffs = (self.sh_degree + 1) ** 2
            if shs_val.shape[-1] == num_coeffs * 3:
                shs_val = shs_val.view(-1, num_coeffs, 3)
            
            # Render this batch element
            rendered_color, radii = rasterizer(
                means3D=gaussian_params["xyz"][b],
                means2D=screenspace_points[b],
                opacities=gaussian_params["opacity"][b],
                shs=shs_val,
                scales=scales[b],
                rotations=rotations[b]
            )

            rendered_images.append(rendered_color)

        # Stack batch
        rendered_images = torch.stack(rendered_images, dim=0)  # [B, 3, H, W]
        # occ 
        return rendered_images


def compute_rendering_loss(
    gaussian_params: Dict[str, torch.Tensor],
    target_image: torch.Tensor,
    camera_params: Dict[str, torch.Tensor],
    renderer: GaussianRenderer,
    M_attn: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute rendering loss with optional attention masking.

    Args:
        gaussian_params: 3D Gaussian parameters
        target_image: [B, 3, H, W] - Ground truth image
        camera_params: Camera parameters
        renderer: GaussianRenderer instance
        M_attn: [B, H, W] - Optional attention mask

    Returns:
        loss: Scalar rendering loss
    """
    # Render from the given viewpoint
    rendered_image = renderer(gaussian_params, camera_params)

    # Compute pixel-wise loss
    pixel_loss = (rendered_image - target_image) ** 2  # [B, 3, H, W]

    if M_attn is not None:
        # Interaction-aware rendering loss
        # L_render = M_attn · ||Î - I||²
        weighted_loss = M_attn.unsqueeze(1) * pixel_loss  # [B, 3, H, W]
        loss = weighted_loss.mean()
    else:
        # Standard rendering loss
        loss = pixel_loss.mean()

    return loss


def compute_multi_view_rendering_loss(
    gaussian_params: Dict[str, torch.Tensor],
    observations: Dict[str, torch.Tensor],
    camera_params_dict: Dict[str, Dict[str, torch.Tensor]],
    renderer: GaussianRenderer,
    M_attn_dict: Optional[Dict[str, torch.Tensor]] = None,
    view_names: list = ["agent", "wrist"]
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute multi-view rendering loss.
    
    Args:
        gaussian_params: 3D Gaussian parameters
        observations: Dictionary with target images for each view
        camera_params_dict: Dictionary of camera parameters for each view
        renderer: GaussianRenderer instance
        M_attn_dict: Optional dictionary of attention masks for each view
        view_names: List of view names to render

    Returns:
        total_loss: Total rendering loss across all views
        loss_dict: Dictionary of per-view losses
    """
    total_loss = 0.0
    loss_dict = {}

    for view_name in view_names:
        # Get target image for this view
        target_image = observations[f"{view_name}_image"]

        # Get camera parameters for this view
        camera_params = camera_params_dict[view_name]

        # Get attention mask for this view (if available)
        M_attn = M_attn_dict.get(view_name) if M_attn_dict is not None else None

        # Compute rendering loss for this view
        view_loss = compute_rendering_loss(
            gaussian_params,
            target_image,
            camera_params,
            renderer,
            M_attn
        )

        loss_dict[f"loss_render_{view_name}"] = view_loss
        total_loss += view_loss

    # Average across views
    total_loss = total_loss / len(view_names)

    return total_loss, loss_dict

def visualize_rendering_comparison(step, gaussian_params, target_obs, cam_params_dict, renderer, view_names, save_dir=None, time_suffix=""):
    """
    Helper to visualize Rendered vs GT images.
    Args:
        step: Current training step (int)
        gaussian_params: Dict of gaussian parameters (batched)
        target_obs: GT target observation dict
        cam_params_dict: Camera parameters dict
        renderer: Instance of GaussianRenderer
        view_names: List of camera names to visualize
        save_dir: Optional directory to save visualizations. Defaults to "./visualizations/rendering"
        time_suffix: Optional suffix to identify time step (e.g., "_t", "_t1_pred", "_t1_gt")
    """
    import matplotlib
    # Use non-interactive backend to avoid X11 authorization issues in headless environments
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    
    if save_dir is None:
        save_dir = "./visualizations/rendering"
    os.makedirs(save_dir, exist_ok=True)
    
    # Take first item in batch
    idx = 0
    
    fig, axes = plt.subplots(len(view_names), 3, figsize=(15, 5 * len(view_names)))
    if len(view_names) == 1:
        axes = axes[None, :] # Ensure 2D array
        
    with torch.no_grad():
        for i, view_name in enumerate(view_names):
            # 1. Render
            # Re-construct raster settings for single item
            cam_params = {k: v[idx:idx+1] if isinstance(v, torch.Tensor) else v for k, v in cam_params_dict[view_name].items()}
            
            # Single item slice for gaussians
            params_single = {
               "xyz": gaussian_params["xyz"][idx:idx+1],
               "sigma": gaussian_params["sigma"][idx:idx+1],
               "sh": gaussian_params["sh"][idx:idx+1],
               "opacity": gaussian_params["opacity"][idx:idx+1]
            }
            
            rendered_img = renderer(params_single, cam_params) # [1, 3, H, W]
            rendered_img = rendered_img.float() # Ensure float for plotting
            
            rendered_np = rendered_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() # [H, W, 3]
            print(f"[Viz] Rendered {view_name} range: min={rendered_np.min():.4f}, max={rendered_np.max():.4f}, mean={rendered_np.mean():.4f}")
            
            # Robust visualization for Rendered
            denom = rendered_np.max() - rendered_np.min()
            if denom > 1e-6:
                rendered_viz = (rendered_np - rendered_np.min()) / denom
            else:
                rendered_viz = rendered_np # All same value
            rendered_viz = np.clip(rendered_viz, 0, 1)

            # 2. GT Image
            gt_img = target_obs[f"{view_name}_image"][idx] # [3, H, W]
            gt_img = gt_img.float()
            
            gt_np = gt_img.permute(1, 2, 0).detach().cpu().numpy()
            print(f"[Viz] GT {view_name} raw range: min={gt_np.min():.4f}, max={gt_np.max():.4f}, mean={gt_np.mean():.4f}")
            
            # Robust visualization for GT
            gt_min, gt_max = gt_np.min(), gt_np.max()
            if gt_max - gt_min > 1e-6:
                 gt_viz = (gt_np - gt_min) / (gt_max - gt_min)
            else:
                 gt_viz = gt_np
            gt_viz = np.clip(gt_viz, 0, 1)
            
            # 3. Difference (on visual properties)
            diff_np = np.abs(rendered_viz - gt_viz)
            
            # Plot
            axes[i, 0].imshow(gt_viz)
            axes[i, 0].set_title(f"{view_name} GT (Norm)")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(rendered_viz)
            axes[i, 1].set_title(f"{view_name} Rendered (Norm)")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(diff_np)
            axes[i, 2].set_title(f"{view_name} Diff (Norm)")
            axes[i, 2].axis('off')
            
    plt.tight_layout()
    # Include time suffix in filename to identify t vs t+1
    if time_suffix:
        save_path = os.path.join(save_dir, f"render_viz_step_{step:06d}{time_suffix}.png")
    else:
        save_path = os.path.join(save_dir, f"render_viz_step_{step:06d}.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved Rendering Visualization to {save_path}")
