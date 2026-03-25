# zijian
# date 2026.01.26
# Description: Gaussian Renderer Module for World Model Supervision
# Purpose: Integrate AD-FFgsStudio's GaussianRasterizer for rendering loss computation
# 3D高斯参数 [B,N,*]
#     ↓
# 数值清理 (NaN/Inf处理, Clamp)
#     ↓
# 相机变换 (viewmatrix, projmatrix)
#     ↓
# 光栅化器 (AD-FFgsStudio)
#     ├─ 投影到2D
#     ├─ Alpha混合
#     └─ 球谐着色
#     ↓
# 渲染图像 [B,3,H,W]
#     ↓
# 损失计算
#     ├─ SSIM + L1 (光度)
#     ├─ 边缘感知平滑 (深度)
#     └─ 正则化 (尺度+不透明度)
#     ↓
# 反向传播 → 更新世界模型

import sys
from pathlib import Path
import numpy as np
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


def validate_camera_params(camera_params: Dict[str, torch.Tensor], batch_idx: int = 0, step: int = None):
    """
    Validate camera parameters for correctness.
    
    Args:
        camera_params: Dictionary containing camera parameters
        batch_idx: Batch index to validate
        step: Step number for conditional logging
    
    Returns:
        is_valid: bool indicating if camera params are valid
        issues: List of issue strings
    """
    issues = []
    is_valid = True
    
    # Check required keys
    required_keys = ["viewmatrix", "projmatrix", "intrinsics", "tanfovx", "tanfovy", "campos"]
    for key in required_keys:
        if key not in camera_params:
            issues.append(f"Missing required key: {key}")
            is_valid = False
    
    if not is_valid:
        return False, issues
    
    # Extract batch element
    viewmatrix = camera_params["viewmatrix"][batch_idx] if camera_params["viewmatrix"].ndim == 3 else camera_params["viewmatrix"]
    projmatrix = camera_params["projmatrix"][batch_idx] if camera_params["projmatrix"].ndim == 3 else camera_params["projmatrix"]
    intrinsics = camera_params["intrinsics"][batch_idx] if camera_params["intrinsics"].ndim == 3 else camera_params["intrinsics"]
    campos = camera_params["campos"][batch_idx] if camera_params["campos"].ndim == 2 else camera_params["campos"]
    
    # 1. Check viewmatrix shape and properties
    if viewmatrix.shape != (4, 4):
        issues.append(f"viewmatrix shape is {viewmatrix.shape}, expected (4, 4)")
        is_valid = False
    
    # Check last row should be [0, 0, 0, 1] (homogeneous coordinate)
    last_row = viewmatrix[3, :]
    expected_last_row = torch.tensor([0.0, 0.0, 0.0, 1.0], device=viewmatrix.device, dtype=viewmatrix.dtype)
    if not torch.allclose(last_row, expected_last_row, atol=1e-4):
        issues.append(f"viewmatrix last row is {last_row.cpu().tolist()}, expected [0, 0, 0, 1]")
        is_valid = False
    
    # Check rotation part (upper-left 3x3) should be orthogonal
    R = viewmatrix[:3, :3]
    RRT = torch.matmul(R, R.transpose(-1, -2))
    identity = torch.eye(3, device=R.device, dtype=R.dtype)
    if not torch.allclose(RRT, identity, atol=1e-3):
        issues.append(f"viewmatrix rotation part is not orthogonal (R @ R^T != I)")
        is_valid = False
    
    # Check determinant of rotation part should be ~1
    det_R = torch.det(R)
    if abs(det_R.item() - 1.0) > 0.1:
        issues.append(f"viewmatrix rotation determinant is {det_R.item():.4f}, expected ~1.0")
        is_valid = False
    
    # 2. Check projmatrix shape
    if projmatrix.shape != (4, 4):
        issues.append(f"projmatrix shape is {projmatrix.shape}, expected (4, 4)")
        is_valid = False
    
    # Check projmatrix last row should be [0, 0, -1, 0] or similar (perspective projection)
    proj_last_row = projmatrix[3, :]
    if torch.allclose(proj_last_row[:3], torch.zeros(3, device=projmatrix.device), atol=1e-3):
        issues.append(f"projmatrix last row is {proj_last_row.cpu().tolist()}, may be invalid")
    
    # 3. Check intrinsics
    if intrinsics.shape != (3, 3):
        issues.append(f"intrinsics shape is {intrinsics.shape}, expected (3, 3)")
        is_valid = False
    
    fx = intrinsics[0, 0].item()
    fy = intrinsics[1, 1].item()
    cx = intrinsics[0, 2].item()
    cy = intrinsics[1, 2].item()
    
    if fx <= 0 or fy <= 0:
        issues.append(f"Invalid focal lengths: fx={fx:.2f}, fy={fy:.2f}")
        is_valid = False
    
    # 4. Test projection with a known 3D point
    # Use camera position as test point (should project to center if campos is correct)
    test_point_world = campos[:3] if campos.shape[0] >= 3 else campos  # [3]
    test_point_homo = torch.cat([test_point_world, torch.ones(1, device=test_point_world.device, dtype=test_point_world.dtype)])
    
    # Transform to camera space
    test_point_cam = torch.matmul(test_point_homo, viewmatrix.transpose(-1, -2))  # [4]
    
    # Project to 2D
    if test_point_cam[2] > 1e-6:  # Only if in front of camera
        x_cam = test_point_cam[0] / test_point_cam[2]
        y_cam = test_point_cam[1] / test_point_cam[2]
        u_test = fx * x_cam + cx
        v_test = fy * y_cam + cy
        
        # Check if projection is reasonable (within image bounds or close)
        image_size = 224  # Default
        if u_test < -image_size or u_test > 2 * image_size or v_test < -image_size or v_test > 2 * image_size:
            issues.append(f"Test point projection is out of bounds: u={u_test:.1f}, v={v_test:.1f}")
    else:
        issues.append(f"Test point (campos) is behind camera: z_cam={test_point_cam[2].item():.4f}")
    
    # 5. Check tanfov values
    tanfovx = camera_params["tanfovx"].item() if isinstance(camera_params["tanfovx"], torch.Tensor) else camera_params["tanfovx"]
    tanfovy = camera_params["tanfovy"].item() if isinstance(camera_params["tanfovy"], torch.Tensor) else camera_params["tanfovy"]
    
    if tanfovx <= 0 or tanfovy <= 0:
        issues.append(f"Invalid tanfov: tanfovx={tanfovx:.4f}, tanfovy={tanfovy:.4f}")
        is_valid = False
    
    # 6. Verify projmatrix = P @ V relationship
    # Reconstruct projection matrix from intrinsics and compare
    # This is a simplified check - full projection matrix depends on znear/zfar
    
    # Print summary if requested
    if step is not None and step % 400 == 0:
        print(f"[Camera Validation] Batch {batch_idx}:")
        print(f"  viewmatrix: shape={viewmatrix.shape}, det(R)={det_R.item():.4f}")
        print(f"  projmatrix: shape={projmatrix.shape}")
        print(f"  intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        print(f"  campos: {campos.cpu().tolist()}")
        print(f"  tanfov: x={tanfovx:.4f}, y={tanfovy:.4f}")
        if issues:
            print(f"  Issues: {len(issues)}")
            for issue in issues[:5]:  # Print first 5 issues
                print(f"    - {issue}")
        else:
            print(f"  ✓ All checks passed")
    
    return is_valid, issues


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
        camera_params: Dict[str, torch.Tensor],
        step: Optional[int] = None
    ) -> torch.Tensor:
        """
        Render 3D Gaussians from a specific camera viewpoint.
        """
        B, N, _ = gaussian_params["xyz"].shape
        device = gaussian_params["xyz"].device

        # Validate and sanitize Gaussian parameters before rendering
        # Check for NaN/Inf in all parameters
        for key, value in gaussian_params.items():
            if torch.isnan(value).any() or torch.isinf(value).any():
                import warnings
                warnings.warn(f"NaN/Inf detected in {key}. Replacing with safe defaults.")
                if key == "xyz":
                    gaussian_params[key] = torch.where(
                        torch.isnan(value) | torch.isinf(value),
                        torch.zeros_like(value),
                        value
                    )
                elif key == "scales":
                    # Replace with small positive values
                    gaussian_params[key] = torch.where(
                        torch.isnan(value) | torch.isinf(value),
                        torch.ones_like(value) * 1e-4,
                        value
                    )
                elif key == "opacity":
                    gaussian_params[key] = torch.clamp(
                        torch.where(
                            torch.isnan(value) | torch.isinf(value),
                            torch.zeros_like(value),
                            value
                        ),
                        min=0.0, max=1.0
                    )
                else:
                    gaussian_params[key] = torch.where(
                        torch.isnan(value) | torch.isinf(value),
                        torch.zeros_like(value),
                        value
                    )

        # Clamp xyz to reasonable range
        # FIX: Handle NaNs in xyz explicitly before clamping
        if torch.isnan(gaussian_params["xyz"]).any() or torch.isinf(gaussian_params["xyz"]).any():
             gaussian_params["xyz"] = torch.nan_to_num(gaussian_params["xyz"], nan=0.0, posinf=100.0, neginf=-100.0)
        gaussian_params["xyz"] = torch.clamp(gaussian_params["xyz"], min=-100.0, max=100.0)
        
        # Clamp opacity to [0, 1]
        # Debug: Print opacity stats (only every 40 steps)
        if step is not None and step % 400 == 0:
            opacity_before = gaussian_params["opacity"]
            print(f"[GaussianRenderer] Opacity before clamp: min={opacity_before.min():.6f}, max={opacity_before.max():.6f}, mean={opacity_before.mean():.6f}")
        gaussian_params["opacity"] = torch.clamp(gaussian_params["opacity"], min=0.0, max=1.0)
        if step is not None and step % 400 == 0:
            opacity_after = gaussian_params["opacity"]
            print(f"[GaussianRenderer] Opacity after clamp: min={opacity_after.min():.6f}, max={opacity_after.max():.6f}, mean={opacity_after.mean():.6f}")

        # Get scales and rotations directly (no sigma fallback)
        scales = gaussian_params["scales"]
        rotations = gaussian_params["rotations"]

        # Clamp scales
        if step is not None and step % 400 == 0:
            print(f"[GaussianRenderer] Scales before clamp: min={scales.min():.6f}, max={scales.max():.6f}, mean={scales.mean():.6f}")
        scales = torch.clamp(scales, min=1e-6, max=10.0)
        if step is not None and step % 400 == 0:
            print(f"[GaussianRenderer] Scales after clamp: min={scales.min():.6f}, max={scales.max():.6f}, mean={scales.mean():.6f}")

        # Apply scale factor
        scales = scales * self.scale_factor
        
        # Normalize quaternions
        quat_norm = rotations.norm(dim=-1, keepdim=True) + 1e-8
        rotations = rotations / quat_norm
        
        # Debug: Print scale statistics (only for first batch, every 40 steps would be too verbose here)
        # You can enable this by checking step number in the calling code

        # Create screenspace points tensor for gradient computation
        # Following AD-FFgsStudio convention: use zeros_like with requires_grad
        # The rasterizer will compute screen-space positions internally
        # FIX: Ensure screenspace_points requires_grad only if we are in training mode or grad is enabled
        requires_grad = gaussian_params["xyz"].requires_grad
        
        # DEBUG: Check xyz gradients
        # try:
        #    print(f"[DEBUG] GaussianRenderer: xyz requires_grad={requires_grad}, grad_fn={gaussian_params['xyz'].grad_fn}")
        # except: pass
        
        screenspace_points = torch.zeros_like(
            gaussian_params["xyz"],
            dtype=gaussian_params["xyz"].dtype,
            device=device,
            requires_grad=requires_grad
        )
        if requires_grad:
            try:
                screenspace_points.retain_grad()
            except:
                pass

        # Prepare for batch processing
        rendered_images = []

        # Validate camera parameters (only for first batch and every 40 steps)
        if step is not None and step % 400 == 0:
            is_valid, issues = validate_camera_params(camera_params, batch_idx=0, step=step)
            if not is_valid and issues:
                import warnings
                warnings.warn(f"Camera parameter validation failed: {issues[0]}")
        
        for b in range(B):
            # Debug: Check if Gaussians are in valid range before rendering
            xyz_b = gaussian_params["xyz"][b]  # [N, 3]
            # Transform to camera space to check visibility
            # viewmatrix transforms world -> camera: xyz_cam = xyz_world @ viewmatrix.T
            viewmatrix_b = camera_params["viewmatrix"][b]  # [4, 4]
            xyz_homo = torch.cat([xyz_b, torch.ones(xyz_b.shape[0], 1, device=device, dtype=xyz_b.dtype)], dim=-1)  # [N, 4]
            xyz_cam = torch.matmul(xyz_homo, viewmatrix_b.transpose(-1, -2))  # [N, 4]
            z_cam = xyz_cam[:, 2]  # [N] - Z in camera space
            
            # Check if any Gaussians are in front of camera (z > 0)
            valid_mask = z_cam > 0.01  # znear threshold
            num_valid = valid_mask.sum().item()
            
            # If no valid Gaussians, render will be black - this is expected for some batches
            # But we should still render to get gradients (rasterizer handles this)
            
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

            # Reshape SH to match diff-gaussian-rasterization expectation
            # For sh_degree=1: (1+1)^2 = 4 coeffs per color, 3 RGB channels = 12 total
            # Our decoder outputs 9 coeffs (3 DC + 6 for 1st order), need to pad to 12
            shs_val = gaussian_params["sh"][b]
            # Calculate num_coeffs dynamically based on sh_degree
            num_coeffs = (self.sh_degree + 1) ** 2
            expected_sh_dim = num_coeffs * 3  # For sh_degree=1: 4 * 3 = 12

            # Debug: Print SH stats (only every 40 steps)
            if step is not None and step % 400 == 0 and b == 0:
                print(f"[GaussianRenderer] SH before reshape: shape={shs_val.shape}, min={shs_val.min():.6f}, max={shs_val.max():.6f}, mean={shs_val.mean():.6f}")
                print(f"[GaussianRenderer] sh_degree={self.sh_degree}, num_coeffs={num_coeffs}, expected_sh_dim={expected_sh_dim}")

            # Handle different SH dimensions
            if shs_val.shape[-1] == expected_sh_dim:
                # Exact match: reshape to [N, num_coeffs, 3]
                shs_val = shs_val.view(-1, num_coeffs, 3)
            elif shs_val.shape[-1] > expected_sh_dim:
                # More coefficients than needed: truncate to first expected_sh_dim
                shs_val = shs_val[..., :expected_sh_dim].view(-1, num_coeffs, 3)
            else:
                # Fewer coefficients: pad with zeros
                # For sh_degree=1: we have 9 coeffs, need 12 (pad 3 zeros)
                sh_padded = torch.zeros(shs_val.shape[0], expected_sh_dim, device=shs_val.device, dtype=shs_val.dtype)
                sh_padded[:, :shs_val.shape[-1]] = shs_val
                shs_val = sh_padded.view(-1, num_coeffs, 3)

            # Debug: Print SH stats after reshape (only every 40 steps)
            if step is not None and step % 400 == 0 and b == 0:
                print(f"[GaussianRenderer] SH after reshape: shape={shs_val.shape}, min={shs_val.min():.6f}, max={shs_val.max():.6f}, mean={shs_val.mean():.6f}")
                # Check SH C0 (DC term) which determines base color
                sh_c0 = shs_val[:, 0, :]  # [N, 3] - DC term for RGB
                print(f"[GaussianRenderer] SH C0 (DC term): min={sh_c0.min():.6f}, max={sh_c0.max():.6f}, mean={sh_c0.mean():.6f}")
            
            # Final validation before rendering
            # Filter out invalid Gaussians (NaN/Inf, out of range, etc.)
            xyz_b_valid = gaussian_params["xyz"][b]
            opacity_b_valid = gaussian_params["opacity"][b]
            scales_b_valid = scales[b]
            rotations_b_valid = rotations[b]
            shs_val_valid = shs_val
            
            # Check for any remaining invalid values
            valid_mask = (
                torch.isfinite(xyz_b_valid).all(dim=-1) &
                torch.isfinite(opacity_b_valid.squeeze(-1)) &
                torch.isfinite(scales_b_valid).all(dim=-1) &
                torch.isfinite(rotations_b_valid).all(dim=-1) &
                (opacity_b_valid.squeeze(-1) > 0.0) &  # Non-zero opacity
                (scales_b_valid.min(dim=-1)[0] > 0.0)  # Positive scales
            )
            
            if not valid_mask.all():
                # Filter to only valid Gaussians
                num_valid = valid_mask.sum().item()
                if num_valid == 0:
                    # No valid Gaussians, return black image
                    rendered_color = torch.zeros(3, self.image_size, self.image_size, device=device)
                    radii = torch.zeros(num_valid, device=device)
                else:
                    xyz_b_valid = xyz_b_valid[valid_mask]
                    opacity_b_valid = opacity_b_valid[valid_mask]
                    scales_b_valid = scales_b_valid[valid_mask]
                    rotations_b_valid = rotations_b_valid[valid_mask]
                    shs_val_valid = shs_val_valid[valid_mask]
                    screenspace_points_b_valid = screenspace_points[b][valid_mask]
                    
                    # Render with filtered Gaussians
                    rendered_color, radii = rasterizer(
                        means3D=xyz_b_valid,
                        means2D=screenspace_points_b_valid,
                        opacities=opacity_b_valid,
                        shs=shs_val_valid,
                        scales=scales_b_valid,
                        rotations=rotations_b_valid
                    )
            else:
                # All Gaussians are valid, render normally
                rendered_color, radii = rasterizer(
                    means3D=gaussian_params["xyz"][b],
                    means2D=screenspace_points[b],
                    opacities=gaussian_params["opacity"][b],
                    shs=shs_val,
                    scales=scales[b],
                    rotations=rotations[b]
                )
            
            # Debug: Log rendering statistics (only for first batch, every 40 steps)
            if b == 0 and step is not None and step % 400 == 0:
                z_min, z_max = z_cam.min().item(), z_cam.max().item()
                rendered_max = rendered_color.max().item()
                rendered_mean = rendered_color.mean().item()
                
                # Manually compute 2D projection coordinates for debugging
                # (screenspace_points is always 0 because rasterizer computes internally)
                xyz_batch = gaussian_params["xyz"][b]  # [N, 3] in world coordinates
                if xyz_batch.shape[0] > 0:
                    # Transform to camera space using viewmatrix
                    xyz_homo = torch.cat([
                        xyz_batch,
                        torch.ones(xyz_batch.shape[0], 1, device=xyz_batch.device, dtype=xyz_batch.dtype)
                    ], dim=-1)  # [N, 4]
                    
                    viewmatrix_b = camera_params["viewmatrix"][b]  # [4, 4]
                    xyz_cam = torch.matmul(xyz_homo, viewmatrix_b.transpose(-1, -2))  # [N, 4]
                    
                    # Project to 2D using intrinsics
                    intrinsics_b = camera_params["intrinsics"][b] if camera_params["intrinsics"].ndim == 3 else camera_params["intrinsics"]
                    fx = intrinsics_b[0, 0]
                    fy = intrinsics_b[1, 1]
                    cx = intrinsics_b[0, 2]
                    cy = intrinsics_b[1, 2]
                    
                    x_cam = xyz_cam[:, 0]
                    y_cam = xyz_cam[:, 1]
                    z_cam_proj = xyz_cam[:, 2].clamp(min=1e-6)
                    
                    u_coords = fx * (x_cam / z_cam_proj) + cx
                    v_coords = fy * (y_cam / z_cam_proj) + cy
                    
                    u_min, u_max = u_coords.min().item(), u_coords.max().item()
                    v_min, v_max = v_coords.min().item(), v_coords.max().item()
                    u_mean, v_mean = u_coords.mean().item(), v_coords.mean().item()
                    
                    # Count points in image bounds
                    image_size = self.image_size
                    in_bounds = ((u_coords >= 0) & (u_coords < image_size) & 
                                (v_coords >= 0) & (v_coords < image_size)).sum().item()
                    
                    print(f"[GaussianRenderer] 2D Projection (manual): "
                          f"u_range=[{u_min:.1f}, {u_max:.1f}], v_range=[{v_min:.1f}, {v_max:.1f}], "
                          f"u_mean={u_mean:.1f}, v_mean={v_mean:.1f}, "
                          f"in_bounds={in_bounds}/{xyz_batch.shape[0]}")
                    print(f"[GaussianRenderer] Camera params: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
                    print(f"[GaussianRenderer] Z_cam range: [{z_cam_proj.min():.3f}, {z_cam_proj.max():.3f}], mean={z_cam_proj.mean():.3f}")
                
                # Check if rendering is mostly black or has unusual distribution
                if rendered_max < 1e-6:
                    print(f"[GaussianRenderer] Warning: Black rendering detected. "
                          f"Valid Gaussians: {num_valid}/{xyz_b.shape[0]}, "
                          f"Z_cam range: [{z_min:.3f}, {z_max:.3f}], "
                          f"Z_world range: [{xyz_b[:, 2].min().item():.3f}, {xyz_b[:, 2].max().item():.3f}]")
                elif rendered_max < 0.5:
                    # Rendering is dim but not completely black
                    print(f"[GaussianRenderer] Info: Dim rendering detected. "
                          f"Max: {rendered_max:.4f}, Mean: {rendered_mean:.4f}, "
                          f"Valid Gaussians: {num_valid}/{xyz_b.shape[0]}, "
                          f"Scale range: [{scales[b].min().item():.6f}, {scales[b].max().item():.6f}]")

            rendered_images.append(rendered_color)

        # Stack batch
        rendered_images = torch.stack(rendered_images, dim=0)  # [B, 3, H, W]
        # occ 
        return rendered_images


def compute_ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """SSIM loss with 3x3 kernel and reflection padding. Returns per-pixel loss map."""
    ref_pad = torch.nn.ReflectionPad2d(1)
    pred = ref_pad(pred)
    target = ref_pad(target)

    mu_pred = F.avg_pool2d(pred, kernel_size=3, stride=1)
    mu_target = F.avg_pool2d(target, kernel_size=3, stride=1)

    musq_pred = mu_pred.pow(2)
    musq_target = mu_target.pow(2)
    mu_pred_target = mu_pred * mu_target

    sigma_pred = F.avg_pool2d(pred.pow(2), kernel_size=3, stride=1) - musq_pred
    sigma_target = F.avg_pool2d(target.pow(2), kernel_size=3, stride=1) - musq_target
    sigma_pred_target = F.avg_pool2d(pred * target, kernel_size=3, stride=1) - mu_pred_target

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) \
               / ((musq_pred + musq_target + C1) * (sigma_pred + sigma_target + C2) + 1e-8)
    return torch.clamp((1 - ssim_map) / 2, 0, 1)


def compute_photometric_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lpips_fn: Optional[callable] = None,
    lpips_weight: float = 0.1
) -> torch.Tensor:
    """
    Combined photometric loss: 0.85 * SSIM + 0.15 * L1 + optional LPIPS.

    Args:
        pred: [B, 3, H, W] predicted image
        target: [B, 3, H, W] target image
        lpips_fn: Optional LPIPS loss function
        lpips_weight: Weight for LPIPS loss (default 0.1)

    Returns:
        Scalar loss
    """
    l1_loss = (target - pred).abs().mean(1, True)
    ssim_loss = compute_ssim_loss(pred, target).mean(1, True)
    base_loss = (0.85 * ssim_loss + 0.15 * l1_loss).mean()

    # Add LPIPS perceptual loss if available
    if lpips_fn is not None:
        # Clamp inputs to [0, 1] range to ensure valid input domain
        pred_clamped = torch.clamp(pred, 0.0, 1.0)
        target_clamped = torch.clamp(target, 0.0, 1.0)
        # LPIPS expects input in [-1, 1] range
        pred_norm = pred_clamped * 2.0 - 1.0
        target_norm = target_clamped * 2.0 - 1.0
        lpips_loss = lpips_fn(pred_norm, target_norm).mean()
        return base_loss + lpips_weight * lpips_loss

    return base_loss


def compute_edge_smooth_loss(rgb: torch.Tensor, disp_map: torch.Tensor) -> torch.Tensor:
    """Edge-aware depth smoothness loss. rgb: [B,3,H,W], disp_map: [B,1,H,W]."""
    grad_rgb_x = (rgb[:, :, :, :-1] - rgb[:, :, :, 1:]).abs().mean(1, True)
    grad_rgb_y = (rgb[:, :, :-1, :] - rgb[:, :, 1:, :]).abs().mean(1, True)

    grad_disp_x = (disp_map[:, :, :, :-1] - disp_map[:, :, :, 1:]).abs()
    grad_disp_y = (disp_map[:, :, :-1, :] - disp_map[:, :, 1:, :]).abs()

    grad_disp_x *= (-1.0 * grad_rgb_x).exp()
    grad_disp_y *= (-1.0 * grad_rgb_y).exp()
    return grad_disp_x.mean() + grad_disp_y.mean()


def compute_gaussian_regularization(
    gaussian_params: Dict[str, torch.Tensor],
    lambda_scale: float = 0.01,
    lambda_opacity: float = 0.01,
) -> torch.Tensor:
    """Scale + opacity regularization (encourages small Gaussians and sparsity)."""
    device = gaussian_params["xyz"].device
    reg = torch.tensor(0.0, device=device)
    if "scales" in gaussian_params:
        reg = reg + lambda_scale * gaussian_params["scales"].norm(dim=-1).mean()
    if "opacity" in gaussian_params:
        reg = reg + lambda_opacity * gaussian_params["opacity"].abs().mean()
    return reg


def compute_rendering_loss(
    gaussian_params: Dict[str, torch.Tensor],
    target_image: torch.Tensor,
    camera_params: Dict[str, torch.Tensor],
    renderer: GaussianRenderer,
    M_attn: Optional[torch.Tensor] = None,
    step: Optional[int] = None,
    lpips_fn: Optional[callable] = None,
    lpips_weight: float = 0.1
) -> torch.Tensor:
    """
    Compute rendering loss (L1 + SSIM photometric + optional LPIPS) with optional attention masking.

    Args:
        gaussian_params: 3D Gaussian parameters
        target_image: [B, 3, H, W] - Ground truth image
        camera_params: Camera parameters
        renderer: GaussianRenderer instance
        M_attn: [B, H, W] - Optional attention mask
        step: Current training step
        lpips_fn: Optional LPIPS loss function
        lpips_weight: Weight for LPIPS loss

    Returns:
        loss: Scalar rendering loss
    """
    # Render from the given viewpoint
    rendered_image = renderer(gaussian_params, camera_params, step=step)

    if M_attn is not None:
        # Masked photometric loss (base: SSIM + L1)
        l1_loss = (rendered_image - target_image).abs()
        ssim_loss = compute_ssim_loss(rendered_image, target_image)
        pixel_loss = 0.85 * ssim_loss + 0.15 * l1_loss
        weighted_loss = M_attn.unsqueeze(1) * pixel_loss
        base_loss = weighted_loss.mean()

        # Add LPIPS perceptual loss if available (applied to full images, not masked)
        if lpips_fn is not None:
            # Clamp inputs to [0, 1] range
            pred_clamped = torch.clamp(rendered_image, 0.0, 1.0)
            target_clamped = torch.clamp(target_image, 0.0, 1.0)
            # LPIPS expects input in [-1, 1] range
            pred_norm = pred_clamped * 2.0 - 1.0
            target_norm = target_clamped * 2.0 - 1.0
            lpips_loss = lpips_fn(pred_norm, target_norm).mean()
            loss = base_loss + lpips_weight * lpips_loss
        else:
            loss = base_loss
    else:
        loss = compute_photometric_loss(rendered_image, target_image, lpips_fn, lpips_weight)

    return loss


def compute_multi_view_rendering_loss(
    gaussian_params: Dict[str, torch.Tensor],
    observations: Dict[str, torch.Tensor],
    camera_params_dict: Dict[str, Dict[str, torch.Tensor]],
    renderer: GaussianRenderer,
    M_attn_dict: Optional[Dict[str, torch.Tensor]] = None,
    view_names: list = ["agent", "wrist"],
    step: Optional[int] = None,
    depth_map: Optional[torch.Tensor] = None,
    lambda_scale: float = 0.001,
    lambda_opacity: float = 0.001,
    lambda_edge_smooth: float = 0.01,
    lpips_fn: Optional[callable] = None,
    lpips_weight: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute multi-view rendering loss with regularization and edge-aware depth smoothness.

    Args:
        gaussian_params: 3D Gaussian parameters
        observations: Dictionary with target images for each view
        camera_params_dict: Dictionary of camera parameters for each view
        renderer: GaussianRenderer instance
        M_attn_dict: Optional dictionary of attention masks for each view
        view_names: List of view names to render
        step: Current training step
        depth_map: [B, 1, H, W] predicted depth map for edge-aware smoothness
        lambda_scale: Weight for scale regularization
        lambda_opacity: Weight for opacity regularization
        lambda_edge_smooth: Weight for edge-aware depth smoothness
        lpips_fn: Optional LPIPS loss function
        lpips_weight: Weight for LPIPS perceptual loss

    Returns:
        total_loss: Total rendering loss across all views
        loss_dict: Dictionary of per-view losses
    """
    device = gaussian_params["xyz"].device
    total_loss = torch.tensor(0.0, device=device, requires_grad=True) if gaussian_params["xyz"].requires_grad \
        else torch.tensor(0.0, device=device)
    loss_dict = {}

    for view_name in view_names:
        try:
            target_image = observations[f"{view_name}_image"]
            camera_params = camera_params_dict[view_name]
            M_attn = M_attn_dict.get(view_name) if M_attn_dict is not None else None

            view_loss = compute_rendering_loss(
                gaussian_params, target_image, camera_params, renderer, M_attn, step=step,
                lpips_fn=lpips_fn, lpips_weight=lpips_weight
            )

            loss_dict[f"loss_render_{view_name}"] = view_loss
            total_loss = total_loss + view_loss
        except Exception as e:
            import warnings
            warnings.warn(f"Rendering failed for view {view_name}: {e}")
            if gaussian_params["xyz"].requires_grad:
                dummy_loss = 0.0 * gaussian_params["xyz"].sum()
            else:
                dummy_loss = torch.tensor(0.0, device=device, requires_grad=False)
            loss_dict[f"loss_render_{view_name}"] = dummy_loss
            total_loss = total_loss + dummy_loss

    # Average photometric loss across views
    total_loss = total_loss / len(view_names)

    # Scale + opacity regularization
    reg_loss = compute_gaussian_regularization(gaussian_params, lambda_scale, lambda_opacity)
    loss_dict["loss_reg"] = reg_loss
    total_loss = total_loss + reg_loss

    # Edge-aware depth smoothness
    if depth_map is not None:
        # Use first view's target image as RGB reference for edge detection
        first_view = view_names[0]
        rgb_ref_key = f"{first_view}_image"
        if rgb_ref_key in observations:
            rgb_ref = observations[rgb_ref_key]
            # Resize depth to match RGB if needed
            if depth_map.shape[2:] != rgb_ref.shape[2:]:
                depth_map = F.interpolate(depth_map, size=rgb_ref.shape[2:], mode="bilinear", align_corners=False)
            # Normalize disparity by mean for stability (AD-FFgsStudio convention)
            disp = 1.0 / (depth_map + 1e-6)
            disp = disp / (disp.mean() + 1e-6)
            edge_loss = lambda_edge_smooth * compute_edge_smooth_loss(rgb_ref, disp)
            loss_dict["loss_edge_smooth"] = edge_loss
            total_loss = total_loss + edge_loss

    if step is not None and step % 400 == 0:
        parts = ", ".join(f"{k}={v.item():.6f}" for k, v in loss_dict.items())
        print(f"[MultiViewLoss] Step {step}: {parts}, total={total_loss.item():.6f}")

    return total_loss, loss_dict

def visualize_rendering_comparison(step, gaussian_params, target_obs, cam_params_dict, renderer, view_names, save_dir=None, time_suffix="", temporal_frames=None, temporal_labels=None, future_label="future"):
    """
    Helper to visualize temporal sequence and rendering comparison.
    Args:
        step: Current training step (int)
        gaussian_params: Dict of gaussian parameters (batched)
        target_obs: GT target observation dict
        cam_params_dict: Camera parameters dict
        renderer: Instance of GaussianRenderer
        view_names: List of camera names to visualize
        save_dir: Optional directory to save visualizations. Defaults to "./visualizations/rendering"
        time_suffix: Optional suffix to identify time step
        temporal_frames: Dict of temporal frames {key: [B, T, C, H, W]}
        temporal_labels: Optional labels for temporal context frames (e.g. ["t-10", "t-5", "t"])
        future_label: Label for the supervised future frame column
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os
    import numpy as np

    if save_dir is None:
        save_dir = "./visualizations/rendering"
    os.makedirs(save_dir, exist_ok=True)

    idx = 0
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    with torch.no_grad():
        for i, view_name in enumerate(view_names):
            if i >= 1:
                break

            temporal_key = None
            if temporal_frames:
                for k in temporal_frames.keys():
                    if view_name == "agent" and ("base" in k or "high" in k or "exterior" in k):
                        temporal_key = k
                        break
                    elif view_name == "wrist" and "wrist" in k:
                        temporal_key = k
                        break
                    elif view_name in k:
                        temporal_key = k
                        break
                if not temporal_key and temporal_frames:
                    temporal_key = next(iter(temporal_frames.keys()))

            if temporal_key and temporal_key in temporal_frames:
                frames = temporal_frames[temporal_key][idx]
                num_temporal_frames = frames.shape[0]
                context_count = min(3, max(1, num_temporal_frames - 1))
                labels = temporal_labels[:context_count] if temporal_labels else None
                context_frames = frames[:context_count]

                if context_count == 1:
                    frame_np = context_frames[0].detach().cpu().numpy()
                    frame_viz = np.clip((frame_np + 1.0) / 2.0, 0, 1)
                    ax = fig.add_subplot(gs[0, 1])
                    ax.imshow(frame_viz)
                    ax.set_title(labels[0] if labels else "t", fontsize=14)
                    ax.axis('off')
                    for col in [0, 2]:
                        ax = fig.add_subplot(gs[0, col])
                        ax.axis('off')
                else:
                    for t_idx in range(min(3, context_count)):
                        frame_np = context_frames[t_idx].detach().cpu().numpy()
                        frame_viz = np.clip((frame_np + 1.0) / 2.0, 0, 1)
                        ax = fig.add_subplot(gs[0, t_idx])
                        ax.imshow(frame_viz)
                        title = labels[t_idx] if labels and t_idx < len(labels) else f"ctx_{t_idx}"
                        ax.set_title(title, fontsize=14)
                        ax.axis('off')

                future_frame_idx = context_count if num_temporal_frames > context_count else num_temporal_frames - 1
                frame_future = frames[future_frame_idx]
                frame_future_np = frame_future.detach().cpu().numpy()
                frame_future_viz = np.clip((frame_future_np + 1.0) / 2.0, 0, 1)
                ax_gt_future = fig.add_subplot(gs[1, 0])
                ax_gt_future.imshow(frame_future_viz)
                ax_gt_future.set_title(future_label, fontsize=14)
                ax_gt_future.axis('off')
            else:
                for row in range(2):
                    for col in range(3 if row == 0 else 1):
                        ax = fig.add_subplot(gs[row, col])
                        ax.text(0.5, 0.5, 'No temporal data', ha='center', va='center', fontsize=12)
                        ax.axis('off')

            cam_params = {k: v[idx:idx+1] if isinstance(v, torch.Tensor) else v for k, v in cam_params_dict[view_name].items()}
            params_single = {
               "xyz": gaussian_params["xyz"][idx:idx+1],
               "sh": gaussian_params["sh"][idx:idx+1],
               "opacity": gaussian_params["opacity"][idx:idx+1],
               "scales": gaussian_params["scales"][idx:idx+1],
               "rotations": gaussian_params["rotations"][idx:idx+1]
            }

            rendered_img = renderer(params_single, cam_params).float()
            rendered_np = rendered_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            rendered_viz = np.clip(rendered_np, 0, 1)

            ax_rendered = fig.add_subplot(gs[1, 1])
            ax_rendered.imshow(rendered_viz)
            ax_rendered.set_title(f"render {future_label}", fontsize=14)
            ax_rendered.axis('off')

            gt_img = target_obs[f"{view_name}_image"][idx].float()
            gt_np = gt_img.permute(1, 2, 0).detach().cpu().numpy()
            gt_viz = np.clip(gt_np, 0, 1)
            diff_np = np.abs(rendered_viz - gt_viz)

            ax_diff = fig.add_subplot(gs[1, 2])
            ax_diff.imshow(diff_np)
            ax_diff.set_title("Diff", fontsize=14)
            ax_diff.axis('off')

    save_path = os.path.join(save_dir, f"render_viz_step_{step:06d}{time_suffix}.png" if time_suffix else f"render_viz_step_{step:06d}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Rendering Visualization to {save_path}")




def _latent_tokens_to_heatmap(tokens) -> np.ndarray:
    """Convert token embeddings [N, D] into a normalized 2D heatmap."""
    if torch.is_tensor(tokens):
        token_tensor = tokens.detach().float().cpu()
    else:
        token_tensor = torch.as_tensor(tokens, dtype=torch.float32)

    if token_tensor.ndim != 2:
        raise ValueError(f"Expected latent tokens with shape [N, D], got {tuple(token_tensor.shape)}")

    num_tokens = token_tensor.shape[0]
    spatial_size = int(round(num_tokens ** 0.5))
    if spatial_size * spatial_size != num_tokens:
        raise ValueError(f"Cannot reshape {num_tokens} latent tokens into a square heatmap")

    token_scores = token_tensor.norm(dim=-1).reshape(spatial_size, spatial_size).numpy()
    finite_mask = np.isfinite(token_scores)
    if not finite_mask.any():
        return np.zeros((spatial_size, spatial_size), dtype=np.float32)

    valid_scores = token_scores[finite_mask]
    lo = float(np.percentile(valid_scores, 5.0))
    hi = float(np.percentile(valid_scores, 95.0))
    if hi <= lo:
        hi = lo + 1e-6

    normalized = np.clip((token_scores - lo) / (hi - lo), 0.0, 1.0)
    normalized[~finite_mask] = 0.0
    return normalized.astype(np.float32)


def _latent_entry_to_heatmap(entry) -> np.ndarray | None:
    if entry is None:
        return None
    if torch.is_tensor(entry):
        if entry.ndim == 3:
            entry = entry[0]
        return _latent_tokens_to_heatmap(entry)

    if isinstance(entry, np.ndarray):
        if entry.ndim == 3:
            entry = entry[0]
        return _latent_tokens_to_heatmap(entry)

    return None


def _blank_latent_like(reference_image: np.ndarray | None, fallback_size: int = 16) -> np.ndarray:
    if reference_image is not None and reference_image.ndim >= 2:
        return np.zeros(reference_image.shape[:2], dtype=np.float32)
    return np.zeros((fallback_size, fallback_size), dtype=np.float32)


def visualize_future_rollout_comparison(
    step,
    target_obs_seq,
    rendered_obs_seq,
    view_names,
    save_dir=None,
    temporal_frames=None,
    time_suffix="_future_rollout",
    horizon_labels=None,
    base_target_obs=None,
    base_rendered_obs=None,
    base_label="t/base",
    motion_weight_seq=None,
    context_labels=None,
    encoded_latent_seq=None,
    future_latent_seq=None,
    gt_future_latent_seq=None,
    pred_gt_latent_diff_seq=None,
    future_delta_seq=None,
):
    """Visualize context + latent diagnostics + multi-horizon future rollout in one figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os

    if save_dir is None:
        save_dir = "./visualizations/rendering"
    os.makedirs(save_dir, exist_ok=True)

    if not target_obs_seq or not rendered_obs_seq:
        return

    idx = 0
    horizon = min(len(target_obs_seq), len(rendered_obs_seq))
    view_name = view_names[0]
    context_frames = []

    temporal_key = None
    if temporal_frames:
        for key in temporal_frames.keys():
            key_lower = key.lower()
            if view_name == "agent" and ("base" in key_lower or "high" in key_lower or "exterior" in key_lower or "agent" in key_lower):
                temporal_key = key
                break
            if view_name == "wrist" and "wrist" in key_lower:
                temporal_key = key
                break
        if temporal_key is None:
            temporal_key = next(iter(temporal_frames.keys()))

    if temporal_key is not None:
        frames = temporal_frames[temporal_key][idx]
        context_count = max(0, frames.shape[0] - horizon)
        for frame_idx in range(context_count):
            frame = frames[frame_idx].detach().cpu().numpy()
            context_frames.append(np.clip((frame + 1.0) / 2.0, 0, 1))

    if horizon_labels is None:
        horizon_labels = [f"t+{idx + 1}" for idx in range(horizon)]

    if context_labels is None:
        if len(context_frames) == 1:
            context_labels = ["t"]
        else:
            context_labels = [
                "t" if label_offset == 0 else f"t-{label_offset}"
                for label_offset in range(len(context_frames) - 1, -1, -1)
            ]
    else:
        context_labels = list(context_labels)

    show_base = base_target_obs is not None or base_rendered_obs is not None
    show_future_vs_base = base_rendered_obs is not None
    show_motion = motion_weight_seq is not None and len(motion_weight_seq) > 0
    show_encoded_latent = encoded_latent_seq is not None and len(encoded_latent_seq) > 0
    show_future_latent = future_latent_seq is not None and len(future_latent_seq) > 0
    show_gt_future_latent = gt_future_latent_seq is not None and len(gt_future_latent_seq) > 0
    show_pred_gt_latent_diff = pred_gt_latent_diff_seq is not None and len(pred_gt_latent_diff_seq) > 0
    show_future_delta = future_delta_seq is not None and len(future_delta_seq) > 0

    total_future_cols = max(
        horizon,
        len(context_frames),
        len(encoded_latent_seq or []),
        len(future_latent_seq or []),
        len(gt_future_latent_seq or []),
        len(pred_gt_latent_diff_seq or []),
        len(future_delta_seq or []),
        1,
    )
    num_cols = total_future_cols + (1 if show_base else 0)

    row_titles = []
    encoded_row = None
    if show_encoded_latent:
        encoded_row = len(row_titles)
        row_titles.append("Encoded Latent")

    context_row = len(row_titles)
    row_titles.append("Context")
    gt_row = len(row_titles)
    row_titles.append("GT")
    rendered_row = len(row_titles)
    row_titles.append("Rendered")
    diff_row = len(row_titles)
    row_titles.append("Abs Diff")

    render_base_row = None
    if show_future_vs_base:
        render_base_row = len(row_titles)
        row_titles.append("|Render-Base|")

    motion_row = None
    if show_motion:
        motion_row = len(row_titles)
        row_titles.append("Motion Weight")

    future_latent_row = None
    if show_future_latent:
        future_latent_row = len(row_titles)
        row_titles.append("Future Latent")

    gt_future_latent_row = None
    if show_gt_future_latent:
        gt_future_latent_row = len(row_titles)
        row_titles.append("GT Future Encoded")

    pred_gt_latent_diff_row = None
    if show_pred_gt_latent_diff:
        pred_gt_latent_diff_row = len(row_titles)
        row_titles.append("|Pred-GT Encoded|")

    future_delta_row = None
    if show_future_delta:
        future_delta_row = len(row_titles)
        row_titles.append("Delta Latent")

    num_rows = len(row_titles)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 3.6 * num_rows))
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = np.array([axes])
    elif num_cols == 1:
        axes = np.array(axes).reshape(num_rows, 1)

    for row_idx, title in enumerate(row_titles):
        axes[row_idx, 0].set_ylabel(title, fontsize=14)

    for col_idx in range(num_cols):
        for row_idx in range(num_rows):
            axes[row_idx, col_idx].axis("off")

    start_col = 1 if show_base else 0

    if show_encoded_latent:
        for col_idx, latent_entry in enumerate((encoded_latent_seq or [])[:total_future_cols]):
            dst_col = start_col + col_idx
            latent_map = _latent_entry_to_heatmap(latent_entry)
            if latent_map is None:
                continue
            axes[encoded_row, dst_col].imshow(latent_map, cmap="viridis", vmin=0.0, vmax=1.0)
            label = context_labels[col_idx] if col_idx < len(context_labels) else f"ctx_{col_idx}"
            axes[encoded_row, dst_col].set_title(label, fontsize=13)

    for col_idx, frame in enumerate(context_frames[:total_future_cols]):
        dst_col = start_col + col_idx
        axes[context_row, dst_col].imshow(frame)
        if not show_encoded_latent:
            if col_idx < len(context_labels):
                axes[context_row, dst_col].set_title(context_labels[col_idx], fontsize=13)
            else:
                axes[context_row, dst_col].set_title(f"ctx_{col_idx}", fontsize=13)

    gt_key = f"{view_name}_image"
    base_gt_img = None
    base_render_img = None

    if show_base:
        base_title_row = encoded_row if show_encoded_latent else context_row
        axes[base_title_row, 0].set_title(base_label, fontsize=13)
        if show_encoded_latent and encoded_latent_seq:
            base_latent_map = _latent_entry_to_heatmap(encoded_latent_seq[min(len(encoded_latent_seq) - 1, len(context_labels) - 1)])
            if base_latent_map is not None:
                axes[encoded_row, 0].imshow(base_latent_map, cmap="viridis", vmin=0.0, vmax=1.0)
        if base_target_obs is not None and gt_key in base_target_obs:
            base_gt_img = base_target_obs[gt_key][idx].permute(1, 2, 0).detach().cpu().numpy()
            base_gt_img = np.clip(base_gt_img, 0, 1)
            axes[context_row, 0].imshow(base_gt_img)
            axes[gt_row, 0].imshow(base_gt_img)
        if base_rendered_obs is not None and gt_key in base_rendered_obs:
            base_render_img = base_rendered_obs[gt_key][0].permute(1, 2, 0).detach().cpu().numpy()
            base_render_img = np.clip(base_render_img, 0, 1)
            axes[rendered_row, 0].imshow(base_render_img)
        if base_gt_img is not None and base_render_img is not None:
            axes[diff_row, 0].imshow(np.abs(base_render_img - base_gt_img))
        if render_base_row is not None and base_render_img is not None:
            axes[render_base_row, 0].imshow(np.zeros_like(base_render_img))
        if motion_row is not None:
            blank_motion = _blank_latent_like(base_gt_img, fallback_size=224)
            axes[motion_row, 0].imshow(blank_motion, cmap="magma", vmin=0.0, vmax=1.0)
        if future_latent_row is not None:
            axes[future_latent_row, 0].text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=16, color="gray")
            axes[future_latent_row, 0].set_facecolor("white")
        if gt_future_latent_row is not None:
            axes[gt_future_latent_row, 0].text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=16, color="gray")
            axes[gt_future_latent_row, 0].set_facecolor("white")
        if pred_gt_latent_diff_row is not None:
            axes[pred_gt_latent_diff_row, 0].text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=16, color="gray")
            axes[pred_gt_latent_diff_row, 0].set_facecolor("white")
        if future_delta_row is not None:
            axes[future_delta_row, 0].text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=16, color="gray")
            axes[future_delta_row, 0].set_facecolor("white")

    for horizon_idx in range(min(horizon, total_future_cols)):
        col = start_col + horizon_idx
        gt_img = target_obs_seq[horizon_idx][gt_key][idx].permute(1, 2, 0).detach().cpu().numpy()
        gt_img = np.clip(gt_img, 0, 1)
        rendered_img = rendered_obs_seq[horizon_idx][gt_key][idx].permute(1, 2, 0).detach().cpu().numpy()
        rendered_img = np.clip(rendered_img, 0, 1)
        diff_img = np.abs(rendered_img - gt_img)

        axes[gt_row, col].imshow(gt_img)
        axes[gt_row, col].set_title(horizon_labels[horizon_idx], fontsize=13)
        axes[rendered_row, col].imshow(rendered_img)
        axes[diff_row, col].imshow(diff_img)
        if render_base_row is not None and base_render_img is not None:
            axes[render_base_row, col].imshow(np.abs(rendered_img - base_render_img))
        if motion_row is not None and horizon_idx < len(motion_weight_seq):
            motion_entry = motion_weight_seq[horizon_idx]
            if gt_key in motion_entry:
                motion_map = motion_entry[gt_key][idx].detach().cpu().numpy()
                axes[motion_row, col].imshow(motion_map, cmap="magma")
        if future_latent_row is not None and horizon_idx < len(future_latent_seq):
            latent_map = _latent_entry_to_heatmap(future_latent_seq[horizon_idx])
            if latent_map is not None:
                axes[future_latent_row, col].imshow(latent_map, cmap="viridis", vmin=0.0, vmax=1.0)
        if gt_future_latent_row is not None and horizon_idx < len(gt_future_latent_seq):
            gt_latent_map = _latent_entry_to_heatmap(gt_future_latent_seq[horizon_idx])
            if gt_latent_map is not None:
                axes[gt_future_latent_row, col].imshow(gt_latent_map, cmap="viridis", vmin=0.0, vmax=1.0)
        if pred_gt_latent_diff_row is not None and horizon_idx < len(pred_gt_latent_diff_seq):
            pred_gt_diff_map = _latent_entry_to_heatmap(pred_gt_latent_diff_seq[horizon_idx])
            if pred_gt_diff_map is not None:
                axes[pred_gt_latent_diff_row, col].imshow(pred_gt_diff_map, cmap="viridis", vmin=0.0, vmax=1.0)
        if future_delta_row is not None and horizon_idx < len(future_delta_seq):
            delta_map = _latent_entry_to_heatmap(future_delta_seq[horizon_idx])
            if delta_map is not None:
                axes[future_delta_row, col].imshow(delta_map, cmap="viridis", vmin=0.0, vmax=1.0)

    fig.suptitle(f"Future Rollout Visualization - Step {step}", fontsize=16)
    save_path = os.path.join(save_dir, f"render_viz_step_{step:06d}{time_suffix}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Future Rollout Visualization to {save_path}")
