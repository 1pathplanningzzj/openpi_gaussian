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
    if step is not None and step % 40 == 0:
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
        if step is not None and step % 40 == 0:
            opacity_before = gaussian_params["opacity"]
            print(f"[GaussianRenderer] Opacity before clamp: min={opacity_before.min():.6f}, max={opacity_before.max():.6f}, mean={opacity_before.mean():.6f}")
        gaussian_params["opacity"] = torch.clamp(gaussian_params["opacity"], min=0.0, max=1.0)
        if step is not None and step % 40 == 0:
            opacity_after = gaussian_params["opacity"]
            print(f"[GaussianRenderer] Opacity after clamp: min={opacity_after.min():.6f}, max={opacity_after.max():.6f}, mean={opacity_after.mean():.6f}")

        # Get scales and rotations directly (no sigma fallback)
        scales = gaussian_params["scales"]
        rotations = gaussian_params["rotations"]

        # Clamp scales
        if step is not None and step % 40 == 0:
            print(f"[GaussianRenderer] Scales before clamp: min={scales.min():.6f}, max={scales.max():.6f}, mean={scales.mean():.6f}")
        scales = torch.clamp(scales, min=1e-6, max=10.0)
        if step is not None and step % 40 == 0:
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
        if step is not None and step % 40 == 0:
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
            if step is not None and step % 40 == 0 and b == 0:
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
            if step is not None and step % 40 == 0 and b == 0:
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
            if b == 0 and step is not None and step % 40 == 0:
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

    if step is not None and step % 40 == 0:
        parts = ", ".join(f"{k}={v.item():.6f}" for k, v in loss_dict.items())
        print(f"[MultiViewLoss] Step {step}: {parts}, total={total_loss.item():.6f}")

    return total_loss, loss_dict

def visualize_rendering_comparison(step, gaussian_params, target_obs, cam_params_dict, renderer, view_names, save_dir=None, time_suffix="", temporal_frames=None):
    """
    Helper to visualize temporal sequence and rendering comparison.
    Args:
        step: Current training step (int)
        gaussian_params: Dict of gaussian parameters (batched)
        target_obs: GT target observation dict (t+1 frame)
        cam_params_dict: Camera parameters dict
        renderer: Instance of GaussianRenderer
        view_names: List of camera names to visualize
        save_dir: Optional directory to save visualizations. Defaults to "./visualizations/rendering"
        time_suffix: Optional suffix to identify time step (e.g., "_t", "_t1_pred", "_t1_gt")
        temporal_frames: Dict of temporal frames {key: [B, T, C, H, W]} where T includes t-2, t-1, t
    """
    import matplotlib
    # Use non-interactive backend to avoid X11 authorization issues in headless environments
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import os
    import numpy as np

    if save_dir is None:
        save_dir = "./visualizations/rendering"
    os.makedirs(save_dir, exist_ok=True)

    # Take first item in batch
    idx = 0

    # Create figure with 2 rows x 3 columns:
    # Multi-frame mode: Row 1: GT t-2, GT t-1, GT t | Row 2: GT t+1, Predicted t+1, Diff
    # Single-frame mode: Row 1: [empty], GT t, [empty] | Row 2: GT t+1, Predicted t+1, Diff
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    with torch.no_grad():
        # Process temporal sequence + rendering comparison
        for i, view_name in enumerate(view_names):
            if i >= 1:  # Only show first view to save space
                break

            # Find matching key in temporal_frames
            temporal_key = None
            if temporal_frames:
                # Try to match view_name with keys in temporal_frames
                # view_name could be "agent" or "wrist"
                # keys could be "base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", etc.
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
                
                # If still not found, just use the first available key
                if not temporal_key and temporal_frames:
                    temporal_key = next(iter(temporal_frames.keys()))
                    print(f"[Viz] Using first available key: {temporal_key} for view {view_name}")

            # Display 4 frames: [t-2, t-1, t, t+1]
            # Row 1: GT t-2, GT t-1, GT t (columns 0, 1, 2) for multi-frame mode
            # Row 1: GT t (column 1 only) for single-frame mode
            # Row 2: GT t+1, Rendered t+1, Diff (columns 0, 1, 2)
            if temporal_key and temporal_key in temporal_frames:
                frames = temporal_frames[temporal_key][idx]  # [T, H, W, C] where T=4 for multi-frame or T=2 for single-frame
                num_temporal_frames = frames.shape[0]

                # Detect single-frame mode: only 2 frames [t, t+1]
                is_single_frame_mode = (num_temporal_frames == 2)

                if is_single_frame_mode:
                    # Single-frame mode: only show t in row 1, center column
                    frame_t = frames[0]  # [H, W, C] - current frame t
                    frame_t_np = frame_t.detach().cpu().numpy()
                    frame_t_viz = np.clip((frame_t_np + 1.0) / 2.0, 0, 1)

                    ax = fig.add_subplot(gs[0, 1])  # Center column
                    ax.imshow(frame_t_viz)
                    ax.set_title("t", fontsize=14)
                    ax.axis('off')

                    # Leave side columns empty
                    for col in [0, 2]:
                        ax = fig.add_subplot(gs[0, col])
                        ax.axis('off')

                    # Row 2, Column 0: Display t+1 frame
                    frame_t1 = frames[1]  # [H, W, C] - future frame t+1
                    frame_t1_np = frame_t1.detach().cpu().numpy()
                    frame_t1_viz = np.clip((frame_t1_np + 1.0) / 2.0, 0, 1)

                    ax_gt_t1 = fig.add_subplot(gs[1, 0])
                    ax_gt_t1.imshow(frame_t1_viz)
                    ax_gt_t1.set_title("t+1", fontsize=14)
                    ax_gt_t1.axis('off')
                else:
                    # Multi-frame mode: Display first 3 frames (t-2, t-1, t)
                    for t_idx in range(3):
                        if t_idx < num_temporal_frames:
                            frame = frames[t_idx]  # [H, W, C]
                            frame_np = frame.detach().cpu().numpy()
                            # Normalize from [-1, 1] to [0, 1]
                            frame_viz = np.clip((frame_np + 1.0) / 2.0, 0, 1)

                            ax = fig.add_subplot(gs[0, t_idx])
                            ax.imshow(frame_viz)
                            ax.set_title(f"GT t-{2-t_idx}", fontsize=14)
                            ax.axis('off')

                    # Row 2, Column 0: Display 4th frame (t+1)
                    if num_temporal_frames >= 4:
                        frame_t1 = frames[3]  # [H, W, C]
                        frame_t1_np = frame_t1.detach().cpu().numpy()
                        frame_t1_viz = np.clip((frame_t1_np + 1.0) / 2.0, 0, 1)

                        ax_gt_t1 = fig.add_subplot(gs[1, 0])
                        ax_gt_t1.imshow(frame_t1_viz)
                        ax_gt_t1.set_title(f"GT t+1", fontsize=14)
                        ax_gt_t1.axis('off')
            else:
                # If no temporal frames, show placeholder
                for row in range(2):
                    for col in range(3 if row == 0 else 1):
                        ax = fig.add_subplot(gs[row, col])
                        ax.text(0.5, 0.5, 'No temporal data', ha='center', va='center', fontsize=12)
                        ax.axis('off')

            # Row 2, Column 1: Render t+1
            cam_params = {k: v[idx:idx+1] if isinstance(v, torch.Tensor) else v for k, v in cam_params_dict[view_name].items()}

            params_single = {
               "xyz": gaussian_params["xyz"][idx:idx+1],
               "sh": gaussian_params["sh"][idx:idx+1],
               "opacity": gaussian_params["opacity"][idx:idx+1],
               "scales": gaussian_params["scales"][idx:idx+1],
               "rotations": gaussian_params["rotations"][idx:idx+1]
            }

            rendered_img = renderer(params_single, cam_params)  # [1, 3, H, W]
            rendered_img = rendered_img.float()
            rendered_np = rendered_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()  # [H, W, 3]
            rendered_viz = np.clip(rendered_np, 0, 1)

            ax_rendered = fig.add_subplot(gs[1, 1])
            ax_rendered.imshow(rendered_viz)
            ax_rendered.set_title("render t+1", fontsize=14)
            ax_rendered.axis('off')

            # Row 2, Column 2: Difference
            # Get GT t+1 for comparison
            gt_img = target_obs[f"{view_name}_image"][idx]  # [3, H, W]
            gt_img = gt_img.float()
            gt_np = gt_img.permute(1, 2, 0).detach().cpu().numpy()
            gt_viz = np.clip(gt_np, 0, 1)
            
            diff_np = np.abs(rendered_viz - gt_viz)

            ax_diff = fig.add_subplot(gs[1, 2])
            ax_diff.imshow(diff_np)
            ax_diff.set_title(f"Diff", fontsize=14)
            ax_diff.axis('off')

            print(f"[Viz] Rendered {view_name} range: min={rendered_np.min():.4f}, max={rendered_np.max():.4f}, mean={rendered_np.mean():.4f}")
            if rendered_np.max() < 1e-6:
                print(f"[Viz] WARNING: Rendered image is all zeros or very small (max={rendered_np.max():.6f})!")

    # Save visualization
    if time_suffix:
        save_path = os.path.join(save_dir, f"render_viz_step_{step:06d}{time_suffix}.png")
    else:
        save_path = os.path.join(save_dir, f"render_viz_step_{step:06d}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Rendering Visualization to {save_path}")
