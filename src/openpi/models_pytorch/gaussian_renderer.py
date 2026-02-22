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

# Import LPIPS for perceptual loss
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    print("Warning: lpips not available. Install with: pip install lpips")
    LPIPS_AVAILABLE = False


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
    # We'll pass z_cam just in case. but ..to be fixed later
    pixels_2d = torch.stack([u, v, z_cam], dim=-1)  # [B, N, 3]

    return pixels_2d


class GaussianRenderer(nn.Module):
    """
    Wrapper for AD-FFgsStudio's GaussianRasterizer.
    Renders 3D Gaussians to 2D images for supervision.
    """

    def __init__(self, image_size: int = 224, sh_degree: int = 3, scale_factor: float = 1.0, use_lpips: bool = False):
        super().__init__()
        self.image_size = image_size
        self.sh_degree = sh_degree
        self.scale_factor = scale_factor  # Scale multiplier to adjust Gaussian sizes
        self.use_lpips = use_lpips

        if not RASTERIZER_AVAILABLE:
            raise ImportError(
                "GaussianRasterizer not available. "
                "Please compile diff-gaussian-rasterization."
            )

        # Initialize LPIPS loss function if available
        self.lpips_fn = None
        if use_lpips and LPIPS_AVAILABLE:
            self.lpips_fn = lpips.LPIPS(net='vgg')
            self.lpips_fn.eval()  # Always in eval mode
            print("[GaussianRenderer] LPIPS loss enabled (VGG backbone)")

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

        # --- Get scales and rotations ---
        # New path: use scales and rotations directly (like gsplat reference in AD-FFgsStudio)
        # Old path (sigma): decompose covariance → loses rotation info (all become identity)
        if "scales" in gaussian_params:
            scales = gaussian_params["scales"]
            rotations = gaussian_params["rotations"]
        else:
            # Legacy fallback: decompose sigma (loses rotation!)
            scales, rotations = convert_sigma_to_scale_rotation(
                gaussian_params["sigma"]
            )

        # Clamp scales — min=1e-4 prevents degenerate Gaussians that cause NaN in rasterizer backward
        if step is not None and step % 40 == 0:
            print(f"[GaussianRenderer] Scales before clamp: min={scales.min():.6f}, max={scales.max():.6f}, mean={scales.mean():.6f}")
        scales = torch.clamp(scales, min=1e-4, max=10.0)
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
            # Degree 3 -> (3+1)^2 = 16 coeffs. 3 RGB channels = 48 total
            shs_val = gaussian_params["sh"][b]
            # Calculate num_coeffs dynamically based on sh_degree
            num_coeffs = (self.sh_degree + 1) ** 2
            expected_sh_dim = num_coeffs * 3  # 16 * 3 = 48 for sh_degree=3
            
            # Debug: Print SH stats (only every 40 steps)
            if step is not None and step % 40 == 0 and b == 0:
                print(f"[GaussianRenderer] SH before reshape: shape={shs_val.shape}, min={shs_val.min():.6f}, max={shs_val.max():.6f}, mean={shs_val.mean():.6f}")
            
            # Handle different SH dimensions (VGGT may use sh_degree=4 -> 75 dims, we need 48 for sh_degree=3)
            if shs_val.shape[-1] == expected_sh_dim:
                # Exact match: reshape to [N, num_coeffs, 3]
                shs_val = shs_val.view(-1, num_coeffs, 3)
            elif shs_val.shape[-1] > expected_sh_dim:
                # More coefficients than needed: truncate to first expected_sh_dim
                shs_val = shs_val[..., :expected_sh_dim].view(-1, num_coeffs, 3)
            else:
                # Fewer coefficients: pad with zeros
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
    dynamic_weight_map: Optional[torch.Tensor] = None,
    lpips_fn: Optional[nn.Module] = None,
    use_lpips: bool = True,
    training_step: Optional[int] = None,
) -> torch.Tensor:
    """Combined photometric loss: 0.6 * L1 + 0.2 * SSIM + 0.2 * LPIPS (if available).

    Args:
        pred: [B, 3, H, W] rendered image
        target: [B, 3, H, W] ground truth image
        dynamic_weight_map: [B, 1, H, W] per-pixel weight that upweights dynamic regions.
            If None, uniform weighting is used (original behavior).
        lpips_fn: Optional LPIPS loss function. If None and use_lpips=True, will create one.
        use_lpips: Whether to use LPIPS loss (default True if available).
        training_step: Current training step. If provided, LPIPS is only enabled after step 10000.
    Returns:
        Scalar loss.
    """
    l1_loss = (target - pred).abs()
    ssim_loss = compute_ssim_loss(pred, target)

    # Disable LPIPS before step 5000 to let action learning converge first
    if training_step is not None and training_step < 5000:
        use_lpips = False

    # Compute LPIPS if available and requested
    if use_lpips and LPIPS_AVAILABLE:
        if lpips_fn is None:
            # Create LPIPS function on first call (will be cached in renderer)
            lpips_fn = lpips.LPIPS(net='vgg').to(pred.device)
            lpips_fn.eval()  # Always in eval mode

        with torch.no_grad() if not pred.requires_grad else torch.enable_grad():
            # LPIPS expects images in [-1, 1] range, our images are in [0, 1]
            pred_normalized = pred * 2.0 - 1.0
            target_normalized = target * 2.0 - 1.0
            lpips_loss = lpips_fn(pred_normalized, target_normalized)  # [B, 1, 1, 1]
            lpips_loss = lpips_loss.squeeze()  # [B] or scalar
            if lpips_loss.dim() == 0:
                lpips_loss = lpips_loss.unsqueeze(0)  # Ensure at least [B]
            # Clamp LPIPS to ensure non-negative values (LPIPS should be in [0, inf))
            lpips_loss = torch.clamp(lpips_loss, min=0.0)
            # Expand to [B, 1, H, W] for consistent weighting
            lpips_loss = lpips_loss.view(-1, 1, 1, 1).expand_as(l1_loss.mean(1, True))

        # Combine: 0.6 * L1 + 0.2 * SSIM + 0.2 * LPIPS
        pixel_loss = 0.6 * l1_loss.mean(1, True) + 0.2 * ssim_loss.mean(1, True) + 0.2 * lpips_loss
    else:
        # Fallback to original: 0.85 * SSIM + 0.15 * L1
        pixel_loss = 0.85 * ssim_loss.mean(1, True) + 0.15 * l1_loss.mean(1, True)

    if dynamic_weight_map is not None:
        return (pixel_loss * dynamic_weight_map).mean()
    return pixel_loss.mean()


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
    dynamic_weight_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute rendering loss (L1 + SSIM + LPIPS photometric) with optional attention masking.

    Args:
        gaussian_params: 3D Gaussian parameters
        target_image: [B, 3, H, W] - Ground truth image
        camera_params: Camera parameters
        renderer: GaussianRenderer instance
        M_attn: [B, H, W] - Optional attention mask
        dynamic_weight_map: [B, 1, H, W] - Per-pixel weight for dynamic regions

    Returns:
        loss: Scalar rendering loss
    """
    # Render from the given viewpoint
    rendered_image = renderer(gaussian_params, camera_params, step=step)

    if M_attn is not None:
        # Masked photometric loss (fallback to old formula when using attention mask)
        l1_loss = (rendered_image - target_image).abs()
        ssim_loss = compute_ssim_loss(rendered_image, target_image)
        pixel_loss = 0.85 * ssim_loss + 0.15 * l1_loss
        weighted_loss = M_attn.unsqueeze(1) * pixel_loss
        loss = weighted_loss.mean()
    else:
        loss = compute_photometric_loss(
            rendered_image, target_image,
            dynamic_weight_map=dynamic_weight_map,
            lpips_fn=renderer.lpips_fn,
            use_lpips=renderer.use_lpips,
            training_step=step
        )

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
    current_observations: Optional[Dict[str, torch.Tensor]] = None,
    dynamic_weight_boost: float = 3.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute multi-view rendering loss with regularization and edge-aware depth smoothness.

    Args:
        gaussian_params: 3D Gaussian parameters
        observations: Dictionary with target (future) images for each view
        camera_params_dict: Dictionary of camera parameters for each view
        renderer: GaussianRenderer instance
        M_attn_dict: Optional dictionary of attention masks for each view
        view_names: List of view names to render
        step: Current training step
        depth_map: [B, 1, H, W] predicted depth map for edge-aware smoothness
        lambda_scale: Weight for scale regularization
        lambda_opacity: Weight for opacity regularization
        lambda_edge_smooth: Weight for edge-aware depth smoothness
        current_observations: Dictionary with current frame images (same keys as observations).
            Used to compute dynamic region weighting.
        dynamic_weight_boost: How much to upweight dynamic regions (default 3.0).
            Final weight = 1.0 + boost * diff_mask, so static=1.0, dynamic=1.0+boost.

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

            # Compute dynamic region weight map from current vs future frame difference
            dynamic_weight_map = None
            if current_observations is not None:
                curr_key = f"{view_name}_image"
                if curr_key in current_observations:
                    curr_img = current_observations[curr_key]  # [B, 3, H, W]
                    # Resize if needed
                    if curr_img.shape[2:] != target_image.shape[2:]:
                        curr_img = F.interpolate(curr_img, size=target_image.shape[2:], mode="bilinear", align_corners=False)
                    # Per-pixel L1 difference → soft mask
                    diff = (target_image - curr_img).abs().mean(dim=1, keepdim=True)  # [B, 1, H, W]
                    # Normalize to [0, 1] per sample
                    diff_max = diff.flatten(1).max(dim=1)[0].view(-1, 1, 1, 1).clamp(min=1e-6)
                    diff_norm = diff / diff_max  # [B, 1, H, W] in [0, 1]
                    # Weight: static=1.0, dynamic=1.0+boost
                    dynamic_weight_map = 1.0 + dynamic_weight_boost * diff_norm

                    if step is not None and step % 200 == 0:
                        import logging
                        logging.info(
                            f"[DynamicWeight] {view_name}: diff_mean={diff.mean():.4f}, "
                            f"weight_mean={dynamic_weight_map.mean():.2f}, "
                            f"weight_max={dynamic_weight_map.max():.2f}"
                        )

            view_loss = compute_rendering_loss(
                gaussian_params, target_image, camera_params, renderer, M_attn,
                step=step, dynamic_weight_map=dynamic_weight_map,
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

def visualize_rendering_comparison(step, gaussian_params, target_obs, cam_params_dict, renderer, view_names, save_dir=None, time_suffix="", observation_images=None):
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
        observation_images: Optional dict of observation frames {view_key: [3, C, H, W]} for t-2, t-1, t
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

    has_obs = observation_images is not None and len(observation_images) > 0
    n_rows = 2 if has_obs else 1
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]  # Ensure 2D array

    with torch.no_grad():
        # Row 1: observation GT frames (t-2, t-1, t) if available
        if has_obs:
            for i, view_name in enumerate(view_names):
                view_key = f"{view_name}_image"
                if view_key in observation_images:
                    frames = observation_images[view_key]  # [3, C, H, W]
                    time_labels = ["t-2", "t-1", "t"]
                    for col in range(min(3, frames.shape[0])):
                        frame_np = frames[col].permute(1, 2, 0).detach().cpu().numpy()
                        frame_viz = np.clip(frame_np, 0, 1)
                        axes[0, col].imshow(frame_viz)
                        axes[0, col].set_title(f"GT {time_labels[col]}")
                        axes[0, col].axis('off')
                break  # Only first view for now

        # Row 2 (or Row 1 if no obs): GT t+1, Rendered t+1, Diff
        render_row = 1 if has_obs else 0
        for i, view_name in enumerate(view_names):
            # 1. Render
            cam_params = {k: v[idx:idx+1] if isinstance(v, torch.Tensor) else v for k, v in cam_params_dict[view_name].items()}

            params_single = {
               "xyz": gaussian_params["xyz"][idx:idx+1],
               "sh": gaussian_params["sh"][idx:idx+1],
               "opacity": gaussian_params["opacity"][idx:idx+1]
            }
            if "scales" in gaussian_params:
                params_single["scales"] = gaussian_params["scales"][idx:idx+1]
                params_single["rotations"] = gaussian_params["rotations"][idx:idx+1]
            else:
                params_single["sigma"] = gaussian_params["sigma"][idx:idx+1]

            rendered_img = renderer(params_single, cam_params)
            rendered_img = rendered_img.float()

            rendered_np = rendered_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            print(f"[Viz] Rendered {view_name} range: min={rendered_np.min():.4f}, max={rendered_np.max():.4f}, mean={rendered_np.mean():.4f}")

            if rendered_np.max() < 1e-6:
                print(f"[Viz] WARNING: Rendered image is all zeros or very small (max={rendered_np.max():.6f})!")

            rendered_viz = np.clip(rendered_np, 0, 1)

            # 2. GT Image
            gt_img = target_obs[f"{view_name}_image"][idx]
            gt_img = gt_img.float()
            gt_np = gt_img.permute(1, 2, 0).detach().cpu().numpy()
            gt_viz = np.clip(gt_np, 0, 1)

            # 3. Difference
            diff_np = np.abs(rendered_viz - gt_viz)

            # Plot
            axes[render_row, 0].imshow(gt_viz)
            axes[render_row, 0].set_title(f"GT t+1")
            axes[render_row, 0].axis('off')

            axes[render_row, 1].imshow(rendered_viz)
            axes[render_row, 1].set_title(f"Rendered t+1")
            axes[render_row, 1].axis('off')

            axes[render_row, 2].imshow(diff_np)
            axes[render_row, 2].set_title(f"Diff")
            axes[render_row, 2].axis('off')
            break  # Only first view

    plt.tight_layout()
    if time_suffix:
        save_path = os.path.join(save_dir, f"render_viz_step_{step:06d}{time_suffix}.png")
    else:
        save_path = os.path.join(save_dir, f"render_viz_step_{step:06d}.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved Rendering Visualization to {save_path}")
