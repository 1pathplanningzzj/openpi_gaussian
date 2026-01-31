# zijian
# date 2026.01.19
# Description: Adapter module for integrating frozen 3D Gaussian Splatting (DF3DGS) features into OpenPI.

import logging
import sys
from pathlib import Path
import itertools

import torch
from torch import nn
import torch.nn.functional as F

# Add AD-FFgsStudio to python path
# Assuming the file is at src/openpi/models_pytorch/pi0_gaussian.py
# We need to go up 3 levels to reach the root (src/openpi/models_pytorch -> src/openpi -> src -> root)
# that ‘s done ！☑️！✅ 
_root_path = Path(__file__).resolve().parents[3]
_ad_ffgs_path = _root_path / "third_party" / "AD-FFgsStudio"
if str(_ad_ffgs_path) not in sys.path:
    sys.path.append(str(_ad_ffgs_path))

try:
    from models.vggt3dgs_model import VGGT3DGSModel
except ImportError:
    logging.warning("Could not import VGGT3DGSModel. 3DGS integration may fail.")
    VGGT3DGSModel = None

# Import LGPD Module
from openpi.models_pytorch.lgpd_module import LanguageGatedPhysicalDistillation


def convert_vggt_params_to_render_format(vggt_params, camera_params=None):
    """
    Convert VGGT decoded parameters to format expected by GaussianRenderer.
    
    Args:
        vggt_params: Dict with keys:
            - depth_maps: [B, S, H, W, 1]
            - rot_maps: [B, S, H, W, 4] (quaternion)
            - scale_maps: [B, S, H, W, 3]
            - opacity_maps: [B, S, H, W, 1]
            - sh_maps: [B, S, H, W, K, 3] where K = (sh_degree+1)^2
        camera_params: Optional dict with intrinsic and extrinsic for unprojection
    
    Returns:
        gaussian_params: Dict with keys:
            - xyz: [B, N, 3] - 3D positions (from depth unprojection)
            - sigma: [B, N, 6] - Covariance parameters (from scale and rotation)
            - opacity: [B, N, 1]
            - sh: [B, N, K*3] - Spherical harmonics coefficients
    """
    B, S, H, W = vggt_params["depth_maps"].shape[:4]
    device = vggt_params["depth_maps"].device
    
    # Flatten spatial dimensions: [B, S, H, W, ...] -> [B, S*H*W, ...]
    depth_flat = vggt_params["depth_maps"].reshape(B, S * H * W, 1)  # [B, N, 1]
    rot_flat = vggt_params["rot_maps"].reshape(B, S * H * W, 4)  # [B, N, 4]
    scale_flat = vggt_params["scale_maps"].reshape(B, S * H * W, 3)  # [B, N, 3]
    opacity_flat = vggt_params["opacity_maps"].reshape(B, S * H * W, 1)  # [B, N, 1]
    sh_flat = vggt_params["sh_maps"].reshape(B, S * H * W, -1)  # [B, N, K*3]
    
    # Unproject depth to 3D points if camera params provided
    if camera_params is not None:
        intrinsic = camera_params.get("intrinsic")
        extrinsic = camera_params.get("extrinsic")
        if intrinsic is not None:
            # Reshape depth back to [B, S, H, W] for unprojection
            depth_reshaped = vggt_params["depth_maps"].squeeze(-1)  # [B, S, H, W]
            xyz_maps = unproject_depth_to_points(depth_reshaped, intrinsic, extrinsic)  # [B, S, H, W, 3]
            xyz_flat = xyz_maps.reshape(B, S * H * W, 3)  # [B, N, 3]
        else:
            # Fallback: use zeros (will need to be set properly)
            xyz_flat = torch.zeros(B, S * H * W, 3, device=device)
    else:
        # No camera params: use zeros (placeholder)
        xyz_flat = torch.zeros(B, S * H * W, 3, device=device)
    
    # Convert scale and rotation to covariance parameters (sigma)
    # For now, use a simple conversion: sigma = scale^2 * rotation_matrix
    # This is a simplified version; proper conversion would use the rotation matrix
    # For simplicity, we'll create a 6D representation from scale
    # In practice, you'd want to convert quaternion to rotation matrix and then to covariance
    scale_sq = scale_flat ** 2  # [B, N, 3]
    # Create upper triangular covariance representation [s11, s12, s13, s22, s23, s33]
    # Simplified: assume diagonal covariance (no rotation for now)
    sigma_flat = torch.zeros(B, S * H * W, 6, device=device)
    sigma_flat[:, :, 0] = scale_sq[:, :, 0]  # s11
    sigma_flat[:, :, 3] = scale_sq[:, :, 1]  # s22
    sigma_flat[:, :, 5] = scale_sq[:, :, 2]  # s33
    # TODO: Properly incorporate rotation into covariance matrix
    
    return {
        "xyz": xyz_flat,  # [B, N, 3]
        "sigma": sigma_flat,  # [B, N, 6]
        "opacity": opacity_flat,  # [B, N, 1]
        "sh": sh_flat,  # [B, N, K*3]
    }


def compute_vggt_decoder_supervision_loss(
    decoder_params, 
    vggt_params_dict, 
    camera_params=None,
    lambda_xyz=1.0,
    lambda_opacity=1.0,
    lambda_sh=1.0,
    lambda_sigma=0.5
):
    """
    Compute supervision loss between World Model decoder output and VGGT decoded parameters.
    
    Args:
        decoder_params: Dict from Privileged4DGSDecoder with keys:
            - xyz: [B, N, 3]
            - sigma: [B, N, 6]
            - opacity: [B, N, 1]
            - sh: [B, N, K*3]
        vggt_params_dict: Dict from VGGT with keys:
            - depth_maps: [B, S, H, W, 1]
            - rot_maps: [B, S, H, W, 4]
            - scale_maps: [B, S, H, W, 3]
            - opacity_maps: [B, S, H, W, 1]
            - sh_maps: [B, S, H, W, K, 3]
        camera_params: Optional dict with intrinsic and extrinsic for unprojection
        lambda_xyz: Weight for xyz loss
        lambda_opacity: Weight for opacity loss
        lambda_sh: Weight for SH loss
        lambda_sigma: Weight for sigma/covariance loss
    
    Returns:
        loss: Scalar tensor - total supervision loss
        loss_dict: Dict with individual loss components
    """
    # Convert VGGT params to decoder format
    vggt_decoder_format = convert_vggt_params_to_render_format(
        vggt_params_dict, 
        camera_params
    )
    
    # Align dimensions if needed
    B_dec, N_dec = decoder_params["xyz"].shape[:2]
    B_vggt, N_vggt = vggt_decoder_format["xyz"].shape[:2]
    
    # If dimensions don't match, we need to handle it
    # For now, assume they should match (both from same batch and same number of tokens)
    if N_dec != N_vggt:
        # Interpolate or sample to match dimensions
        # Simple approach: take first N_dec points from vggt
        min_n = min(N_dec, N_vggt)
        vggt_decoder_format = {
            k: v[:, :min_n] if v.ndim >= 2 else v 
            for k, v in vggt_decoder_format.items()
        }
        decoder_params = {
            k: v[:, :min_n] if v.ndim >= 2 else v 
            for k, v in decoder_params.items()
        }
        N = min_n
    else:
        N = N_dec
    
    # Compute individual losses
    loss_dict = {}
    
    # 1. XYZ loss (L2)
    xyz_loss = F.mse_loss(
        decoder_params["xyz"], 
        vggt_decoder_format["xyz"]
    )
    loss_dict["xyz_loss"] = xyz_loss
    
    # 2. Opacity loss (L2)
    opacity_loss = F.mse_loss(
        decoder_params["opacity"], 
        vggt_decoder_format["opacity"]
    )
    loss_dict["opacity_loss"] = opacity_loss
    
    # 3. SH loss (L2)
    # Handle dimension mismatch: VGGT uses SH degree 4 (75 dims), decoder uses SH degree 3 (48 dims)
    decoder_sh = decoder_params["sh"]  # [B, N, 48]
    vggt_sh = vggt_decoder_format["sh"]  # [B, N, 75]
    
    # Take first 48 dimensions from VGGT SH to match decoder
    # SH coefficients are ordered: degree 0, 1, 2, 3, 4...
    # So taking first 48 gives us degrees 0-3, which matches decoder
    if vggt_sh.shape[-1] > decoder_sh.shape[-1]:
        vggt_sh_truncated = vggt_sh[..., :decoder_sh.shape[-1]]  # [B, N, 48]
    elif vggt_sh.shape[-1] < decoder_sh.shape[-1]:
        # Pad with zeros if VGGT has fewer dimensions (unlikely)
        padding = torch.zeros(
            *vggt_sh.shape[:-1], 
            decoder_sh.shape[-1] - vggt_sh.shape[-1],
            device=vggt_sh.device,
            dtype=vggt_sh.dtype
        )
        vggt_sh_truncated = torch.cat([vggt_sh, padding], dim=-1)
    else:
        vggt_sh_truncated = vggt_sh
    
    sh_loss = F.mse_loss(decoder_sh, vggt_sh_truncated)
    loss_dict["sh_loss"] = sh_loss
    
    # 4. Sigma/Covariance loss (L2)
    # Note: This is simplified since we're not properly converting rotation to covariance
    sigma_loss = F.mse_loss(
        decoder_params["sigma"], 
        vggt_decoder_format["sigma"]
    )
    loss_dict["sigma_loss"] = sigma_loss
    
    # Weighted total loss
    total_loss = (
        lambda_xyz * xyz_loss +
        lambda_opacity * opacity_loss +
        lambda_sh * sh_loss +
        lambda_sigma * sigma_loss
    )
    
    return total_loss, loss_dict


def unproject_depth_to_points(depth_map, intrinsic, extrinsic=None):
    """
    Unproject depth map to 3D points (Gaussian centers).
    
    Args:
        depth_map: [B, S, H, W, 1] or [B, S, H, W] - Depth values
        intrinsic: [B, S, 3, 3] or [B, 3, 3] - Camera intrinsic matrix
        extrinsic: Optional [B, S, 4, 4] or [B, 4, 4] - Camera extrinsic (camera to world/ego)
                  If None, returns points in camera coordinate
    
    Returns:
        points: [B, S, H, W, 3] - 3D points (in world/ego coordinate if extrinsic provided, else camera coordinate)
    """
    # Handle shape variations
    if depth_map.ndim == 4:
        depth_map = depth_map.unsqueeze(-1)  # [B, S, H, W] -> [B, S, H, W, 1]
    
    B, S, H, W, _ = depth_map.shape
    device = depth_map.device
    dtype = depth_map.dtype
    
    # Ensure depth_map is 2D: [B, S, H, W]
    depth_map = depth_map.squeeze(-1)  # [B, S, H, W]
    
    # Handle intrinsic shape
    if intrinsic.ndim == 3:
        intrinsic = intrinsic.unsqueeze(1).expand(-1, S, -1, -1)  # [B, 3, 3] -> [B, S, 3, 3]
    
    # Create pixel grid
    u = torch.arange(W, device=device, dtype=dtype)
    v = torch.arange(H, device=device, dtype=dtype)
    u_grid, v_grid = torch.meshgrid(u, v, indexing='xy')  # u_grid: (H, W), v_grid: (H, W)
    u_grid = u_grid.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)  # [B, S, H, W]
    v_grid = v_grid.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)  # [B, S, H, W]
    
    # Extract intrinsic parameters
    fx = intrinsic[:, :, 0, 0].unsqueeze(-1).unsqueeze(-1)  # [B, S, 1, 1]
    fy = intrinsic[:, :, 1, 1].unsqueeze(-1).unsqueeze(-1)  # [B, S, 1, 1]
    cx = intrinsic[:, :, 0, 2].unsqueeze(-1).unsqueeze(-1)  # [B, S, 1, 1]
    cy = intrinsic[:, :, 1, 2].unsqueeze(-1).unsqueeze(-1)  # [B, S, 1, 1]
    
    # Unproject to camera coordinate
    x_cam = (u_grid - cx) * depth_map / fx
    y_cam = (v_grid - cy) * depth_map / fy
    z_cam = depth_map
    
    cam_points = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # [B, S, H, W, 3]
    
    # Transform to world/ego coordinate if extrinsic provided
    if extrinsic is not None:
        # Handle extrinsic shape
        if extrinsic.ndim == 3:
            extrinsic = extrinsic.unsqueeze(1).expand(-1, S, -1, -1)  # [B, 4, 4] -> [B, S, 4, 4]
        
        R = extrinsic[:, :, :3, :3]  # [B, S, 3, 3]
        t = extrinsic[:, :, :3, 3]   # [B, S, 3]
        
        # Reshape for batch matrix multiplication
        cam_points_flat = cam_points.reshape(B, S, -1, 3)  # [B, S, H*W, 3]
        
        # Transform: P_world = R @ P_cam^T + t
        world_points_flat = torch.matmul(cam_points_flat, R.transpose(-1, -2)) + t.unsqueeze(2)
        
        # Reshape back
        world_points = world_points_flat.reshape(B, S, H, W, 3)  # [B, S, H, W, 3]
        
        # Mask invalid depth points
        mask = (depth_map == 0).unsqueeze(-1).expand(-1, -1, -1, -1, 3)
        world_points[mask] = 0
        
        return world_points
    else:
        # Return camera coordinate points
        mask = (depth_map == 0).unsqueeze(-1).expand(-1, -1, -1, -1, 3)
        cam_points[mask] = 0
        return cam_points


class GaussianAdapter(nn.Module):
    """
    Adapter for integrating VGGT (Transformer-based 3DGS) features into OpenPI models.
    """
    def __init__(self, use_gaussian: bool, action_expert_width: int, use_lgpd: bool = True, 
                 view_selection: str = "both"):
        """
        Args:
            use_gaussian: Whether to use Gaussian features
            action_expert_width: Width of action expert
            use_lgpd: Whether to use Language-Gated Physical Distillation
            view_selection: View selection strategy
                - "both": Use both agent and wrist views (default)
                - "agent_only": Use only agent/global view
                - "wrist_only": Use only wrist view
                - "closest": Use two views with smallest position difference (if available)
        """
        super().__init__()
        self.use_gaussian = use_gaussian
        self.encoder = None
        self.proj = None
        self.lgpd = None  # Language-Gated Physical Distillation
        self.use_lgpd = use_lgpd
        self.view_selection = view_selection

        if self.use_gaussian and VGGT3DGSModel is not None:
            logging.info("Initializing VGGT 3DGS Components in Adapter...")
            
            try:
                # Initialize VGGT Model
                # Parameters based on vggt3dgs_model.py defaults or typical values
                self.encoder = VGGT3DGSModel(sh_degree=4, min_depth=1.5, max_depth=100.0)
                
                # Freeze Encoder
                for param in self.encoder.parameters():
                    param.requires_grad = False
                self.encoder.eval()
                
                # Projection and Head
                # VGGT embed_dim: The aggregator seems to return 2048 dim (concatenated? or large DINO)
                self.gaussian_feat_dim = 2048 
                self.proj = nn.Linear(self.gaussian_feat_dim, action_expert_width)
                
                # Pooling
                # Input: 2 views (Agent + Wrist)
                # Target: ~200 tokens total
                # 10x10 -> 100 tokens/view * 2 views = 200 tokens
                self.pool = nn.AdaptiveAvgPool2d((10, 10))
                
                # LGPD Module
                if self.use_lgpd:
                    logging.info("Initializing LGPD Module...")
                    self.lgpd = LanguageGatedPhysicalDistillation(
                        token_dim=action_expert_width,
                        text_dim=action_expert_width,  # Assuming pooled text emb has same dim as visual proj
                        num_context_tokens=16,
                        background_weight=0.1
                    )

            except Exception as e:
                logging.error(f"Failed to initialize VGGT components: {e}")
                self.use_gaussian = False
                self.encoder = None
        else:
            self.use_gaussian = False

    def encode(self, observation, device, batch_size):
        """Pure encoding step to get Gaussian features."""
        # This is mainly for inference/debugging checks
        inputs = self.prepare_inputs(observation, device, batch_size)
        if inputs is None:
            return None
        return self.forward(inputs)[0] # Just return embs

    def prepare_inputs(self, observation, device, batch_size):
        """Helper to prepare inputs for VGGT encoder from observation.
        VGGT expects [Batch_size, view_num, 3, H, W]
        
        View selection strategies:
        - "both": Agent + Wrist (may have large position difference)
        - "agent_only": Only agent/global view (more stable, less view diversity)
        - "wrist_only": Only wrist view (close-up, may miss global context)
        - "closest": Two views with smallest position difference (if available)
        """
        if not self.use_gaussian:
            return None
            
        images_dict = observation.images
        keys = list(images_dict.keys())
        
        # 1. Find Wrist Camera
        wrist_key = next((k for k in keys if "wrist" in k or "eye" in k or "hand" in k), None)
        
        # 2. Find Agent/Global Camera
        # Heuristic: Look for 'agent', 'high', 'env', 'cam0', 'base'
        agent_key = next((k for k in keys if k != wrist_key and any(x in k for x in ["agent", "high", "env", "front", "cam0", "base"])), None)
        
        # Fallback if no specific keyword found: pick first non-wrist key
        if not agent_key and len(keys) > (1 if wrist_key else 0):
             agent_key = next((k for k in keys if k != wrist_key), None)
        
        selected_imgs = []
        
        # Apply view selection strategy
        if self.view_selection == "agent_only":
            # Only use agent view (single view, more stable)
            if agent_key:
                selected_imgs.append(images_dict[agent_key])
                logging.debug(f"Using single view: {agent_key}")
        elif self.view_selection == "wrist_only":
            # Only use wrist view
            if wrist_key:
                selected_imgs.append(images_dict[wrist_key])
                logging.debug(f"Using single view: {wrist_key}")
        elif self.view_selection == "closest":
            # Try to find two views with smallest position difference
            # For now, fallback to agent + wrist, but could be enhanced with camera pose info
            if agent_key and wrist_key:
                # TODO: If camera poses are available, select closest pair
                # For now, use agent + wrist but log a warning
                logging.warning("'closest' view selection not fully implemented, using agent+wrist")
                selected_imgs.append(images_dict[agent_key])
                selected_imgs.append(images_dict[wrist_key])
            elif agent_key:
                selected_imgs.append(images_dict[agent_key])
            elif wrist_key:
                selected_imgs.append(images_dict[wrist_key])
        else:  # "both" (default)
            # Use both views (original behavior)
            if agent_key: 
                selected_imgs.append(images_dict[agent_key])
            if wrist_key: 
                selected_imgs.append(images_dict[wrist_key])
            if len(selected_imgs) == 2:
                logging.debug(f"Using both views: {agent_key} + {wrist_key} (may have large position difference)")
        
        if not selected_imgs:
            return None
            
        processed_imgs = []
        # Use the image size defined in the encoder if available, otherwise default to 518 (VGGT standard)
        target_size = getattr(self.encoder, "img_size", 518)
        
        for img in selected_imgs:
            # Handle Temporal Dimension
            # If 5D: [B, T, ...] -> Take last frame [B, ...]
            # Note: T is usually dim 1.
            if img.ndim == 5: 
                img = img[:, -1] # [B, ...]
            
            # Start with [B, ...] (4D)
            # Check for Channel Last [B, H, W, C] (C=3)
            if img.shape[-1] == 3:
                img = img.permute(0, 3, 1, 2) # [B, 3, H, W]
                
            # Normalize
            if img.dtype == torch.uint8:
                img = img.to(torch.float32) / 255.0 
            
            # Resize
            # F.interpolate expects [B, C, H, W]
            if img.shape[-2:] != (target_size, target_size):
                img = F.interpolate(img, size=(target_size, target_size), mode='bilinear', align_corners=False)
                
            processed_imgs.append(img)
            
        # Stack Views: [B, V=2, C, H, W]
        # If only 1 camera found, V=1
        imgs_stacked = torch.stack(processed_imgs, dim=1)
             
        return imgs_stacked.to(device)

    def forward(self, gaussian_inputs, text_embedding=None, return_gaussian_params=False):
        """
        Processes gaussian inputs and returns embeddings.
        Input: 
            gaussian_inputs: [B, S, 3, H, W]
            text_embedding: [B, D] Optional text embedding for LGPD.
            return_gaussian_params: If True, also return decoded Gaussian parameters from VGGT.
        Returns:
            gaussian_embs: [B, N, D] - Token embeddings for World Model
            g_mask: [B, N] - Mask for tokens
            gaussian_params (optional): Dict with depth, rot, scale, opacity, sh if return_gaussian_params=True
        """
        if not self.use_gaussian or gaussian_inputs is None:
            return (None, None) if not return_gaussian_params else (None, None, None)

        with torch.no_grad():
             outputs = self.encoder(gaussian_inputs)
             # Extract all outputs from VGGT
             # outputs: depth_maps, rot_maps, scale_maps, opacity_maps, sh_maps, aggregated_tokens_list, patch_start_idx
             depth_maps = outputs[0]  # [B, S, H, W, 1]
             rot_maps = outputs[1]     # [B, S, H, W, 4]
             scale_maps = outputs[2]  # [B, S, H, W, 3]
             opacity_maps = outputs[3] # [B, S, H, W, 1]
             sh_maps = outputs[4]      # [B, S, H, W, K, 3] where K = (sh_degree+1)^2
             aggregated_tokens_list = outputs[-2]
             patch_start_idx = outputs[-1]

             if isinstance(aggregated_tokens_list, (list, tuple)):
                 raw_tokens = aggregated_tokens_list[-1]
             else:
                 raw_tokens = aggregated_tokens_list
             
        if raw_tokens is None:
            logging.warning("No features extracted from Gaussian Encoder.")
            return (None, None) if not return_gaussian_params else (None, None, None)
        
        # Store decoded Gaussian parameters if requested
        gaussian_params_dict = None
        if return_gaussian_params:
            gaussian_params_dict = {
                "depth_maps": depth_maps,      # [B, S, H, W, 1]
                "rot_maps": rot_maps,          # [B, S, H, W, 4]
                "scale_maps": scale_maps,      # [B, S, H, W, 3]
                "opacity_maps": opacity_maps,  # [B, S, H, W, 1]
                "sh_maps": sh_maps,            # [B, S, H, W, K, 3]
            }
            
        # raw_tokens shape: [B, S, N_total, D]
        # Remove register tokens
        if patch_start_idx > 0:
            raw_tokens = raw_tokens[:, :, patch_start_idx:, :]
            
        # raw_tokens shape: [B, S, N_patches, D]
        # e.g., [B, 2, 1369, 1024]
        B, S, N, D = raw_tokens.shape

        
        # Reshape to spatial for pooling
        # N = 37*37 = 1369 (assuming 518/14)
        H_feat = int(N**0.5) 
        
        # [B, S, N, D] -> [B*S, D, H_feat, W_feat]
        tokens_spatial = raw_tokens.view(B*S, H_feat, H_feat, D).permute(0, 3, 1, 2)
        
        # Pool
        # [B*S, D, 10, 10]
        pooled = self.pool(tokens_spatial)
        
        # Flatten back
        # [B, S, D, 100] -> [B, S, 100, D]
        tokens_pooled = pooled.flatten(2).transpose(1, 2).view(B, -1, D)
        
        # Project [B, S*100, D] -> [B, TotalTokens, ProjDataset]
        gaussian_embs = self.proj(tokens_pooled)
        
        # Prepare masks (Default Uniform)
        g_bs = gaussian_embs.shape[0]
        g_len = gaussian_embs.shape[1]
        
        # === Apply LGPD ===
        # zijian 0126: Refine tokens based on language if available
        if self.use_lgpd and self.lgpd is not None and text_embedding is not None:
            # LGPD returns refined tokens. 
            # We can also get the gate for visualization if needed, but here we just update tokens.
            # gaussian_embs: [B, N, D]
            # text_embedding: [B, D]
            
            # Ensure dims match (proj might have changed expected dim)
            # LGPD inited with action_expert_width.
            # text_embedding might need projection if it comes from PaliGemma (2048) -> LGPD expects same dim.
            
            gaussian_embs, gate = self.lgpd(gaussian_embs, text_embedding, return_gate=True)
            
            # Use gate to create a soft attention mask or keep specific mask logic?
            # Standard Pi0 uses binary mask for valid/padding.
            # 3DGS tokens are all "valid" (not padding).
            # But we can use the gate to suppress attention in the Main Transformer later if we wanted to?
            # For now, we trust LGPD filtered the features themselves, so mask remains all 1s (all valid).
            g_mask = torch.ones(g_bs, g_len, dtype=torch.bool, device=gaussian_embs.device)
            
            # Optional: Return gate for visualization upstream? 
            # Currently forward only returns (embs, mask).
        else:
            g_mask = torch.ones(g_bs, g_len, dtype=torch.bool, device=gaussian_embs.device)

        if return_gaussian_params:
            return gaussian_embs, g_mask, gaussian_params_dict
        else:
            return gaussian_embs, g_mask
