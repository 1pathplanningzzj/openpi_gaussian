# zijian
# date 2026.01.24
# v2 refactored: GaussianDecoder — lightweight decode-only module
# Todo zijian 2026.0123 ：to fix linear attach anything ？？
import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
from einops import rearrange


class GaussianDecoder(nn.Module):
    """
    Lightweight decoder that converts VLM-predicted latent tokens to 3D Gaussian parameters.

    Pipeline:
        z_t1_pred [B, 100, D]
        → upsample to [B, 1369, D]
        → project to VGGT token dim (2 * embed_dim)
        → replace last-layer patch tokens in VGGT aggregated_tokens_list
        → run VGGT gs_head → raw Gaussian maps [B, S, H, W, C]
        → run VGGT depth_head → depth maps [B, S, H, W, 1]
        → depth2pc with real camera intrinsics → xyz [B, N, 3]
        → pack (xyz, sigma, opacity, sh)
    """

    def __init__(
        self,
        token_dim: int,
        use_vggt_decoder: bool = False,
        vggt_decoder=None,
        input_num_tokens: int = 100,
        target_num_tokens: int = 1369,
        vggt_embed_dim: int = 1024,
    ):
        super().__init__()
        self.use_vggt_decoder = use_vggt_decoder
        self.vggt_decoder = vggt_decoder
        self.input_num_tokens = input_num_tokens
        self.target_num_tokens = target_num_tokens
        self.token_dim = token_dim
        self.vggt_embed_dim = vggt_embed_dim

        # Projection: VLM width → 2 * vggt_embed_dim (aggregated_tokens_list dim)
        if use_vggt_decoder and vggt_decoder is not None:
            target_dim = 2 * vggt_embed_dim  # 2048
            self.token_proj = nn.Linear(token_dim, target_dim)
        else:
            self.token_proj = None

    # ------------------------------------------------------------------
    # Token upsampling  100 → 1369
    # ------------------------------------------------------------------
    def _upsample_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Upsample [B, 100, D] → [B, 1369, D] via bilinear interpolation on a 10×10 grid."""
        B, N, D = tokens.shape
        if N != self.input_num_tokens:
            return tokens
        # [B, 100, D] → [B, D, 10, 10]  (permute first to keep spatial structure correct)
        tokens_2d = tokens.permute(0, 2, 1).reshape(B, D, 10, 10)
        # bilinear upsample → [B, D, 37, 37]
        tokens_2d = F.interpolate(tokens_2d, size=(37, 37), mode="bilinear", align_corners=False)
        # → [B, 1369, D]
        return tokens_2d.permute(0, 2, 3, 1).reshape(B, 37 * 37, D)

    # ------------------------------------------------------------------
    # depth2pc — real-camera unprojection  (adapted from AD-FFgsStudio)
    # ------------------------------------------------------------------
    @staticmethod
    def depth2pc(
        depth: torch.Tensor,
        fx: float, fy: float, cx: float, cy: float,
        downsample_factor: int = 1,
    ) -> torch.Tensor:
        """
        Unproject depth map to 3D points using real camera intrinsics.

        Args:
            depth: [B, H, W] depth values
            fx, fy, cx, cy: camera intrinsic parameters
            downsample_factor: if depth was downsampled, scale intrinsics accordingly
        Returns:
            xyz: [B, H*W, 3] camera-space 3D points
        """
        B, H, W = depth.shape
        device, dtype = depth.device, depth.dtype

        # Scale intrinsics if downsampled
        fx_s = fx / downsample_factor
        fy_s = fy / downsample_factor
        cx_s = cx / downsample_factor
        cy_s = cy / downsample_factor

        # Pixel grid with half-pixel offset (matching AD-FFgsStudio convention)
        u = torch.arange(0.5, W + 0.5, device=device, dtype=dtype)  # [W]
        v = torch.arange(0.5, H + 0.5, device=device, dtype=dtype)  # [H]
        v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")  # [H, W] each

        # Camera-space coordinates: x = (u - cx) * depth / fx
        x = ((u_grid[None] - cx_s) * depth) / fx_s  # [B, H, W]
        y = ((v_grid[None] - cy_s) * depth) / fy_s  # [B, H, W]
        z = depth  # [B, H, W]

        xyz = torch.stack([x, y, z], dim=-1)  # [B, H, W, 3]
        return xyz.reshape(B, H * W, 3)

    # ------------------------------------------------------------------
    # Main decode entry point
    # ------------------------------------------------------------------
    def decode(
        self,
        z: torch.Tensor,
        future_observation=None,
        gaussian_adapter=None,
        camera_params=None,
        return_2d_maps: bool = False,
        step=None,
    ):
        """
        Decode latent tokens → Gaussian parameters.

        Args:
            z: [B, N, D] predicted latent tokens from VLM future query
            future_observation: observation with images for VGGT encoder context
            gaussian_adapter: GaussianAdapter instance (provides VGGT encoder)
            camera_params: dict with 'fx', 'fy', 'cx', 'cy' (real intrinsics)
            return_2d_maps: if True, return raw 2D maps before 3D conversion
            step: training step (for periodic logging)
        """
        if not (self.use_vggt_decoder and self.vggt_decoder is not None):
            raise ValueError(
                "GaussianDecoder requires use_vggt_decoder=True and a valid vggt_decoder."
            )
        if future_observation is None or gaussian_adapter is None:
            raise ValueError(
                "future_observation and gaussian_adapter are required for VGGT decoder"
            )

        # 1. Prepare VGGT inputs from future observation
        vggt_inputs = gaussian_adapter.prepare_inputs(
            future_observation, z.device, z.shape[0], is_training=False
        )
        if vggt_inputs is None:
            raise ValueError("Failed to prepare VGGT inputs from future_observation")

        # 2. Run VGGT aggregator to get multi-layer tokens
        with torch.no_grad():
            aggregated_tokens_list, patch_start_idx = gaussian_adapter.encoder.aggregator(
                vggt_inputs.to(torch.bfloat16)
            )

        # Ensure all tokens are [B, S, P, C] (4D)
        B_vggt, S_vggt = vggt_inputs.shape[:2]
        for i in range(len(aggregated_tokens_list)):
            if aggregated_tokens_list[i].ndim == 3:
                _bs, _p, _c = aggregated_tokens_list[i].shape
                aggregated_tokens_list[i] = aggregated_tokens_list[i].view(B_vggt, S_vggt, _p, _c)

        # 3. Upsample predicted tokens 100 → 1369
        z_up = self._upsample_tokens(z)  # [B, 1369, D]

        # 4. Replace last-layer patch tokens with predicted tokens
        self._replace_patch_tokens(aggregated_tokens_list, z_up, patch_start_idx, step)

        # 5. Run VGGT gs_head decoder → raw Gaussian maps
        raw_gaussian = self.vggt_decoder(
            aggregated_tokens_list, images=vggt_inputs, patch_start_idx=patch_start_idx
        )  # [B, S, H, W, C]

        # 6. Run VGGT depth_head → depth maps
        with torch.no_grad():
            depth_maps, _ = gaussian_adapter.encoder.depth_head(
                aggregated_tokens_list, images=vggt_inputs, patch_start_idx=patch_start_idx
            )
            depth_maps = torch.sigmoid(torch.log(depth_maps))
            min_depth = gaussian_adapter.encoder.min_depth
            max_depth = gaussian_adapter.encoder.max_depth
            depth_maps = min_depth + (max_depth - min_depth) * depth_maps  # [B, S, H, W, 1]

        # 7. Parse raw Gaussian output
        d_sh = gaussian_adapter.encoder.d_sh
        rot_maps, scale_maps, opacity_maps, sh_maps = raw_gaussian.split(
            (4, 3, 1, 3 * d_sh), dim=-1
        )

        # Process maps — each activation applied exactly ONCE
        rot_maps = rot_maps / (rot_maps.norm(dim=-1, keepdim=True) + 1e-8)
        scale_maps = F.softplus(scale_maps, beta=1) * 0.001  # AD-FFgsStudio convention
        opacity_maps = torch.sigmoid(opacity_maps)
        sh_maps = rearrange(sh_maps, "b s h w (i c) -> b s h w i c", i=3, c=d_sh)

        # Apply SH mask (higher-order attenuation)
        if hasattr(gaussian_adapter.encoder, "sh_mask"):
            sh_mask = gaussian_adapter.encoder.sh_mask  # [d_sh]
            sh_maps = sh_maps * sh_mask.view(1, 1, 1, 1, 1, -1)

        if return_2d_maps:
            return {
                "rot_maps": rot_maps,
                "scale_maps": scale_maps,
                "opacity_maps": opacity_maps,
                "sh_maps": sh_maps,
                "depth_maps": depth_maps,
                "is_2d_maps": True,
            }

        # 8. Convert 2D maps → 3D Gaussians
        return self._convert_2d_maps_to_3d_gaussians(
            depth_maps, rot_maps, scale_maps, opacity_maps, sh_maps,
            camera_params=camera_params,
            downsample_factor=4,
            step=step,
        )

    # ------------------------------------------------------------------
    # Replace patch tokens in aggregated_tokens_list
    # ------------------------------------------------------------------
    def _replace_patch_tokens(self, aggregated_tokens_list, z_up, patch_start_idx, step=None):
        """Replace last-layer patch tokens with world-model predicted tokens."""
        if not aggregated_tokens_list:
            return

        last = aggregated_tokens_list[-1]
        is_4d = last.ndim == 4

        # Flatten to [B*S, P, D] for uniform handling
        if is_4d:
            B_t, S_t, P, D = last.shape
            last = last.reshape(B_t * S_t, P, D)
        elif last.ndim == 3:
            B_t, S_t = None, None
            _, P, D = last.shape
        else:
            warnings.warn(f"Unexpected last_layer_tokens shape: {last.shape}. Skipping replacement.")
            return

        B_S = last.shape[0]
        B = z_up.shape[0]
        S = B_S // B
        if S == 0 or B_S % B != 0:
            warnings.warn(f"Shape mismatch: B_S={B_S}, B={B}. Skipping replacement.")
            return

        # Expand z_up to match sequence dim
        if S > 1:
            z_exp = z_up.unsqueeze(1).expand(B, S, -1, -1).reshape(B * S, -1, z_up.shape[-1])
        else:
            z_exp = z_up

        # Project to VGGT dim
        z_proj = self.token_proj(z_exp) if self.token_proj is not None else z_exp

        expected = P - patch_start_idx
        actual = z_proj.shape[1]

        if actual != expected:
            warnings.warn(
                f"Token count mismatch: predicted {actual}, expected {expected}. Skipping replacement."
            )
            return
        if z_proj.shape[-1] != D:
            warnings.warn(
                f"Dim mismatch: projected {z_proj.shape[-1]} vs aggregated {D}. Skipping replacement."
            )
            return

        new_last = torch.cat([last[:, :patch_start_idx], z_proj], dim=1)

        if is_4d:
            aggregated_tokens_list[-1] = new_last.view(B_t, S_t, P, D)
        else:
            aggregated_tokens_list[-1] = new_last

    # ------------------------------------------------------------------
    # 2D maps → 3D Gaussian point cloud
    # ------------------------------------------------------------------
    def _convert_2d_maps_to_3d_gaussians(
        self,
        depth_maps,      # [B, S, H, W, 1]
        rot_maps,        # [B, S, H, W, 4]
        scale_maps,      # [B, S, H, W, 3]  (already activated)
        opacity_maps,    # [B, S, H, W, 1]  (already activated)
        sh_maps,         # [B, S, H, W, 3, d_sh]
        camera_params=None,
        downsample_factor: int = 4,
        step=None,
    ):
        B, S, H, W, _ = depth_maps.shape
        device, dtype = depth_maps.device, depth_maps.dtype
        frame_idx = S - 1  # use last frame

        # Extract single frame
        depth = depth_maps[:, frame_idx, :, :, 0]   # [B, H, W]
        rot = rot_maps[:, frame_idx]                 # [B, H, W, 4]
        scale = scale_maps[:, frame_idx]             # [B, H, W, 3]
        opacity = opacity_maps[:, frame_idx, :, :, 0]  # [B, H, W]
        sh = sh_maps[:, frame_idx]                   # [B, H, W, 3, d_sh]

        # Optional downsample
        if downsample_factor > 1:
            H_ds, W_ds = H // downsample_factor, W // downsample_factor
            depth = F.interpolate(depth.unsqueeze(1), (H_ds, W_ds), mode="bilinear", align_corners=False).squeeze(1)
            rot = F.interpolate(rot.permute(0, 3, 1, 2), (H_ds, W_ds), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
            scale = F.interpolate(scale.permute(0, 3, 1, 2), (H_ds, W_ds), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
            opacity = F.interpolate(opacity.unsqueeze(1), (H_ds, W_ds), mode="bilinear", align_corners=False).squeeze(1)
            K_sh, C_sh = sh.shape[-2], sh.shape[-1]
            sh_flat = sh.permute(0, 3, 4, 1, 2).reshape(B, K_sh * C_sh, H, W)
            sh_flat = F.interpolate(sh_flat, (H_ds, W_ds), mode="bilinear", align_corners=False)
            sh = sh_flat.reshape(B, K_sh, C_sh, H_ds, W_ds).permute(0, 3, 4, 1, 2)
            H, W = H_ds, W_ds

        # Re-normalize rotation after interpolation
        rot = rot / (rot.norm(dim=-1, keepdim=True) + 1e-8)

        # Depth → 3D positions using real camera intrinsics
        if camera_params is not None and "fx" in camera_params:
            xyz = self.depth2pc(
                depth,
                fx=camera_params["fx"], fy=camera_params["fy"],
                cx=camera_params["cx"], cy=camera_params["cy"],
                downsample_factor=downsample_factor,
            )
        else:
            # Fallback: LIBERO default intrinsics (fx=fy=221.7025, cx=cy=128, 256x256 → VGGT 518x518)
            # VGGT resizes to 518x518, so scale intrinsics: factor = 518/256 ≈ 2.0234
            vggt_scale = 518.0 / 256.0
            xyz = self.depth2pc(
                depth,
                fx=221.7025 * vggt_scale, fy=221.7025 * vggt_scale,
                cx=128.0 * vggt_scale, cy=128.0 * vggt_scale,
                downsample_factor=downsample_factor,
            )

        # Flatten all params
        N = H * W
        rot_flat = rot.reshape(B, N, 4)
        scale_flat = scale.reshape(B, N, 3)
        opacity_flat = opacity.reshape(B, N, 1)
        sh_flat = sh.reshape(B, N, -1)

        # Sanitize
        scale_flat = torch.clamp(scale_flat, min=1e-7, max=1.0)
        xyz = torch.clamp(xyz, min=-100.0, max=100.0)
        xyz = torch.where(torch.isnan(xyz) | torch.isinf(xyz), torch.zeros_like(xyz), xyz)

        # Build 6D covariance from scale (diagonal, no rotation coupling for now)
        sigma = torch.zeros(B, N, 6, device=device, dtype=dtype)
        sigma[:, :, 0] = scale_flat[:, :, 0] ** 2  # s11
        sigma[:, :, 3] = scale_flat[:, :, 1] ** 2  # s22
        sigma[:, :, 5] = scale_flat[:, :, 2] ** 2  # s33

        return {
            "xyz": xyz,            # [B, N, 3]
            "sigma": sigma,        # [B, N, 6]
            "opacity": opacity_flat,  # [B, N, 1]
            "sh": sh_flat,         # [B, N, K*3]
            "rotations": rot_flat, # [B, N, 4]
        }
