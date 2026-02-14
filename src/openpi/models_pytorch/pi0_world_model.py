# zijian
# date 2026.01.24
# v2 refactored: GaussianDecoder — lightweight decode-only module
# Todo zijian 2026.0123 ：to fix linear attach anything ？？
import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
from einops import rearrange


class _UpsampleBlock(nn.Module):
    """ConvTranspose upsample block with GroupNorm + GELU + residual."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        # Residual projection when channels change
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = F.interpolate(self.skip(x), scale_factor=2, mode="bilinear", align_corners=False)
        h = F.gelu(self.norm1(self.up(x)))
        h = F.gelu(self.norm2(self.conv(h)))
        return h + skip


class _FeatureFusionBlock(nn.Module):
    """DPT-style feature fusion block for multi-scale feature integration."""
    
    def __init__(self, features: int, has_residual: bool = True):
        super().__init__()
        self.has_residual = has_residual
        
        if has_residual:
            self.residual_conv = nn.Sequential(
                nn.Conv2d(features, features, 3, padding=1, bias=True),
                nn.GroupNorm(min(32, features), features),
                nn.GELU(),
                nn.Conv2d(features, features, 3, padding=1, bias=True),
            )
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1, bias=True),
            nn.GroupNorm(min(32, features), features),
            nn.GELU(),
            nn.Conv2d(features, features, 1, bias=True),
        )
    
    def forward(self, x, residual=None, size=None):
        """
        Args:
            x: Main feature [B, C, H, W]
            residual: Optional residual feature [B, C, H_res, W_res]
            size: Target size (H, W) for upsampling
        """
        if self.has_residual and residual is not None:
            # Upsample residual to match x if needed
            if residual.shape[2:] != x.shape[2:]:
                residual = F.interpolate(residual, size=x.shape[2:], mode='bilinear', align_corners=True)
            x = x + self.residual_conv(residual)
        
        x = self.fusion_conv(x)
        
        # Upsample to target size if specified
        if size is not None and x.shape[2:] != size:
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        
        return x


class IndependentGaussianHead(nn.Module):
    """
    Decodes VLM future tokens [B, 256, D] directly to dense Gaussian parameter maps.
    
    Architecture: DPT-style multi-scale feature fusion + image feature fusion.
    - Multi-scale features: 16×16 → 32×32 → 64×64 → 128×128
    - Image feature fusion: Residual connection with input image features
    Output: rot(4) + scale(3) + opacity(1) + RGB(3) + depth_delta(1) = 12 channels
    """

    def __init__(self, token_dim: int = 2048, grid_size: int = 16, 
                 use_image_fusion: bool = True, img_dim: int = 3):
        super().__init__()
        self.grid_size = grid_size
        self.use_image_fusion = use_image_fusion
        out_ch = 4 + 3 + 1 + 3 + 1  # rot + scale + opacity + RGB + depth_delta
        
        # Multi-scale feature extraction (DPT-style)
        # Layer 1: 16×16 → 32×32
        self.layer1 = _UpsampleBlock(token_dim, 512)
        
        # Layer 2: 32×32 → 64×64
        self.layer2 = _UpsampleBlock(512, 256)
        
        # Layer 3: 64×64 → 128×128
        self.layer3 = _UpsampleBlock(256, 128)
        
        # DPT-style feature fusion blocks
        # All fusion blocks output 128 channels for consistency
        self.fusion1 = _FeatureFusionBlock(128, has_residual=False)  # Final layer
        # Project residual features to 128 channels before fusion
        self.fusion2_proj = nn.Conv2d(256, 128, 1)  # Project layer2 (256) → 128
        self.fusion2 = _FeatureFusionBlock(128, has_residual=True)   # Fuse layer2
        self.fusion3_proj = nn.Conv2d(512, 128, 1)  # Project layer1 (512) → 128
        self.fusion3 = _FeatureFusionBlock(128, has_residual=True)   # Fuse layer1
        
        # Image feature fusion (if enabled)
        if use_image_fusion:
            self.img_merger = nn.Sequential(
                nn.Conv2d(img_dim, 128, 7, padding=3),
                nn.GELU(),
            )
        
        # Final projection to output channels
        self.head = nn.Conv2d(128, out_ch, 3, padding=1)

        # Initialize output bias for neutral starting point
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, tokens: torch.Tensor, images: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            tokens: [B, 256, D] VLM future tokens
            images: [B, 3, H, W] Optional input images for feature fusion
        Returns:
            [B, 12, 128, 128] raw Gaussian parameter maps
        """
        B, N, D = tokens.shape
        g = self.grid_size
        x = tokens.permute(0, 2, 1).reshape(B, D, g, g)  # [B, D, 16, 16]
        
        # Multi-scale feature extraction
        feat1 = self.layer1(x)      # [B, 512, 32, 32]
        feat2 = self.layer2(feat1)  # [B, 256, 64, 64]
        feat3 = self.layer3(feat2)  # [B, 128, 128, 128]
        
        # DPT-style feature fusion (bottom-up)
        # Start from deepest layer and fuse with shallower layers
        fused = self.fusion1(feat3)                    # [B, 128, 128, 128]
        # Fuse with layer2: project to 128 channels first
        feat2_proj = self.fusion2_proj(feat2)         # [B, 256, 64, 64] → [B, 128, 64, 64]
        fused = self.fusion2(fused, residual=feat2_proj)  # [B, 128, 128, 128]
        # Fuse with layer1: project to 128 channels first
        feat1_proj = self.fusion3_proj(feat1)         # [B, 512, 32, 32] → [B, 128, 32, 32]
        fused = self.fusion3(fused, residual=feat1_proj)  # [B, 128, 128, 128]
        
        # Image feature fusion (residual connection)
        if self.use_image_fusion and images is not None:
            # Ensure images are at correct resolution
            if images.shape[2:] != fused.shape[2:]:
                images = F.interpolate(images, size=fused.shape[2:], mode='bilinear', align_corners=True)
            img_feat = self.img_merger(images)  # [B, 128, 128, 128]
            fused = fused + img_feat  # Residual connection
        
        # Final projection
        x = self.head(fused)  # [B, 12, 128, 128]
        return x


class GaussianDecoder(nn.Module):
    """
    Lightweight decoder that converts VLM-predicted latent tokens to 3D Gaussian parameters.

    Supports two modes:
      - "vggt_dpt": inject tokens into VGGT DPT decoder (legacy)
      - "independent": decode directly via ConvNet (no VGGT DPT dependency)
    """

    def __init__(
        self,
        token_dim: int,
        use_vggt_decoder: bool = False,
        vggt_decoder=None,
        input_num_tokens: int = 256,
        target_num_tokens: int = 1369,
        vggt_embed_dim: int = 1024,
        upsample_bottleneck: int = 128,
        decode_mode: str = "vggt_dpt",
        d_sh: int = 25,
    ):
        super().__init__()
        self.use_vggt_decoder = use_vggt_decoder
        self.vggt_decoder = vggt_decoder
        self.input_num_tokens = input_num_tokens
        self.target_num_tokens = target_num_tokens
        self.token_dim = token_dim
        self.vggt_embed_dim = vggt_embed_dim
        self.decode_mode = decode_mode
        self.d_sh = d_sh

        if decode_mode == "independent":
            # Independent ConvNet decoder — RGB output, no VGGT DPT dependency
            grid_size = int(input_num_tokens ** 0.5)  # 256 → 16
            self.gaussian_head = IndependentGaussianHead(
                token_dim=token_dim, grid_size=grid_size,
                use_image_fusion=True,  # Enable image feature fusion
                img_dim=3,
            )
            self.token_proj = None
            self.upsample_refine = None
        else:
            # Legacy: inject into VGGT DPT
            if use_vggt_decoder and vggt_decoder is not None:
                target_dim = 2 * vggt_embed_dim  # 2048
                self.token_proj = nn.Linear(token_dim, target_dim)
            else:
                self.token_proj = None
            self.upsample_refine = nn.Sequential(
                nn.Conv2d(token_dim, upsample_bottleneck, 1),
                nn.GELU(),
                nn.Conv2d(upsample_bottleneck, upsample_bottleneck, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(upsample_bottleneck, token_dim, 1),
            )
            self.gaussian_head = None

    # ------------------------------------------------------------------
    # Token upsampling  100 → 1369
    # ------------------------------------------------------------------
    def _upsample_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Upsample [B, 100, D] → [B, 1369, D] via bilinear interpolation + learnable refinement."""
        B, N, D = tokens.shape
        if N != self.input_num_tokens:
            return tokens
        # [B, 100, D] → [B, D, 10, 10]
        tokens_2d = tokens.permute(0, 2, 1).reshape(B, D, 10, 10)
        # bilinear upsample → [B, D, 37, 37]
        tokens_2d = F.interpolate(tokens_2d, size=(37, 37), mode="bilinear", align_corners=False)
        # learnable refinement (residual) to smooth grid artifacts
        tokens_2d = tokens_2d + self.upsample_refine(tokens_2d)
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
        current_observation=None,
    ):
        """Decode latent tokens → Gaussian parameters. Routes based on decode_mode."""
        if self.decode_mode == "independent":
            return self._decode_independent(
                z, gaussian_adapter=gaussian_adapter, camera_params=camera_params,
                current_observation=current_observation, future_observation=future_observation,
                step=step,
            )
        return self._decode_vggt_dpt(
            z, future_observation=future_observation, gaussian_adapter=gaussian_adapter,
            camera_params=camera_params, return_2d_maps=return_2d_maps, step=step,
            current_observation=current_observation,
        )

    # ------------------------------------------------------------------
    # Independent ConvNet decoder (RGB output)
    # ------------------------------------------------------------------
    def _decode_independent(
        self, z, gaussian_adapter=None, camera_params=None,
        current_observation=None, future_observation=None, step=None,
    ):
        """Decode VLM tokens via independent ConvNet head + VGGT depth."""
        vggt_obs = current_observation if current_observation is not None else future_observation
        if vggt_obs is None or gaussian_adapter is None:
            raise ValueError("current_observation and gaussian_adapter are required")

        # 1. Prepare VGGT inputs from current frames
        vggt_inputs = gaussian_adapter.prepare_inputs(
            vggt_obs, z.device, z.shape[0], is_training=False
        )
        if vggt_inputs is None:
            raise ValueError("Failed to prepare VGGT inputs")

        # 2. VGGT aggregator → unmodified features (for depth only)
        with torch.no_grad():
            aggregated_tokens_list, patch_start_idx = gaussian_adapter.encoder.aggregator(
                vggt_inputs.to(torch.bfloat16)
            )
        B_vggt, S_vggt = vggt_inputs.shape[:2]
        for i in range(len(aggregated_tokens_list)):
            if aggregated_tokens_list[i].ndim == 3:
                _bs, _p, _c = aggregated_tokens_list[i].shape
                aggregated_tokens_list[i] = aggregated_tokens_list[i].view(B_vggt, S_vggt, _p, _c)

        # 3. Depth from unmodified VGGT features (trainable depth head)
        # Use AD-FFgsStudio style depth normalization: sigmoid(log(depth)) for stability
        depth_maps, _ = gaussian_adapter.encoder.depth_head(
            aggregated_tokens_list, images=vggt_inputs, patch_start_idx=patch_start_idx
        )
        # AD-FFgsStudio style: sigmoid(log(depth)) ensures positive depth values
        depth_maps = torch.sigmoid(torch.log(depth_maps))
        min_depth = gaussian_adapter.encoder.min_depth
        max_depth = gaussian_adapter.encoder.max_depth
        depth_maps = min_depth + (max_depth - min_depth) * depth_maps  # [B, S, H, W, 1]

        # 4. Independent decoder: VLM tokens → Gaussian params (RGB) + depth_delta
        # Extract current frame image for feature fusion
        frame_idx = S_vggt - 1
        current_frame_img = vggt_inputs[:, frame_idx]  # [B, 3, H_vggt, W_vggt]
        # Resize to decoder resolution if needed (will be handled inside gaussian_head)
        raw = self.gaussian_head(z, images=current_frame_img)  # [B, 12, 128, 128]
        rot_raw, scale_raw, opa_raw, rgb_raw, depth_delta = raw.split(
            [4, 3, 1, 3, 1], dim=1
        )

        # 5. Combine depth: base (VGGT, t-time) + delta (predicted motion)
        frame_idx = S_vggt - 1
        base_depth = depth_maps[:, frame_idx, :, :, 0]  # [B, H_vggt, W_vggt]
        H_dec, W_dec = raw.shape[2], raw.shape[3]
        base_depth_resized = F.interpolate(
            base_depth.unsqueeze(1), size=(H_dec, W_dec), mode="bilinear", align_corners=False
        ).squeeze(1)  # [B, H_dec, W_dec]
        depth_delta_2d = depth_delta.squeeze(1)  # [B, H_dec, W_dec]
        final_depth = base_depth_resized + depth_delta_2d

        # 6. Activations
        B = z.shape[0]
        rot_maps = rot_raw.permute(0, 2, 3, 1)  # [B, H, W, 4]
        rot_maps = rot_maps / (rot_maps.norm(dim=-1, keepdim=True) + 1e-8)
        scale_maps = F.softplus(scale_raw.permute(0, 2, 3, 1), beta=1)  # [B, H, W, 3]
        opacity_maps = torch.sigmoid(opa_raw.permute(0, 2, 3, 1))  # [B, H, W, 1]
        # RGB: Directly predict SH DC coefficients (more stable than RGB → SH DC conversion)
        # Renderer computes: color = SH_C0 * sh_dc + 0.5
        # We predict sh_dc directly, which allows for both positive (bright) and negative (dark) values
        # But we initialize to output neutral gray (sh_dc ≈ 0) by default
        SH_C0 = 0.28209479177387814
        # Option 1: Direct SH DC prediction (tanh → [-1, 1] range, then scale)
        # This gives us control over brightness: sh_dc = 0 → color = 0.5 (neutral gray)
        sh_dc = torch.tanh(rgb_raw.permute(0, 2, 3, 1)) * 2.0  # [B, H, W, 3], range [-2, 2]
        # Clamp to reasonable range to avoid extreme colors
        sh_dc = torch.clamp(sh_dc, min=-1.5, max=1.5)
        
        # For visualization/debugging: convert back to RGB
        rgb_maps = SH_C0 * sh_dc + 0.5  # [B, H, W, 3], range [0, 1]
        rgb_maps = torch.clamp(rgb_maps, min=0.0, max=1.0)

        # 7. Depth → xyz
        # LIBERO original camera: 256×256, fx=fy=221.7025, cx=cy=128.0
        # Scale intrinsics to match decoder output resolution (H_dec × W_dec)
        if camera_params is not None and "fx" in camera_params:
            xyz = self.depth2pc(
                final_depth,
                fx=camera_params["fx"], fy=camera_params["fy"],
                cx=camera_params["cx"], cy=camera_params["cy"],
                downsample_factor=1,
            )
        else:
            # Scale LIBERO intrinsics from 256×256 to decoder output resolution
            decoder_scale = H_dec / 256.0  # 128/256 = 0.5
            xyz = self.depth2pc(
                final_depth,
                fx=221.7025 * decoder_scale, fy=221.7025 * decoder_scale,
                cx=128.0 * decoder_scale, cy=128.0 * decoder_scale,
                downsample_factor=1,
            )

        # 8. Flatten and sanitize
        N = H_dec * W_dec
        rot_flat = rot_maps.reshape(B, N, 4)
        scale_flat = scale_maps.reshape(B, N, 3)
        opacity_flat = opacity_maps.reshape(B, N, 1)
        sh_flat = sh_dc.reshape(B, N, 3)  # [B, N, 3] — DC-only SH
        
        # Note: Currently using SH DC only (degree 0). If future support for higher-order SH:
        # - sh_flat would be [B, N, K*3] where K = (sh_degree+1)^2
        # - Apply SH mask: sh_mask = [1.0, 0.1*0.25^1, 0.1*0.25^2, ...] for degrees 0,1,2,...
        # - sh_flat = sh_flat * sh_mask.view(1, 1, -1)  # Attenuate higher-order coefficients

        scale_flat = torch.clamp(scale_flat, min=1e-7, max=10.0)
        xyz = torch.clamp(xyz, min=-100.0, max=100.0)
        xyz = torch.where(torch.isnan(xyz) | torch.isinf(xyz), torch.zeros_like(xyz), xyz)

        if step is not None and step % 40 == 0:
            import logging
            logging.info(f"[IndependentDecoder] Step {step}: depth_delta=[{depth_delta_2d.min():.3f}, {depth_delta_2d.max():.3f}], "
                         f"scales=[{scale_flat.min():.3f}, {scale_flat.max():.3f}], rgb=[{rgb_maps.min():.3f}, {rgb_maps.max():.3f}], N={N}")

        return {
            "xyz": xyz,
            "scales": scale_flat,
            "opacity": opacity_flat,
            "sh": sh_flat,
            "rotations": rot_flat,
            "depth_map": final_depth.unsqueeze(1),  # [B, 1, H, W] for edge-aware smoothness
        }

    # ------------------------------------------------------------------
    # Legacy VGGT DPT decode path
    # ------------------------------------------------------------------
    def _decode_vggt_dpt(
        self, z, future_observation=None, gaussian_adapter=None,
        camera_params=None, return_2d_maps=False, step=None, current_observation=None,
    ):
        if not (self.use_vggt_decoder and self.vggt_decoder is not None):
            raise ValueError(
                "GaussianDecoder requires use_vggt_decoder=True and a valid vggt_decoder."
            )
        # Use current observation (no future leak); fall back to future_observation for compat
        vggt_obs = current_observation if current_observation is not None else future_observation
        if vggt_obs is None or gaussian_adapter is None:
            raise ValueError(
                "current_observation (or future_observation) and gaussian_adapter are required"
            )

        # 1. Prepare VGGT inputs from current observation (excludes future frame)
        #    prepare_inputs selects [t-2, t-1, t] when given [B, T, H, W, C]
        vggt_inputs = gaussian_adapter.prepare_inputs(
            vggt_obs, z.device, z.shape[0], is_training=False
        )
        if vggt_inputs is None:
            raise ValueError("Failed to prepare VGGT inputs from observation")

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
        # softplus output ~0.7 → 2D std ≈ 0.7*194/50 ≈ 2.7 px per Gaussian.
        # 37×37 grid spacing ≈ 6 px, so ~2.7 px std gives good coverage without
        # excessive overlap. Clamp max=10.0 downstream ensures gradient flow.
        scale_maps = F.softplus(scale_maps, beta=1)
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
        scale_flat = torch.clamp(scale_flat, min=1e-7, max=10.0)
        xyz = torch.clamp(xyz, min=-100.0, max=100.0)
        xyz = torch.where(torch.isnan(xyz) | torch.isinf(xyz), torch.zeros_like(xyz), xyz)

        # Pass scales and rotations directly to renderer (like gsplat reference).
        # Previous code built a diagonal sigma and then decomposed it back,
        # which discarded the rotation quaternions entirely.
        return {
            "xyz": xyz,            # [B, N, 3]
            "scales": scale_flat,  # [B, N, 3]  — direct scales for rasterizer
            "opacity": opacity_flat,  # [B, N, 1]
            "sh": sh_flat,         # [B, N, K*3]
            "rotations": rot_flat, # [B, N, 4]  — quaternions for rasterizer
        }
