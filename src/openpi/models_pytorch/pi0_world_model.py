# zijian
# date 2026.01.24
# v2 refactored: GaussianDecoder — lightweight decode-only module
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    - Multi-scale features: 16×16 → 32×32 → 64×64 → 128×128 → 256×256
    - Image feature fusion: Residual connection with input image features
    - Output: rot(4) + scale(3) + opacity(1) + SH(9) = 17 channels
    - SH(9): DC(3) + 1st order(6) for view-dependent appearance

    Reference: AD-FFgsStudio architecture (without depth prediction)
    """

    def __init__(self, token_dim: int = 2048, grid_size: int = 16,
                 use_image_fusion: bool = True, img_dim: int = 3,
                 predict_depth: bool = True):
        super().__init__()
        self.grid_size = grid_size
        self.use_image_fusion = use_image_fusion
        self.predict_depth = predict_depth

        # Output channels: rot(4) + scale(3) + opacity(1) + SH(9) + xy_delta(2)
        # SH(9) = DC(3) + 1st order(6) for basic view-dependent effects
        # xy_delta(2) allows explicit lateral motion avoiding conflict with existing depth prediction.
        out_ch = 4 + 3 + 1 + 9 + 2

        # Multi-scale feature extraction (DPT-style)
        # Layer 1: 16×16 → 32×32
        self.layer1 = _UpsampleBlock(token_dim, 512)

        # Layer 2: 32×32 → 64×64
        self.layer2 = _UpsampleBlock(512, 256)

        # Layer 3: 64×64 → 128×128
        self.layer3 = _UpsampleBlock(256, 128)

        # Layer 4: 128×128 → 256×256 (NEW: higher resolution)
        self.layer4 = _UpsampleBlock(128, 128)

        # DPT-style feature fusion blocks (4 layers now)
        # All fusion blocks expect 128 channels for both input and residual
        self.fusion1 = _FeatureFusionBlock(128, has_residual=False)  # Final layer
        self.fusion2 = _FeatureFusionBlock(128, has_residual=True)   # Fuse layer3 (128 ch)
        self.fusion3 = _FeatureFusionBlock(128, has_residual=True)   # Fuse layer2 (256 ch)
        self.fusion4 = _FeatureFusionBlock(128, has_residual=True)   # Fuse layer1 (512 ch)

        # Projection layers to unify channel dimensions to 128
        # Required because layer2 has 256 channels and layer1 has 512 channels
        self.proj_feat2 = nn.Conv2d(256, 128, 1)  # Project layer2: 256 → 128
        self.proj_feat1 = nn.Conv2d(512, 128, 1)  # Project layer1: 512 → 128

        # Image feature fusion (if enabled)
        if use_image_fusion:
            self.img_merger = nn.Sequential(
                nn.Conv2d(img_dim, 128, 7, padding=3),
                nn.GELU(),
            )

        # Final projection to output channels
        self.head = nn.Conv2d(128, out_ch, 3, padding=1)

        # Independent depth prediction branch (方案 A - 借鉴 VGGT DPT)
        if predict_depth:
            # Lightweight refinement network for depth prediction
            self.depth_refine = nn.Sequential(
                # First refinement block
                nn.Conv2d(128, 64, 3, padding=1),
                nn.GroupNorm(min(32, 64), 64),
                nn.GELU(),
                # Second refinement block with residual
                nn.Conv2d(64, 64, 3, padding=1),
                nn.GroupNorm(min(32, 64), 64),
                nn.GELU(),
                # Final projection to depth
                nn.Conv2d(64, 1, 3, padding=1),
            )
            # Initialize depth refinement layers
            for m in self.depth_refine.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight, gain=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # Initialize output with small random values for better gradient flow
        # Using Xavier/Glorot initialization scaled down for stability
        nn.init.xavier_uniform_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)

        # SH mask for attenuating higher-order coefficients (following AD-FFgsStudio)
        # DC (degree 0): weight = 1.0
        # 1st order (degree 1): weight = 0.1 * 0.25 = 0.025
        self.register_buffer(
            "sh_mask",
            torch.tensor([1.0, 1.0, 1.0,  # DC (3 coefficients)
                         0.025, 0.025, 0.025, 0.025, 0.025, 0.025],  # 1st order (6 coefficients)
                        dtype=torch.float32),
            persistent=False,
        )

    def forward(self, tokens: torch.Tensor, images: torch.Tensor = None):
        """
        Args:
            tokens: [B, 256, D] VLM future tokens
            images: [B, 3, H, W] Optional input images for feature fusion
        Returns:
            dict with:
                - 'gaussian_params': [B, 17, 256, 256] - rot(4) + scale(3) + opacity(1) + SH(9)
                - 'depth': [B, 1, 256, 256] - predicted depth (if predict_depth=True)
        """
        B, N, D = tokens.shape
        g = self.grid_size
        x = tokens.permute(0, 2, 1).reshape(B, D, g, g)  # [B, D, 16, 16]

        # Multi-scale feature extraction (4 layers)
        feat1 = self.layer1(x)      # [B, 512, 32, 32]
        feat2 = self.layer2(feat1)  # [B, 256, 64, 64]
        feat3 = self.layer3(feat2)  # [B, 128, 128, 128]
        feat4 = self.layer4(feat3)  # [B, 128, 256, 256]

        # DPT-style feature fusion (bottom-up, 4 layers)
        # Start from deepest layer and fuse with shallower layers
        fused = self.fusion1(feat4)                              # [B, 128, 256, 256]
        fused = self.fusion2(fused, residual=feat3)              # [B, 128, 256, 256] feat3: 128 ch
        fused = self.fusion3(fused, residual=self.proj_feat2(feat2))  # [B, 128, 256, 256] feat2: 256→128
        fused = self.fusion4(fused, residual=self.proj_feat1(feat1))  # [B, 128, 256, 256] feat1: 512→128

        # Image feature fusion (residual connection)
        if self.use_image_fusion and images is not None:
            # Ensure images are at correct resolution (256×256)
            if images.shape[2:] != fused.shape[2:]:
                images = F.interpolate(images, size=fused.shape[2:], mode='bilinear', align_corners=True)
            img_feat = self.img_merger(images)  # [B, 128, 256, 256]
            fused = fused + img_feat  # Residual connection

        # Final projection to Gaussian parameters
        gaussian_params = self.head(fused)  # [B, 17, 256, 256]

        # Independent depth prediction with refinement (方案 A - 借鉴 VGGT DPT)
        result = {'gaussian_params': gaussian_params}
        if self.predict_depth:
            depth_raw = self.depth_refine(fused)  # [B, 1, 256, 256]
            result['depth'] = depth_raw

        return result


class GaussianDecoder(nn.Module):
    """
    Lightweight decoder that converts VLM-predicted latent tokens to 3D Gaussian parameters.
    Uses independent ConvNet decoder with no VGGT DPT dependency.

    Pipeline:
    --------
    VLM Token [B,256,2048]
        ↓
    IndependentGaussianHead (ConvNet)
        ├─ 16×16 → 32×32 → 64×64 → 128×128 (multi-scale)
        ├─ DPT feature fusion
        └─ Image feature fusion (current frame)
        ↓
    Raw Maps [B,12,128,128]
        ├─ rot(4) + scale(3) + opacity(1) + RGB(3) + depth_delta(1) = 12 channels
        ↓
    Transformations:
        ├─ rot(4) → normalize → rotations [B,N,4] (quaternions)
        ├─ scale(3) → softplus → scales [B,N,3]
        ├─ opacity(1) → sigmoid → opacity [B,N,1]
        ├─ RGB(3) → tanh*2.0 → sh [B,N,3] (SH DC coefficients)
        └─ depth_delta(1) + base_depth → final_depth
        ↓
    3D Unprojection (depth2pc)
        final_depth + camera_intrinsics → xyz [B,N,3]
        ↓
    Gaussian Point Cloud [B,N=16384,*]
        ├─ xyz [B,N,3]        - 3D positions
        ├─ scales [B,N,3]     - Gaussian scales
        ├─ rotations [B,N,4]  - Rotation quaternions
        ├─ opacity [B,N,1]    - Alpha values
        └─ sh [B,N,3]         - Spherical harmonics (DC only)
    """

    def __init__(
        self,
        token_dim: int,
        input_num_tokens: int = 256,
        future_input_num_tokens: int | None = None,
        action_dim: int = 7,
        use_action_conditioning: bool = True,
        predict_depth: bool = True,
        use_incremental_depth: bool = True,
        future_prediction_horizon: int = 1,
        use_velocity_future_gaussians: bool = False,
        velocity_world_model_scale: float = 2.0,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.input_num_tokens = input_num_tokens
        self.static_input_num_tokens = input_num_tokens
        self.future_input_num_tokens = future_input_num_tokens or input_num_tokens
        self.action_dim = action_dim
        self.use_action_conditioning = use_action_conditioning
        self.predict_depth = predict_depth
        self.use_incremental_depth = use_incremental_depth
        self.future_prediction_horizon = max(1, int(future_prediction_horizon))
        self.use_velocity_future_gaussians = use_velocity_future_gaussians
        self.velocity_world_model_scale = float(velocity_world_model_scale)

        # Horizon embedding helps the decoder distinguish t+1 vs t+H.
        self.horizon_embed = nn.Embedding(self.future_prediction_horizon, token_dim)

        # Action embedding projection
        if use_action_conditioning:
            self.action_proj = nn.Sequential(
                nn.Linear(action_dim, 128),
                nn.GELU(),
                nn.Linear(128, token_dim),  # 输出维度应该匹配 token_dim (2048)
            )

        # Full Gaussian decode is used to build the current/base template.
        # Future horizons reuse that template and only predict gated dynamic xyz updates.
        static_grid_size = int(self.static_input_num_tokens ** 0.5)
        self.static_grid_size = static_grid_size
        self.future_grid_size = int(self.future_input_num_tokens ** 0.5)
        self.gaussian_head = IndependentGaussianHead(
            token_dim=token_dim, grid_size=static_grid_size,
            use_image_fusion=True,  # Enable image feature fusion
            img_dim=3,
            predict_depth=predict_depth,
        )

        # Future-query dynamics head: decode 32×32 motion tokens into a denser 128×128
        # velocity field before projecting to the static Gaussian grid.
        if use_velocity_future_gaussians:
            self.velocity_head = nn.Sequential(
                _UpsampleBlock(token_dim, 256),   # 32×32 -> 64×64
                _UpsampleBlock(256, 128),         # 64×64 -> 128×128
                nn.Conv2d(128, 3, 3, padding=1),
            )
            nn.init.xavier_uniform_(self.velocity_head[-1].weight, gain=0.01)
            nn.init.zeros_(self.velocity_head[-1].bias)
        else:
            self.velocity_head = None

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
        step=None,
        current_observation=None,
        actions=None,
        base_depth: torch.Tensor | None = None,
        horizon_idx: int = 0,
        static_reference_params: dict | None = None,
        velocity_time_factor: float = 1.0,
        motion_gate: torch.Tensor | None = None,
    ):
        """Decode latent tokens → Gaussian parameters."""
        return self._decode_independent(
            z, gaussian_adapter=gaussian_adapter, camera_params=camera_params,
            current_observation=current_observation, future_observation=future_observation,
            step=step, actions=actions, base_depth=base_depth, horizon_idx=horizon_idx,
            static_reference_params=static_reference_params,
            velocity_time_factor=velocity_time_factor,
            motion_gate=motion_gate,
        )

    def decode_gaussian_prefix_template(
        self,
        z: torch.Tensor,
        gaussian_adapter,
        current_observation,
        camera_params=None,
        base_depth: torch.Tensor | None = None,
        step=None,
    ):
        """Decode the current/base Gaussian template from prefix Gaussian tokens.

        Used by the future dynamics branch: current Gaussian tokens produce one static
        template, and future latents only predict delta xyz / velocity on top of it.
        z: [B, 256, D] or [B, 768, D]; if 768, uses last 256 (t frame) for decode.
        """
        if z.shape[1] == 768:
            z = z[:, -256:, :]
        elif z.shape[1] != 256:
            raise ValueError(f"decode_gaussian_prefix_template expects 256 or 768 tokens, got {z.shape[1]}")
        return self._decode_independent(
            z,
            gaussian_adapter=gaussian_adapter,
            camera_params=camera_params,
            current_observation=current_observation,
            future_observation=None,
            step=step,
            actions=None,
            base_depth=base_depth,
            horizon_idx=0,
            static_reference_params=None,
            velocity_time_factor=1.0,
            skip_horizon_embedding=True,
            skip_action_conditioning=True,
        )

    # ------------------------------------------------------------------
    # Independent ConvNet decoder (RGB output)
    # ------------------------------------------------------------------
    def _decode_velocity_from_static(
        self,
        z: torch.Tensor,
        static_reference_params: dict,
        velocity_time_factor: float,
        step: int | None,
        horizon_idx: int = 0,
        base_depth: torch.Tensor | None = None,
        motion_gate: torch.Tensor | None = None,
    ) -> dict:
        """Reuse the base Gaussian template and predict future dynamics via delta xyz only."""
        B, num_tokens, D = z.shape
        g = self.future_grid_size
        if g * g != num_tokens:
            raise ValueError(
                f"velocity path expects {self.future_input_num_tokens} future tokens, got N={num_tokens}"
            )

        motion_feat = z.permute(0, 2, 1).reshape(B, D, g, g)  # [B, D, 32, 32]
        vel_map = self.velocity_head(motion_feat)  # [B, 3, 128, 128]

        xyz0 = static_reference_params["xyz"]
        Npts = xyz0.shape[1]
        H = W = int(math.sqrt(Npts))
        if H * W != Npts:
            raise ValueError(f"static xyz N={Npts} is not a square grid")

        vel_up = F.interpolate(vel_map, size=(H, W), mode="bilinear", align_corners=False)
        vel_flat = vel_up.permute(0, 2, 3, 1).reshape(B, Npts, 3)

        raw_delta = torch.tanh(vel_flat) * self.velocity_world_model_scale * float(velocity_time_factor)
        motion_gate_up = None
        if motion_gate is not None:
            if motion_gate.ndim == 3:
                motion_gate = motion_gate.unsqueeze(-1)
            elif motion_gate.ndim != 4:
                raise ValueError(f"motion_gate must have shape [B, N, 1] or [B, H, W, 1], got {tuple(motion_gate.shape)}")

            if motion_gate.shape[0] != B:
                raise ValueError(f"motion_gate batch mismatch: expected {B}, got {motion_gate.shape[0]}")

            if motion_gate.shape[1] == num_tokens:
                motion_gate_up = motion_gate.reshape(B, g, g, -1).permute(0, 3, 1, 2)
                motion_gate_up = F.interpolate(motion_gate_up, size=(H, W), mode="bilinear", align_corners=False)
                motion_gate_up = motion_gate_up.permute(0, 2, 3, 1).reshape(B, Npts, -1)
            elif motion_gate.shape[1] * motion_gate.shape[2] == Npts:
                motion_gate_up = motion_gate.reshape(B, motion_gate.shape[1], motion_gate.shape[2], -1)
                motion_gate_up = motion_gate_up.permute(0, 3, 1, 2)
                if motion_gate_up.shape[-2:] != (H, W):
                    motion_gate_up = F.interpolate(motion_gate_up, size=(H, W), mode="bilinear", align_corners=False)
                motion_gate_up = motion_gate_up.permute(0, 2, 3, 1).reshape(B, Npts, -1)
            else:
                raise ValueError(
                    f"motion_gate token/spatial size mismatch: got {tuple(motion_gate.shape)}, expected token count {num_tokens} or point count {Npts}"
                )

            motion_gate_up = torch.clamp(motion_gate_up.to(device=raw_delta.device, dtype=raw_delta.dtype), 0.0, 1.0)
            raw_delta = raw_delta * motion_gate_up

        delta = raw_delta
        xyz = xyz0 + delta.to(dtype=xyz0.dtype)
        xyz = torch.clamp(xyz, min=-100.0, max=100.0)
        xyz = torch.where(torch.isnan(xyz) | torch.isinf(xyz), torch.zeros_like(xyz), xyz)

        z_cam = xyz[..., 2].reshape(B, H, W)
        depth_map = z_cam.unsqueeze(1).clamp(min=0.0, max=8.0)
        depth_delta_map = None
        if base_depth is not None:
            if base_depth.ndim == 4:
                base_depth_map = base_depth.squeeze(1)
            else:
                base_depth_map = base_depth
            if base_depth_map.shape[-2:] != depth_map.shape[-2:]:
                base_depth_map = F.interpolate(
                    base_depth_map.unsqueeze(1), size=depth_map.shape[-2:], mode="bilinear", align_corners=False
                ).squeeze(1)
            depth_delta_map = depth_map - base_depth_map.unsqueeze(1)

        if step is not None and step % 400 == 0:
            import logging

            static_scale_mean = static_reference_params["scales"].float().mean().item()
            static_scale_max = static_reference_params["scales"].float().max().item()
            gate_mean = motion_gate_up.mean().item() if motion_gate_up is not None else 1.0
            gate_max = motion_gate_up.max().item() if motion_gate_up is not None else 1.0
            logging.info(
                f"[VelocityDecoder][h={horizon_idx}][t+~{horizon_idx + 1}] delta_xyz: "
                f"motion_scale={self.velocity_world_model_scale}, "
                f"time_factor={velocity_time_factor:.4f}, |delta|_mean={delta.abs().mean().item():.6f}, "
                f"|delta|_max={delta.abs().max().item():.6f}, gate_mean={gate_mean:.6f}, gate_max={gate_max:.6f}, "
                f"static_gaussian_scale_mean={static_scale_mean:.6f}, "
                f"static_gaussian_scale_max={static_scale_max:.6f}"
            )

        return {
            "xyz": xyz,
            "scales": static_reference_params["scales"],
            "opacity": static_reference_params["opacity"],
            "sh": static_reference_params["sh"],
            "rotations": static_reference_params["rotations"],
            "depth_map": depth_map,
            "depth_delta_map": depth_delta_map,
            "raw_delta_xyz": raw_delta,
        }

    def decode_dynamic_gaussians_from_static(
        self,
        z: torch.Tensor,
        static_reference_params: dict,
        velocity_time_factor: float,
        step: int | None,
        horizon_idx: int = 0,
        base_depth: torch.Tensor | None = None,
        motion_gate: torch.Tensor | None = None,
    ) -> dict:
        """Decode shared motion-query tokens into a constant-velocity dynamic Gaussian update."""
        return self._decode_velocity_from_static(
            z,
            static_reference_params,
            velocity_time_factor,
            step,
            horizon_idx=horizon_idx,
            base_depth=base_depth,
            motion_gate=motion_gate,
        )

    def _decode_independent(
        self, z, gaussian_adapter=None, camera_params=None,
        current_observation=None, future_observation=None, step=None, actions=None,
        base_depth: torch.Tensor | None = None, horizon_idx: int = 0,
        static_reference_params: dict | None = None,
        velocity_time_factor: float = 1.0,
        skip_horizon_embedding: bool = False,
        skip_action_conditioning: bool = False,
        motion_gate: torch.Tensor | None = None,
    ):
        """Decode VLM tokens into Gaussian parameters.

        Current/base template decode uses the full ConvNet head.
        Future decode reuses a provided static template and predicts only delta xyz.
        """
        if not skip_horizon_embedding:
            horizon_idx = max(0, min(int(horizon_idx), self.future_prediction_horizon - 1))
            horizon_ids = torch.full((z.shape[0],), horizon_idx, device=z.device, dtype=torch.long)
            z = z + self.horizon_embed(horizon_ids).unsqueeze(1).to(dtype=z.dtype)

        if not skip_action_conditioning and self.use_action_conditioning and actions is not None:
            # Transform actions to camera frame
            action_cam = self._transform_action_to_camera(actions, camera_params)
            # Extract only first 7 meaningful dimensions for projection
            action_cam_7d = action_cam[:, :7]  # [B, 7]
            # Embed and add to tokens
            action_embed = self.action_proj(action_cam_7d)  # [B, token_dim]
            z = z + action_embed.unsqueeze(1)  # [B, 256, D] + [B, 1, D] → [B, 256, D]

        if (
            self.use_velocity_future_gaussians
            and static_reference_params is not None
            and self.velocity_head is not None
        ):
            return self.decode_dynamic_gaussians_from_static(
                z,
                static_reference_params,
                velocity_time_factor,
                step,
                horizon_idx=horizon_idx,
                base_depth=base_depth,
                motion_gate=motion_gate,
            )

        # For future-horizon decoding, prefer the matched future observation so image
        # fusion is anchored to the target horizon instead of the current frame.
        vggt_obs = future_observation if future_observation is not None else current_observation
        if vggt_obs is None or gaussian_adapter is None:
            raise ValueError("current_observation and gaussian_adapter are required")

        # Prepare VGGT inputs for image feature fusion
        vggt_inputs = gaussian_adapter.prepare_inputs(
            vggt_obs, z.device, z.shape[0], is_training=False
        )
        if vggt_inputs is None:
            raise ValueError("Failed to prepare VGGT inputs")

        # Extract current frame image for residual feature fusion
        B_vggt, S_vggt = vggt_inputs.shape[:2]
        frame_idx = S_vggt - 1
        current_frame_img = vggt_inputs[:, frame_idx]  # [B, 3, H_vggt, W_vggt]

        # Decode VLM tokens → Gaussian params + depth (with current frame residual)
        decoder_output = self.gaussian_head(z, images=current_frame_img)
        raw = decoder_output['gaussian_params']  # [B, 19, 256, 256]
        rot_raw, scale_raw, opa_raw, sh_raw, xy_delta_raw = raw.split([4, 3, 1, 9, 2], dim=1)

        # Get depth: use predicted incremental depth if available, otherwise fallback to VGGT
        depth_delta_map = None
        if self.predict_depth and 'depth' in decoder_output:
            depth_raw = decoder_output['depth']  # [B, 1, 256, 256]

            if base_depth is not None and self.use_incremental_depth:
                if base_depth.ndim == 4:
                    base_depth_map = base_depth.squeeze(1)
                else:
                    base_depth_map = base_depth
                if base_depth_map.shape[-2:] != depth_raw.shape[-2:]:
                    base_depth_map = F.interpolate(
                        base_depth_map.unsqueeze(1), size=depth_raw.shape[-2:], mode="bilinear", align_corners=False
                    ).squeeze(1)
                depth_delta_map = torch.tanh(depth_raw.squeeze(1))
                final_depth = torch.clamp(base_depth_map + depth_delta_map, min=0.0, max=8.0)  # [B, 256, 256]
                H_dec, W_dec = final_depth.shape[1], final_depth.shape[2]
            else:
                # Absolute-depth fallback path
                min_depth = 0.0
                max_depth = 8.0
                final_depth = min_depth + (max_depth - min_depth) * torch.sigmoid(depth_raw.squeeze(1))  # [B, 256, 256]
                H_dec, W_dec = final_depth.shape[1], final_depth.shape[2]
        else:
            # Fallback: use VGGT depth from current frame (old behavior)
            with torch.no_grad():
                aggregated_tokens_list, patch_start_idx = gaussian_adapter.encoder.aggregator(
                    vggt_inputs.to(torch.bfloat16)
                )
            for i in range(len(aggregated_tokens_list)):
                if aggregated_tokens_list[i].ndim == 3:
                    _bs, _p, _c = aggregated_tokens_list[i].shape
                    aggregated_tokens_list[i] = aggregated_tokens_list[i].view(B_vggt, S_vggt, _p, _c)

            depth_maps, depth_conf = gaussian_adapter.encoder.depth_head(
                aggregated_tokens_list, images=vggt_inputs, patch_start_idx=patch_start_idx
            )
            depth_maps = torch.sigmoid(torch.log(depth_maps + 1e-6))
            min_depth = gaussian_adapter.encoder.min_depth
            max_depth = gaussian_adapter.encoder.max_depth
            depth_maps = min_depth + (max_depth - min_depth) * depth_maps
            final_depth = depth_maps[:, frame_idx, :, :, 0]  # [B, H_vggt, W_vggt]
            H_dec, W_dec = raw.shape[2], raw.shape[3]
            final_depth = F.interpolate(
                final_depth.unsqueeze(1), size=(H_dec, W_dec), mode="bilinear", align_corners=False
            ).squeeze(1)  # [B, 256, 256]

        # 6. Activations
        B = z.shape[0]

        # Rotation: normalize quaternions
        rot_maps = rot_raw.permute(0, 2, 3, 1)  # [B, H, W, 4]
        rot_maps = rot_maps / (rot_maps.norm(dim=-1, keepdim=True) + 1e-8)

        # Scale: softplus for positive values (following AD-FFgsStudio)
        # Changed from 0.001 to 0.01 to improve rendering quality (larger Gaussians)
        scale_maps = F.softplus(scale_raw.permute(0, 2, 3, 1), beta=1) * 0.01  # [B, H, W, 3]

        # Opacity: sigmoid to [0, 1]
        opacity_maps = torch.sigmoid(opa_raw.permute(0, 2, 3, 1))  # [B, H, W, 1]

        # Spherical Harmonics: 9 coefficients (DC + 1st order)
        # Reshape to [B, H, W, 9] and apply SH mask (following AD-FFgsStudio)
        sh_maps = sh_raw.permute(0, 2, 3, 1)  # [B, H, W, 9]
        sh_mask = self.gaussian_head.sh_mask.view(1, 1, 1, 9)  # [1, 1, 1, 9]
        sh_maps = sh_maps * sh_mask  # Attenuate higher-order coefficients

        # Reshape to [B, H, W, 3, 3] for rendering (3 colors × 3 SH basis per color)
        # Note: For 1st order SH, we have DC(3) + 1st(6) = 9 total
        # Renderer expects [B, H, W, K, 3] where K = (sh_degree+1)^2 / 3
        # For degree 1: K = 4 (DC + 3 for 1st order), but we have 9 coefficients
        # We'll keep it as [B, H, W, 9] and reshape in rendering if needed

        # 7. Depth → xyz
        # LIBERO original camera: 256×256, fx=fy=221.7025, cx=cy=128.0
        # No scaling needed since decoder outputs 256×256
        if camera_params is not None and "fx" in camera_params:
            xyz_base = self.depth2pc(
                final_depth,
                fx=camera_params["fx"], fy=camera_params["fy"],
                cx=camera_params["cx"], cy=camera_params["cy"],
                downsample_factor=1,
            )
        else:
            # LIBERO intrinsics for 256×256 resolution
            xyz_base = self.depth2pc(
                final_depth,
                fx=221.7025, fy=221.7025,
                cx=128.0, cy=128.0,
                downsample_factor=1,
            )

        # Add explicit spatial delta prediction to Break 2D ray lock
        xy_delta_maps = xy_delta_raw.permute(0, 2, 3, 1)  # [B, H, W, 2]
        N = H_dec * W_dec  # 256 * 256 = 65536
        xy_delta_flat = xy_delta_maps.reshape(B, N, 2)
        # Scale lateral movement explicitly
        xy_delta_flat = torch.tanh(xy_delta_flat) * 0.5  # constrain to max 0.5m movement

        # Keep Z unchanged here because Z (depth) movement is completely handled and supervised by depth_refine.
        z_zeros = torch.zeros(B, N, 1, device=xyz_base.device, dtype=xyz_base.dtype)
        xyz_delta_flat = torch.cat([xy_delta_flat, z_zeros], dim=-1)
        xyz = xyz_base + xyz_delta_flat

        # 8. Flatten and sanitize
        rot_flat = rot_maps.reshape(B, N, 4)
        scale_flat = scale_maps.reshape(B, N, 3)
        opacity_flat = opacity_maps.reshape(B, N, 1)
        sh_flat = sh_maps.reshape(B, N, 9)  # [B, N, 9] — DC + 1st order SH

        scale_flat = torch.clamp(scale_flat, min=1e-7, max=10.0)
        xyz = torch.clamp(xyz, min=-100.0, max=100.0)
        xyz = torch.where(torch.isnan(xyz) | torch.isinf(xyz), torch.zeros_like(xyz), xyz)

        if step is not None and step % 100 == 0:
            import logging
            logging.info(f"[IndependentDecoder] Step {step}: "
                         f"depth=[{final_depth.min():.3f}, {final_depth.max():.3f}], "
                         f"scales=[{scale_flat.min():.3f}, {scale_flat.max():.3f}], "
                         f"sh=[{sh_flat.min():.3f}, {sh_flat.max():.3f}], N={N}")

        return {
            "xyz": xyz,
            "scales": scale_flat,
            "opacity": opacity_flat,
            "sh": sh_flat,  # [B, N, 9] for 1st order SH
            "rotations": rot_flat,
            "depth_map": final_depth.unsqueeze(1),  # [B, 1, H, W] for edge-aware smoothness
            "depth_delta_map": None if depth_delta_map is None else depth_delta_map.unsqueeze(1),
        }

    def _transform_action_to_camera(self, actions, camera_params):
        """
        quit ** 0313
        Transform actions from world frame to camera frame.

        Args:
            actions: [B, action_dim] - padded actions (may be 32-dim, but only first 7 are used)
                     (delta_x, delta_y, delta_z, quat_w, quat_x, quat_y, quat_z, ...)
            camera_params: dict with camera_pos and camera_quat
        Returns:
            action_cam: [B, action_dim] - actions in camera frame with X-axis flipped
        """
        if camera_params is None or "camera_pos" not in camera_params:
            # No transformation, return as-is
            return actions

        B = actions.shape[0]
        device = actions.device
        action_dim = actions.shape[1]

        # Extract camera pose
        cam_pos = torch.tensor(camera_params["camera_pos"], device=device, dtype=torch.float32)
        cam_quat = torch.tensor(camera_params["camera_quat"], device=device, dtype=torch.float32)

        # Build camera rotation matrix from quaternion
        cam_rot = self._quat_to_rotation_matrix(cam_quat)  # [3, 3]

        # Handle temporal dimension: actions could be [B, T, D] or [B, D]
        if actions.ndim == 3:
            # actions is [B, T, D], take the first timestep
            actions = actions[:, 0, :]  # [B, D]

        # Extract position and rotation from actions (only first 7 dims are meaningful)
        eef_pos = actions[:, :3]  # [B, 3] - delta position in world frame
        eef_quat = actions[:, 3:7]  # [B, 4] - quaternion (w, x, y, z)

        # Transform position to camera frame
        eef_pos_cam = torch.matmul(eef_pos, cam_rot.T)  # [B, 3]

        # CRITICAL: Flip X-axis to match image coordinate convention
        eef_pos_cam[:, 0] = -eef_pos_cam[:, 0]

        # For rotation, we keep it as-is (quaternion transformation is complex)
        # In practice, position is more important for action conditioning

        # Reconstruct action with transformed position
        action_cam = actions.clone()
        action_cam[:, :3] = eef_pos_cam
        # Keep quaternion and padding unchanged

        return action_cam

    @staticmethod
    def _quat_to_rotation_matrix(quat):
        """
        Convert quaternion to rotation matrix.

        Args:
            quat: [4] - (w, x, y, z)
        Returns:
            R: [3, 3] rotation matrix
        """
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]

        R = torch.stack([
            torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y]),
            torch.stack([2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x]),
            torch.stack([2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]),
        ])

        return R