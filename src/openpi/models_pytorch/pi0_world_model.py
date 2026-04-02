# zijian
# date 2026.01.24
# v2 refactored: GaussianDecoder — lightweight decode-only module
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    m00 = matrix[..., 0, 0]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]
    m11 = matrix[..., 1, 1]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m22 = matrix[..., 2, 2]

    trace = m00 + m11 + m22
    q = torch.zeros(*matrix.shape[:-2], 4, device=matrix.device, dtype=matrix.dtype)

    mask_trace = trace > 0.0
    if mask_trace.any():
        s = torch.sqrt(trace[mask_trace] + 1.0) * 2.0
        q_trace = q[mask_trace]
        q_trace[..., 0] = 0.25 * s
        q_trace[..., 1] = (m21[mask_trace] - m12[mask_trace]) / s
        q_trace[..., 2] = (m02[mask_trace] - m20[mask_trace]) / s
        q_trace[..., 3] = (m10[mask_trace] - m01[mask_trace]) / s
        q[mask_trace] = q_trace

    mask_x = (~mask_trace) & (m00 > m11) & (m00 > m22)
    if mask_x.any():
        s = torch.sqrt(1.0 + m00[mask_x] - m11[mask_x] - m22[mask_x]) * 2.0
        q_x = q[mask_x]
        q_x[..., 0] = (m21[mask_x] - m12[mask_x]) / s
        q_x[..., 1] = 0.25 * s
        q_x[..., 2] = (m01[mask_x] + m10[mask_x]) / s
        q_x[..., 3] = (m02[mask_x] + m20[mask_x]) / s
        q[mask_x] = q_x

    mask_y = (~mask_trace) & (~mask_x) & (m11 > m22)
    if mask_y.any():
        s = torch.sqrt(1.0 + m11[mask_y] - m00[mask_y] - m22[mask_y]) * 2.0
        q_y = q[mask_y]
        q_y[..., 0] = (m02[mask_y] - m20[mask_y]) / s
        q_y[..., 1] = (m01[mask_y] + m10[mask_y]) / s
        q_y[..., 2] = 0.25 * s
        q_y[..., 3] = (m12[mask_y] + m21[mask_y]) / s
        q[mask_y] = q_y

    mask_z = (~mask_trace) & (~mask_x) & (~mask_y)
    if mask_z.any():
        s = torch.sqrt(1.0 + m22[mask_z] - m00[mask_z] - m11[mask_z]) * 2.0
        q_z = q[mask_z]
        q_z[..., 0] = (m10[mask_z] - m01[mask_z]) / s
        q_z[..., 1] = (m02[mask_z] + m20[mask_z]) / s
        q_z[..., 2] = (m12[mask_z] + m21[mask_z]) / s
        q_z[..., 3] = 0.25 * s
        q[mask_z] = q_z

    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    return q


def quaternion_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    quat = quat / (quat.norm(dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = quat.unbind(dim=-1)

    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    return torch.stack(
        [
            torch.stack([ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
            torch.stack([2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)], dim=-1),
            torch.stack([2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz], dim=-1),
        ],
        dim=-2,
    )


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    out = torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )
    return out / (out.norm(dim=-1, keepdim=True) + 1e-8)


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


class SharedGaussianBackbone(nn.Module):
    """Shared token-to-feature backbone for both static and velocity decoding."""

    def __init__(self, token_dim: int = 2048):
        super().__init__()
        self.layer1 = _UpsampleBlock(token_dim, 512)   # 32×32 -> 64×64
        self.layer2 = _UpsampleBlock(512, 256)         # 64×64 -> 128×128
        self.layer3 = _UpsampleBlock(256, 128)         # 128×128 -> 256×256

        self.fusion1 = _FeatureFusionBlock(128, has_residual=False)
        self.fusion2 = _FeatureFusionBlock(128, has_residual=True)
        self.fusion3 = _FeatureFusionBlock(128, has_residual=True)

        self.proj_feat2 = nn.Conv2d(256, 128, 1)
        self.proj_feat1 = nn.Conv2d(512, 128, 1)

    def forward(self, token_grid: torch.Tensor) -> torch.Tensor:
        feat1 = self.layer1(token_grid)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)

        fused = self.fusion1(feat3)
        fused = self.fusion2(fused, residual=self.proj_feat2(feat2))
        fused = self.fusion3(fused, residual=self.proj_feat1(feat1))
        return fused


class StaticGaussianHead(nn.Module):
    """Decode shared features into static Gaussian parameters and absolute depth.

    Geometry maps F_g from ``shared_features``; optional RGB is added only before the
    Gaussian-parameter head. Depth is refined from F_g alone so appearance does not
    leak into the depth branch.
    """

    def __init__(self, use_image_fusion: bool = True, img_dim: int = 3, predict_depth: bool = True):
        super().__init__()
        self.use_image_fusion = use_image_fusion
        self.predict_depth = predict_depth

        out_ch = 4 + 3 + 1 + 9  # rot(4) + scale(3) + opacity(1) + SH(9)

        if use_image_fusion:
            self.img_merger = nn.Sequential(
                nn.Conv2d(img_dim, 128, 7, padding=3),
                nn.GELU(),
            )

        self.head = nn.Conv2d(128, out_ch, 3, padding=1)

        if predict_depth:
            self.depth_refine = nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1),
                nn.GroupNorm(min(32, 64), 64),
                nn.GELU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.GroupNorm(min(32, 64), 64),
                nn.GELU(),
                nn.Conv2d(64, 1, 3, padding=1),
            )
            for m in self.depth_refine.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight, gain=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)

        self.register_buffer(
            "sh_mask",
            torch.tensor(
                [1.0, 1.0, 1.0, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025],
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def forward(self, shared_features: torch.Tensor, images: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        f_g = shared_features
        g = f_g
        if self.use_image_fusion and images is not None:
            if images.shape[2:] != f_g.shape[2:]:
                images = F.interpolate(images, size=f_g.shape[2:], mode="bilinear", align_corners=True)
            g = f_g + self.img_merger(images)

        result = {"gaussian_params": self.head(g)}
        if self.predict_depth:
            result["depth"] = self.depth_refine(f_g)
        return result


class VelocityGaussianHead(nn.Module):
    """Decode shared features into horizon-conditioned slot-rigid motion parameters."""

    def __init__(
        self,
        future_prediction_horizon: int,
        model_dim: int = 128,
        memory_grid_size: int = 32,
        num_motion_queries: int = 8,
        slot_assignment_temperature: float = 1.0,
        slot_translation_scale: float = 2.0,
        slot_rotation_scale: float = 1.0,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.memory_grid_size = max(4, int(memory_grid_size))
        self.num_motion_queries = max(1, int(num_motion_queries))
        self.slot_assignment_temperature = max(1e-4, float(slot_assignment_temperature))
        self.slot_translation_scale = float(slot_translation_scale)
        self.slot_rotation_scale = float(slot_rotation_scale)
        self.horizon_proj = nn.Embedding(max(1, int(future_prediction_horizon)), model_dim)
        self.motion_queries = nn.Embedding(self.num_motion_queries, model_dim)
        self.input_proj = nn.Conv2d(128, model_dim, kernel_size=1)
        self.pos_proj = nn.Conv2d(2, model_dim, kernel_size=1)
        self.query_norm = nn.LayerNorm(model_dim)
        self.memory_norm = nn.LayerNorm(model_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=8,
            dropout=0.0,
            batch_first=True,
        )
        self.query_mlp = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Linear(model_dim * 4, model_dim),
        )
        self.slot_param_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim * 2),
            nn.GELU(),
            nn.Linear(model_dim * 2, 12),
        )
        self.spatial_proj = nn.Conv2d(model_dim, model_dim, kernel_size=1)
        self.query_proj = nn.Linear(model_dim, model_dim)
        self.query_bias = nn.Linear(model_dim, 1)
        nn.init.xavier_uniform_(self.slot_param_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.slot_param_head[-1].bias)
        nn.init.xavier_uniform_(self.spatial_proj.weight, gain=1.0)
        nn.init.zeros_(self.spatial_proj.bias)
        nn.init.xavier_uniform_(self.query_proj.weight, gain=1.0)
        nn.init.zeros_(self.query_proj.bias)
        nn.init.zeros_(self.query_bias.weight)
        nn.init.zeros_(self.query_bias.bias)

    def _positional_grid(self, batch_size: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        pos = torch.stack([grid_y, grid_x], dim=0).unsqueeze(0)
        return pos.expand(batch_size, -1, -1, -1)

    def forward(
        self,
        shared_features: torch.Tensor,
        horizon_idx: int,
    ) -> dict[str, torch.Tensor]:
        horizon_idx = max(0, min(int(horizon_idx), self.horizon_proj.num_embeddings - 1))
        batch_size, _, height, width = shared_features.shape
        feat = self.input_proj(shared_features)
        feat = F.adaptive_avg_pool2d(feat, output_size=(self.memory_grid_size, self.memory_grid_size))
        feat = feat + self.pos_proj(self._positional_grid(batch_size, feat.shape[-2], feat.shape[-1], feat.device, feat.dtype))

        memory = feat.flatten(2).transpose(1, 2)
        memory = self.memory_norm(memory)

        horizon_ids = torch.full((batch_size,), horizon_idx, device=shared_features.device, dtype=torch.long)
        horizon_query = self.horizon_proj(horizon_ids).unsqueeze(1)
        learned_queries = self.motion_queries.weight.unsqueeze(0).expand(batch_size, -1, -1)
        query = learned_queries + horizon_query
        query = self.query_norm(query)
        attn_out, _ = self.cross_attn(query, memory, memory, need_weights=False)
        query = query + attn_out
        query = query + self.query_mlp(query)

        slot_params = self.slot_param_head(query)
        slot_rot_6d = slot_params[..., :6] * self.slot_rotation_scale
        slot_trans = torch.tanh(slot_params[..., 6:9]) * self.slot_translation_scale
        slot_pivot_offset = torch.tanh(slot_params[..., 9:12]) * self.slot_translation_scale

        spatial = self.spatial_proj(feat)
        query_feat = self.query_proj(query)
        slot_logits = torch.einsum("bkc,bchw->bkhw", query_feat, spatial)
        slot_logits = slot_logits + self.query_bias(query).unsqueeze(-1)
        if slot_logits.shape[-2:] != (height, width):
            slot_logits = F.interpolate(slot_logits, size=(height, width), mode="bilinear", align_corners=False)

        slot_probs = F.softmax(slot_logits / self.slot_assignment_temperature, dim=1)
        slot_probs_flat = slot_probs.flatten(2).transpose(1, 2)

        slot_entropy = -(slot_probs_flat * (slot_probs_flat.clamp_min(1e-8).log())).sum(dim=-1).mean()
        slot_usage = slot_probs_flat.mean(dim=1)
        balance_target = torch.full_like(slot_usage, 1.0 / self.num_motion_queries)
        slot_balance_loss = F.mse_loss(slot_usage, balance_target)
        slot_trans_reg = slot_trans.pow(2).mean()
        slot_rot_reg = slot_rot_6d.pow(2).mean()

        return {
            "slot_logits": slot_logits,
            "slot_probs": slot_probs_flat,
            "slot_rot_6d": slot_rot_6d,
            "slot_trans": slot_trans,
            "slot_pivot_offset": slot_pivot_offset,
            "slot_entropy": slot_entropy,
            "slot_balance_loss": slot_balance_loss,
            "slot_trans_reg": slot_trans_reg,
            "slot_rot_reg": slot_rot_reg,
            "slot_usage": slot_usage,
        }


class GaussianDecoder(nn.Module):
    """
    Decoder that converts shared future-token features into 3D Gaussian parameters.

    The shared token block is normalized onto the canonical future-token grid once,
    passed through one shared spatial backbone, then split into two heads:
    - static head: predicts current/base Gaussian parameters + absolute depth
    - velocity head: predicts horizon-conditioned raw_delta_xyz rollout updates

    Future rollouts reuse the detached static template for scale / opacity / SH /
    rotation while applying horizon-scaled xyz motion from the velocity head.
    """

    def __init__(
        self,
        token_dim: int,
        input_num_tokens: int = 256,
        future_input_num_tokens: int | None = None,
        predict_depth: bool = True,
        use_incremental_depth: bool = True,
        future_prediction_horizon: int = 1,
        use_velocity_future_gaussians: bool = False,
        velocity_world_model_scale: float = 2.0,
        use_action_conditioning: bool = False,
        num_motion_slots: int = 8,
        slot_assignment_temperature: float = 1.0,
        slot_translation_scale: float | None = None,
        slot_rotation_scale: float = 1.0,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.input_num_tokens = input_num_tokens
        self.static_input_num_tokens = input_num_tokens
        self.future_input_num_tokens = future_input_num_tokens or input_num_tokens
        self.predict_depth = predict_depth
        self.use_incremental_depth = use_incremental_depth
        self.future_prediction_horizon = max(1, int(future_prediction_horizon))
        self.use_velocity_future_gaussians = use_velocity_future_gaussians
        self.velocity_world_model_scale = float(velocity_world_model_scale)
        self.use_action_conditioning = use_action_conditioning
        self.num_motion_slots = max(1, int(num_motion_slots))
        self.slot_assignment_temperature = float(slot_assignment_temperature)
        self.slot_translation_scale = float(slot_translation_scale if slot_translation_scale is not None else velocity_world_model_scale)
        self.slot_rotation_scale = float(slot_rotation_scale)

        # Horizon embedding helps the decoder distinguish t+1 vs t+H.
        self.horizon_embed = nn.Embedding(self.future_prediction_horizon, token_dim)

        self.static_grid_size = int(self.static_input_num_tokens ** 0.5)
        self.future_grid_size = int(self.future_input_num_tokens ** 0.5)

        # One shared decoder backbone consumes the canonical 32×32 future-token grid.
        # Static/base reconstruction and future velocity rollout both branch from this shared feature map.
        self.canonical_grid_size = self.future_grid_size
        self.shared_backbone = SharedGaussianBackbone(token_dim=token_dim)
        self.static_head = StaticGaussianHead(
            use_image_fusion=True,
            img_dim=3,
            predict_depth=predict_depth,
        )

        if use_velocity_future_gaussians:
            self.velocity_head = VelocityGaussianHead(
                self.future_prediction_horizon,
                num_motion_queries=self.num_motion_slots,
                slot_assignment_temperature=self.slot_assignment_temperature,
                slot_translation_scale=self.slot_translation_scale,
                slot_rotation_scale=self.slot_rotation_scale,
            )
        else:
            self.velocity_head = None

    def _normalize_tokens_to_canonical_grid(self, z: torch.Tensor) -> torch.Tensor:
        """Normalize 256/768/1024-token inputs onto the canonical future-token grid."""
        if z.ndim != 3:
            raise ValueError(f"Expected token tensor [B, N, D], got {tuple(z.shape)}")

        if z.shape[1] == self.static_input_num_tokens * max(1, self.future_prediction_horizon):
            z = z[:, -self.static_input_num_tokens :, :]
        elif z.shape[1] == self.static_input_num_tokens * 3:
            z = z[:, -self.static_input_num_tokens :, :]

        src_tokens = z.shape[1]
        src_grid = int(math.isqrt(src_tokens))
        if src_grid * src_grid != src_tokens:
            raise ValueError(f"Cannot reshape token count {src_tokens} into a square grid")

        if src_grid == self.canonical_grid_size:
            return z

        token_grid = z.transpose(1, 2).reshape(z.shape[0], z.shape[2], src_grid, src_grid)
        token_grid = F.interpolate(
            token_grid,
            size=(self.canonical_grid_size, self.canonical_grid_size),
            mode="bilinear",
            align_corners=False,
        )
        return token_grid.reshape(z.shape[0], z.shape[2], -1).transpose(1, 2)

    def _prepare_shared_token_features(
        self,
        z: torch.Tensor,
        *,
        horizon_idx: int = 0,
    ) -> dict[str, torch.Tensor | int]:
        """Normalize token count and run the shared decoder backbone once.

        Horizon-specific rollout variation now lives in VelocityGaussianHead.
        The shared backbone stays horizon-agnostic so one decoder state can be reused.
        """
        horizon_idx = max(0, min(int(horizon_idx), self.future_prediction_horizon - 1))

        z_canonical = self._normalize_tokens_to_canonical_grid(z)
        token_grid = z_canonical.transpose(1, 2).reshape(
            z_canonical.shape[0], z_canonical.shape[2], self.canonical_grid_size, self.canonical_grid_size
        )
        shared_features = self.shared_backbone(token_grid)
        return {
            "tokens": z_canonical,
            "token_grid": token_grid,
            "shared_features": shared_features,
            "horizon_idx": horizon_idx,
        }

    def prepare_decoder_state(
        self,
        z: torch.Tensor,
        *,
        horizon_idx: int = 0,
    ) -> dict[str, torch.Tensor | int]:
        return self._prepare_shared_token_features(
            z,
            horizon_idx=horizon_idx,
        )

    def _decode_static_from_shared(
        self,
        shared_state: dict[str, torch.Tensor | int],
        *,
        gaussian_adapter=None,
        camera_params=None,
        current_observation=None,
        future_observation=None,
        base_depth: torch.Tensor | None = None,
    ) -> dict:
        shared_features = shared_state["shared_features"]
        assert isinstance(shared_features, torch.Tensor)

        current_frame_img = None
        if future_observation is None and current_observation is not None and gaussian_adapter is not None:
            vggt_inputs = gaussian_adapter.prepare_inputs(
                current_observation, shared_features.device, shared_features.shape[0], is_training=True
            )
            if vggt_inputs is not None:
                current_frame_img = vggt_inputs[:, -1]

        decoder_output = self.static_head(shared_features, images=current_frame_img)
        raw = decoder_output["gaussian_params"]
        rot_raw, scale_raw, opa_raw, sh_raw = raw.split([4, 3, 1, 9], dim=1)

        depth_delta_map = None
        if self.predict_depth and "depth" in decoder_output:
            depth_raw = decoder_output["depth"]
            final_depth = 8.0 * torch.sigmoid(depth_raw.squeeze(1))
        elif base_depth is not None:
            final_depth = base_depth.squeeze(1) if base_depth.ndim == 4 else base_depth
        else:
            raise ValueError("Decoder requires predicted depth or base_depth")

        H_dec, W_dec = final_depth.shape[-2:]
        B = final_depth.shape[0]

        rot_maps = rot_raw.permute(0, 2, 3, 1)
        rot_maps = rot_maps / (rot_maps.norm(dim=-1, keepdim=True) + 1e-8)

        scale_maps = F.softplus(scale_raw.permute(0, 2, 3, 1), beta=1) * 0.01
        opacity_maps = torch.sigmoid(opa_raw.permute(0, 2, 3, 1))

        sh_maps = sh_raw.permute(0, 2, 3, 1)
        sh_mask = self.static_head.sh_mask.view(1, 1, 1, 9)
        sh_maps = sh_maps * sh_mask

        if camera_params is not None and "fx" in camera_params:
            xyz_base = self.depth2pc(
                final_depth,
                fx=camera_params["fx"], fy=camera_params["fy"],
                cx=camera_params["cx"], cy=camera_params["cy"],
                downsample_factor=1,
            )
        else:
            xyz_base = self.depth2pc(
                final_depth,
                fx=221.7025, fy=221.7025,
                cx=128.0, cy=128.0,
                downsample_factor=1,
            )

        N = H_dec * W_dec
        xyz = xyz_base

        rot_flat = rot_maps.reshape(B, N, 4)
        scale_flat = torch.clamp(scale_maps.reshape(B, N, 3), min=1e-7, max=10.0)
        opacity_flat = opacity_maps.reshape(B, N, 1)
        sh_flat = sh_maps.reshape(B, N, 9)

        xyz = torch.clamp(xyz, min=-100.0, max=100.0)
        xyz = torch.where(torch.isnan(xyz) | torch.isinf(xyz), torch.zeros_like(xyz), xyz)

        return {
            "xyz": xyz,
            "scales": scale_flat,
            "opacity": opacity_flat,
            "sh": sh_flat,
            "rotations": rot_flat,
            "depth_map": final_depth.unsqueeze(1),
            "depth_delta_map": None if depth_delta_map is None else depth_delta_map.unsqueeze(1),
        }

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

        fx_s = fx / downsample_factor
        fy_s = fy / downsample_factor
        cx_s = cx / downsample_factor
        cy_s = cy / downsample_factor

        u = torch.arange(0.5, W + 0.5, device=device, dtype=dtype)
        v = torch.arange(0.5, H + 0.5, device=device, dtype=dtype)
        v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")

        x = ((u_grid[None] - cx_s) * depth) / fx_s
        y = ((v_grid[None] - cy_s) * depth) / fy_s
        z = depth

        xyz = torch.stack([x, y, z], dim=-1)
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
        base_depth: torch.Tensor | None = None,
        horizon_idx: int = 0,
        static_reference_params: dict | None = None,
        velocity_time_factor: float = 1.0,
        shared_state: dict[str, torch.Tensor | int] | None = None,
    ):
        """Decode latent tokens → Gaussian parameters."""
        return self._decode_independent(
            z,
            gaussian_adapter=gaussian_adapter,
            camera_params=camera_params,
            current_observation=current_observation,
            future_observation=future_observation,
            step=step,
            base_depth=base_depth,
            horizon_idx=horizon_idx,
            static_reference_params=static_reference_params,
            velocity_time_factor=velocity_time_factor,
            shared_state=shared_state,
        )

    def decode_gaussian_prefix_template(
        self,
        z: torch.Tensor,
        gaussian_adapter,
        current_observation,
        camera_params=None,
        base_depth: torch.Tensor | None = None,
        step=None,
        shared_state: dict[str, torch.Tensor | int] | None = None,
    ):
        """Decode the current/base Gaussian template from shared future-token features."""
        return self._decode_independent(
            z,
            gaussian_adapter=gaussian_adapter,
            camera_params=camera_params,
            current_observation=current_observation,
            future_observation=None,
            step=step,
            base_depth=base_depth,
            horizon_idx=0,
            static_reference_params=None,
            velocity_time_factor=1.0,
            skip_horizon_embedding=True,
            shared_state=shared_state,
        )

    def _decode_velocity_from_static(
        self,
        shared_state: dict[str, torch.Tensor | int],
        static_reference_params: dict,
        velocity_time_factor: float,
        step: int | None,
        horizon_idx: int = 0,
        base_depth: torch.Tensor | None = None,
    ) -> dict:
        """Reuse the base Gaussian template and predict future dynamics via slot-rigid motion."""
        shared_features = shared_state["shared_features"]
        assert isinstance(shared_features, torch.Tensor)

        if self.velocity_head is None:
            raise ValueError("Velocity head is not initialized")

        motion_outputs = self.velocity_head(
            shared_features,
            horizon_idx=horizon_idx,
        )

        xyz0 = static_reference_params["xyz"]
        B, Npts, _ = xyz0.shape
        H = W = int(math.sqrt(Npts))
        if H * W != Npts:
            raise ValueError(f"static xyz N={Npts} is not a square grid")

        slot_probs = motion_outputs["slot_probs"].to(dtype=xyz0.dtype)
        slot_rot_6d = motion_outputs["slot_rot_6d"].to(dtype=xyz0.dtype)
        slot_trans = motion_outputs["slot_trans"].to(dtype=xyz0.dtype) * float(velocity_time_factor)
        slot_pivot_offset = motion_outputs["slot_pivot_offset"].to(dtype=xyz0.dtype)
        slot_usage = slot_probs.mean(dim=1)
        slot_weight_sum = slot_probs.sum(dim=1)
        slot_centers = torch.einsum("bnk,bnc->bkc", slot_probs, xyz0)
        slot_centers = slot_centers / slot_weight_sum.unsqueeze(-1).clamp_min(1e-6)
        slot_pivots = slot_centers + slot_pivot_offset

        rot_mats = rotation_6d_to_matrix(slot_rot_6d.reshape(-1, 6)).reshape(B, self.num_motion_slots, 3, 3)
        xyz_rel = xyz0.unsqueeze(2) - slot_pivots.unsqueeze(1)
        xyz_slots = torch.einsum("bkdc,bnkc->bnkd", rot_mats, xyz_rel) + slot_pivots.unsqueeze(1) + slot_trans.unsqueeze(1)
        xyz = (slot_probs.unsqueeze(-1) * xyz_slots).sum(dim=2)
        raw_delta = xyz - xyz0

        xyz = torch.clamp(xyz, min=-100.0, max=100.0)
        xyz = torch.where(torch.isnan(xyz) | torch.isinf(xyz), torch.zeros_like(xyz), xyz)
        raw_delta = torch.where(torch.isnan(raw_delta) | torch.isinf(raw_delta), torch.zeros_like(raw_delta), raw_delta)

        z_cam = xyz[..., 2].reshape(B, H, W)
        depth_map = z_cam.unsqueeze(1).clamp(min=0.0, max=8.0)
        depth_delta_map = None
        if base_depth is not None:
            base_depth_map = base_depth.squeeze(1) if base_depth.ndim == 4 else base_depth
            if base_depth_map.shape[-2:] != depth_map.shape[-2:]:
                base_depth_map = F.interpolate(
                    base_depth_map.unsqueeze(1), size=depth_map.shape[-2:], mode="bilinear", align_corners=False
                ).squeeze(1)
            depth_delta_map = depth_map - base_depth_map.unsqueeze(1)

        scales = static_reference_params["scales"]
        opacity = static_reference_params["opacity"]
        sh = static_reference_params["sh"]
        rotations = static_reference_params["rotations"]

        slot_quat = matrix_to_quaternion(rot_mats)
        slot_assign = slot_probs.argmax(dim=-1)
        point_slot_quat = torch.gather(slot_quat, 1, slot_assign.unsqueeze(-1).expand(-1, -1, 4))
        rotations = quaternion_multiply(point_slot_quat, rotations.to(dtype=xyz0.dtype))

        if step is not None and step % 400 == 0:
            import logging

            static_scale_mean = static_reference_params["scales"].float().mean().item()
            static_scale_max = static_reference_params["scales"].float().max().item()
            trans_norm = slot_trans.norm(dim=-1)
            logging.info(
                f"[VelocityDecoder][h={horizon_idx}][t+~{horizon_idx + 1}] slot_rigid: "
                f"motion_scale={self.slot_translation_scale}, "
                f"time_factor={velocity_time_factor:.4f}, |delta|_mean={raw_delta.abs().mean().item():.6f}, "
                f"|delta|_max={raw_delta.abs().max().item():.6f}, slot_entropy={motion_outputs['slot_entropy'].item():.6f}, "
                f"slot_usage_mean={slot_usage.mean().item():.6f}, slot_trans_norm_mean={trans_norm.mean().item():.6f}, "
                f"static_gaussian_scale_mean={static_scale_mean:.6f}, "
                f"static_gaussian_scale_max={static_scale_max:.6f}"
            )

        return {
            "xyz": xyz,
            "scales": scales,
            "opacity": opacity,
            "sh": sh,
            "rotations": rotations,
            "depth_map": depth_map,
            "depth_delta_map": depth_delta_map,
            "raw_delta_xyz": raw_delta,
            "slot_probs": slot_probs,
            "slot_usage": slot_usage,
            "slot_entropy": motion_outputs["slot_entropy"],
            "slot_balance_loss": motion_outputs["slot_balance_loss"],
            "slot_trans_reg": motion_outputs["slot_trans_reg"],
            "slot_rot_reg": motion_outputs["slot_rot_reg"],
            "slot_trans": slot_trans,
            "slot_rot_6d": slot_rot_6d,
            "slot_pivots": slot_pivots,
        }

    def decode_dynamic_gaussians_from_static(
        self,
        shared_state: dict[str, torch.Tensor | int],
        static_reference_params: dict,
        velocity_time_factor: float,
        step: int | None,
        horizon_idx: int = 0,
        base_depth: torch.Tensor | None = None,
    ) -> dict:
        """Decode shared motion-query features into a constant-velocity dynamic Gaussian update."""
        return self._decode_velocity_from_static(
            shared_state,
            static_reference_params,
            velocity_time_factor,
            step,
            horizon_idx=horizon_idx,
            base_depth=base_depth,
        )

    def _decode_independent(
        self,
        z,
        gaussian_adapter=None,
        camera_params=None,
        current_observation=None,
        future_observation=None,
        step=None,
        base_depth: torch.Tensor | None = None,
        horizon_idx: int = 0,
        static_reference_params: dict | None = None,
        velocity_time_factor: float = 1.0,
        skip_horizon_embedding: bool = False,
        shared_state: dict[str, torch.Tensor | int] | None = None,
    ):
        """Decode VLM tokens into Gaussian parameters using the shared backbone state."""
        if shared_state is None:
            shared_state = self._prepare_shared_token_features(
                z,
                horizon_idx=horizon_idx,
            )

        if (
            self.use_velocity_future_gaussians
            and static_reference_params is not None
            and self.velocity_head is not None
        ):
            return self.decode_dynamic_gaussians_from_static(
                shared_state,
                static_reference_params,
                velocity_time_factor,
                step,
                horizon_idx=horizon_idx,
                base_depth=base_depth,
            )

        gaussian_params = self._decode_static_from_shared(
            shared_state,
            gaussian_adapter=gaussian_adapter,
            camera_params=camera_params,
            current_observation=current_observation,
            future_observation=future_observation,
            base_depth=base_depth,
        )

        if step is not None and step % 100 == 0:
            import logging

            depth_map = gaussian_params["depth_map"]
            scale_flat = gaussian_params["scales"]
            sh_flat = gaussian_params["sh"]
            logging.info(
                f"[IndependentDecoder] Step {step}: "
                f"depth=[{depth_map.min():.3f}, {depth_map.max():.3f}], "
                f"scales=[{scale_flat.min():.3f}, {scale_flat.max():.3f}], "
                f"sh=[{sh_flat.min():.3f}, {sh_flat.max():.3f}], N={scale_flat.shape[1]}"
            )

        return gaussian_params