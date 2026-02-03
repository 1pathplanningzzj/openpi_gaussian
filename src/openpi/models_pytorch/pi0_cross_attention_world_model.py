# zijian
# date 2026.02.03
# v1 to be debug and much test to do 
# Description: Cross-Attention based World Model for predicting z_{t+1} from temporal tokens.
"""
Cross-Attention based World Model for predicting z_{t+1} from temporal tokens.

Architecture:
    Q = Q_learnable + Linear(z_t ⊕ a_gt)
    z_{t+1} = CrossAttention(Q, K=[z_{t-2}, z_{t-1}, z_t], V=[z_{t-2}, z_{t-1}, z_t])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

logger = logging.getLogger(__name__)


class CrossAttentionPredictor(nn.Module):
    """
    Cross-Attention based predictor for z_{t+1}.
    
    Query: Q = Q_learnable + Linear(z_t ⊕ a_gt)
    Keys/Values: K/V = [z_{t-2}, z_{t-1}, z_t] (concatenated temporal tokens)
    Output: z_{t+1} = CrossAttention(Q, K, V)
    """
    
    def __init__(
        self,
        token_dim: int,
        action_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
    ):
        """
        Args:
            token_dim: Dimension of tokens (z_t, z_{t-1}, etc.)
            action_dim: Dimension of action vector
            num_heads: Number of attention heads
            num_layers: Number of cross-attention layers
            mlp_ratio: MLP expansion ratio
            dropout: Dropout rate
            use_positional_encoding: Whether to add positional encoding to temporal tokens
        """
        super().__init__()
        
        self.token_dim = token_dim
        self.action_dim = action_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.head_dim = token_dim // num_heads
        
        assert token_dim % num_heads == 0, f"token_dim ({token_dim}) must be divisible by num_heads ({num_heads})"
        
        # Number of query tokens (should match output tokens per frame)
        # We predict 100 tokens per frame (10x10 grid from pooling)
        self.num_query_tokens = 100
        
        # Learnable query tokens (similar to DETR object queries)
        # Shape: [1, 100, token_dim] - will be broadcasted to [B, 100, token_dim]
        # Each query token can learn to attend to different spatial/temporal locations
        self.query_tokens = nn.Parameter(torch.randn(1, self.num_query_tokens, token_dim) * 0.02)
        
        # Action + z_t fusion for query modulation
        # Instead of creating query from scratch, we modulate the learnable queries
        # Input: z_t [B, N, D] + action [B, A] -> Output: [B, 100, D] (modulation vector)
        # Strategy: Pool z_t to [B, D], concatenate with action [B, A] -> [B, D+A], then project to [B, 100, D]
        self.action_proj = nn.Linear(action_dim, token_dim)
        self.z_t_pool = nn.AdaptiveAvgPool1d(1)  # [B, N, D] -> [B, D, 1] -> [B, D]
        # Improved query_modulation with better numerical stability
        # Use a more gradual expansion to prevent large outputs
        modulation_layer1 = nn.Linear(token_dim * 2, token_dim * 2)
        modulation_norm = nn.LayerNorm(token_dim * 2)
        modulation_act = nn.GELU()
        modulation_layer2 = nn.Linear(token_dim * 2, token_dim * self.num_query_tokens)
        
        # Initialize the last layer with smaller weights for stability
        nn.init.normal_(modulation_layer2.weight, mean=0.0, std=0.01)
        nn.init.zeros_(modulation_layer2.bias)
        
        self.query_modulation = nn.Sequential(
            modulation_layer1,
            modulation_norm,
            modulation_act,
            modulation_layer2
        )
        
        # Positional encoding for temporal tokens (optional)
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            # Learnable positional embeddings for temporal positions: [t-2, t-1, t]
            self.temporal_pos_emb = nn.Parameter(torch.randn(3, token_dim) * 0.02)
        
        # Cross-attention layers
        self.layers = nn.ModuleList([
            CrossAttentionLayer(
                token_dim=token_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Output projection (optional, for residual connection)
        self.output_norm = nn.LayerNorm(token_dim)
        self.output_proj = nn.Linear(token_dim, token_dim)
        
    def forward(
        self,
        z_t: torch.Tensor,
        action: torch.Tensor,
        z_temporal: torch.Tensor = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Args:
            z_t: Current state tokens [B, 300, D] (3 frames) or [B, 100, D] (single frame)
                 If 300 tokens, extracts current frame (last 100 tokens) for prediction
            action: Action vector [B, A]
            z_temporal: Temporal tokens [B, 3, 100, D] or [B, 300, D] (optional)
                       Used as Keys/Values for cross-attention. If None, uses z_t.
            return_attention: Whether to return attention weights
        
        Returns:
            z_t1_pred: Predicted next state tokens [B, 100, D] (single frame prediction)
            attention_weights: (optional) Attention weights from last layer [B, H, 1, K]
        """
        B, N, D = z_t.shape
        
        # Extract current frame from z_t if it contains multiple frames
        # z_t might be [B, 300, D] (3 frames) or [B, 100, D] (single frame)
        # We want to predict the next frame, so we use the current frame (last 100 tokens)
        if N == 300:  # 3 frames × 100 tokens
            # Extract current frame (t): last 100 tokens
            z_t_current = z_t[:, -100:, :]  # [B, 100, D] - current frame tokens
            N_per_frame = 100
        else:
            # Single frame case
            z_t_current = z_t  # [B, N, D]
            N_per_frame = N
        
        # 1. Construct Query: Q = Q_learnable + Modulation(z_t_current ⊕ a_gt)
        # Start with learnable query tokens: [1, 100, D] -> [B, 100, D]
        query = self.query_tokens.expand(B, -1, -1)  # [B, 100, D]
        
        # Pool z_t_current: [B, N_per_frame, D] -> [B, D]
        z_t_pooled = z_t_current.mean(dim=1)  # [B, D]
        
        # Project action: [B, A] -> [B, D]
        action_emb = self.action_proj(action)  # [B, D]
        
        # Concatenate and project to modulation vector: [B, 2*D] -> [B, 100*D] -> [B, 100, D]
        query_input = torch.cat([z_t_pooled, action_emb], dim=-1)  # [B, 2*D]
        modulation = self.query_modulation(query_input)  # [B, 100*D]
        modulation = modulation.view(B, self.num_query_tokens, D)  # [B, 100, D]
        
        # Numerical stability: clamp modulation to prevent extreme values
        modulation = torch.clamp(modulation, min=-10.0, max=10.0)
        
        # Add modulation to learnable queries
        query = query + modulation  # [B, 100, D]
        
        # Numerical stability: check for NaN/Inf in query
        if torch.isnan(query).any() or torch.isinf(query).any():
            logger.warning("NaN/Inf detected in query after modulation. Replacing with zeros.")
            query = torch.where(torch.isnan(query) | torch.isinf(query), torch.zeros_like(query), query)
        
        # 2. Prepare Keys/Values from temporal tokens
        if z_temporal is not None:
            if z_temporal.ndim == 4:  # [B, 3, N, D] - separate frames
                B_temp, T, N_temp, D_temp = z_temporal.shape
                # Reshape to [B, 3*N, D]
                kv = z_temporal.view(B_temp, T * N_temp, D_temp)  # [B, 3*N, D]
                
                # Add positional encoding if enabled
                if self.use_positional_encoding:
                    # Expand temporal pos emb: [3, D] -> [B, 3, D] -> [B, 3*N, D] (broadcast)
                    pos_emb = self.temporal_pos_emb.unsqueeze(1).expand(3, N_temp, D_temp)  # [3, N, D]
                    pos_emb = pos_emb.reshape(3 * N_temp, D_temp).unsqueeze(0).expand(B_temp, -1, -1)  # [B, 3*N, D]
                    kv = kv + pos_emb
            else:  # [B, 3*N, D] - already concatenated
                kv = z_temporal
        else:
            # Fallback: use z_t as KV (all frames if z_t has 300 tokens, or single frame if 100)
            # If z_t has 300 tokens, use all frames for temporal context
            # If z_t has 100 tokens, use it directly
            kv = z_t  # [B, N, D] where N can be 300 or 100
        
        # 3. Apply cross-attention layers
        x = query  # [B, 1, D]
        attention_weights = None
        
        for i, layer in enumerate(self.layers):
            x, attn_weights = layer(x, kv, return_attention=True)
            if i == len(self.layers) - 1:
                attention_weights = attn_weights
        
        # 4. Query output is already [B, 100, D] (no need to expand)
        # x from cross-attention: [B, 100, D]
        # Add residual connection with z_t_current (current frame)
        # Output is single frame prediction: [B, 100, D]
        output_delta = self.output_proj(self.output_norm(x))
        
        # Numerical stability: clamp output delta to prevent extreme values
        output_delta = torch.clamp(output_delta, min=-10.0, max=10.0)
        
        # Check for NaN/Inf before residual connection
        if torch.isnan(output_delta).any() or torch.isinf(output_delta).any():
            logger.warning("NaN/Inf detected in output_delta. Replacing with zeros.")
            output_delta = torch.where(torch.isnan(output_delta) | torch.isinf(output_delta), torch.zeros_like(output_delta), output_delta)
        
        z_t1_pred = z_t_current + output_delta
        
        # Final check for NaN/Inf in output
        if torch.isnan(z_t1_pred).any() or torch.isinf(z_t1_pred).any():
            logger.warning("NaN/Inf detected in z_t1_pred. Replacing with z_t_current.")
            z_t1_pred = torch.where(torch.isnan(z_t1_pred) | torch.isinf(z_t1_pred), z_t_current, z_t1_pred)
        
        if return_attention:
            return z_t1_pred, attention_weights
        return z_t1_pred


class CrossAttentionLayer(nn.Module):
    """
    Single cross-attention layer with MLP.
    """
    
    def __init__(
        self,
        token_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        
        # Cross-attention
        self.norm_q = nn.LayerNorm(token_dim)
        self.norm_kv = nn.LayerNorm(token_dim)
        self.q_proj = nn.Linear(token_dim, token_dim)
        self.k_proj = nn.Linear(token_dim, token_dim)
        self.v_proj = nn.Linear(token_dim, token_dim)
        self.out_proj = nn.Linear(token_dim, token_dim)
        self.dropout = nn.Dropout(dropout)
        
        # MLP
        self.norm_mlp = nn.LayerNorm(token_dim)
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, int(token_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(token_dim * mlp_ratio), token_dim),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Args:
            query: Query tokens [B, Q, D]
            kv: Key/Value tokens [B, K, D]
            return_attention: Whether to return attention weights
        
        Returns:
            output: [B, Q, D]
            attention_weights: (optional) [B, H, Q, K]
        """
        B, Q, D = query.shape
        _, K, _ = kv.shape
        
        # Cross-attention
        q = self.q_proj(self.norm_q(query))  # [B, Q, D]
        k = self.k_proj(self.norm_kv(kv))   # [B, K, D]
        v = self.v_proj(self.norm_kv(kv))   # [B, K, D]
        
        # Reshape for multi-head: [B, Q, D] -> [B, Q, H, head_dim] -> [B, H, Q, head_dim]
        q = q.view(B, Q, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Q, head_dim]
        k = k.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, K, head_dim]
        v = v.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, K, head_dim]
        
        # Attention: [B, H, Q, head_dim] @ [B, H, head_dim, K] -> [B, H, Q, K]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Numerical stability: clamp attention scores to prevent overflow
        scores = torch.clamp(scores, min=-50.0, max=50.0)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values: [B, H, Q, K] @ [B, H, K, head_dim] -> [B, H, Q, head_dim]
        attn_output = torch.matmul(attn_weights, v)  # [B, H, Q, head_dim]
        
        # Reshape back: [B, H, Q, head_dim] -> [B, Q, H, head_dim] -> [B, Q, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Q, D)
        attn_output = self.out_proj(attn_output)
        
        # Residual connection
        # Numerical stability: clamp attn_output before adding
        attn_output = torch.clamp(attn_output, min=-10.0, max=10.0)
        output = query + attn_output
        
        # Check for NaN/Inf after attention
        if torch.isnan(output).any() or torch.isinf(output).any():
            logger.warning("NaN/Inf detected in output after attention. Replacing with query.")
            output = torch.where(torch.isnan(output) | torch.isinf(output), query, output)
        
        # MLP
        mlp_output = self.mlp(self.norm_mlp(output))
        # Numerical stability: clamp MLP output
        mlp_output = torch.clamp(mlp_output, min=-10.0, max=10.0)
        
        # Check for NaN/Inf in MLP output
        if torch.isnan(mlp_output).any() or torch.isinf(mlp_output).any():
            logger.warning("NaN/Inf detected in mlp_output. Replacing with zeros.")
            mlp_output = torch.where(torch.isnan(mlp_output) | torch.isinf(mlp_output), torch.zeros_like(mlp_output), mlp_output)
        
        output = output + mlp_output
        
        # Final check for NaN/Inf
        if torch.isnan(output).any() or torch.isinf(output).any():
            logger.warning("NaN/Inf detected in final output. Replacing with query.")
            output = torch.where(torch.isnan(output) | torch.isinf(output), query, output)
        
        if return_attention:
            return output, attn_weights
        return output


class CrossAttentionWorldModel(nn.Module):
    """
    World Model using Cross-Attention for predicting z_{t+1}.
    
    This replaces the flow-based ForwardInteractionPredictor with a cross-attention mechanism.
    """
    
    def __init__(
        self,
        token_dim: int,
        action_dim: int,
        use_vggt_decoder: bool = False,
        vggt_decoder=None,
        input_num_tokens: int = 100,
        target_num_tokens: int = 1369,
        vggt_embed_dim: int = 1024,
        cross_attn_config: dict = None,
    ):
        super().__init__()
        
        self.token_dim = token_dim
        self.action_dim = action_dim
        self.use_vggt_decoder = use_vggt_decoder
        self.vggt_decoder = vggt_decoder
        self.input_num_tokens = input_num_tokens
        self.target_num_tokens = target_num_tokens
        self.vggt_embed_dim = vggt_embed_dim
        
        # Cross-attention predictor
        default_config = {
            "num_heads": 8,
            "num_layers": 2,
            "mlp_ratio": 4.0,
            "dropout": 0.1,
            "use_positional_encoding": True,
        }
        if cross_attn_config:
            default_config.update(cross_attn_config)
        
        self.predictor = CrossAttentionPredictor(
            token_dim=token_dim,
            action_dim=action_dim,
            **default_config
        )
        
        # Keep inverse model for consistency
        from .pi0_world_model import InverseModel
        self.inverse_model = InverseModel(token_dim, action_dim)
        
        # Legacy decoder (None for VGGT decoder, kept for compatibility)
        self.decoder = None
        
        # Token upsampling for VGGT decoder (same as BiDirectionalWorldModel)
        if use_vggt_decoder and input_num_tokens != target_num_tokens:
            self.token_upsampler = True
            # Learnable upsampling: 10x10 -> 37x37 using transposed convolution
            # This learns how to recover spatial information from pooled tokens
            # Input: [B, D, 10, 10], Output: [B, D, 37, 37]
            # Strategy: Use interpolation to get to 37x37, then refine with conv
            self.token_upsampler_conv = nn.Sequential(
                # First interpolate to target size (learnable via conv after)
                # We'll do interpolation in forward, then apply conv refinement
                nn.Conv2d(token_dim, token_dim, kernel_size=3, padding=1),
                nn.GroupNorm(1, token_dim),  # LayerNorm equivalent for 4D (normalize over channels)
                nn.GELU(),
                nn.Conv2d(token_dim, token_dim, kernel_size=3, padding=1),
            )
        else:
            self.token_upsampler = None
            self.token_upsampler_conv = None
        
        # Projection for VGGT decoder
        if use_vggt_decoder and vggt_decoder is not None:
            target_dim = 2 * vggt_embed_dim  # 2048
            self.token_proj = nn.Linear(token_dim, target_dim)
        else:
            self.token_proj = None
    
    def forward(
        self,
        z_t: torch.Tensor,
        action: torch.Tensor,
        z_temporal: torch.Tensor = None,
        t: float = 0.0,
    ):
        """
        Predict z_{t+1} using cross-attention.
        
        Args:
            z_t: Current state tokens [B, N, D]
            action: Action vector [B, A]
            z_temporal: Temporal tokens [B, 3, N, D] or [B, 3*N, D]
                       If None, will use z_t only
            t: Time (kept for compatibility, not used in cross-attention)
        
        Returns:
            z_next_pred: [B, N, D] or [B, target_num_tokens, D] if upsampled
            details: Dict with attention weights and other info
        """
        # Predict using cross-attention
        z_next_pred, attention_weights = self.predictor(
            z_t, action, z_temporal, return_attention=True
        )
        
        # Upsample if needed for VGGT decoder
        if self.token_upsampler and z_next_pred.shape[1] == self.input_num_tokens:
            z_next_pred = self._upsample_tokens(z_next_pred)
        
        details = {
            "attention_weights": attention_weights,
            "z_t": z_t,
            "action": action,
        }
        
        return z_next_pred, details
    
    def _upsample_tokens(self, tokens):
        """
        Learnable upsampling from [B, 100, D] to [B, 1369, D].
        
        Uses transposed convolution to learn how to upsample tokens,
        which is better than simple bilinear interpolation as it can
        learn spatial relationships and recover information.
        """
        B, N, D = tokens.shape
        spatial_size = int(math.sqrt(N))
        
        if spatial_size * spatial_size == N:
            # Reshape to 2D: [B, N, D] -> [B, D, spatial_size, spatial_size]
            tokens_2d = tokens.view(B, D, spatial_size, spatial_size)
            
            # Use learnable upsampling if available
            if hasattr(self, 'token_upsampler_conv') and self.token_upsampler_conv is not None:
                # Strategy: Interpolate first, then refine with learnable conv
                # This allows the conv to learn how to recover spatial details
                tokens_interp = F.interpolate(
                    tokens_2d,
                    size=(37, 37),
                    mode='bilinear',
                    align_corners=False
                )
                # Refine with learnable convolution
                tokens_upsampled = self.token_upsampler_conv(tokens_interp)
            else:
                # Fallback: bilinear interpolation (less ideal)
                tokens_upsampled = F.interpolate(
                    tokens_2d,
                    size=(37, 37),
                    mode='bilinear',
                    align_corners=False
                )
            
            # Reshape back: [B, D, 37, 37] -> [B, 1369, D]
            tokens_final = tokens_upsampled.permute(0, 2, 3, 1).reshape(B, 37 * 37, D)
            return tokens_final
        else:
            # Fallback: simple expansion (not ideal)
            logger.warning(f"Cannot upsample {N} tokens to 1369. Returning as-is.")
            return tokens
    
    def compute_full_loss(
        self,
        z_t: torch.Tensor,
        action_t: torch.Tensor,
        z_t1_gt: torch.Tensor,
        z_temporal: torch.Tensor = None,
        lambda_fwd: float = 1.0,
        lambda_inv: float = 0.1,
    ):
        """
        Compute training loss.
        
        Args:
            z_t: Current state [B, 300, D] (3 frames) or [B, 100, D] (single frame)
            action_t: Action [B, A]
            z_t1_gt: Ground truth next state [B, 100, D] (single frame)
            z_temporal: Temporal tokens [B, 3, 100, D] or [B, 300, D] (optional)
            lambda_fwd: Forward loss weight
            lambda_inv: Inverse loss weight
        """
        # Forward prediction
        z_t1_pred_raw, attention_weights = self.predictor(
            z_t, action_t, z_temporal, return_attention=True
        )
        
        # Forward loss
        loss_fwd = F.mse_loss(z_t1_pred_raw, z_t1_gt)
        
        # Upsample for VGGT decoder if needed
        if self.token_upsampler and z_t1_pred_raw.shape[1] == self.input_num_tokens:
            z_t1_pred = self._upsample_tokens(z_t1_pred_raw)
        else:
            z_t1_pred = z_t1_pred_raw
        
        # Inverse consistency
        action_rec_from_gt = self.inverse_model(z_t, z_t1_gt)
        loss_inv = F.mse_loss(action_rec_from_gt, action_t)
        
        loss_total = lambda_fwd * loss_fwd + lambda_inv * loss_inv
        
        return {
            "loss_total": loss_total,
            "loss_fwd": loss_fwd,
            "loss_inv": loss_inv,
            "z_t1_pred": z_t1_pred,
            "action_recovered": action_rec_from_gt,
            "attention_weights": attention_weights,
        }
    
    def decode(self, z, future_observation=None, gaussian_adapter=None, camera_params=None, return_2d_maps=False, step=None):
        """
        Decode z to Gaussian parameters (same as BiDirectionalWorldModel).
        Reuses the decode logic from BiDirectionalWorldModel by importing and calling it.
        
        Note: This method is typically used to decode future frames (t+1), not current frames (t).
        The temporal information from t-2, t-1, t is already used during prediction.
        
        Args:
            z: [B, N, D] - latent tokens (will be upsampled if N != target_num_tokens)
                         Typically [B, 100, D] (single frame: z_t1_pred or z_t1_gt)
                         Can also be [B, 300, D] (3 frames: z_t), in which case extracts last frame
            future_observation: Observation object with images (required for VGGT decoder)
            gaussian_adapter: GaussianAdapter instance (required for VGGT decoder)
            camera_params: Optional camera parameters for unprojection
            return_2d_maps: If True, return 2D maps directly instead of converting to 3D
        """
        # Handle multi-frame tokens: if z has 300 tokens (3 frames × 100), extract last frame
        # This is only needed if z_t (current state) is passed, but typically we decode z_t1 (future)
        B, N, D = z.shape
        tokens_per_frame = 100  # Each frame has 100 tokens after pooling
        if N == 300:  # 3 frames × 100 tokens (unlikely for decode, but handle gracefully)
            # Extract last frame for decoding (if z_t was passed instead of z_t1)
            z = z[:, -100:, :]  # [B, 100, D]
        
        # Upsample tokens if needed (before calling decode)
        # VGGT decoder expects target_num_tokens (1369) patch tokens
        # Check if z has the expected single-frame token count (100 tokens per frame)
        if self.token_upsampler and z.shape[1] == tokens_per_frame:
            z_upsampled = self._upsample_tokens(z)
            # Verify upsampling succeeded
            if z_upsampled.shape[1] == self.target_num_tokens:
                z = z_upsampled
                # Debug: Log successful upsampling
                if step is not None and step % 40 == 0:
                    print(f"[DEBUG] Token upsampling: {tokens_per_frame} -> {self.target_num_tokens}, "
                          f"z.shape={z.shape}, z_upsampled.shape={z_upsampled.shape}")
            else:
                import warnings
                warnings.warn(
                    f"Upsampling failed: expected {self.target_num_tokens} tokens, got {z_upsampled.shape[1]}. "
                    f"Using original {z.shape[1]} tokens."
                )
                if step is not None and step % 40 == 0:
                    print(f"[WARNING] Token upsampling failed! Expected {self.target_num_tokens}, got {z_upsampled.shape[1]}. "
                          f"This may cause rendering issues.")
        elif self.token_upsampler:
            # Debug: Log why upsampling was skipped
            import logging
            logging.debug(
                f"[CrossAttentionWorldModel.decode] Skipping upsampling: "
                f"token_upsampler={self.token_upsampler}, z.shape[1]={z.shape[1]}, tokens_per_frame={tokens_per_frame}"
            )
        
        # Import BiDirectionalWorldModel to access its decode method
        # We'll create a minimal instance just for the decode method
        from .pi0_world_model import BiDirectionalWorldModel
        
        # Check if we have vggt_embed_dim attribute
        vggt_embed_dim = getattr(self, 'vggt_embed_dim', 1024)
        
        # Create a minimal BiDirectionalWorldModel instance for decode
        # We only need the decode method, so we can skip some initialization
        class MinimalWorldModel(BiDirectionalWorldModel):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
        
        temp_wm = MinimalWorldModel(
            token_dim=self.token_dim,
            action_dim=self.action_dim,
            use_vggt_decoder=self.use_vggt_decoder,
            vggt_decoder=self.vggt_decoder,
            input_num_tokens=self.input_num_tokens,
            target_num_tokens=self.target_num_tokens,
            vggt_embed_dim=vggt_embed_dim
        )
        
        # Copy token_proj if it exists
        if hasattr(self, 'token_proj') and self.token_proj is not None:
            temp_wm.token_proj = self.token_proj
        
        # Call decode method
        return temp_wm.decode(z, future_observation, gaussian_adapter, camera_params, return_2d_maps, step=step)
