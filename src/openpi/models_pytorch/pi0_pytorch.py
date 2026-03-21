# zijian
# date 2026.01.19
# Description: Main PI0 PyTorch model implementation with integrated 3D Gaussian Splatting (DF3DGS) support.
import logging
import math
from types import SimpleNamespace

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
from openpi.models_pytorch.pi0_vggt import GaussianAdapter
from openpi.models_pytorch.pi0_world_model import GaussianDecoder
# Import Gaussian Renderer
from openpi.models_pytorch.gaussian_renderer import GaussianRenderer, compute_multi_view_rendering_loss


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def _resolve_future_prediction_offsets(config, default_horizon: int) -> list[int]:
    """Resolve future rollout offsets in frame steps for logging and supervision labels."""
    raw_offsets = getattr(config, "future_prediction_offsets", None)
    if raw_offsets:
        return [max(1, int(value)) for value in raw_offsets]
    return list(range(1, max(1, int(default_horizon)) + 1))


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)

# Timestep Sampling Schedule
def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        # torch.compile disabled due to graph break issues with VGGT

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Visualization save directory for rendering comparisons
        # Can be set via VIS_SAVE_DIR environment variable, or defaults to ./visualizations/rendering
        import os
        self.vis_save_dir = os.environ.get("VIS_SAVE_DIR", "./visualizations/rendering_independent_decoder_test0320_lpips_future_5_frame")

        # --- 3D Gaussian Integration ---
        use_gaussian = getattr(config, "use_gaussian", False)
        use_single_frame_mode = getattr(config, "use_single_frame_mode", False)
        # Typically Gaussian features join the prefix, so they must match the VLM width
        # Disable LGPD for now to test training stability
        # Option to unfreeze VGGT encoder/decoder for reconstruction loss training
        unfreeze_vggt_encoder = getattr(config, "unfreeze_vggt_encoder", False)
        unfreeze_vggt_decoder_only = getattr(config, "unfreeze_vggt_decoder_only", True)  # Default: train decoder only
        use_lora = getattr(config, "use_lora", True)  # Default: use LoRA for VGGT encoder
        self.gaussian_adapter = GaussianAdapter(
            use_gaussian,
            paligemma_config.width,
            use_lgpd=False,
            use_single_frame_mode=use_single_frame_mode,
            unfreeze_encoder=unfreeze_vggt_encoder,
            unfreeze_decoder_only=unfreeze_vggt_decoder_only,
            use_lora=use_lora
        )
        
        # Current frame reconstruction loss weight
        self.current_frame_recon_loss_weight = getattr(config, "current_frame_recon_loss_weight", 0.5)
        
        # --- World Model Tokens in Prefix (NEW Architecture) ---
        # Add future query tokens to prefix for unified VLM processing
        self.use_world_tokens_in_prefix = getattr(config, "use_world_model", False) and use_gaussian
        self.future_prediction_offsets = _resolve_future_prediction_offsets(
            config, 5 if self.use_world_tokens_in_prefix else 1
        )
        self.future_prediction_horizon = len(self.future_prediction_offsets)
        if self.use_world_tokens_in_prefix:
            # === Priority 1: Aligned Future Query Tokens ===
            # 256 future query tokens (matches 16×16 decoder grid)
            self.future_token_count = 256
            self.future_grid_size = 16  # 16×16 spatial structure

            # World tokens removed — redundant with Gaussian tokens (768)
            self.world_token_count = 0
            self.world_token_proj = None

            # Future query tokens: learnable embeddings predicted by VLM
            # These represent "delta queries" for predicting changes from current frame
            self.future_query_tokens = nn.Parameter(
                torch.randn(1, self.future_token_count, paligemma_config.width) * 0.02
            )

            # Temporal base weights: learnable weights for combining [t-2, t-1, t] to form base
            # Initialized to favor recent frames: [0.1, 0.2, 0.7]
            self.temporal_base_w = nn.Parameter(torch.tensor([0.1, 0.2, 0.7]))

            # Delta scale raw parameter. We optimize a raw scalar and map it via softplus
            # so the effective scale stays positive. Initialize so softplus(raw) = 0.3.
            init_delta_scale = torch.tensor(0.3, dtype=torch.float32)
            self.delta_scale = nn.Parameter(torch.log(torch.expm1(init_delta_scale)))

            # Delta role embedding: signals that future tokens predict changes, not full state
            self.future_delta_embed = nn.Parameter(
                torch.randn(1, 1, paligemma_config.width) * 0.02
            )

            # === NEW: Spatial Positional Encoding (16×16 grid) ===
            # This gives Future tokens explicit spatial structure
            # matching the decoder's 16×16 input grid
            self.future_spatial_pos = nn.Parameter(
                torch.randn(self.future_grid_size, self.future_grid_size, paligemma_config.width) * 0.02
            )

            # Optional: Sinusoidal spatial encoding (more stable)
            self.use_sinusoidal_spatial = True
            if self.use_sinusoidal_spatial:
                self.register_buffer(
                    'future_spatial_sinusoidal',
                    self._create_2d_sinusoidal_encoding(
                        self.future_grid_size,
                        self.future_grid_size,
                        paligemma_config.width
                    )
                )

            # Roll out a single future seed into short-horizon latents [t+1, ..., t+H].
            self.future_horizon_embed = nn.Parameter(
                torch.randn(self.future_prediction_horizon, paligemma_config.width) * 0.02
            )
            # Add a self-attention layer for spatial consistency during rollout
            self.future_rollout_attn = nn.MultiheadAttention(
                embed_dim=paligemma_config.width,
                num_heads=8,
                batch_first=True
            )
            self.future_rollout_mlp = nn.Sequential(
                nn.LayerNorm(paligemma_config.width),
                nn.Linear(paligemma_config.width, paligemma_config.width),
                nn.SiLU(),
                nn.Linear(paligemma_config.width, paligemma_config.width),
            )

            logging.info(
                f"Initialized aligned future query tokens:\n"
                f"  - Token count: {self.future_token_count}\n"
                f"  - Spatial structure: {self.future_grid_size}×{self.future_grid_size}\n"
                f"  - Spatial positional encoding: {'Sinusoidal + Learnable' if self.use_sinusoidal_spatial else 'Learnable only'}\n"
                f"  - Aligned with VGGT tokens: 768 (3 frames × 256 tokens/frame)\n"
                f"  - Future rollout horizon: {self.future_prediction_horizon}\n"
                f"  - Future rollout offsets: {self.future_prediction_offsets}"
            )
        else:
            self.world_token_count = 0
            self.future_token_count = 0
            self.world_token_proj = None
            self.future_query_tokens = None
            self.future_horizon_embed = None
            self.future_rollout_mlp = None
        
        # --- VAE Token Compressor (Option B: Hybrid) ---
        # VAE is used only for reconstruction supervision (auxiliary loss)
        # Main pipeline still uses pooling/upsampling for efficiency
        use_vae_supervision = getattr(config, "use_vae_supervision", False)
        self.use_vae_supervision = use_vae_supervision
        if use_vae_supervision and use_gaussian:
            from .vae_token_compressor import VAETokenCompressor
            # VAE compresses 1369 tokens to 100 tokens
            # token_dim is the VGGT embed_dim (2048) or action_expert_width
            vae_token_dim = paligemma_config.width  # Match action_expert_width
            vae_beta = getattr(config, "vae_beta", 0.01)  # KL divergence weight
            self.vae_compressor = VAETokenCompressor(
                token_dim=vae_token_dim,
                latent_tokens=100,  # Match pooled tokens
                hidden_dim=512,
                beta=vae_beta
            )
            logging.info(f"Initialized VAE Token Compressor for supervision (beta={vae_beta})")
        else:
            self.vae_compressor = None

        # --- GaussianDecoder (replaces CrossAttentionWorldModel) ---
        self.use_velocity_future_gaussians = False
        if hasattr(config, "use_world_model") and config.use_world_model:
            logging.info("Initializing GaussianDecoder...")

            _vel_g = bool(getattr(config, "use_velocity_future_gaussians", False)) and self.use_world_tokens_in_prefix
            self.use_velocity_future_gaussians = _vel_g
            if _vel_g:
                logging.info(
                    "Velocity future Gaussians: horizon 0 = full decode; h>0 = static + v(z_h) "
                    f"(velocity_world_model_scale={getattr(config, 'velocity_world_model_scale', 0.15)})"
                )

            self.world_model = GaussianDecoder(
                token_dim=paligemma_config.width,
                input_num_tokens=256,
                # Disable action conditioning for future-query decoding.
                # This removes the action coupling path while keeping future-query tokens enabled.
                use_action_conditioning=False,
                use_incremental_depth=getattr(config, "use_incremental_depth", True),
                future_prediction_horizon=self.future_prediction_horizon,
                use_velocity_future_gaussians=_vel_g,
                velocity_world_model_scale=float(getattr(config, "velocity_world_model_scale", 0.15)),
            )

            # Initialize Gaussian Renderer (sh_degree=1 for DC + 1st order SH)
            try:
                self.gaussian_renderer = GaussianRenderer(image_size=224, sh_degree=1, scale_factor=1.0)
                logging.info("Gaussian Renderer initialized with sh_degree=1 (DC + 1st order) for World Model supervision.")
            except ImportError:
                self.gaussian_renderer = None
                logging.warning("Gaussian Renderer not available. Skipping rendering loss.")

            # Initialize LPIPS perceptual loss (optional)
            self.lpips_fn = None
            self.lpips_weight = getattr(config, "lpips_weight", 0.1)
            if getattr(config, "use_lpips", False):
                try:
                    import lpips
                    # Initialize LPIPS and move to the same device as the model
                    self.lpips_fn = lpips.LPIPS(net='vgg')
                    # Set to eval mode and freeze parameters
                    self.lpips_fn.eval()
                    for param in self.lpips_fn.parameters():
                        param.requires_grad = False
                    logging.info(f"LPIPS perceptual loss initialized with weight={self.lpips_weight}")
                except ImportError:
                    logging.warning("lpips package not installed. Install with: pip install lpips")
        else:
            self.world_model = None
            self.gaussian_renderer = None

        # Initialize render loss weight (can be changed dynamically for staged training)
        self.render_loss_weight = getattr(config, "render_loss_weight", 0.1)  # 降低render loss权重，让action loss主导
        self.depth_loss_weight = getattr(config, "depth_loss_weight", 0.1)
        self.delta_depth_loss_weight = getattr(config, "delta_depth_loss_weight", 0.0)
        # Default to supervising delta depth on every predicted future horizon.
        self.delta_depth_first_horizon_only = bool(getattr(config, "delta_depth_first_horizon_only", False))
        self.future_delta_reg_weight = getattr(config, "future_delta_reg_weight", 1e-4)
        self.future_horizon_curriculum_steps = max(0, int(getattr(config, "future_horizon_curriculum_steps", 5_000)))
        self.future_horizon_early_min_weight = float(getattr(config, "future_horizon_early_min_weight", 0.2))
        self._action_loss_enabled = True  # Can be toggled by freeze/unfreeze_action_expert
        # -------------------------------

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def freeze_world_model(self):
        """Freeze world model (Gaussian Decoder) for stage 1 training (action-only)."""
        if self.world_model is not None:
            for param in self.world_model.parameters():
                param.requires_grad = False
            logging.info("Froze World Model (Gaussian Decoder) - Stage 1: Action-only training")

        if hasattr(self, 'gaussian_adapter') and self.gaussian_adapter is not None:
            for param in self.gaussian_adapter.parameters():
                param.requires_grad = False
            logging.info("Froze Gaussian Adapter")

    def unfreeze_world_model(self):
        """Unfreeze world model for stage 2 training (joint training)."""
        if self.world_model is not None:
            for param in self.world_model.parameters():
                param.requires_grad = True
            logging.info("Unfroze World Model (Gaussian Decoder) - Stage 2: Joint training")

        if hasattr(self, 'gaussian_adapter') and self.gaussian_adapter is not None:
            for param in self.gaussian_adapter.parameters():
                param.requires_grad = True
            logging.info("Unfroze Gaussian Adapter")

    def _get_action_mlp_modules(self):
        """Get the time/state MLP modules based on pi05 mode."""
        if self.pi05:
            return [self.time_mlp_in, self.time_mlp_out]
        else:
            return [self.state_proj, self.action_time_mlp_in, self.action_time_mlp_out]

    def freeze_action_expert(self):
        """Freeze action expert and related projections for Stage 1 (render+depth only)."""
        for param in self.paligemma_with_expert.gemma_expert.parameters():
            param.requires_grad = False
        for module in [self.action_in_proj, self.action_out_proj] + self._get_action_mlp_modules():
            for param in module.parameters():
                param.requires_grad = False
        self._action_loss_enabled = False
        logging.info("Froze Action Expert + projections, disabled action loss")

    def unfreeze_action_expert(self):
        """Unfreeze action expert for Stage 2 (joint training)."""
        for param in self.paligemma_with_expert.gemma_expert.parameters():
            param.requires_grad = True
        for module in [self.action_in_proj, self.action_out_proj] + self._get_action_mlp_modules():
            for param in module.parameters():
                param.requires_grad = True
        self._action_loss_enabled = True
        logging.info("Unfroze Action Expert + projections, enabled action loss")

    def set_render_loss_weight(self, weight: float):
        """Dynamically set render loss weight for stage-based training."""
        self.render_loss_weight = weight
        logging.info(f"Set render_loss_weight = {weight}")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)


    def _create_2d_sinusoidal_encoding(self, height, width, embed_dim):
        """Create 2D sinusoidal positional encoding for spatial grid.

        Args:
            height: Grid height (e.g., 16)
            width: Grid width (e.g., 16)
            embed_dim: Embedding dimension

        Returns:
            [height, width, embed_dim] - 2D positional encoding
        """
        # Create position indices
        y_pos = torch.arange(height).unsqueeze(1).float()  # [H, 1]
        x_pos = torch.arange(width).unsqueeze(0).float()   # [1, W]

        # Expand to grid
        y_grid = y_pos.expand(height, width)  # [H, W]
        x_grid = x_pos.expand(height, width)  # [H, W]

        # Frequency bands
        half_dim = embed_dim // 4  # Split embed_dim into 4 parts (y_sin, y_cos, x_sin, x_cos)
        div_term = torch.exp(torch.arange(0, half_dim).float() * -(math.log(10000.0) / half_dim))

        # Compute sinusoidal encoding for y and x
        pe = torch.zeros(height, width, embed_dim)

        # Y position encoding (first half of embed_dim)
        pe[:, :, 0:half_dim] = torch.sin(y_grid.unsqueeze(-1) * div_term)
        pe[:, :, half_dim:2*half_dim] = torch.cos(y_grid.unsqueeze(-1) * div_term)

        # X position encoding (second half of embed_dim)
        pe[:, :, 2*half_dim:3*half_dim] = torch.sin(x_grid.unsqueeze(-1) * div_term)
        pe[:, :, 3*half_dim:4*half_dim] = torch.cos(x_grid.unsqueeze(-1) * div_term)

        return pe

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _temporal_context_future_indices(self, time_dim: int, use_single_frame_mode: bool) -> tuple[list[int], list[int], int]:
        """Pick context / future frame indices along T.

        When world-model tokens are enabled, futures are at **current + offset** for each entry in
        ``future_prediction_offsets`` (e.g. [2,5,10,15,20] → frames c+2, c+5, ...), with
        ``c`` chosen as the last context index so that the largest offset still lies in [0, T-1].

        Otherwise (no world model), keeps the legacy **consecutive** tail window.
        """
        context_frames = 1 if use_single_frame_mode else min(3, time_dim)

        def _legacy_consecutive() -> tuple[list[int], list[int], int]:
            future_steps = max(0, min(self.future_prediction_horizon, time_dim - context_frames))
            context_start = max(0, time_dim - future_steps - context_frames)
            ctx = list(range(context_start, context_start + context_frames))
            fut = list(range(context_start + context_frames, context_start + context_frames + future_steps))
            return ctx, fut, ctx[-1]

        if not self.use_world_tokens_in_prefix:
            return _legacy_consecutive()

        offsets = self.future_prediction_offsets
        max_off = max(offsets)
        c_end = time_dim - 1 - max_off
        need_t = max_off + context_frames
        bad = c_end < 0 or (context_frames > 1 and c_end < context_frames - 1)
        ctx_start = c_end - (context_frames - 1)
        fut_try = [c_end + o for o in offsets]
        if not bad:
            bad = any(i < 0 or i >= time_dim for i in fut_try)

        if bad:
            if not getattr(self, "_warned_future_offset_fallback", False):
                logging.warning(
                    "World model: time_dim=%s cannot fit future_prediction_offsets=%s (need T >= %s with "
                    "this context_frames=%s). Using consecutive tail fallback — increase sequence length in data.",
                    time_dim,
                    offsets,
                    need_t,
                    context_frames,
                )
                self._warned_future_offset_fallback = True
            return _legacy_consecutive()

        ctx = list(range(ctx_start, c_end + 1))
        return ctx, fut_try, c_end

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""

        # Check if single-frame mode is enabled
        use_single_frame_mode = getattr(self.config, "use_single_frame_mode", False)

        # --- Handle Future Split for 3DGS World Model ---
        future_observation = None
        # Check if first image has T dimension (ndim=5 for B,T,H,W,C now due to model.py fix)
        if observation.images:
            # Inspect first image to detect time dimension
            img_val = next(iter(observation.images.values()))
            # DEBUG PRINT
            if train and torch.rand(1).item() < 0.01:
                 print(f"DEBUG: _preprocess_observation. img_val ndim={img_val.ndim}, shape={img_val.shape}, use_single_frame_mode={use_single_frame_mode}")

            if img_val.ndim == 5:
                # [B, T, H, W, C]
                time_dim = img_val.shape[1]
                context_indices, future_indices, idx_curr = self._temporal_context_future_indices(
                    time_dim, use_single_frame_mode
                )

                # DEBUG PRINT
                if train and torch.rand(1).item() < 0.01:
                     print(
                         f"DEBUG: Found Time Dim {time_dim}. "
                         f"context_indices={context_indices}, future_indices={future_indices}, "
                         f"offsets={self.future_prediction_offsets if self.use_world_tokens_in_prefix else 'n/a'}"
                     )

                if future_indices:
                    curr_imgs, fut_imgs = {}, {}
                    # Save original temporal images for visualization before slicing
                    raw_temporal_images = {}

                    def _select_time_slices(tensor, indices, *, collapse_single=False):
                        if tensor is None or tensor.ndim < 2:
                            return tensor
                        selected = tensor[:, indices]
                        if collapse_single and len(indices) == 1:
                            return selected[:, 0]
                        return selected

                    # Handle single-frame mode vs multi-frame mode
                    for k, v in observation.images.items():
                        raw_ix = sorted(set(context_indices + future_indices))
                        raw_temporal_images[k] = _select_time_slices(v, raw_ix)
                        curr_imgs[k] = _select_time_slices(v, context_indices, collapse_single=use_single_frame_mode)
                        fut_imgs[k] = _select_time_slices(v, future_indices, collapse_single=len(future_indices) == 1)

                    curr_state = observation.state
                    fut_state = observation.state

                    # Handle state [B, T, D]
                    if observation.state is not None:
                        if observation.state.ndim == 3:
                            # State has temporal dimension [B, T, D]
                            if observation.state.shape[1] == time_dim:
                                curr_state = observation.state[:, idx_curr]
                                fut_state = _select_time_slices(
                                    observation.state, future_indices, collapse_single=len(future_indices) == 1
                                )
                        elif observation.state.ndim == 2:
                            curr_state = observation.state
                            if len(future_indices) == 1:
                                fut_state = observation.state
                            else:
                                fut_state = observation.state.unsqueeze(1).expand(-1, len(future_indices), -1)

                    # Clone observation for future
                    # Note: We must also slice the masks and prompts to match the single-step batch dimension,
                    # otherwise jaxtyping will complain about mismatched *b dimensions (e.g. mask [B, T] vs image [B, H, W, C])

                    # 1. Slice Image Masks
                    fut_masks = {}
                    curr_masks = {}
                    for k, v in observation.image_masks.items():
                        if v.ndim == 2: # [B, T]
                            curr_masks[k] = _select_time_slices(v, context_indices, collapse_single=use_single_frame_mode)
                            fut_masks[k] = _select_time_slices(v, future_indices, collapse_single=len(future_indices) == 1)
                        else: # [B] - assume valid for all steps
                            curr_masks[k] = v
                            fut_masks[k] = v

                    # 2. Slice Prompts
                    curr_prompt = observation.tokenized_prompt
                    fut_prompt = observation.tokenized_prompt
                    if curr_prompt is not None:
                        if curr_prompt.ndim == 3: # [B, T, L]
                            curr_prompt = _select_time_slices(
                                curr_prompt, context_indices, collapse_single=use_single_frame_mode
                            )
                            fut_prompt = _select_time_slices(
                                observation.tokenized_prompt,
                                future_indices,
                                collapse_single=len(future_indices) == 1,
                            )
                        elif curr_prompt.ndim == 2: # [B, L]
                            if not use_single_frame_mode:
                                curr_prompt = curr_prompt.unsqueeze(1).expand(-1, len(context_indices), -1)
                            if len(future_indices) > 1:
                                fut_prompt = observation.tokenized_prompt.unsqueeze(1).expand(-1, len(future_indices), -1)

                    curr_prompt_mask = observation.tokenized_prompt_mask
                    fut_prompt_mask = observation.tokenized_prompt_mask
                    if curr_prompt_mask is not None:
                        if curr_prompt_mask.ndim == 3: # [B, T, L]
                            curr_prompt_mask = _select_time_slices(
                                curr_prompt_mask, context_indices, collapse_single=use_single_frame_mode
                            )
                            fut_prompt_mask = _select_time_slices(
                                observation.tokenized_prompt_mask,
                                future_indices,
                                collapse_single=len(future_indices) == 1,
                            )
                        elif curr_prompt_mask.ndim == 2: # [B, L]
                            if not use_single_frame_mode:
                                curr_prompt_mask = curr_prompt_mask.unsqueeze(1).expand(-1, len(context_indices), -1)
                            if len(future_indices) > 1:
                                fut_prompt_mask = observation.tokenized_prompt_mask.unsqueeze(1).expand(
                                    -1, len(future_indices), -1
                                )

                    # Handle depth data if available
                    fut_depth = None
                    curr_depth = None
                    if hasattr(observation, 'depth') and observation.depth is not None:
                        # observation.depth: [B, T, 1, H, W] or [B, 1, H, W]
                        if observation.depth.ndim == 5:  # [B, T, 1, H, W]
                            if observation.depth.shape[1] == time_dim:
                                curr_depth = _select_time_slices(
                                    observation.depth, context_indices, collapse_single=use_single_frame_mode
                                )
                                fut_depth = _select_time_slices(
                                    observation.depth, future_indices, collapse_single=len(future_indices) == 1
                                )
                        elif observation.depth.ndim == 4:  # [B, 1, H, W]
                            # Single frame depth, use as is for both
                            curr_depth = observation.depth
                            if len(future_indices) == 1:
                                fut_depth = observation.depth
                            else:
                                fut_depth = observation.depth.unsqueeze(1).expand(-1, len(future_indices), -1, -1, -1)
                        else:
                            logging.warning(f"Unexpected depth shape: {observation.depth.shape}, ndim={observation.depth.ndim}")

                    future_observation = observation.replace(
                        images=fut_imgs,
                        state=fut_state,
                        image_masks=fut_masks,
                        tokenized_prompt=fut_prompt,
                        tokenized_prompt_mask=fut_prompt_mask,
                        depth=fut_depth  # Set depth directly to avoid type check error
                    )
                    # Update current observation
                    observation = observation.replace(
                        images=curr_imgs,
                        state=curr_state,
                        image_masks=curr_masks,
                        tokenized_prompt=curr_prompt,
                        tokenized_prompt_mask=curr_prompt_mask,
                        depth=curr_depth  # Set depth for current frame
                    )
                    
                    # Also preprocess future observation (normalization etc)
                    future_observation = _preprocessing.preprocess_observation_pytorch(future_observation, train=False)

        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)

        # Store the preprocessed observation for later use in World Model
        # This ensures that when _prepare_gaussian_inputs is called later, it uses the observation
        # with preserved temporal dimension, not the original one
        preprocessed_observation = observation

        # Attach raw temporal images for visualization if available
        if 'raw_temporal_images' in locals() and raw_temporal_images:
            # Store as a separate attribute for visualization
            preprocessed_observation.raw_temporal_images = raw_temporal_images

        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
            future_observation,
            preprocessed_observation  # Return preprocessed observation for World Model
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def _get_future_horizon_loss_weights(
        self, step: int | None, horizon: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Anneal future-rollout horizon weights from near-heavy to uniform."""
        if horizon <= 0:
            return torch.zeros(0, device=device, dtype=dtype)

        uniform = torch.ones(horizon, device=device, dtype=torch.float32)
        if step is None or horizon == 1 or self.future_horizon_curriculum_steps <= 0:
            return uniform.to(dtype=dtype)

        tail_weight = min(max(self.future_horizon_early_min_weight, 1e-3), 1.0)
        early = torch.linspace(1.0, tail_weight, horizon, device=device, dtype=torch.float32)
        early = early / early.mean().clamp_min(1e-6)

        progress = min(max(step, 0), self.future_horizon_curriculum_steps) / float(self.future_horizon_curriculum_steps)
        weights = (1.0 - progress) * early + progress * uniform
        return weights.to(dtype=dtype)

    def _log_future_rollout_diagnostics(
        self,
        step: int | None,
        future_seed_tokens: torch.Tensor,
        z_future_pred_tokens: torch.Tensor,
        per_step_delta: torch.Tensor,
        horizon_weights: torch.Tensor,
    ) -> None:
        """Log latent-space rollout diagnostics on the visualization cadence."""
        if step is None or step % 400 != 0:
            return

        import torch.distributed as dist

        if dist.is_initialized() and dist.get_rank() != 0:
            return

        delta_fp32 = per_step_delta.float()
        future_fp32 = z_future_pred_tokens.float()
        delta_rms_by_h = delta_fp32.pow(2).mean(dim=(0, 2, 3)).sqrt()

        inter_horizon_l2 = torch.zeros(0, device=future_fp32.device, dtype=torch.float32)
        t1_tH_l2 = torch.zeros((), device=future_fp32.device, dtype=torch.float32)
        if future_fp32.shape[1] > 1:
            inter_horizon_l2 = (future_fp32[:, 1:] - future_fp32[:, :-1]).pow(2).mean(dim=(0, 2, 3)).sqrt()
            t1_tH_l2 = (future_fp32[:, -1] - future_fp32[:, 0]).pow(2).mean().sqrt()

        seed_to_tH_l2 = (future_fp32[:, -1] - future_seed_tokens.float()).pow(2).mean().sqrt()
        weights_str = ", ".join(f"{value:.3f}" for value in horizon_weights.detach().cpu().tolist())
        delta_rms_str = ", ".join(f"{value:.6f}" for value in delta_rms_by_h.detach().cpu().tolist())
        inter_l2_str = ", ".join(f"{value:.6f}" for value in inter_horizon_l2.detach().cpu().tolist()) or "N/A"
        logging.info(
            f"Step {step}: Future Rollout Diagnostics | "
            f"raw_delta_abs_mean={delta_fp32.abs().mean().item():.6f}, "
            f"raw_delta_rms={delta_fp32.pow(2).mean().sqrt().item():.6f}, "
            f"raw_delta_abs_max={delta_fp32.abs().max().item():.6f}, "
            f"delta_rms_by_h=[{delta_rms_str}], "
            f"inter_horizon_l2=[{inter_l2_str}], "
            f"t1_tH_l2={t1_tH_l2.item():.6f}, "
            f"seed_to_tH_l2={seed_to_tH_l2.item():.6f}, "
            f"horizon_weights=[{weights_str}]"
        )

    def _log_future_rollout_pixel_lpips(self, step: int | None, rendered_obs_seq: list[dict[str, torch.Tensor]]) -> None:
        """Log pixel-space LPIPS between the first and last rendered future predictions."""
        if step is None or step % 400 != 0 or len(rendered_obs_seq) < 2:
            return

        import torch.distributed as dist

        if dist.is_initialized() and dist.get_rank() != 0:
            return

        first_render = rendered_obs_seq[0]
        last_render = rendered_obs_seq[-1]
        shared_keys = list(set(first_render.keys()) & set(last_render.keys()))
        if not shared_keys:
            return

        view_key = "agent_image" if "agent_image" in shared_keys else shared_keys[0]
        if self.lpips_fn is None:
            logging.info(
                f"Step {step}: Future Rollout Pixel LPIPS(t1,t{len(rendered_obs_seq)})[{view_key}] unavailable (LPIPS disabled)"
            )
            return

        pred_t1 = torch.clamp(first_render[view_key], 0.0, 1.0)
        pred_tH = torch.clamp(last_render[view_key], 0.0, 1.0)
        with torch.no_grad():
            lpips_value = self.lpips_fn(pred_t1 * 2.0 - 1.0, pred_tH * 2.0 - 1.0).mean()
        logging.info(
            f"Step {step}: Future Rollout Pixel LPIPS(t1,t{len(rendered_obs_seq)})[{view_key}] = {lpips_value.item():.6f}"
        )

    def _rollout_future_latents(self, future_seed_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive rollout: z_{t+h} = z_{t+h-1} + delta_h.

        Each step's MLP input is the *previous* latent + horizon embedding,
        so the model can condition on accumulated state rather than
        predicting all horizons independently from the same seed.
        """
        if self.future_rollout_mlp is None or self.future_horizon_embed is None:
            zero_delta = torch.zeros_like(future_seed_tokens[:, None, :, :])
            return future_seed_tokens[:, None, :, :], zero_delta

        bsize, num_tokens, width = future_seed_tokens.shape
        horizon = self.future_horizon_embed.shape[0]

        rollout_tokens = []
        rollout_deltas = []
        running = future_seed_tokens  # [B, N, D]
        for h in range(horizon):
            mlp_input = running + self.future_horizon_embed[h][None, None, :]  # [B, N, D]
            
            # 1. Spatial Self-Attention for structural consistency
            attn_out, _ = self.future_rollout_attn(mlp_input, mlp_input, mlp_input)
            mlp_input = mlp_input + attn_out  # Residual connection
            
            # 2. Point-wise MLP
            delta_h = self.future_rollout_mlp(mlp_input.reshape(bsize * num_tokens, width))
            delta_h = delta_h.view(bsize, num_tokens, width)
            
            running = running + delta_h
            rollout_tokens.append(running)
            rollout_deltas.append(delta_h)

        return torch.stack(rollout_tokens, dim=1), torch.stack(rollout_deltas, dim=1)  # [B, H, N, D]

    def _get_temporal_observation_length(self, observation) -> int:
        """Infer the temporal length stored in a future observation sequence."""
        if observation is None:
            return 0

        if hasattr(observation, "images"):
            for value in observation.images.values():
                if value.ndim == 5:
                    return value.shape[1]
        if hasattr(observation, "depth") and observation.depth is not None and observation.depth.ndim == 5:
            return observation.depth.shape[1]
        if hasattr(observation, "state") and observation.state is not None and observation.state.ndim == 3:
            return observation.state.shape[1]
        return 0

    def _slice_temporal_observation(self, observation, index: int):
        """Extract one horizon from a temporally-stacked observation container."""
        if observation is None:
            return None

        images = {}
        for key, value in observation.images.items():
            if value.ndim == 5:
                images[key] = value[:, index]
            else:
                images[key] = value

        image_masks = {}
        for key, value in observation.image_masks.items():
            if value.ndim == 2:
                image_masks[key] = value[:, index]
            else:
                image_masks[key] = value

        state = observation.state[:, index] if observation.state is not None and observation.state.ndim == 3 else observation.state
        tokenized_prompt = (
            observation.tokenized_prompt[:, index]
            if observation.tokenized_prompt is not None and observation.tokenized_prompt.ndim == 3
            else observation.tokenized_prompt
        )
        tokenized_prompt_mask = (
            observation.tokenized_prompt_mask[:, index]
            if observation.tokenized_prompt_mask is not None and observation.tokenized_prompt_mask.ndim == 3
            else observation.tokenized_prompt_mask
        )
        depth = observation.depth[:, index] if getattr(observation, "depth", None) is not None and observation.depth.ndim == 5 else getattr(observation, "depth", None)

        return SimpleNamespace(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
            token_ar_mask=getattr(observation, "token_ar_mask", None),
            token_loss_mask=getattr(observation, "token_loss_mask", None),
            depth=depth,
        )

    def _compute_world_model_frame_loss(
        self,
        z_next: torch.Tensor,
        future_target,
        preprocessed_observation,
        step: int | None = None,
        time_suffix: str = "_t1_pred_vlm",
        visualize: bool = True,
        horizon_idx: int = 0,
        static_gaussian_params: dict | None = None,
        velocity_time_factor: float = 1.0,
        return_gaussian_params: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Decode one future latent and supervise it with rendering + depth losses."""
        device = z_next.device
        total_loss = torch.zeros((), dtype=torch.float32, device=device)
        if self.world_model is None or self.gaussian_renderer is None or future_target is None:
            return (total_loss, {}) if return_gaussian_params else total_loss

        import torch.distributed as dist

        is_main_process = not dist.is_initialized() or dist.get_rank() == 0

        gaussian_params: dict = {}
        try:
            camera_params_for_decode = self._get_camera_params_for_view("agent", device, z_next.shape[0])
            base_depth = getattr(preprocessed_observation, "depth", None)
            if base_depth is not None and base_depth.ndim == 5:
                base_depth = base_depth[:, -1]

            gaussian_params = self.world_model.decode(
                z_next.float(),
                future_observation=future_target,
                gaussian_adapter=self.gaussian_adapter,
                camera_params=camera_params_for_decode,
                step=step,
                current_observation=preprocessed_observation,
                base_depth=base_depth,
                horizon_idx=horizon_idx,
                static_reference_params=static_gaussian_params,
                velocity_time_factor=velocity_time_factor,
            )

            depth_map = gaussian_params.pop("depth_map", None)
            depth_delta_map = gaussian_params.pop("depth_delta_map", None)

            if step is not None and step % 400 == 0:
                depth_delta_mean = float("nan")
                depth_delta_abs_mean = float("nan")
                depth_delta_abs_max = float("nan")
                if depth_delta_map is not None:
                    delta_stats_map = depth_delta_map.float()
                    depth_delta_mean = delta_stats_map.mean().item()
                    depth_delta_abs_mean = delta_stats_map.abs().mean().item()
                    depth_delta_abs_max = delta_stats_map.abs().max().item()
                logging.info(
                    f"Step {step}{time_suffix}: depth_map={depth_map is not None}, "
                    f"depth_delta_map={depth_delta_map is not None}, "
                    f"has_depth_attr={hasattr(future_target, 'depth')}, "
                    f"depth_value={future_target.depth is not None if hasattr(future_target, 'depth') else 'N/A'}, "
                    f"depth_delta_mean={depth_delta_mean:.6f}, "
                    f"depth_delta_abs_mean={depth_delta_abs_mean:.6f}, "
                    f"depth_delta_abs_max={depth_delta_abs_max:.6f}"
                )

            if depth_map is not None and hasattr(future_target, "depth") and future_target.depth is not None:
                gt_depth = future_target.depth
                if depth_map.shape != gt_depth.shape:
                    depth_map = F.interpolate(
                        depth_map, size=gt_depth.shape[-2:], mode="bilinear", align_corners=False
                    )
                depth_loss = F.l1_loss(depth_map, gt_depth)
                if torch.isfinite(depth_loss):
                    total_loss = total_loss + self.depth_loss_weight * depth_loss
                if step is not None and step % 400 == 0:
                    logging.info(
                        f"Step {step}{time_suffix}: Depth Loss = {depth_loss.item():.6f}, "
                        f"weight={self.depth_loss_weight}"
                    )

                if (
                    depth_delta_map is not None
                    and base_depth is not None
                    and self.delta_depth_loss_weight > 0.0
                    and (not self.delta_depth_first_horizon_only or horizon_idx == 0)
                ):
                    if base_depth.ndim == 3:
                        base_depth_for_delta = base_depth.unsqueeze(1)
                    else:
                        base_depth_for_delta = base_depth
                    if base_depth_for_delta.shape[-2:] != gt_depth.shape[-2:]:
                        base_depth_for_delta = F.interpolate(
                            base_depth_for_delta, size=gt_depth.shape[-2:], mode="bilinear", align_corners=False
                        )
                    gt_depth_delta = gt_depth - base_depth_for_delta
                    if depth_delta_map.shape != gt_depth_delta.shape:
                        depth_delta_map = F.interpolate(
                            depth_delta_map, size=gt_depth_delta.shape[-2:], mode="bilinear", align_corners=False
                        )
                    delta_depth_loss = F.smooth_l1_loss(depth_delta_map, gt_depth_delta)
                    if torch.isfinite(delta_depth_loss):
                        total_loss = total_loss + self.delta_depth_loss_weight * delta_depth_loss
                    if step is not None and step % 400 == 0:
                        logging.info(
                            f"Step {step}{time_suffix}: Delta Depth Loss = {delta_depth_loss.item():.6f}, "
                            f"weight={self.delta_depth_loss_weight}, horizon_idx={horizon_idx}"
                        )

            for key, value in gaussian_params.items():
                if torch.isnan(value).any() or torch.isinf(value).any():
                    gaussian_params[key] = torch.where(torch.isfinite(value), value, torch.zeros_like(value))

            target_obs = {}
            cam_params_dict = {}
            valid_views = []

            for key, value in future_target.images.items():
                img_tensor = value
                view_name = None
                if key == "image" or "agent" in key or "high" in key or "cam_high" in key or "exterior" in key or "base" in key:
                    view_name = "agent"
                elif "left_wrist" in key or "wrist_left" in key:
                    view_name = "wrist"
                elif "right_wrist" in key or "wrist_right" in key:
                    if img_tensor.min() == img_tensor.max() == -1.0:
                        continue
                    view_name = "wrist"
                elif "wrist" in key or "bravo" in key:
                    view_name = "wrist"

                if view_name:
                    if img_tensor.shape[1] != 3 and img_tensor.shape[-1] == 3:
                        img_tensor = img_tensor.permute(0, 3, 1, 2)
                    img_tensor = (img_tensor + 1.0) / 2.0
                    view_key = f"{view_name}_image"
                    if view_key not in target_obs:
                        target_obs[view_key] = img_tensor
                        cam_params_dict[view_name] = self._get_camera_params_for_view(view_name, device, z_next.shape[0])
                        valid_views.append(view_name)

            if valid_views:
                render_views = ["agent"] if "agent" in valid_views else valid_views
                render_loss, render_loss_dict = compute_multi_view_rendering_loss(
                    gaussian_params,
                    target_obs,
                    cam_params_dict,
                    self.gaussian_renderer,
                    view_names=render_views,
                    step=step,
                    depth_map=depth_map,
                    lambda_scale=0.001,
                    lambda_opacity=0.01,
                    lambda_edge_smooth=0.01,
                    lpips_fn=self.lpips_fn,
                    lpips_weight=self.lpips_weight,
                )
                if torch.isfinite(render_loss):
                    total_loss = total_loss + self.render_loss_weight * render_loss.to(total_loss.dtype)

                if "sh" in gaussian_params:
                    sh_dc = gaussian_params["sh"]
                    sh_dc_reg = (sh_dc ** 2).mean() * 0.01
                    total_loss = total_loss + sh_dc_reg
                    if step is not None and step % 400 == 0:
                        logging.info(
                            f"Step {step}{time_suffix}: SH DC Reg = {sh_dc_reg.item():.6f}, "
                            f"SH DC mean = {sh_dc.mean().item():.6f}"
                        )

                should_log = step is not None and step % 400 == 0
                if should_log:
                    loss_parts = ", ".join(f"{k}={v.item():.6f}" for k, v in render_loss_dict.items())
                    logging.info(
                        f"Step {step}{time_suffix}: Render Loss = {render_loss.item():.4f}, "
                        f"weight={self.render_loss_weight}, breakdown: {loss_parts}"
                    )

                if visualize and should_log and is_main_process:
                    with torch.no_grad():
                        try:
                            temporal_frames = {}
                            if hasattr(preprocessed_observation, "raw_temporal_images"):
                                temporal_frames = preprocessed_observation.raw_temporal_images
                                for key, value in temporal_frames.items():
                                    logging.info(f"[Viz] Extracted raw temporal {key} with shape {value.shape}")
                            elif hasattr(preprocessed_observation, "images"):
                                for key, value in preprocessed_observation.images.items():
                                    if value.ndim == 5 and value.shape[1] >= 2:
                                        temporal_frames[key] = value
                                        logging.info(f"[Viz] Extracted {key} with shape {value.shape}")

                            self._visualize_rendering_comparison(
                                step,
                                gaussian_params,
                                target_obs,
                                cam_params_dict,
                                view_names=render_views,
                                time_suffix=time_suffix,
                                temporal_frames=temporal_frames,
                            )
                        except Exception as viz_error:
                            logging.warning(f"Step {step}{time_suffix}: Visualization failed: {viz_error}")
        except Exception as error:
            import traceback

            logging.warning(f"Step {step}{time_suffix}: Decode/render failed: {error}")
            if step is not None and step % 400 == 0:
                traceback.print_exc()
                logging.warning(traceback.format_exc())

        if return_gaussian_params:
            return total_loss, gaussian_params
        return total_loss


    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, gaussian_inputs=None,
        return_segment_lengths=False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        
        NEW: Also includes world tokens and future query tokens if enabled.
        
        Returns:
            If return_segment_lengths=False: (embs, pad_masks, att_masks)
            If return_segment_lengths=True: (embs, pad_masks, att_masks, segment_lengths)
                where segment_lengths is a dict with keys: 'gaussian', 'images', 'language', 'world', 'future'
        """
        embs = []
        pad_masks = []
        att_masks = []
        segment_lengths = {}  # Track lengths of each segment for extracting future tokens later
        
        # Process language tokens first to get text embedding for LGPD
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
        
        # Handle temporal dimension in lang_emb if present
        # lang_emb might be [B, SeqLen, D] or [B, T, SeqLen, D]
        # For concatenation with other embeddings, we need [B, SeqLen, D] or [B, T*SeqLen, D]
        lang_emb_original = lang_emb
        lang_masks_original = lang_masks
        if lang_emb.ndim == 4:  # [B, T, SeqLen, D] - has temporal dimension
            B, T, S, D = lang_emb.shape
            # Flatten temporal and sequence dimensions: [B, T, SeqLen, D] -> [B, T*SeqLen, D]
            lang_emb = lang_emb.view(B, T * S, D)  # [B, T*SeqLen, D]
            lang_masks = lang_masks.view(B, T * S)  # [B, T*SeqLen]
            # For text embedding pooling, use original shape
            lang_emb_for_pooling = lang_emb_original
            lang_masks_for_pooling = lang_masks_original
        else:
            lang_emb_for_pooling = lang_emb
            lang_masks_for_pooling = lang_masks
        
        # --- 3D Gaussian Encoder ---
        # Delegated to Adapter
        # Pass Pooled Text Embedding for LGPD
        # lang_emb: [B, SeqLen, D] or [B, T, SeqLen, D] -> [B, D] or [B, T, D] via Mean/Max Pooling
        # Need to consider masks? lang_masks [B, SeqLen] or [B, T, SeqLen] (1=valid, 0=pad)
        if self.gaussian_adapter.use_lgpd:
            # Handle temporal dimension if present
            if lang_emb_for_pooling.ndim == 4:  # [B, T, SeqLen, D] - has temporal dimension
                # Flatten temporal and sequence dimensions for pooling
                B, T, S, D = lang_emb_for_pooling.shape
                lang_emb_flat = lang_emb_for_pooling.view(B * T, S, D)  # [B*T, S, D]
                lang_masks_flat = lang_masks_for_pooling.view(B * T, S)  # [B*T, S]
                mask_float = lang_masks_flat.unsqueeze(-1).float()  # [B*T, S, 1]
                sum_emb = (lang_emb_flat * mask_float).sum(dim=1)  # [B*T, D]
                sum_mask = mask_float.sum(dim=1).clamp(min=1e-6)  # [B*T, 1]
                text_embedding_flat = sum_emb / sum_mask  # [B*T, D]
                # Reshape back to [B, T, D] and take mean over time dimension
                text_embedding = text_embedding_flat.view(B, T, D).mean(dim=1)  # [B, D]
            else:  # [B, SeqLen, D] - no temporal dimension
                # Masked Mean Pooling
                mask_float = lang_masks_for_pooling.unsqueeze(-1).float()  # [B, S, 1]
                sum_emb = (lang_emb_for_pooling * mask_float).sum(dim=1)  # [B, D]
                sum_mask = mask_float.sum(dim=1).clamp(min=1e-6)  # [B, 1]
                text_embedding = sum_emb / sum_mask  # [B, D]
        else:
            text_embedding = None

        # --- Get World Tokens from GaussianAdapter (NEW) ---
        world_tokens = None
        world_mask = None
        # World tokens removed — redundant with Gaussian tokens (300).
        # The block below is intentionally empty; kept for code structure clarity.

        # Original Gaussian embeddings (for backward compatibility, if not using world tokens)
        gaussian_embs, g_mask = self.gaussian_adapter(gaussian_inputs, text_embedding=text_embedding)
        
        if gaussian_embs is not None:
             embs.append(gaussian_embs)
             pad_masks.append(g_mask)
             # Attention: 3DGS tokens act as context
             g_len = gaussian_embs.shape[1]
             att_masks += [0] * g_len
             segment_lengths['gaussian'] = g_len

        # Process images
        total_img_tokens = 0
        for img, img_mask in zip(images, img_masks, strict=True):
            # Handle temporal dimension: [B, T, H, W, C] or [B, H, W, C]
            has_temporal = img.ndim == 5
            if has_temporal:
                B, T = img.shape[0], img.shape[1]
                # Flatten temporal dimension: [B, T, H, W, C] -> [B*T, H, W, C]
                img_flat = img.view(B * T, *img.shape[2:])  # [B*T, H, W, C]
                # Encode all frames at once
                def image_embed_func(img_flat):
                    return self.paligemma_with_expert.embed_image(img_flat)

                img_emb_flat = self._apply_checkpoint(image_embed_func, img_flat)
                # img_emb_flat: [B*T, N, D] where N is number of image tokens
                num_img_embs = img_emb_flat.shape[1]
                # Reshape back: [B*T, N, D] -> [B, T, N, D]
                img_emb = img_emb_flat.view(B, T, num_img_embs, img_emb_flat.shape[-1])
                # Flatten time and token dimensions: [B, T, N, D] -> [B, T*N, D]
                img_emb = img_emb.view(B, T * num_img_embs, img_emb_flat.shape[-1])

                # Handle mask: [B, T] -> expand to [B, T*N]
                img_mask_expanded = img_mask.unsqueeze(-1).expand(-1, -1, num_img_embs)  # [B, T, N]
                img_mask_flat = img_mask_expanded.reshape(B, T * num_img_embs)  # [B, T*N]
                n_tokens = T * num_img_embs
            else:
                # No temporal dimension: [B, H, W, C]
                def image_embed_func(img):
                    return self.paligemma_with_expert.embed_image(img)

                img_emb = self._apply_checkpoint(image_embed_func, img)
                bsize, num_img_embs = img_emb.shape[:2]
                img_mask_flat = img_mask[:, None].expand(bsize, num_img_embs)
                n_tokens = num_img_embs

            embs.append(img_emb)
            pad_masks.append(img_mask_flat)
            total_img_tokens += n_tokens

            # Create attention masks so that image tokens attend to each other
            if has_temporal:
                # For temporal images, each time frame's tokens attend to each other
                att_masks += [0] * (T * num_img_embs)
            else:
                att_masks += [0] * num_img_embs
        segment_lengths['images'] = total_img_tokens

        # Append language tokens (already computed)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs
        segment_lengths['language'] = num_lang_embs
        
        # --- Add World Tokens (NEW) ---
        if world_tokens is not None:
            embs.append(world_tokens)
            pad_masks.append(world_mask)
            # World tokens can attend to all previous tokens (images, language, gaussian)
            att_masks += [0] * self.world_token_count
        
        # --- Add Future Query Tokens (Coupled Mask) with Temporal Base ---
        if self.use_world_tokens_in_prefix and self.future_query_tokens is not None:
            B = pad_masks[0].shape[0] if pad_masks else 1
            device = pad_masks[0].device if pad_masks else next(self.parameters()).device
            token_dim = self.future_query_tokens.shape[-1]
            future_dtype = self.future_query_tokens.dtype
            use_single_frame_mode = getattr(self.config, "use_single_frame_mode", False)

            # 1. Construct temporal base from gaussian_embs
            if gaussian_embs is not None:
                num_tokens = gaussian_embs.shape[1]
                D = gaussian_embs.shape[-1]

                if use_single_frame_mode:
                    # Single-frame mode: use current frame t as base
                    # gaussian_embs: [B, 256, D] (1 frame × 256 tokens)
                    # Semantic: future = current_latent + learnable_delta
                    if num_tokens == 256:
                        z_base = gaussian_embs  # [B, 256, D]
                    else:
                        # Unexpected shape in single-frame mode
                        logging.warning(f"Single-frame mode expects 256 tokens, got {num_tokens}. Using zero base.")
                        z_base = torch.zeros(B, self.future_token_count, D, device=device, dtype=future_dtype)
                else:
                    # Multi-frame mode: weighted fusion of [t-2, t-1, t]
                    # gaussian_embs: [B, 768, D] (3 frames × 256 tokens)
                    # Semantic: future = weighted_history + learnable_delta
                    if num_tokens == 768:  # 3 frames * 256 tokens (training)
                        # Reshape to [B, 3, 256, D] for [t-2, t-1, t]
                        g = gaussian_embs.view(B, 3, 256, -1)
                        z_t2, z_t1, z_t = g[:, 0], g[:, 1], g[:, 2]  # Each [B, 256, D]

                        # Learnable temporal weighting (softmax to ensure sum=1, stable)
                        w = torch.softmax(self.temporal_base_w, dim=0)  # [3]
                        z_base = w[0] * z_t2 + w[1] * z_t1 + w[2] * z_t  # [B, 256, D]

                    elif num_tokens == 256:  # Single frame (inference fallback or single-frame mode)
                        # Use the single frame as base
                        z_base = gaussian_embs  # [B, 256, D]

                    else:
                        # Unexpected shape: use zero base
                        logging.warning(f"Multi-frame mode expects 768 tokens, got {num_tokens}. Using zero base.")
                        z_base = torch.zeros(B, self.future_token_count, D, device=device, dtype=future_dtype)
            else:
                # No gaussian_embs: use zero base
                z_base = torch.zeros(B, self.future_token_count, token_dim, device=device, dtype=future_dtype)

            # 2. Delta query tokens (learnable, predict changes)
            delta_q = self.future_query_tokens.expand(B, -1, -1)  # [B, 256, D]
            z_base = z_base.to(delta_q.dtype)
            delta_scale = F.softplus(self.delta_scale)

            # 3. Future delta without action coupling.
            future_tokens = z_base + delta_scale * delta_q + self.future_delta_embed  # [B, 256, D]

            # 5. Add spatial positional encoding (16×16 grid structure)
            # Reshape spatial pos: [16, 16, D] -> [256, D]
            spatial_pos = self.future_spatial_pos.reshape(1, self.future_token_count, -1)  # [1, 256, D]

            # Add sinusoidal encoding if enabled
            if hasattr(self, 'use_sinusoidal_spatial') and self.use_sinusoidal_spatial:
                sinusoidal_pos = self.future_spatial_sinusoidal.reshape(1, self.future_token_count, -1)  # [1, 256, D]
                spatial_pos = spatial_pos + sinusoidal_pos

            # Add spatial positional encoding to future tokens
            future_tokens = future_tokens + spatial_pos  # [B, 256, D]

            future_mask = torch.ones(B, self.future_token_count, dtype=torch.bool, device=device)

            embs.append(future_tokens)
            pad_masks.append(future_mask)
            # Coupled mask: first future token creates a causal boundary (att=1),
            # remaining future tokens are bidirectional among themselves (att=0).
            # Effect: future tokens can see all context, but context cannot see future tokens.
            att_masks += [1] + [0] * (self.future_token_count - 1)
            segment_lengths['future'] = self.future_token_count

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        if return_segment_lengths:
            return embs, pad_masks, att_masks, segment_lengths
        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def _prepare_gaussian_inputs(self, observation, device, batch_size, is_training=None):
        """Helper to prepare inputs for 3DGS encoder from observation.
        
        Args:
            is_training: If True, uses training mode (more frames). If False, uses inference mode (fewer frames).
                        If None, auto-detects from model.training.
        """
        if is_training is None:
            is_training = self.training
        return self.gaussian_adapter.prepare_inputs(observation, device, batch_size, is_training=is_training)

    def _get_camera_params_for_view(self, view_name, device, batch_size):
        """Get camera parameters for specific view (agent or wrist)."""
        # LIBERO camera intrinsics for 256×256 depth resolution
        # Must match depth2pc input size (256×256), NOT the image encoder size (224)
        W = 256.0
        H = 256.0
        fx = 221.7025
        fy = 221.7025
        cx = W / 2.0   # 128.0
        cy = H / 2.0   # 128.0
        fov_deg = 2.0 * math.degrees(math.atan(W / (2.0 * fx)))  # ~60° derived from fx
        tanfov = math.tan(0.5 * math.radians(fov_deg))
        
        # Helper for Projection Matrix (OpenGL style)
        def getProjectionMatrix(znear, zfar, fovX, fovY):
            tanHalfFovY = math.tan((fovY / 2))
            tanHalfFovX = math.tan((fovX / 2))
            
            P = torch.zeros(4, 4, device=device)
            z_sign = 1.0 

            P[0, 0] = 1 / tanHalfFovX
            P[1, 1] = 1 / tanHalfFovY
            P[3, 2] = z_sign
            P[2, 2] = z_sign * zfar / (zfar - znear)
            P[2, 3] = -(zfar * znear) / (zfar - znear)
            return P

        # Intrinsics (same for both views)
        intrinsics = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        if view_name == "agent":
            # Agent camera: identity viewmatrix (no translation)
            # depth2pc already outputs points in camera space (z = depth > 0),
            # so no Z-axis translation is needed. Adding translation would
            # push Gaussians further away and cause uniform gray rendering.
            viewmatrix = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

        elif view_name == "wrist":
            # Wrist camera: also identity (same reasoning as agent)
            viewmatrix = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

        else:
            # Default: identity
            viewmatrix = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

        # Create proper Projection Matrix
        # Note: rasterizer usually expects P @ V, i.e. Full MVP for "projmatrix" argument 
        # or just P if it does V * pos separately? 
        # The diff-gaussian-rasterization cuda code does: p_hom = projmatrix * p_orig
        # So "projmatrix" passed to settings MUST BE the full View+Projection matrix (MVP).
        
        proj_base = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=math.radians(fov_deg), fovY=math.radians(fov_deg))
        proj_base = proj_base.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # MVP = P @ V
        # Torch matmul is (..., N, M) x (..., M, P) -> (..., N, P)
        # We need standard multiplication order P * V
        projmatrix = torch.bmm(proj_base, viewmatrix)

        # Calculate camera position from viewmatrix
        # viewmatrix transforms world to camera: xyz_cam = xyz_world @ R^T + T
        # So viewmatrix = [R^T | T; 0 0 0 1]
        # Inverse: viewmatrix_inv = [R | -R^T @ T; 0 0 0 1]
        # Camera position in world: campos = -R^T @ T = viewmatrix_inv[:3, 3]
        viewmatrix_inv = torch.inverse(viewmatrix)  # [B, 4, 4]
        campos = viewmatrix_inv[:, :3, 3]  # [B, 3] - camera position in world coordinates

        # LIBERO canonical_agentview camera pose (for action coordinate transformation)
        camera_pos = [0.5386131746834771, 0.0, 0.7903500240372423]
        camera_quat = [0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]  # [w, x, y, z]

        return {
            "viewmatrix": viewmatrix,
            "projmatrix": projmatrix,
            "tanfovx": tanfov,
            "tanfovy": tanfov,
            "campos": campos,  # Now correctly computed from viewmatrix
            "intrinsics": intrinsics,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "camera_pos": camera_pos,
            "camera_quat": camera_quat,
        }
    
    def _compute_2d_maps_loss(self, pred_2d_maps, gt_2d_maps):
        """
        Compute loss between predicted and GT 2D maps from VGGT decoder.
        
        Args:
            pred_2d_maps: Dict with keys:
                - rot_maps: [B, S, H, W, 4]
                - scale_maps: [B, S, H, W, 3]
                - opacity_maps: [B, S, H, W, 1]
                - sh_maps: [B, S, H, W, K, 3]
                - depth_maps: [B, S, H, W, 1]
            gt_2d_maps: Same structure as pred_2d_maps
        
        Returns:
            loss: Scalar loss value
        """
        import torch.nn.functional as F
        
        # Use last frame (S-1) for single frame prediction
        frame_idx = -1
        
        # Extract single frame
        pred_rot = pred_2d_maps["rot_maps"][:, frame_idx]  # [B, H, W, 4]
        pred_scale = pred_2d_maps["scale_maps"][:, frame_idx]  # [B, H, W, 3]
        pred_opacity = pred_2d_maps["opacity_maps"][:, frame_idx]  # [B, H, W, 1]
        pred_sh = pred_2d_maps["sh_maps"][:, frame_idx]  # [B, H, W, K, 3]
        pred_depth = pred_2d_maps["depth_maps"][:, frame_idx]  # [B, H, W, 1]
        
        gt_rot = gt_2d_maps["rot_maps"][:, frame_idx]
        gt_scale = gt_2d_maps["scale_maps"][:, frame_idx]
        gt_opacity = gt_2d_maps["opacity_maps"][:, frame_idx]
        gt_sh = gt_2d_maps["sh_maps"][:, frame_idx]
        gt_depth = gt_2d_maps["depth_maps"][:, frame_idx]
        
        # Compute losses for each component
        # Rotation: L2 loss on quaternions (already normalized)
        loss_rot = F.mse_loss(pred_rot, gt_rot)
        
        # Scale: L2 loss
        loss_scale = F.mse_loss(pred_scale, gt_scale)
        
        # Opacity: L2 loss
        loss_opacity = F.mse_loss(pred_opacity, gt_opacity)
        
        # SH: L2 loss
        loss_sh = F.mse_loss(pred_sh, gt_sh)
        
        # Depth: L2 loss (with optional weighting for closer objects)
        loss_depth = F.mse_loss(pred_depth, gt_depth)
        
        # Weighted combination
        total_loss = (
            0.1 * loss_rot +      # Rotation is less critical
            1.0 * loss_scale +    # Scale is important
            1.0 * loss_opacity +  # Opacity is important
            1.0 * loss_sh +       # SH (color) is important
            2.0 * loss_depth      # Depth is very important for 3D structure
        )
        
        return total_loss

    def _visualize_rendering_comparison(self, step, gaussian_params, target_obs, cam_params_dict, view_names, time_suffix="", temporal_frames=None):
        """Helper to visualize Rendered vs GT images. Delegated to GaussianRenderer."""
        # Cleanly moved to gaussian_renderer.py
        from openpi.models_pytorch.gaussian_renderer import visualize_rendering_comparison

        visualize_rendering_comparison(
            step,
            gaussian_params,
            target_obs,
            cam_params_dict,
            self.gaussian_renderer,
            view_names,
            save_dir=self.vis_save_dir,
            time_suffix=time_suffix,
            temporal_frames=temporal_frames
        )

    def _get_temporal_frames_for_viz(self, preprocessed_observation):
        temporal_frames = {}
        if hasattr(preprocessed_observation, "raw_temporal_images"):
            temporal_frames = preprocessed_observation.raw_temporal_images
            for key, value in temporal_frames.items():
                logging.info(f"[Viz] Extracted raw temporal {key} with shape {value.shape}")
        elif hasattr(preprocessed_observation, "images"):
            for key, value in preprocessed_observation.images.items():
                if value.ndim == 5 and value.shape[1] >= 2:
                    temporal_frames[key] = value
                    logging.info(f"[Viz] Extracted {key} with shape {value.shape}")
        return temporal_frames

    def _visualize_future_rollout(
        self,
        step,
        z_future_pred_tokens,
        future_observation,
        preprocessed_observation,
        static_template: dict | None = None,
    ):
        """Visualize context + future rollout GT/render/diff in one grid.

        static_template: Prefix-derived base Gaussian template reused for all horizons.
        When None, fallback to h=0 full decode then reuse for later horizons.
        """
        from openpi.models_pytorch.gaussian_renderer import visualize_future_rollout_comparison

        if self.world_model is None or self.gaussian_renderer is None or future_observation is None:
            return

        temporal_frames = self._get_temporal_frames_for_viz(preprocessed_observation)
        available_future_steps = self._get_temporal_observation_length(future_observation)
        rollout_horizon = min(z_future_pred_tokens.shape[1], available_future_steps or 1)
        if rollout_horizon <= 0:
            return

        first_future_target = (
            self._slice_temporal_observation(future_observation, 0)
            if available_future_steps > 0
            else future_observation
        )
        view_names = []
        for key in first_future_target.images.keys():
            key_lower = key.lower()
            if key == "image" or "agent" in key_lower or "high" in key_lower or "cam_high" in key_lower or "exterior" in key_lower or "base" in key_lower:
                if "agent" not in view_names:
                    view_names.append("agent")
            elif "wrist" in key_lower or "bravo" in key_lower:
                if "wrist" not in view_names:
                    view_names.append("wrist")
        if not view_names:
            view_names = ["agent"]

        target_obs_seq = []
        rendered_obs_seq = []
        render_views = None

        with torch.no_grad():
            base_depth = getattr(preprocessed_observation, "depth", None)
            if base_depth is not None and base_depth.ndim == 5:
                base_depth = base_depth[:, -1]

            viz_static_template = static_template
            for horizon_idx in range(rollout_horizon):
                future_target = (
                    self._slice_temporal_observation(future_observation, horizon_idx)
                    if available_future_steps > 0
                    else future_observation
                )
                camera_params_for_decode = self._get_camera_params_for_view(
                    "agent", z_future_pred_tokens.device, z_future_pred_tokens.shape[0]
                )
                reused_template = None
                vtf = 1.0
                if getattr(self, "use_velocity_future_gaussians", False):
                    reused_template = viz_static_template
                    o0 = self.future_prediction_offsets[0]
                    oh = self.future_prediction_offsets[horizon_idx]
                    vtf = float(oh) / float(o0) if o0 else 1.0

                gaussian_params = self.world_model.decode(
                    z_future_pred_tokens[:, horizon_idx].float(),
                    future_observation=future_target,
                    gaussian_adapter=self.gaussian_adapter,
                    camera_params=camera_params_for_decode,
                    step=step,
                    current_observation=preprocessed_observation,
                    base_depth=base_depth,
                    horizon_idx=horizon_idx,
                    static_reference_params=reused_template,
                    velocity_time_factor=vtf,
                )
                if getattr(self, "use_velocity_future_gaussians", False) and horizon_idx == 0 and viz_static_template is None:
                    viz_static_template = {
                        k: v.detach() if torch.is_tensor(v) else v for k, v in gaussian_params.items()
                    }

                target_obs = {}
                cam_params_dict = {}
                valid_views = []
                for key, value in future_target.images.items():
                    img_tensor = value
                    view_name = None
                    key_lower = key.lower()
                    if key == "image" or "agent" in key_lower or "high" in key_lower or "cam_high" in key_lower or "exterior" in key_lower or "base" in key_lower:
                        view_name = "agent"
                    elif "left_wrist" in key_lower or "wrist_left" in key_lower:
                        view_name = "wrist"
                    elif "right_wrist" in key_lower or "wrist_right" in key_lower:
                        if img_tensor.min() == img_tensor.max() == -1.0:
                            continue
                        view_name = "wrist"
                    elif "wrist" in key_lower or "bravo" in key_lower:
                        view_name = "wrist"

                    if view_name:
                        if img_tensor.shape[1] != 3 and img_tensor.shape[-1] == 3:
                            img_tensor = img_tensor.permute(0, 3, 1, 2)
                        img_tensor = (img_tensor + 1.0) / 2.0
                        view_key = f"{view_name}_image"
                        if view_key not in target_obs:
                            target_obs[view_key] = img_tensor
                            cam_params_dict[view_name] = self._get_camera_params_for_view(
                                view_name, z_future_pred_tokens.device, z_future_pred_tokens.shape[0]
                            )
                            valid_views.append(view_name)

                if valid_views:
                    render_views = ["agent"] if "agent" in valid_views else valid_views
                elif render_views is None:
                    render_views = ["agent"]

                rendered_obs = {}
                params_single = {
                    "xyz": gaussian_params["xyz"][:1],
                    "sh": gaussian_params["sh"][:1],
                    "opacity": gaussian_params["opacity"][:1],
                    "scales": gaussian_params["scales"][:1],
                    "rotations": gaussian_params["rotations"][:1],
                }
                for view_name in render_views:
                    cam_params = {
                        key: value[:1] if isinstance(value, torch.Tensor) else value
                        for key, value in cam_params_dict[view_name].items()
                    }
                    rendered_obs[f"{view_name}_image"] = self.gaussian_renderer(params_single, cam_params).float()

                target_obs_seq.append({key: value[:1].float() for key, value in target_obs.items()})
                rendered_obs_seq.append(rendered_obs)

        visualize_future_rollout_comparison(
            step,
            target_obs_seq,
            rendered_obs_seq,
            render_views or view_names,
            save_dir=self.vis_save_dir,
            temporal_frames=temporal_frames,
            time_suffix="_future_rollout",
            horizon_labels=[f"t+{offset}" for offset in self.future_prediction_offsets[:rollout_horizon]],
        )
        self._log_future_rollout_pixel_lpips(step, rendered_obs_seq)

    def forward(self, observation, actions, noise=None, time=None, step=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state, future_observation, preprocessed_observation = self._preprocess_observation(observation, train=True)

        # Move LPIPS to the correct device if initialized
        if self.lpips_fn is not None and hasattr(self.lpips_fn, 'net'):
            device = actions.device
            if next(self.lpips_fn.parameters()).device != device:
                self.lpips_fn = self.lpips_fn.to(device)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        
        # Prepare Gaussian Inputs
        # IMPORTANT: Use the preprocessed observation (with temporal dimension preserved) for VGGT
        # The original observation passed to forward() may not have the correct temporal structure
        # after _preprocess_observation processing. We need to reconstruct it from the processed data.
        gaussian_inputs = None
        # Check against adapter flag
        if self.gaussian_adapter.use_gaussian:
            # FIX: Use preprocessed_observation instead of original observation
            # The preprocessed_observation has temporal dimension preserved, while original observation may have been modified
            gaussian_inputs = self._prepare_gaussian_inputs(preprocessed_observation, actions.device, actions.shape[0])

        # Get prefix embeddings with segment lengths for extracting future tokens
        prefix_result = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, gaussian_inputs=gaussian_inputs,
            return_segment_lengths=self.use_world_tokens_in_prefix
        )
        if self.use_world_tokens_in_prefix:
            prefix_embs, prefix_pad_masks, prefix_att_masks, segment_lengths = prefix_result
        else:
            prefix_embs, prefix_pad_masks, prefix_att_masks = prefix_result
            segment_lengths = {}
        
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return prefix_out, suffix_out

        prefix_out, suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        suffix_out = self.action_out_proj(suffix_out)
        
        # Base Action Loss (Flow Matching)
        if self._action_loss_enabled:
            loss = F.mse_loss(suffix_out, u_t)
        else:
            # Use suffix_out * 0 to keep grad_fn alive even when action loss is disabled,
            # so backward() won't fail if render/depth branches are skipped for a batch.
            loss = (suffix_out * 0).sum()
        
        # --- Extract Future Frame Tokens from Prefix Output (NEW) ---
        z_t1_pred_tokens = None
        z_future_pred_tokens = None
        if self.use_world_tokens_in_prefix and 'future' in segment_lengths:
            # Calculate the start position of future tokens in prefix_output
            # Order: gaussian (if exists) -> images -> language -> future
            # (world tokens removed — no longer in prefix)
            future_start = 0
            if 'gaussian' in segment_lengths:
                future_start += segment_lengths['gaussian']
            if 'images' in segment_lengths:
                future_start += segment_lengths['images']
            if 'language' in segment_lengths:
                future_start += segment_lengths['language']
            if 'world' in segment_lengths:
                future_start += segment_lengths['world']

            future_end = future_start + segment_lengths['future']
            z_t1_pred_tokens = prefix_out[:, future_start:future_end, :]  # [B, future_token_count, D]

            if step is not None and step % 400 == 0:
                print(f"[DEBUG] Extracted future tokens: shape={z_t1_pred_tokens.shape}, "
                      f"start={future_start}, end={future_end}, segment_lengths={segment_lengths}")

            z_future_pred_tokens, per_step_delta = self._rollout_future_latents(z_t1_pred_tokens)

            # Future-rollout regularization: penalise per-step delta magnitude.
            # For autoregressive rollout, regularise each step's increment separately.
            horizon = z_future_pred_tokens.shape[1]
            if horizon > 0:
                delta_reg_weights = self._get_future_horizon_loss_weights(
                    step, horizon, per_step_delta.device, torch.float32
                )
                raw_delta_reg = (
                    per_step_delta.float().pow(2).mean(dim=(0, 2, 3)) * delta_reg_weights
                ).sum() / delta_reg_weights.sum().clamp_min(1e-6)
                delta_reg = self.future_delta_reg_weight * raw_delta_reg
                if torch.isfinite(delta_reg):
                    loss = loss + delta_reg.to(loss.dtype)
                self._log_future_rollout_diagnostics(
                    step,
                    z_t1_pred_tokens,
                    z_future_pred_tokens,
                    per_step_delta,
                    delta_reg_weights,
                )

                if step is not None and step % 400 == 0:
                    logging.info(
                        f"Step {step}: Future Rollout Delta Reg = {delta_reg.item():.6f}, "
                        f"raw={raw_delta_reg.item():.6f}, weight={self.future_delta_reg_weight}"
                    )

        # --- World Model Render Loss (render-only, no forward loss) ---
        if self.use_world_tokens_in_prefix and z_future_pred_tokens is not None and future_observation is not None:
            available_future_steps = self._get_temporal_observation_length(future_observation)
            rollout_horizon = min(z_future_pred_tokens.shape[1], available_future_steps or 1)
            world_model_loss = torch.zeros((), dtype=torch.float32, device=loss.device)
            horizon_loss_weights = self._get_future_horizon_loss_weights(
                step, rollout_horizon, loss.device, torch.float32
            )

            # Decode the current/base Gaussian template once from prefix Gaussian tokens.
            static_template: dict | None = None
            if getattr(self, "use_velocity_future_gaussians", False) and "gaussian" in segment_lengths:
                g_len = segment_lengths["gaussian"]
                z_gaussian_vlm = prefix_out[:, :g_len, :]
                try:
                    camera_params_for_template = self._get_camera_params_for_view(
                        "agent", prefix_out.device, prefix_out.shape[0]
                    )
                    base_depth_for_template = getattr(preprocessed_observation, "depth", None)
                    if base_depth_for_template is not None and base_depth_for_template.ndim == 5:
                        base_depth_for_template = base_depth_for_template[:, -1]
                    static_template = self.world_model.decode_gaussian_prefix_template(
                        z_gaussian_vlm.float(),
                        gaussian_adapter=self.gaussian_adapter,
                        current_observation=preprocessed_observation,
                        camera_params=camera_params_for_template,
                        base_depth=base_depth_for_template,
                        step=step,
                    )
                except Exception as e:
                    logging.warning(f"decode_gaussian_prefix_template failed: {e}, fallback to h=0 full decode")
                    static_template = None

            for horizon_idx in range(rollout_horizon):
                future_target = (
                    self._slice_temporal_observation(future_observation, horizon_idx)
                    if available_future_steps > 0
                    else future_observation
                )
                reused_template = None
                vtf = 1.0
                if self.use_velocity_future_gaussians:
                    reused_template = static_template
                    o0 = self.future_prediction_offsets[0]
                    oh = self.future_prediction_offsets[horizon_idx]
                    vtf = float(oh) / float(o0) if o0 else 1.0

                if self.use_velocity_future_gaussians and horizon_idx == 0 and reused_template is None:
                    horizon_loss, gp = self._compute_world_model_frame_loss(
                        z_future_pred_tokens[:, horizon_idx],
                        future_target,
                        preprocessed_observation,
                        step=step,
                        time_suffix=f"_tplus{self.future_prediction_offsets[horizon_idx]}_pred_vlm",
                        visualize=False,
                        horizon_idx=horizon_idx,
                        static_gaussian_params=reused_template,
                        velocity_time_factor=vtf,
                        return_gaussian_params=True,
                    )
                    static_template = {
                        k: v.detach() if torch.is_tensor(v) else v for k, v in gp.items()
                    }
                else:
                    horizon_loss = self._compute_world_model_frame_loss(
                        z_future_pred_tokens[:, horizon_idx],
                        future_target,
                        preprocessed_observation,
                        step=step,
                        time_suffix=f"_tplus{self.future_prediction_offsets[horizon_idx]}_pred_vlm",
                        visualize=False,
                        horizon_idx=horizon_idx,
                        static_gaussian_params=reused_template,
                        velocity_time_factor=vtf,
                    )
                world_model_loss = world_model_loss + horizon_loss_weights[horizon_idx] * horizon_loss

            if rollout_horizon > 0:
                loss = loss + (world_model_loss / horizon_loss_weights.sum().clamp_min(1e-6)).to(loss.dtype)

                import torch.distributed as dist

                should_log = step is not None and step % 400 == 0
                is_main_process = not dist.is_initialized() or dist.get_rank() == 0
                if should_log and is_main_process:
                    try:
                        self._visualize_future_rollout(
                            step,
                            z_future_pred_tokens[:, :rollout_horizon],
                            future_observation,
                            preprocessed_observation,
                            static_template=static_template,
                        )
                    except Exception as viz_error:
                        logging.warning(f"Step {step}: Future rollout visualization failed: {viz_error}")

        return loss

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state, _, preprocessed_observation = self._preprocess_observation(observation, train=False)

        gaussian_inputs = None
        if self.gaussian_adapter.use_gaussian:
            # For inference, use inference mode (fewer frames, more flexible)
            # Use preprocessed_observation to ensure temporal dimension is preserved
            gaussian_inputs = self._prepare_gaussian_inputs(preprocessed_observation, device, bsize, is_training=False)

        # Get prefix embeddings (with segment lengths if using world tokens)
        prefix_result = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, gaussian_inputs=gaussian_inputs,
            return_segment_lengths=self.use_world_tokens_in_prefix
        )
        if self.use_world_tokens_in_prefix:
            prefix_embs, prefix_pad_masks, prefix_att_masks, segment_lengths = prefix_result
        else:
            prefix_embs, prefix_pad_masks, prefix_att_masks = prefix_result
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        timestep = torch.tensor(1.0, dtype=torch.float32, device=device)
        while timestep >= -dt / 2:
            expanded_time = timestep.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            timestep += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
