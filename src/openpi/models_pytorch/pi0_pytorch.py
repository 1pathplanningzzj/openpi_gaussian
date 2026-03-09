# zijian
# date 2026.01.19
# Description: Main PI0 PyTorch model implementation with integrated 3D Gaussian Splatting (DF3DGS) support.
import logging
import math

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
        self.vis_save_dir = os.environ.get("VIS_SAVE_DIR", "./visualizations/rendering_independent_decoder_test5")

        # --- 3D Gaussian Integration ---
        use_gaussian = getattr(config, "use_gaussian", False)
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
            unfreeze_encoder=unfreeze_vggt_encoder,
            unfreeze_decoder_only=unfreeze_vggt_decoder_only,
            use_lora=use_lora
        )
        
        # Current frame reconstruction loss weight
        self.current_frame_recon_loss_weight = getattr(config, "current_frame_recon_loss_weight", 0.5)
        
        # --- World Model Tokens in Prefix (NEW Architecture) ---
        # Add future query tokens to prefix for unified VLM processing
        self.use_world_tokens_in_prefix = getattr(config, "use_world_model", False) and use_gaussian
        if self.use_world_tokens_in_prefix:
            # === Priority 1: Aligned Future Query Tokens ===
            # 256 future query tokens (matches 16×16 decoder grid)
            self.future_token_count = 256
            self.future_grid_size = 16  # 16×16 spatial structure

            # World tokens removed — redundant with Gaussian tokens (768)
            self.world_token_count = 0
            self.world_token_proj = None

            # Future query tokens: learnable embeddings predicted by VLM
            # Initialize with spatial structure awareness
            self.future_query_tokens = nn.Parameter(
                torch.randn(1, self.future_token_count, paligemma_config.width) * 0.02
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

            logging.info(
                f"Initialized aligned future query tokens:\n"
                f"  - Token count: {self.future_token_count}\n"
                f"  - Spatial structure: {self.future_grid_size}×{self.future_grid_size}\n"
                f"  - Spatial positional encoding: {'Sinusoidal + Learnable' if self.use_sinusoidal_spatial else 'Learnable only'}\n"
                f"  - Aligned with VGGT tokens: 768 (3 frames × 256 tokens/frame)"
            )
        else:
            self.world_token_count = 0
            self.future_token_count = 0
            self.world_token_proj = None
            self.future_query_tokens = None
        
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
        if hasattr(config, "use_world_model") and config.use_world_model:
            logging.info("Initializing GaussianDecoder...")

            self.world_model = GaussianDecoder(
                token_dim=paligemma_config.width,
                input_num_tokens=256,
            )

            # Initialize Gaussian Renderer (sh_degree=1 for DC + 1st order SH)
            try:
                self.gaussian_renderer = GaussianRenderer(image_size=224, sh_degree=1, scale_factor=1.0)
                logging.info("Gaussian Renderer initialized with sh_degree=1 (DC + 1st order) for World Model supervision.")
            except ImportError:
                self.gaussian_renderer = None
                logging.warning("Gaussian Renderer not available. Skipping rendering loss.")
        else:
            self.world_model = None
            self.gaussian_renderer = None

        # Initialize render loss weight (can be changed dynamically for staged training)
        self.render_loss_weight = getattr(config, "render_loss_weight", 0.1)  # 降低render loss权重，让action loss主导
        # Initialize depth supervision loss weight
        self.depth_loss_weight = getattr(config, "depth_loss_weight", 0.02)  # 降低depth loss权重，让action loss主导
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

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        
        # --- Handle Future Split for 3DGS World Model ---
        future_observation = None
        # Check if first image has T dimension (ndim=5 for B,T,H,W,C now due to model.py fix)
        if observation.images:
            # Inspect first image to detect time dimension
            img_val = next(iter(observation.images.values()))
            # DEBUG PRINT
            if train and torch.rand(1).item() < 0.01:
                 print(f"DEBUG: _preprocess_observation. img_val ndim={img_val.ndim}, shape={img_val.shape}")

            if img_val.ndim == 5:
                # [B, T, H, W, C]
                time_dim = img_val.shape[1]
                idx_curr, idx_fut = 0, 0

                if time_dim == 2:  # Curr, Next (from delta_timestamps=[0, 1])
                    idx_curr, idx_fut = 0, 1
                elif time_dim == 3:  # Prev, Curr, Next (from delta_timestamps=[-1, 0, 1])
                    idx_curr, idx_fut = 1, 2
                elif time_dim == 4:  # [t-2, t-1, t, t+1] (format for VGGT 3 frames + World Model)
                    idx_curr, idx_fut = 2, 3  # Current is at idx=2 (t), future is at idx=3 (t+1)
                elif time_dim == 6:  # [t-4, t-3, t-2, t-1, t, t+1] (old format for VGGT 5 frames + World Model)
                    idx_curr, idx_fut = 4, 5  # Current is at idx=4 (t), future is at idx=5 (t+1)
                
                # DEBUG PRINT
                if train and torch.rand(1).item() < 0.01:
                     print(f"DEBUG: Found Time Dim {time_dim}. indices: curr={idx_curr}, fut={idx_fut}")

                if idx_fut > 0:
                    curr_imgs, fut_imgs = {}, {}
                    # For VGGT, we need multiple frames, so keep the temporal dimension
                    # For World Model and other processing, we extract single frames
                    for k, v in observation.images.items():
                        # Keep temporal dimension for VGGT: [B, T, H, W, C]
                        # VGGT will extract the needed frames in prepare_inputs
                        curr_imgs[k] = v  # Keep full temporal sequence [B, T, H, W, C]
                        # Extract single frame for future observation [B, H, W, C]
                        fut_imgs[k] = v[:, idx_fut]
                    
                    curr_state = observation.state
                    fut_state = observation.state
                    
                    # Handle state [B, T, D]
                    # For VGGT, we need to keep temporal dimension for state to match images
                    if observation.state is not None:
                        if observation.state.ndim == 3:
                            # State has temporal dimension [B, T, D]
                            if observation.state.shape[1] == time_dim:
                                # Keep full temporal dimension for current (VGGT needs it)
                                curr_state = observation.state  # Keep [B, T, D] for VGGT
                                fut_state = observation.state[:, idx_fut]  # Single frame for future
                        elif observation.state.ndim == 2:
                            # State is [B, D], expand to [B, T, D] to match images
                            # Repeat state for all time frames
                            curr_state = observation.state.unsqueeze(1).expand(-1, time_dim, -1)  # [B, T, D]
                            fut_state = observation.state  # Keep [B, D] for future

                    # Clone observation for future
                    # Note: We must also slice the masks and prompts to match the single-step batch dimension,
                    # otherwise jaxtyping will complain about mismatched *b dimensions (e.g. mask [B, T] vs image [B, H, W, C])
                    
                    # 1. Slice Image Masks
                    fut_masks = {}
                    curr_masks = {}
                    for k, v in observation.image_masks.items():
                        if v.ndim == 2: # [B, T]
                            # Keep full temporal dimension for current (VGGT needs it)
                            curr_masks[k] = v  # Keep [B, T] for VGGT
                            fut_masks[k] = v[:, idx_fut]  # Single frame for future
                        else: # [B] - assume valid for all steps
                            curr_masks[k] = v
                            fut_masks[k] = v
                            
                    # 2. Slice Prompts
                    # Prompts might be expanded to [B, T, L] in model.py
                    # For VGGT, we need to keep temporal dimension to match images
                    curr_prompt = observation.tokenized_prompt
                    fut_prompt = observation.tokenized_prompt
                    if curr_prompt is not None:
                        if curr_prompt.ndim == 3: # [B, T, L]
                            # Keep full temporal dimension for current (VGGT needs it)
                            curr_prompt = curr_prompt  # Keep [B, T, L] for VGGT
                            fut_prompt = curr_prompt[:, idx_fut]  # Single frame for future
                        elif curr_prompt.ndim == 2: # [B, L]
                            # Expand to [B, T, L] to match images
                            curr_prompt = curr_prompt.unsqueeze(1).expand(-1, time_dim, -1)  # [B, T, L]
                            fut_prompt = observation.tokenized_prompt  # Keep [B, L] for future
                        
                    curr_prompt_mask = observation.tokenized_prompt_mask
                    fut_prompt_mask = observation.tokenized_prompt_mask
                    if curr_prompt_mask is not None:
                        if curr_prompt_mask.ndim == 3: # [B, T, L]
                            # Keep full temporal dimension for current (VGGT needs it)
                            curr_prompt_mask = curr_prompt_mask  # Keep [B, T, L] for VGGT
                            fut_prompt_mask = curr_prompt_mask[:, idx_fut]  # Single frame for future
                        elif curr_prompt_mask.ndim == 2: # [B, L]
                            # Expand to [B, T, L] to match images
                            curr_prompt_mask = curr_prompt_mask.unsqueeze(1).expand(-1, time_dim, -1)  # [B, T, L]
                            fut_prompt_mask = observation.tokenized_prompt_mask  # Keep [B, L] for future

                    # Handle depth data if available
                    fut_depth = None
                    curr_depth = None
                    if hasattr(observation, 'depth') and observation.depth is not None:
                        # observation.depth: [B, T, 1, H, W] or [B, 1, H, W]
                        if observation.depth.ndim == 5:  # [B, T, 1, H, W]
                            if observation.depth.shape[1] == time_dim:
                                # For current observation: keep full temporal dimension [B, T, 1, H, W]
                                # For future observation: extract single frame [B, 1, H, W]
                                curr_depth = observation.depth  # Keep [B, T, 1, H, W] for VGGT
                                fut_depth = observation.depth[:, idx_fut]  # [B, 1, H, W]
                        elif observation.depth.ndim == 4:  # [B, 1, H, W]
                            # Single frame depth, use as is for both
                            curr_depth = observation.depth
                            fut_depth = observation.depth
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
        
        # --- Add Future Query Tokens (Coupled Mask) with Spatial Positional Encoding ---
        if self.use_world_tokens_in_prefix and self.future_query_tokens is not None:
            B = pad_masks[0].shape[0] if pad_masks else 1
            device = pad_masks[0].device if pad_masks else next(self.parameters()).device

            # 1. Expand future query tokens to batch
            future_tokens = self.future_query_tokens.expand(B, -1, -1)  # [B, 256, D]

            # 2. Add spatial positional encoding (16×16 grid structure)
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
        fov_deg = 60.0
        tanfov = math.tan(0.5 * math.radians(fov_deg))
        W = 224.0
        fx = W / (2.0 * tanfov)
        cx = W / 2.0
        
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
        intrinsics[:, 1, 1] = fx
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cx

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
            "fy": fx,
            "cx": cx,
            "cy": cx,
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

    def forward(self, observation, actions, noise=None, time=None, step=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state, future_observation, preprocessed_observation = self._preprocess_observation(observation, train=True)

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
        loss = F.mse_loss(suffix_out, u_t)
        
        # --- Extract Future Frame Tokens from Prefix Output (NEW) ---
        z_t1_pred_tokens = None
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
            
            if step is not None and step % 100 == 0:
                print(f"[DEBUG] Extracted future tokens: shape={z_t1_pred_tokens.shape}, "
                      f"start={future_start}, end={future_end}, segment_lengths={segment_lengths}")

        # --- World Model Render Loss (render-only, no forward loss) ---
        if self.use_world_tokens_in_prefix and z_t1_pred_tokens is not None and future_observation is not None:
            z_next = z_t1_pred_tokens  # [B, 256, D]

            import torch.distributed as dist
            is_main_process = not dist.is_initialized() or dist.get_rank() == 0

            # Decode + render every step for proper gradient flow
            if self.world_model is not None and self.gaussian_renderer is not None:
                try:
                    camera_params_for_decode = self._get_camera_params_for_view(
                        "agent", z_next.device, z_next.shape[0]
                    )
                    z_next_float32 = z_next.float()

                    # Decode predicted tokens to 3D Gaussians (with gradients)
                    gaussian_params = self.world_model.decode(
                        z_next_float32,
                        actions=actions,  # NEW: Pass ground-truth actions for conditioning
                        future_observation=future_observation,
                        gaussian_adapter=self.gaussian_adapter,
                        camera_params=camera_params_for_decode,
                        step=step,
                        current_observation=preprocessed_observation,
                    )

                    # Sanitize Gaussian params
                    depth_map = gaussian_params.pop("depth_map", None)  # [B, 1, H, W]

                    # --- Depth Supervision Loss (NEW) ---
                    depth_loss = None
                    # Debug: check if depth data is available
                    if step is not None and step % 100 == 0:
                        logging.info(f"Step {step}: depth_map={depth_map is not None}, "
                                   f"has_depth_attr={hasattr(future_observation, 'depth')}, "
                                   f"depth_value={future_observation.depth is not None if hasattr(future_observation, 'depth') else 'N/A'}")

                    if depth_map is not None and hasattr(future_observation, 'depth') and future_observation.depth is not None:
                        # future_observation.depth: [B, 1, H, W] - ground truth depth from Depth Anything V2
                        gt_depth = future_observation.depth

                        # Resize predicted depth to match GT depth if needed
                        if depth_map.shape != gt_depth.shape:
                            depth_map_resized = F.interpolate(
                                depth_map, size=gt_depth.shape[-2:], mode='bilinear', align_corners=False
                            )
                        else:
                            depth_map_resized = depth_map

                        # Compute L1 loss for depth (more robust than L2 for depth estimation)
                        depth_loss = F.l1_loss(depth_map_resized, gt_depth)

                        # Add depth loss to total loss with weight
                        depth_loss_weight = getattr(self, 'depth_loss_weight', 0.1)
                        if torch.isfinite(depth_loss):
                            loss = loss + depth_loss_weight * depth_loss

                        # Logging
                        if step is not None and step % 100 == 0:
                            logging.info(f"Step {step}: Depth Loss = {depth_loss.item():.6f}, weight={depth_loss_weight}")

                    for k, v in gaussian_params.items():
                        if torch.isnan(v).any() or torch.isinf(v).any():
                            gaussian_params[k] = torch.where(
                                torch.isfinite(v), v, torch.zeros_like(v)
                            )

                    # Prepare target images for rendering loss
                    target_obs = {}
                    cam_params_dict = {}
                    valid_views = []

                    for k, v in future_observation.images.items():
                        img_tensor = v
                        view_name = None
                        if k == "image" or "agent" in k or "high" in k or "cam_high" in k or "exterior" in k or "base" in k:
                            view_name = "agent"
                        elif "left_wrist" in k or "wrist_left" in k:
                            view_name = "wrist"
                        elif "right_wrist" in k or "wrist_right" in k:
                            if img_tensor.min() == img_tensor.max() == -1.0:
                                continue
                            view_name = "wrist"
                        elif "wrist" in k or "bravo" in k:
                            view_name = "wrist"

                        if view_name:
                            if img_tensor.shape[1] != 3 and img_tensor.shape[-1] == 3:
                                img_tensor = img_tensor.permute(0, 3, 1, 2)
                            img_tensor = (img_tensor + 1.0) / 2.0
                            view_key = f"{view_name}_image"
                            if view_key not in target_obs:
                                target_obs[view_key] = img_tensor
                                cam_params_dict[view_name] = self._get_camera_params_for_view(
                                    view_name, z_next.device, z_next.shape[0]
                                )
                                valid_views.append(view_name)

                    if valid_views:
                        render_views = ["agent"] if "agent" in valid_views else valid_views

                        # Render loss every step (gradient supervision for decoder)
                        render_loss, render_loss_dict = compute_multi_view_rendering_loss(
                            gaussian_params,
                            target_obs,
                            cam_params_dict,
                            self.gaussian_renderer,
                            view_names=render_views,
                            step=step,
                            depth_map=depth_map,
                            lambda_scale=0.001,
                            lambda_opacity=0.01,  # 增大10倍: 0.001 → 0.01 (防止opacity过大导致模糊)
                            lambda_edge_smooth=0.01,
                        )
                        # Use dynamic render_loss_weight (can be changed for staged training)
                        if torch.isfinite(render_loss):
                            loss = loss + self.render_loss_weight * render_loss.to(loss.dtype)
                        
                        # SH DC regularization: encourage neutral gray (sh_dc ≈ 0) to prevent dark rendering
                        # This prevents the model from learning to output negative SH DC values
                        if "sh" in gaussian_params:
                            sh_dc = gaussian_params["sh"]  # [B, N, 3] - SH DC coefficients
                            # L2 regularization: penalize SH DC deviating from 0 (neutral gray)
                            sh_dc_reg = (sh_dc ** 2).mean() * 0.01  # Small weight to avoid over-constraining
                            loss = loss + sh_dc_reg
                            if step is not None and step % 100 == 0:
                                logging.info(f"Step {step}: SH DC Reg = {sh_dc_reg.item():.6f}, SH DC mean = {sh_dc.mean().item():.6f}")

                        # Periodic logging + visualization
                        should_log = step is not None and step % 100 == 0
                        if should_log:
                            loss_parts = ", ".join(f"{k}={v.item():.6f}" for k, v in render_loss_dict.items())
                            logging.info(f"Step {step}: Render Loss = {render_loss.item():.4f}, weight={self.render_loss_weight}, breakdown: {loss_parts}")

                        if should_log and is_main_process:
                            with torch.no_grad():
                                try:
                                    # Extract temporal frames from preprocessed_observation
                                    # Expected: 4 consecutive frames [t-2, t-1, t, t+1]
                                    temporal_frames = {}
                                    if hasattr(preprocessed_observation, 'images'):
                                        for k, v in preprocessed_observation.images.items():
                                            # v shape: [B, T, H, W, C] where T=4 for [t-2, t-1, t, t+1]
                                            if v.ndim == 5 and v.shape[1] == 4:  # Has 4 temporal frames
                                                temporal_frames[k] = v  # Keep all 4 frames [B, 4, H, W, C]
                                                logging.info(f"[Viz] Extracted {k} with shape {v.shape}")

                                    self._visualize_rendering_comparison(
                                        step,
                                        gaussian_params,
                                        target_obs,
                                        cam_params_dict,
                                        view_names=render_views,
                                        time_suffix="_t1_pred_vlm",
                                        temporal_frames=temporal_frames
                                    )
                                except Exception as viz_e:
                                    logging.warning(f"Step {step}: Visualization failed: {viz_e}")

                        del gaussian_params, target_obs, cam_params_dict, z_next_float32, depth_map
                        torch.cuda.empty_cache()
                except Exception as e:
                    import traceback
                    logging.warning(f"Step {step}: Decode/render failed: {e}")
                    if step is not None and step % 100 == 0:
                        traceback.print_exc()
                        logging.warning(traceback.format_exc())

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
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
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
