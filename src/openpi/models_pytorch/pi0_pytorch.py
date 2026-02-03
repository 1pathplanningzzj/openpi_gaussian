# zijian
# date 2026.01.19
# Description: Main PI0 PyTorch model implementation with integrated 3D Gaussian Splatting (DF3DGS) support.
import json
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
# Import the new World Model
from openpi.models_pytorch.pi0_cross_attention_world_model import CrossAttentionWorldModel
from openpi.models_pytorch.pi0_world_model import visualize_world_model_prediction
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
        # self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")
        # Disable compile for now due to graph break issues in inference with VGGT
        self.sample_actions = self.sample_actions

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Visualization save directory for rendering comparisons
        # Can be set via VIS_SAVE_DIR environment variable, or defaults to ./visualizations/rendering
        import os
        self.vis_save_dir = os.environ.get("VIS_SAVE_DIR", "./visualizations/rendering_scale_vggt_3frames_1futureframe")

        # --- 3D Gaussian Integration ---
        use_gaussian = getattr(config, "use_gaussian", False)
        # Typically Gaussian features join the prefix, so they must match the VLM width
        # Disable LGPD for now to test training stability
        self.gaussian_adapter = GaussianAdapter(use_gaussian, paligemma_config.width, use_lgpd=False)
        
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
        
        # --- BiDirectional World Model ---
        # Using same width as adaptation layer (e.g. 2048 or projected width)
        # Note: GaussianAdapter typically projects to VLM width (action_expert_config.width)
        # so z_t has dimension `paligemma_config.width`.
        # Use world model config
        if hasattr(config, "use_world_model") and config.use_world_model:
            logging.info("Initializing Cross-Attention World Model...")
            print(f"DEBUG: Initializing World Model. Config has use_world_model={config.use_world_model}")
            # Use VGGT decoder if available
            use_vggt_decoder = (self.gaussian_adapter.use_gaussian and 
                                self.gaussian_adapter.encoder is not None and
                                hasattr(self.gaussian_adapter.encoder, 'gs_head'))
            vggt_decoder = self.gaussian_adapter.encoder.gs_head if use_vggt_decoder else None
            
            # Get VGGT embed_dim from encoder if available
            vggt_embed_dim = 1024  # Default VGGT embed_dim
            if use_vggt_decoder and hasattr(self.gaussian_adapter.encoder, 'embed_dim'):
                vggt_embed_dim = self.gaussian_adapter.encoder.embed_dim
            
            self.world_model = CrossAttentionWorldModel(
                 token_dim=paligemma_config.width, # Latents are projected to VLM width
                 action_dim=32, # Action dim is standard for Pi0
                 use_vggt_decoder=use_vggt_decoder,
                 vggt_decoder=vggt_decoder,
                input_num_tokens=300,  # GaussianAdapter: 100 tokens/frame * 3 frames = 300 tokens (with frame pos encoding)
                                    # If frame_pos_encoding disabled, this should be 100
                 target_num_tokens=1369,  # VGGT decoder expects 37x37=1369 patch tokens
                vggt_embed_dim=vggt_embed_dim,  # VGGT encoder's embed_dim (1024)
                cross_attn_config={
                    "num_heads": 8,
                    "num_layers": 2,
                    "mlp_ratio": 4.0,
                    "dropout": 0.1,
                    "use_positional_encoding": True,
                }
            )
            
            if use_vggt_decoder:
                logging.info("World Model will use VGGT decoder instead of Privileged4DGSDecoder")
            
            # Initialize Gaussian Renderer (World Model Supervision)
            try:
                print("DEBUG: Initializing Gaussian Renderer...")
                # Scale factor to reduce Gaussian sizes for sharper rendering
                # If rendering is blurry, reduce this value (e.g., 0.1, 0.05, 0.01)
                # Default 0.1 based on empirical testing - adjust if needed
                scale_factor = 0.1  # Reduced from 1.0 to improve rendering sharpness
                self.gaussian_renderer = GaussianRenderer(image_size=224, sh_degree=3, scale_factor=scale_factor)
                logging.info(f"Gaussian Renderer initialized for World Model supervision (scale_factor={scale_factor}).")
                print(f"DEBUG: Gaussian Renderer initialized successfully with scale_factor={scale_factor}.")
            except ImportError:
                self.gaussian_renderer = None
                logging.warning("Gaussian Renderer not available. Skipping rendering loss.")
                print("DEBUG: Gaussian Renderer FAILED to initialize (ImportError).")
        else:
            print(f"DEBUG: Skipping World Model initialization. config.use_world_model={getattr(config, 'use_world_model', 'MISSING')}")
            self.world_model = None
            self.gaussian_renderer = None
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
                        # #region agent log
                        try:
                            with open('/home/zijianzhang/openpi/.cursor/debug.log', 'a') as f:
                                log_entry = {
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "B",
                                    "location": "pi0_pytorch.py:252",
                                    "message": "after keeping temporal dimension in curr_imgs",
                                    "data": {
                                        "key": k,
                                        "curr_img_shape": list(v.shape),
                                        "curr_img_ndim": v.ndim
                                    },
                                    "timestamp": int(__import__('time').time() * 1000)
                                }
                                f.write(json.dumps(log_entry) + '\n')
                        except: pass
                        # #endregion
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

                    future_observation = observation.replace(
                        images=fut_imgs, 
                        state=fut_state,
                        image_masks=fut_masks,
                        tokenized_prompt=fut_prompt,
                        tokenized_prompt_mask=fut_prompt_mask
                    )
                    # Update current observation
                    observation = observation.replace(
                        images=curr_imgs, 
                        state=curr_state,
                        image_masks=curr_masks,
                        tokenized_prompt=curr_prompt,
                        tokenized_prompt_mask=curr_prompt_mask
                    )
                    
                    # Also preprocess future observation (normalization etc)
                    future_observation = _preprocessing.preprocess_observation_pytorch(future_observation, train=False)

        # #region agent log
        try:
            with open('/home/zijianzhang/openpi/.cursor/debug.log', 'a') as f:
                log_entry = {
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "B",
                    "location": "pi0_pytorch.py:before_preprocess",
                    "message": "before preprocess_observation_pytorch - check observation.images",
                    "data": {
                        "observation_images_keys": list(observation.images.keys()) if hasattr(observation, 'images') else None,
                        "observation_images_shapes": {k: list(v.shape) if hasattr(v, 'shape') else None for k, v in observation.images.items()} if hasattr(observation, 'images') else None,
                        "observation_images_ndims": {k: v.ndim if hasattr(v, 'ndim') else None for k, v in observation.images.items()} if hasattr(observation, 'images') else None
                    },
                    "timestamp": int(__import__('time').time() * 1000)
                }
                f.write(json.dumps(log_entry) + '\n')
        except: pass
        # #endregion
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        # #region agent log
        try:
            with open('/home/zijianzhang/openpi/.cursor/debug.log', 'a') as f:
                log_entry = {
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "B",
                    "location": "pi0_pytorch.py:after_preprocess",
                    "message": "after preprocess_observation_pytorch - check observation.images",
                    "data": {
                        "observation_images_keys": list(observation.images.keys()) if hasattr(observation, 'images') else None,
                        "observation_images_shapes": {k: list(v.shape) if hasattr(v, 'shape') else None for k, v in observation.images.items()} if hasattr(observation, 'images') else None,
                        "observation_images_ndims": {k: v.ndim if hasattr(v, 'ndim') else None for k, v in observation.images.items()} if hasattr(observation, 'images') else None
                    },
                    "timestamp": int(__import__('time').time() * 1000)
                }
                f.write(json.dumps(log_entry) + '\n')
        except: pass
        # #endregion
        
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
        self, images, img_masks, lang_tokens, lang_masks, gaussian_inputs=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []
        
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

        gaussian_embs, g_mask = self.gaussian_adapter(gaussian_inputs, text_embedding=text_embedding)
        
        if gaussian_embs is not None:
             embs.append(gaussian_embs)
             pad_masks.append(g_mask)
             # Attention: 3DGS tokens act as context
             g_len = gaussian_embs.shape[1]
             att_masks += [0] * g_len

        # Process images
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
            else:
                # No temporal dimension: [B, H, W, C]
                def image_embed_func(img):
                    return self.paligemma_with_expert.embed_image(img)

                img_emb = self._apply_checkpoint(image_embed_func, img)
                bsize, num_img_embs = img_emb.shape[:2]
                img_mask_flat = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask_flat)

            # Create attention masks so that image tokens attend to each other
            if has_temporal:
                # For temporal images, each time frame's tokens attend to each other
                att_masks += [0] * (T * num_img_embs)
            else:
                att_masks += [0] * num_img_embs

        # Append language tokens (already computed)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

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
            # Agent camera: Try identity first (canonical view)
            # If Gaussians are in world coordinates, identity might work better
            viewmatrix = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
            # CRITICAL FIX: Translate world away from camera (Z-axis)
            # 
            # What this does:
            # - viewmatrix transforms world coordinates to camera coordinates
            # - viewmatrix[:, 2, 3] = T means: shift scene by +T along Z-axis
            # - This is equivalent to: camera at Z=-T, looking at +Z direction
            # 
            # Why we need this:
            # - Gaussian Z values can be negative (e.g., [-10, 0])
            # - In camera space, Z < 0 means "behind camera" → frustum culled → black rendering
            # - By translating +10.0, we ensure: Z_new = Z_old + 10.0 > 0 (in front of camera)
            # 
            # Constraints:
            # - Must be large enough: T > -min(Z_gaussian) to avoid negative Z_cam
            # - Must be within [znear, zfar]: translated Z must be in [0.01, 100.0]
            # - Cannot be arbitrary: too small → black rendering, too large → poor quality
            # 
            # Current setting: 10.0 (covers Z range [-10, 0] with margin)
            # If Gaussian Z range changes significantly, adjust this value accordingly
            viewmatrix[:, 2, 3] = 10.0 

        elif view_name == "wrist":
            # Wrist camera: Also try identity for now
            viewmatrix = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
            # Shift wrist view similarly to ensure visibility
            viewmatrix[:, 2, 3] = 10.0
            
        else:
            # Default: identity
            viewmatrix = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
            viewmatrix[:, 2, 3] = 10.0

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

        return {
            "viewmatrix": viewmatrix,
            "projmatrix": projmatrix,
            "tanfovx": tanfov,
            "tanfovy": tanfov,
            "campos": torch.zeros(batch_size, 3, device=device),
            "intrinsics": intrinsics
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

    def _visualize_rendering_comparison(self, step, gaussian_params, target_obs, cam_params_dict, view_names, time_suffix=""):
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
            time_suffix=time_suffix
        )

    def forward(self, observation, actions, noise=None, time=None, step=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        # #region agent log
        try:
            with open('/home/zijianzhang/openpi/.cursor/debug.log', 'a') as f:
                log_entry = {
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "A",
                    "location": "pi0_pytorch.py:672",
                    "message": "forward entry - check observation.images from dataloader",
                    "data": {
                        "observation_images_keys": list(observation.images.keys()) if hasattr(observation, 'images') else None,
                        "observation_images_shapes": {k: list(v.shape) if hasattr(v, 'shape') else None for k, v in observation.images.items()} if hasattr(observation, 'images') else None,
                        "observation_images_ndims": {k: v.ndim if hasattr(v, 'ndim') else None for k, v in observation.images.items()} if hasattr(observation, 'images') else None
                    },
                    "timestamp": int(__import__('time').time() * 1000)
                }
                f.write(json.dumps(log_entry) + '\n')
        except: pass
        # #endregion
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
            # #region agent log
            try:
                with open('/home/zijianzhang/openpi/.cursor/debug.log', 'a') as f:
                    log_entry = {
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "C",
                        "location": "pi0_pytorch.py:688",
                        "message": "before _prepare_gaussian_inputs - check observation.images",
                        "data": {
                            "observation_images_keys": list(observation.images.keys()) if hasattr(observation, 'images') else None,
                            "observation_images_shapes": {k: list(v.shape) if hasattr(v, 'shape') else None for k, v in observation.images.items()} if hasattr(observation, 'images') else None,
                            "observation_images_ndims": {k: v.ndim if hasattr(v, 'ndim') else None for k, v in observation.images.items()} if hasattr(observation, 'images') else None
                        },
                        "timestamp": int(__import__('time').time() * 1000)
                    }
                    f.write(json.dumps(log_entry) + '\n')
            except: pass
            # #endregion
            # FIX: Use preprocessed_observation instead of original observation
            # The preprocessed_observation has temporal dimension preserved, while original observation may have been modified
            gaussian_inputs = self._prepare_gaussian_inputs(preprocessed_observation, actions.device, actions.shape[0])

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, gaussian_inputs=gaussian_inputs
        )
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
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        suffix_out = self.action_out_proj(suffix_out)
        
        # Base Action Loss (Flow Matching)
        loss = F.mse_loss(suffix_out, u_t)

        # Date 2026.01.24 🏀 🏀 🏀 zijian todo fix it for better
        # OLD Logic: Simple Predictor inside GaussianAdapter
        # NEW Logic: BiDirectional World Model  
        
        if self.gaussian_adapter.use_gaussian and future_observation is not None:
            # 1. Encode Current State Z_t
            # Re-encode strictly the gaussian part from current observation? 
            # Actually we already have `gaussian_embs` inside embed_prefix, but that is private.
            # We can re-call adapter.encode or refactor to expose it.
            # Since adapter.encode is lightweight (just forward pass of small parts if encoder is frozen), calling again is acceptable
            # BUT wait, the encoder is heavy (VGGT). We should ideally reuse it.
            # However, `embs` in embed_prefix is complex. 
            # Let's temporarily re-encode to be safe and clean.
            
            # Z_t: [B, N, D]
            # Also get decoded Gaussian parameters from VGGT for supervision
            # Visualize every 100 steps, but only on rank 0 to avoid NCCL timeout
            import torch.distributed as dist
            is_main_process = not dist.is_initialized() or dist.get_rank() == 0
            visualize = (step is not None and step % 100 == 0 and is_main_process) if step is not None else False
            
            # Compute text_embedding for LGPD (same logic as in embed_prefix)
            text_embedding_for_vggt = None
            if self.gaussian_adapter.use_lgpd and lang_tokens is not None and lang_masks is not None:
                def lang_embed_func(lang_tokens):
                    lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
                    lang_emb_dim = lang_emb.shape[-1]
                    return lang_emb * math.sqrt(lang_emb_dim)
                
                lang_emb_for_pooling = self._apply_checkpoint(lang_embed_func, lang_tokens)
                lang_masks_for_pooling = lang_masks
                
                # Handle temporal dimension if present
                if lang_emb_for_pooling.ndim == 4:  # [B, T, SeqLen, D] - has temporal dimension
                    B, T, S, D = lang_emb_for_pooling.shape
                    lang_emb_flat = lang_emb_for_pooling.view(B * T, S, D)  # [B*T, S, D]
                    lang_masks_flat = lang_masks_for_pooling.view(B * T, S)  # [B*T, S]
                    mask_float = lang_masks_flat.unsqueeze(-1).float()  # [B*T, S, 1]
                    sum_emb = (lang_emb_flat * mask_float).sum(dim=1)  # [B*T, D]
                    sum_mask = mask_float.sum(dim=1).clamp(min=1e-6)  # [B*T, 1]
                    text_embedding_flat = sum_emb / sum_mask  # [B*T, D]
                    # Reshape back to [B, T, D] and take mean over time dimension
                    text_embedding_for_vggt = text_embedding_flat.view(B, T, D).mean(dim=1)  # [B, D]
                else:  # [B, SeqLen, D] - no temporal dimension
                    # Masked Mean Pooling
                    mask_float = lang_masks_for_pooling.unsqueeze(-1).float()  # [B, S, 1]
                    sum_emb = (lang_emb_for_pooling * mask_float).sum(dim=1)  # [B, D]
                    sum_mask = mask_float.sum(dim=1).clamp(min=1e-6)  # [B, 1]
                    text_embedding_for_vggt = sum_emb / sum_mask  # [B, D]
            
            # #region agent log
            try:
                with open('/home/zijianzhang/openpi/.cursor/debug.log', 'a') as f:
                    log_entry = {
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "C",
                        "location": "pi0_pytorch.py:848",
                        "message": "before _prepare_gaussian_inputs (World Model) - check preprocessed_observation.images",
                        "data": {
                            "preprocessed_observation_images_keys": list(preprocessed_observation.images.keys()) if hasattr(preprocessed_observation, 'images') else None,
                            "preprocessed_observation_images_shapes": {k: list(v.shape) if hasattr(v, 'shape') else None for k, v in preprocessed_observation.images.items()} if hasattr(preprocessed_observation, 'images') else None,
                            "preprocessed_observation_images_ndims": {k: v.ndim if hasattr(v, 'ndim') else None for k, v in preprocessed_observation.images.items()} if hasattr(preprocessed_observation, 'images') else None
                        },
                        "timestamp": int(__import__('time').time() * 1000)
                    }
                    f.write(json.dumps(log_entry) + '\n')
            except: pass
            # #endregion
            # FIX: Use preprocessed_observation instead of original observation
            # The preprocessed_observation has temporal dimension preserved, while original observation may have been modified
            # Get raw tokens for VAE supervision if enabled
            return_raw_tokens = self.use_vae_supervision and self.vae_compressor is not None
            adapter_output = self.gaussian_adapter(
                self._prepare_gaussian_inputs(preprocessed_observation, actions.device, actions.shape[0]),
                text_embedding=text_embedding_for_vggt,
                return_gaussian_params=True,
                return_raw_tokens=return_raw_tokens,
                step=step,
                visualize=visualize
            )
            if return_raw_tokens:
                z_t, _, gaussian_params_t, raw_tokens_t = adapter_output
            else:
                z_t, _, gaussian_params_t = adapter_output
                raw_tokens_t = None
            
            # Fix NaN: Check and sanitize z_t
            if z_t is not None:
                if torch.isnan(z_t).any() or torch.isinf(z_t).any():
                    import warnings
                    warnings.warn("NaN/Inf detected in z_t from gaussian_adapter. Replacing with zeros.")
                    z_t = torch.where(
                        torch.isnan(z_t) | torch.isinf(z_t),
                        torch.zeros_like(z_t),
                        z_t
            )
            
            # Clear cache between encodings to save memory
            torch.cuda.empty_cache()
            
            # 2. Encode Ground Truth Future Z_{t+1}
            # Get decoded Gaussian parameters for future frame (ground truth)
            # Note: future_observation is intentionally single frame [B, H, W, C] (extracted from v[:, idx_fut])
            # This is correct for World Model ground truth, which only needs t+1 frame
            # #region agent log
            try:
                with open('/home/zijianzhang/openpi/.cursor/debug.log', 'a') as f:
                    log_entry = {
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "C",
                        "location": "pi0_pytorch.py:891",
                        "message": "before _prepare_gaussian_inputs (future_observation) - check future_observation.images",
                        "data": {
                            "future_observation_images_keys": list(future_observation.images.keys()) if hasattr(future_observation, 'images') else None,
                            "future_observation_images_shapes": {k: list(v.shape) if hasattr(v, 'shape') else None for k, v in future_observation.images.items()} if hasattr(future_observation, 'images') else None,
                            "future_observation_images_ndims": {k: v.ndim if hasattr(v, 'ndim') else None for k, v in future_observation.images.items()} if hasattr(future_observation, 'images') else None,
                            "note": "future_observation is intentionally single frame for World Model ground truth"
                        },
                        "timestamp": int(__import__('time').time() * 1000)
                    }
                    f.write(json.dumps(log_entry) + '\n')
            except: pass
            # #endregion
            # future_observation is single frame [B, H, W, C] for World Model ground truth
            # Pass is_training=False to avoid repeating the frame (World Model only needs t+1 frame)
            adapter_output_t1 = self.gaussian_adapter(
                self._prepare_gaussian_inputs(future_observation, actions.device, actions.shape[0], is_training=False),
                return_gaussian_params=True,
                return_raw_tokens=return_raw_tokens
            )
            if return_raw_tokens:
                z_t1_gt, _, gaussian_params_t1_gt, raw_tokens_t1_gt = adapter_output_t1
            else:
                z_t1_gt, _, gaussian_params_t1_gt = adapter_output_t1
                raw_tokens_t1_gt = None
            
            # Fix NaN: Check and sanitize z_t1_gt
            if z_t1_gt is not None:
                if torch.isnan(z_t1_gt).any() or torch.isinf(z_t1_gt).any():
                    import warnings
                    warnings.warn("NaN/Inf detected in z_t1_gt from gaussian_adapter. Replacing with zeros.")
                    z_t1_gt = torch.where(
                        torch.isnan(z_t1_gt) | torch.isinf(z_t1_gt),
                        torch.zeros_like(z_t1_gt),
                        z_t1_gt
            )
            
            # Clear cache after encoding future frame
            torch.cuda.empty_cache()
            
            if z_t is not None and z_t1_gt is not None:
                # 3. World Model Forward
                if self.world_model is not None:
                    # Flatten actions for the world model (it expects [B, A_dim] usually, but here we have [B, T, A_dim])
                    # We might want to use the first action, or average, or feed the sequence logic.
                    # The prompt implies: H_t + A_t -> H_{t+1}. Single step.
                    # So we take the first action in the horizon? Or the action at time t?
                    # `actions` passed here is the Ground Truth action sequence [B, Horizon, D].
                    # We should probably take actions[:, 0, :] corresponding to current step.
                    action_t = actions[:, 0, :]
                    
                    # Prepare temporal tokens for cross-attention
                    # z_t is [B, 300, D] (3 frames × 100 tokens) from gaussian_adapter
                    # Reshape to [B, 3, 100, D] for cross-attention
                    B, total_tokens, D = z_t.shape
                    if total_tokens == 300:  # 3 frames × 100 tokens
                        z_temporal = z_t.view(B, 3, 100, D)  # [B, 3, 100, D]
                    else:
                        # Fallback: if not 300 tokens, use z_t as single frame
                        # Or reshape based on actual structure
                        z_temporal = z_t.unsqueeze(1)  # [B, 1, N, D] - single frame
                        logging.warning(f"z_t has {total_tokens} tokens, expected 300. Using single frame for temporal.")
                    
                    wm_outputs = self.world_model.compute_full_loss(
                        z_t=z_t, 
                        action_t=action_t, 
                        z_t1_gt=z_t1_gt,
                        z_temporal=z_temporal  # Pass temporal tokens for cross-attention
                    )
                    
                    # Add to total loss
                    # wm_outputs["loss_total"] is a scalar (mean). We need to broadcast strictness slightly or just add.
                    loss = loss + wm_outputs["loss_total"]
                    
                    # === VAE Supervision Loss (Option B: Hybrid) ===
                    # Use VAE as auxiliary supervision to improve token compression/reconstruction
                    if self.use_vae_supervision and self.vae_compressor is not None:
                        if raw_tokens_t is not None and raw_tokens_t1_gt is not None:
                            # raw_tokens_t: [B, S, 1369, D] (S=3 frames)
                            # raw_tokens_t1_gt: [B, S, 1369, D] (S=1 frame for future)
                            # Use the last frame from t (current frame) and t1_gt for VAE supervision
                            B_t, S_t, N_t, D_t = raw_tokens_t.shape
                            B_t1, S_t1, N_t1, D_t1 = raw_tokens_t1_gt.shape
                            
                            # Extract current frame (last frame) from t: [B, 1369, D]
                            raw_tokens_current = raw_tokens_t[:, -1, :, :]  # [B, 1369, D]
                            # Extract future frame from t1_gt: [B, 1369, D]
                            raw_tokens_future = raw_tokens_t1_gt[:, 0, :, :]  # [B, 1369, D]
                            
                            # Compute VAE loss on current frame (reconstruction supervision)
                            _, _, vae_loss_dict_current = self.vae_compressor(
                                raw_tokens_current, return_loss=True
                            )
                            
                            # Compute VAE loss on future frame (reconstruction supervision)
                            _, _, vae_loss_dict_future = self.vae_compressor(
                                raw_tokens_future, return_loss=True
                            )
                            
                            # Average VAE losses
                            vae_recon_loss = (vae_loss_dict_current['recon_loss'] + vae_loss_dict_future['recon_loss']) / 2.0
                            vae_kl_loss = (vae_loss_dict_current['kl_loss'] + vae_loss_dict_future['kl_loss']) / 2.0
                            vae_total_loss = (vae_loss_dict_current['total_loss'] + vae_loss_dict_future['total_loss']) / 2.0
                            
                            # Add VAE loss as auxiliary supervision (with small weight)
                            vae_weight = getattr(self, 'vae_loss_weight', 0.1)  # Default weight
                            loss = loss + vae_weight * vae_total_loss
                            
                            # Log VAE losses periodically
                            if step is not None and step % 100 == 0:
                                print(f"Step {step}: VAE Loss - Recon: {vae_recon_loss.item():.4f}, KL: {vae_kl_loss.item():.4f}, Total: {vae_total_loss.item():.4f}")

                    # === 4. Gaussian Rendering Supervision === #
                    # We supervise the decoded Gaussians using the Ground Truth Future Images
                    
                    if step is not None and step % 40 == 0:
                        print(f"DEBUG: Step {step} - Checking Render Loss conditions...")
                        print(f"DEBUG: self.gaussian_renderer is {type(self.gaussian_renderer)}")
                        if "z_t1_pred" in wm_outputs:
                            print("DEBUG: z_t1_pred IS in wm_outputs")
                        else:
                            print("DEBUG: z_t1_pred IS NOT in wm_outputs. Keys:", wm_outputs.keys())
                    
                    if self.gaussian_renderer is not None and "z_t1_pred" in wm_outputs:
                        if step is not None and step % 40 == 0:
                            print("DEBUG: Entering Rendering Loss Block")
                        try:
                            z_next = wm_outputs["z_t1_pred"]
                            # z_next: [B, N, D]
                            
                            # DEBUG: Check z_next values and gradients
                            if torch.isnan(z_next).any():
                                print(f"[CRITICAL] Step {step}: z_next contains NaNs!")
                            if torch.isinf(z_next).any():
                                print(f"[CRITICAL] Step {step}: z_next contains Infs!")
                                
                            if step is not None and step % 40 == 0:
                                print(f"[DEBUG] z_next info: shape={z_next.shape}, requires_grad={z_next.requires_grad}, grad_fn={z_next.grad_fn}")
                                print(f"[DEBUG] z_next stats: min={z_next.min():.4f}, max={z_next.max():.4f}, mean={z_next.mean():.4f}")

                            # Prepare Target Images and Camera Parameters FIRST
                            # We need camera params for VGGT decoder conversion
                            target_obs = {}
                            cam_params_dict = {}

                            # Map dataset keys to loss keys & Identify views
                            valid_views = []
                            for k, v in future_observation.images.items():
                                img_tensor = v

                                # DEBUG: Print image info at step 0
                                if step is not None and step % 40 == 0:
                                    print(f"[DEBUG] Processing image key: '{k}'")
                                    print(f"[DEBUG]   Shape: {img_tensor.shape}")
                                    print(f"[DEBUG]   Range BEFORE conversion: min={img_tensor.min():.4f}, max={img_tensor.max():.4f}, mean={img_tensor.mean():.4f}")

                                # Heuristic to identify views
                                view_name = None
                                if "agent" in k or "high" in k or "cam_high" in k or "exterior" in k or "base" in k:
                                    view_name = "agent"
                                elif "left_wrist" in k or "wrist_left" in k:
                                    view_name = "wrist"  # Prefer left wrist
                                elif "right_wrist" in k or "wrist_right" in k:
                                    # Skip right wrist if it's all black or if left wrist already exists
                                    if img_tensor.min() == img_tensor.max() == -1.0:
                                        if step is not None and step % 40 == 0:
                                            print(f"[DEBUG] Skipping {k} - all black image")
                                        continue
                                    view_name = "wrist"
                                elif "wrist" in k or "bravo" in k:
                                    view_name = "wrist"

                                if view_name:
                                    # Ensure format [B, 3, H, W]
                                    if img_tensor.shape[1] != 3 and img_tensor.shape[-1] == 3:
                                        img_tensor = img_tensor.permute(0, 3, 1, 2)

                                    # IMPORTANT: Convert from [-1, 1] to [0, 1] range
                                    # Preprocessing normalizes to [-1, 1], but GaussianRenderer outputs [0, 1]
                                    img_tensor = (img_tensor + 1.0) / 2.0

                                    # DEBUG: Print after conversion
                                    if step is not None and step % 40 == 0:
                                        print(f"[DEBUG]   Identified as: {view_name}")
                                        print(f"[DEBUG]   Range AFTER conversion: min={img_tensor.min():.4f}, max={img_tensor.max():.4f}, mean={img_tensor.mean():.4f}")

                                    # Only add if not already present (avoid duplicates from multiple cameras)
                                    view_key = f"{view_name}_image"
                                    if view_key not in target_obs:
                                        target_obs[view_key] = img_tensor

                                        # Get view-specific camera parameters
                                        cam_params_dict[view_name] = self._get_camera_params_for_view(
                                            view_name, z_next.device, z_next.shape[0]
                                        )

                                        valid_views.append(view_name)
                                    else:
                                        if step is not None and step % 40 == 0:
                                            print(f"[DEBUG]   Skipping duplicate view: {view_name}")
                            
                            # Decode to Gaussian Parameters
                            # Use VGGT decoder if available, otherwise use legacy decoder
                            # Use first valid view's camera params (typically "agent")
                            camera_params_for_decode = None
                            if valid_views:
                                # Use agent view camera params if available, otherwise use first view
                                if "agent" in cam_params_dict:
                                    camera_params_for_decode = cam_params_dict["agent"]
                                else:
                                    camera_params_for_decode = cam_params_dict[valid_views[0]]
                            
                            # === Option 1: Direct 2D Maps Supervision (NEW) ===
                            # Get predicted 2D maps from World Model
                            if self.world_model.use_vggt_decoder:
                                pred_2d_maps = self.world_model.decode(
                                    z_next,
                                    future_observation=future_observation,
                                    gaussian_adapter=self.gaussian_adapter,
                                    camera_params=camera_params_for_decode,
                                    return_2d_maps=True,  # Return 2D maps for direct supervision
                                    step=step
                                )
                                
                                # Get GT 2D maps from future_observation using VGGT encoder
                                # Simply call encoder.forward() which returns all 2D maps directly
                                vggt_inputs_gt = self.gaussian_adapter.prepare_inputs(
                                    future_observation,
                                    z_next.device,
                                    z_next.shape[0],
                                    is_training=False
                                )
                                
                                if vggt_inputs_gt is not None:
                                    with torch.no_grad():
                                        # VGGT3DGSModel.forward() returns:
                                        # depth_maps, rot_maps, scale_maps, opacity_maps, sh_maps, aggregated_tokens_list, patch_start_idx
                                        outputs_gt = self.gaussian_adapter.encoder(vggt_inputs_gt)
                                        
                                        depth_maps_gt = outputs_gt[0]  # [B, S, H, W, 1]
                                        rot_maps_gt = outputs_gt[1]    # [B, S, H, W, 4]
                                        scale_maps_gt = outputs_gt[2]  # [B, S, H, W, 3]
                                        opacity_maps_gt = outputs_gt[3] # [B, S, H, W, 1]
                                        sh_maps_gt = outputs_gt[4]      # [B, S, H, W, K, 3]
                                        
                                        gt_2d_maps = {
                                            "rot_maps": rot_maps_gt,
                                            "scale_maps": scale_maps_gt,
                                            "opacity_maps": opacity_maps_gt,
                                            "sh_maps": sh_maps_gt,
                                            "depth_maps": depth_maps_gt
                                        }
                                        
                                        # Compute 2D maps loss
                                        loss_2d_maps = self._compute_2d_maps_loss(pred_2d_maps, gt_2d_maps)
                                        
                                        if step is not None and step % 40 == 0:
                                            print(f"Step {step}: 2D Maps Loss = {loss_2d_maps.item():.4f}")
                                        
                                        # Add to total loss
                                        loss = loss + 0.1 * loss_2d_maps  # Weight: 0.1
                                
                                # Also convert to 3D for rendering loss
                                gaussian_params = self.world_model.decode(
                                    z_next,
                                    future_observation=future_observation,
                                    gaussian_adapter=self.gaussian_adapter,
                                    camera_params=camera_params_for_decode,
                                    return_2d_maps=False,  # Convert to 3D for rendering
                                    step=step
                                )
                            else:
                                # Fallback: Use legacy decoder if VGGT decoder not available
                                # Note: This path should rarely be used if VGGT is properly initialized
                                gaussian_params = self.world_model.decode(
                                    z_next,
                                    future_observation=future_observation,
                                    gaussian_adapter=self.gaussian_adapter,
                                    camera_params=camera_params_for_decode,
                                    return_2d_maps=False
                                )
                            
                            # DEBUG: Check Gaussian parameters at step 0
                            if step is not None and step % 40 == 0:
                                # Convert sigma to scales for debugging
                                from openpi.models_pytorch.gaussian_renderer import convert_sigma_to_scale_rotation
                                scales_debug, _ = convert_sigma_to_scale_rotation(gaussian_params['sigma'])
                                
                                print(f"\n[DEBUG] Gaussian Parameters:")
                                print(f"  xyz: shape={gaussian_params['xyz'].shape}, min={gaussian_params['xyz'].min():.4f}, max={gaussian_params['xyz'].max():.4f}, mean={gaussian_params['xyz'].mean():.4f}")
                                print(f"  opacity: shape={gaussian_params['opacity'].shape}, min={gaussian_params['opacity'].min():.4f}, max={gaussian_params['opacity'].max():.4f}, mean={gaussian_params['opacity'].mean():.4f}")
                                print(f"  sh: shape={gaussian_params['sh'].shape}, min={gaussian_params['sh'].min():.4f}, max={gaussian_params['sh'].max():.4f}, mean={gaussian_params['sh'].mean():.4f}")
                                print(f"  sigma: shape={gaussian_params['sigma'].shape}, min={gaussian_params['sigma'].min():.4f}, max={gaussian_params['sigma'].max():.4f}, mean={gaussian_params['sigma'].mean():.4f}")
                                print(f"  scales (from sigma): min={scales_debug.min():.6f}, max={scales_debug.max():.6f}, mean={scales_debug.mean():.6f}, median={scales_debug.median():.6f}")
                                print(f"  [NOTE] If scales are too large (>0.1), rendering will be blurry. Try setting renderer.scale_factor=0.1 or smaller.")
                            
                            if valid_views:
                                # Only compute rendering loss for agent view (more stable, wider FOV)
                                # Still collect both views for potential future use
                                render_views = ["agent"] if "agent" in valid_views else valid_views

                                # DEBUG: Print which views are being rendered
                                if step is not None and step % 40 == 0:
                                    print(f"[DEBUG] valid_views: {valid_views}")
                                    print(f"[DEBUG] render_views: {render_views}")
                                    print(f"[DEBUG] target_obs keys: {list(target_obs.keys())}")
                                    
                                render_loss, _ = compute_multi_view_rendering_loss(
                                    gaussian_params,
                                    target_obs,
                                    cam_params_dict,
                                    self.gaussian_renderer,
                                    view_names=render_views,
                                    step=step
                                )
                                # Weight the render loss
                                loss = loss + 0.1 * render_loss
                                
                                # FIX: Ensure render_loss is connected to computation graph even if rendering fails (black image)
                                # This prevents "element 0 of tensors does not require grad" error when num_valid_gaussians=0
                                if z_next.requires_grad:
                                    loss = loss + 0.0 * z_next.sum()
                                
                                if step is not None and step % 40 == 0:
                                     print(f"Step {step}: Render Loss = {render_loss.item():.4f} (views: {render_views})")
                                     
                                     # Only visualize on main process (rank 0) in distributed training
                                     # to avoid file conflicts and reduce overhead
                                     try:
                                         import torch.distributed as dist
                                         is_main_process = not dist.is_initialized() or dist.get_rank() == 0
                                     except:
                                         is_main_process = True  # Fallback if DDP not available
                                     
                                     if is_main_process:
                                         try:
                                             # === Visualization 1: Predicted next state z_{t+1}^{pred} ===
                                             # This is the original visualization (future frame prediction).
                                             self._visualize_rendering_comparison(
                                                 step,
                                                 gaussian_params,
                                                 target_obs,
                                                 cam_params_dict,
                                                 view_names=render_views,  # Only visualize rendered views
                                                 time_suffix="_t1_pred"  # z_{t+1}^{pred}
                                             )

                                             # === Visualization 2: Decode and render z_{t+1}^{GT} ===
                                             # 只可视化 GT 的下一帧，用于对比预测结果
                                             # 不再可视化 z_t (当前帧)，减少可视化开销
                                             # Use VGGT decoder if available
                                             if self.world_model.use_vggt_decoder:
                                                 # Use agent view camera params if available
                                                 camera_params_viz = cam_params_dict.get("agent") if cam_params_dict else None
                                                 gaussian_params_t1_gt = self.world_model.decode(
                                                     z_t1_gt,
                                                     future_observation=future_observation,
                                                     gaussian_adapter=self.gaussian_adapter,
                                                     camera_params=camera_params_viz
                                                 )
                                             else:
                                                 # Fallback: Use decode method (will use legacy decoder if VGGT not available)
                                                 camera_params_viz = cam_params_dict.get("agent") if cam_params_dict else None
                                                 gaussian_params_t1_gt = self.world_model.decode(
                                                     z_t1_gt,
                                                     future_observation=future_observation,
                                                     gaussian_adapter=self.gaussian_adapter,
                                                     camera_params=camera_params_viz,
                                                     return_2d_maps=False
                                                 )

                                             # Only visualize GT next frame (t+1)
                                             self._visualize_rendering_comparison(
                                                 step,
                                                 gaussian_params_t1_gt,
                                                 target_obs,
                                                 cam_params_dict,
                                                 view_names=render_views,
                                                 time_suffix="_t1_gt"  # z_{t+1}^{GT} (ground truth next frame)
                                             )
                                         except Exception as viz_e:
                                             print(f"Rendering visualization failed: {viz_e}")
                        except Exception as e:
                            # Do not crash training if rendering fails
                            if step is not None and step % 100 == 0:
                                import traceback
                                traceback.print_exc()
                                print(f"Rendering failed: {e}")

                    # Visualization (every 10 steps, only on rank 0 to avoid NCCL timeout)
                    import torch.distributed as dist
                    is_main_process = not dist.is_initialized() or dist.get_rank() == 0
                    if step is not None and step % 10 == 0 and is_main_process:
                        print(f"DEBUG: Attempting visualization at step {step}")
                        try:
                            visualize_world_model_prediction(
                                self.world_model,
                                z_t, 
                                action_t, 
                                z_t1_gt,
                                batch_idx=0,
                                step=step
                            )
                        except Exception as e:
                            print(f"Visualization failed: {e}")
                    
                else: 
                    if step is not None and step % 10 == 0:
                         print(f"DEBUG: World Model is None at step {step}")
        
        elif step is not None and step % 10 == 0:
            if not self.gaussian_adapter.use_gaussian:
                print(f"DEBUG: Step {step} - use_gaussian is False")
            if future_observation is None:
                # Get shape for debug
                shape_info = "No Images"
                if observation.images:
                    shape_info = str(next(iter(observation.images.values())).shape)
                print(f"DEBUG: Step {step} - future_observation is None. Image Shape: {shape_info}") 
                # Fallback to old simple predictor logic if WorldModel not init
                pass
                     
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

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, gaussian_inputs=gaussian_inputs
        )
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
