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
# Import the new World Model
from openpi.models_pytorch.pi0_world_model import BiDirectionalWorldModel, visualize_world_model_prediction


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

        # --- 3D Gaussian Integration ---
        use_gaussian = getattr(config, "use_gaussian", False)
        # Typically Gaussian features join the prefix, so they must match the VLM width
        self.gaussian_adapter = GaussianAdapter(use_gaussian, paligemma_config.width)
        
        # --- BiDirectional World Model ---
        # Using same width as adaptation layer (e.g. 2048 or projected width)
        # Note: GaussianAdapter typically projects to VLM width (action_expert_config.width)
        # so z_t has dimension `paligemma_config.width`.
        if hasattr(config, "use_world_model") and config.use_world_model:
             logging.info("Initializing BiDirectional World Model...")
             self.world_model = BiDirectionalWorldModel(
                 token_dim=paligemma_config.width, # Latents are projected to VLM width
                 action_dim=32 # Action dim is standard for Pi0
             )
        else:
             self.world_model = None
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
            if img_val.ndim == 5:
                # [B, T, H, W, C]
                time_dim = img_val.shape[1]
                idx_curr, idx_fut = 0, 0

                if time_dim == 2:  # Curr, Next (from delta_timestamps=[0, 1])
                    idx_curr, idx_fut = 0, 1
                elif time_dim == 3:  # Prev, Curr, Next (from delta_timestamps=[-1, 0, 1])
                    idx_curr, idx_fut = 1, 2

                if idx_fut > 0:
                    curr_imgs, fut_imgs = {}, {}
                    for k, v in observation.images.items():
                        # Extract single frame [B, H, W, C]
                        curr_imgs[k] = v[:, idx_curr]
                        fut_imgs[k] = v[:, idx_fut]
                    
                    curr_state = observation.state
                    fut_state = observation.state
                    
                    # Handle state [B, T, D]
                    if observation.state is not None and observation.state.ndim == 3:
                         if observation.state.shape[1] == time_dim:
                             curr_state = observation.state[:, idx_curr]
                             fut_state = observation.state[:, idx_fut]

                    # Clone observation for future
                    # Note: We must also slice the masks and prompts to match the single-step batch dimension,
                    # otherwise jaxtyping will complain about mismatched *b dimensions (e.g. mask [B, T] vs image [B, H, W, C])
                    
                    # 1. Slice Image Masks
                    fut_masks = {}
                    curr_masks = {}
                    for k, v in observation.image_masks.items():
                        if v.ndim == 2: # [B, T]
                            curr_masks[k] = v[:, idx_curr]
                            fut_masks[k] = v[:, idx_fut]
                        else: # [B] - assume valid for all steps
                            curr_masks[k] = v
                            fut_masks[k] = v
                            
                    # 2. Slice Prompts
                    # Prompts might be expanded to [B, T, L] in model.py
                    curr_prompt = observation.tokenized_prompt
                    fut_prompt = observation.tokenized_prompt
                    if curr_prompt is not None and curr_prompt.ndim == 3: # [B, T, L]
                        curr_prompt = curr_prompt[:, idx_curr]
                        fut_prompt = fut_prompt[:, idx_fut]
                        
                    curr_prompt_mask = observation.tokenized_prompt_mask
                    fut_prompt_mask = observation.tokenized_prompt_mask
                    if curr_prompt_mask is not None and curr_prompt_mask.ndim == 3: # [B, T, L]
                        curr_prompt_mask = curr_prompt_mask[:, idx_curr]
                        fut_prompt_mask = fut_prompt_mask[:, idx_fut]

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

        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
            future_observation
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
        
        # --- 3D Gaussian Encoder ---
        # Delegated to Adapter
        # Pass Pooled Text Embedding for LGPD
        # lang_emb: [B, SeqLen, D] -> [B, D] via Mean/Max Pooling
        # Need to consider masks? lang_masks [B, SeqLen] (1=valid, 0=pad)
        if self.gaussian_adapter.use_lgpd:
            # Masked Mean Pooling
            mask_float = lang_masks.unsqueeze(-1).float() # [B, S, 1]
            sum_emb = (lang_emb * mask_float).sum(dim=1)
            sum_mask = mask_float.sum(dim=1).clamp(min=1e-6)
            text_embedding = sum_emb / sum_mask # [B, D]
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

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
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

    def _prepare_gaussian_inputs(self, observation, device, batch_size):
        """Helper to prepare inputs for 3DGS encoder from observation."""
        return self.gaussian_adapter.prepare_inputs(observation, device, batch_size)

    def forward(self, observation, actions, noise=None, time=None, step=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state, future_observation = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        
        # Prepare Gaussian Inputs
        gaussian_inputs = None
        # Check against adapter flag
        if self.gaussian_adapter.use_gaussian:
            gaussian_inputs = self._prepare_gaussian_inputs(observation, actions.device, actions.shape[0])

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
            z_t, _ = self.gaussian_adapter(self._prepare_gaussian_inputs(observation, actions.device, actions.shape[0]))
            
            # 2. Encode Ground Truth Future Z_{t+1}
            z_t1_gt, _ = self.gaussian_adapter(self._prepare_gaussian_inputs(future_observation, actions.device, actions.shape[0]))
            
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
                    
                    wm_outputs = self.world_model.compute_full_loss(
                        z_t=z_t, 
                        action_t=action_t, 
                        z_t1_gt=z_t1_gt
                    )
                    
                    # Add to total loss
                    # wm_outputs["loss_total"] is a scalar (mean). We need to broadcast strictness slightly or just add.
                    loss = loss + wm_outputs["loss_total"]

                    # Visualization (every 10 steps)
                    if step is not None and step % 10 == 0:
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

        images, img_masks, lang_tokens, lang_masks, state, _ = self._preprocess_observation(observation, train=False)
        
        gaussian_inputs = None
        if self.gaussian_adapter.use_gaussian:
            # For inference, observation images should be sufficient
            gaussian_inputs = self._prepare_gaussian_inputs(observation, device, bsize)

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
