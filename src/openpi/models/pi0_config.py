import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


#
@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # Whether to use 3D Gaussian Splatting Encoder as additional conditioning
    use_gaussian: bool = False
    # Whether to use single-frame mode for VGGT (only use current frame t, not t-2, t-1)
    # This removes temporal history and tests if temporal context is necessary
    # When True: VGGT uses [t] only (1 frame)
    # When False: VGGT uses [t-2, t-1, t] (3 frames, default)
    use_single_frame_mode: bool = False
    # Whether to use BiDirectional World Model for aux loss
    use_world_model: bool = False
    # Number of future frames to supervise in the world-model branch when using dense rollout.
    # When set to 5, the model predicts t+1 ... t+5 from a single future seed.
    future_prediction_horizon: int = 5
    # Optional sparse temporal context offsets (in frame steps) for Gaussian/VGGT conditioning.
    # These define the history slots packed before future supervision, e.g. (-10, -5, 0).
    # The last offset should correspond to the current frame.
    temporal_context_offsets: tuple[int, ...] | None = (-2, -1, 0)
    # Optional sparse future offsets (in frame steps) for world-model supervision, e.g. (2, 5, 10, 15, 20).
    # If provided, this overrides the dense t+1...t+H schedule while keeping H=len(offsets).
    future_prediction_offsets: tuple[int, ...] | None = None
    # VGGT encoder/decoder training options (for reconstruction loss)
    unfreeze_vggt_encoder: bool = False  # If True, unfreeze entire VGGT encoder for end-to-end training
    unfreeze_vggt_decoder_only: bool = True  # If True (default), only unfreeze decoder (gs_head) while keeping encoder frozen
    # Current frame reconstruction loss weight (for training VGGT encoder+decoder)
    current_frame_recon_loss_weight: float = 0.5  # Weight for current frame reconstruction loss
    # Render loss weight for 3DGS rendering supervision
    render_loss_weight: float = 0.1  # Weight for rendering loss (RGB + regularization) - reduced to let action loss dominate
    # Depth supervision loss weight
    depth_loss_weight: float = 0.02  # Weight for depth supervision loss (from Depth Anything V2) - reduced to let action loss dominate
    # Incremental-depth auxiliary supervision weight. 0 disables delta-depth loss.
    delta_depth_loss_weight: float = 0.0
    # If True, only supervise delta depth for t->t+1. Default False supervises t->t+h for all horizons.
    delta_depth_first_horizon_only: bool = False
    # Make world-model depth prediction residual wrt current depth.
    use_incremental_depth: bool = True
    # If True, horizon 0 uses full Gaussian decode; horizons h>1 reuse detached static
    # (rot/scale/opacity/SH + template xyz) and only predict per-token velocity v so that
    # xyz_h = xyz_0 + v(z_h) * velocity_world_model_scale * (offset[h]/offset[0]).
    use_velocity_future_gaussians: bool = False
    # Scale for camera-space displacement from predicted velocity (meters-scale heuristic).
    velocity_world_model_scale: float = 2.0
    # LPIPS perceptual loss options
    use_lpips: bool = False  # Whether to use LPIPS perceptual loss for rendering
    lpips_weight: float = 0.1  # Weight for LPIPS perceptual loss
    # Regularization weight for future-token delta magnitude (encourages stable delta prediction).
    future_delta_reg_weight: float = 1e-4
    # Number of steps to anneal future-rollout horizon weights from near-heavy to uniform.
    future_horizon_curriculum_steps: int = 5_000
    # Tail weight used by the early near-heavy horizon weighting profile.
    future_horizon_early_min_weight: float = 0.2
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
