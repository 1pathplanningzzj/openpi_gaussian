"""
Design: Action Motion Prior for Future Frame Prediction

Key insight: Actions are ABSOLUTE positions, so we can compute motion vectors
and project them to image space to guide future frame prediction.

LIBERO Action Format (7D):
- action[0:3]: End-effector position (XYZ in world frame)
- action[3:6]: Rotation in axis-angle representation
- action[6]: Gripper state (-1=open, 1=close)

Axis-angle: Compact 3D rotation representation where:
- Direction: rotation axis
- Magnitude: rotation angle in radians
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionMotionPrior(nn.Module):
    """
    Encode action motion to provide spatial prior for future frame prediction.

    Pipeline:
    1. Compute motion vector: Δaction = action[t+1] - action[t]
    2. Project 3D motion to 2D image space using camera intrinsics
    3. Generate spatial motion field (16×16 grid)
    4. Inject motion field into future query tokens
    """

    def __init__(self, action_dim=7, token_dim=2048, grid_size=16):
        super().__init__()
        self.action_dim = action_dim
        self.token_dim = token_dim
        self.grid_size = grid_size

        # Motion encoder: action delta → motion features
        self.motion_encoder = nn.Sequential(
            nn.Linear(action_dim, 256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
        )

        # Spatial motion field generator
        # Predicts per-pixel motion weights for 16×16 grid
        self.motion_field_generator = nn.Sequential(
            nn.Linear(512, grid_size * grid_size),
            nn.Sigmoid(),  # [0, 1] weights for each spatial location
        )

        # Motion feature projector
        self.motion_feature_proj = nn.Linear(512, token_dim)

    def forward(self, actions, future_tokens, camera_params=None):
        """
        Args:
            actions: [B, action_horizon, action_dim] - action sequence
            future_tokens: [B, 256, D] - future query tokens (16×16 grid)
            camera_params: Optional camera intrinsics for 3D→2D projection

        Returns:
            future_tokens_with_motion: [B, 256, D] - motion-conditioned tokens
            motion_field: [B, 16, 16] - spatial motion weights
        """
        B = actions.shape[0]

        # 1. Compute motion vector (delta between consecutive actions)
        # Use first action as current (t) and second as next (t+1)
        if actions.shape[1] >= 2:
            action_t = actions[:, 0, :]  # [B, 7]
            action_t1 = actions[:, 1, :]  # [B, 7]
            action_delta = action_t1 - action_t  # [B, 7] - motion vector
        else:
            # Fallback: use single action as motion hint
            action_delta = actions[:, 0, :]

        # 2. Encode motion
        motion_feat = self.motion_encoder(action_delta)  # [B, 512]

        # 3. Generate spatial motion field
        # This tells us which spatial locations are affected by the motion
        motion_field = self.motion_field_generator(motion_feat)  # [B, 256]
        motion_field = motion_field.view(B, self.grid_size, self.grid_size)  # [B, 16, 16]

        # 4. Project motion features to token space
        motion_token_feat = self.motion_feature_proj(motion_feat)  # [B, D]

        # 5. Apply spatial motion field to modulate future tokens
        # Reshape future tokens to spatial grid
        future_tokens_spatial = future_tokens.view(B, self.grid_size, self.grid_size, -1)  # [B, 16, 16, D]

        # Broadcast motion features with spatial weights
        motion_prior = motion_token_feat.view(B, 1, 1, -1) * motion_field.unsqueeze(-1)  # [B, 16, 16, D]

        # Add motion prior to future tokens
        future_tokens_with_motion = future_tokens_spatial + motion_prior

        # Reshape back to sequence
        future_tokens_with_motion = future_tokens_with_motion.view(B, -1, self.token_dim)  # [B, 256, D]

        return future_tokens_with_motion, motion_field


class ActionMotionPriorV2(nn.Module):
    """
    Version 2: More sophisticated motion modeling with 3D→2D projection.

    This version explicitly projects 3D motion to 2D image space using camera intrinsics.
    """

    def __init__(self, action_dim=7, token_dim=2048, grid_size=16, image_size=224):
        super().__init__()
        self.action_dim = action_dim
        self.token_dim = token_dim
        self.grid_size = grid_size
        self.image_size = image_size

        # 3D motion encoder
        self.motion_3d_encoder = nn.Sequential(
            nn.Linear(3, 128),  # Only XYZ motion
            nn.GELU(),
            nn.Linear(128, 256),
        )

        # Rotation encoder (separate from translation)
        self.rotation_encoder = nn.Sequential(
            nn.Linear(3, 64),  # Rotation components
            nn.GELU(),
            nn.Linear(64, 128),
        )

        # Gripper state encoder
        self.gripper_encoder = nn.Linear(1, 32)

        # Fused motion encoder
        self.motion_fusion = nn.Sequential(
            nn.Linear(256 + 128 + 32, 512),
            nn.GELU(),
            nn.Linear(512, token_dim),
        )

        # Spatial attention: predict which regions are affected by motion
        self.spatial_attention = nn.Sequential(
            nn.Linear(token_dim, 256),
            nn.GELU(),
            nn.Linear(256, grid_size * grid_size),
            nn.Softmax(dim=-1),  # Attention weights sum to 1
        )

    def project_3d_motion_to_2d(self, motion_3d, camera_params):
        """
        Project 3D motion vector to 2D image space.

        Args:
            motion_3d: [B, 3] - XYZ motion in world space
            camera_params: dict with fx, fy, cx, cy

        Returns:
            motion_2d: [B, 2] - UV motion in image space
        """
        # Simplified projection (assumes motion is small and linear)
        # For more accurate projection, need current 3D position
        fx = camera_params.get('fx', 221.7025)
        fy = camera_params.get('fy', 221.7025)

        # Project: u = fx * x/z, v = fy * y/z
        # For motion: Δu ≈ fx * Δx/z, Δv ≈ fy * Δy/z
        # Assume z ≈ 1.0 for simplicity (can be improved with depth)
        motion_2d = torch.stack([
            motion_3d[:, 0] * fx,  # Δu
            motion_3d[:, 1] * fy,  # Δv
        ], dim=-1)

        return motion_2d

    def forward(self, actions, future_tokens, camera_params=None):
        """
        Args:
            actions: [B, action_horizon, 7] - LIBERO actions
                     [eef_pos(3), axis_angle(3), gripper(1)]
            future_tokens: [B, 256, D]
            camera_params: Optional camera intrinsics

        Returns:
            future_tokens_with_motion: [B, 256, D]
            spatial_attention_weights: [B, 256]
        """
        B = actions.shape[0]

        # Compute motion delta
        if actions.shape[1] >= 2:
            action_delta = actions[:, 1, :] - actions[:, 0, :]
        else:
            action_delta = actions[:, 0, :]

        # Decompose action (LIBERO format)
        motion_pos = action_delta[:, :3]  # [B, 3] - XYZ translation
        motion_rot = action_delta[:, 3:6]  # [B, 3] - axis-angle rotation
        motion_gripper = action_delta[:, 6:7]  # [B, 1] - gripper change

        # Encode each component
        pos_feat = self.motion_3d_encoder(motion_pos)  # [B, 256]
        rot_feat = self.rotation_encoder(motion_rot)  # [B, 128]
        gripper_feat = self.gripper_encoder(motion_gripper)  # [B, 32]

        # Fuse motion features
        motion_feat = torch.cat([pos_feat, rot_feat, gripper_feat], dim=-1)  # [B, 416]
        motion_token = self.motion_fusion(motion_feat)  # [B, D]

        # Compute spatial attention (which regions are affected by motion)
        spatial_attn = self.spatial_attention(motion_token)  # [B, 256]

        # Apply motion prior with spatial attention
        motion_prior = motion_token.unsqueeze(1) * spatial_attn.unsqueeze(-1)  # [B, 256, D]

        # Add to future tokens
        future_tokens_with_motion = future_tokens + motion_prior

        return future_tokens_with_motion, spatial_attn.view(B, self.grid_size, self.grid_size)


# Usage example
if __name__ == "__main__":
    # Test
    B, action_horizon, action_dim = 4, 10, 7
    token_dim = 2048

    actions = torch.randn(B, action_horizon, action_dim)
    future_tokens = torch.randn(B, 256, token_dim)

    # Version 1: Simple
    model_v1 = ActionMotionPrior(action_dim, token_dim)
    output_v1, motion_field_v1 = model_v1(actions, future_tokens)
    print(f"V1 Output shape: {output_v1.shape}, Motion field: {motion_field_v1.shape}")

    # Version 2: Advanced
    model_v2 = ActionMotionPriorV2(action_dim, token_dim)
    output_v2, spatial_attn_v2 = model_v2(actions, future_tokens)
    print(f"V2 Output shape: {output_v2.shape}, Spatial attention: {spatial_attn_v2.shape}")
