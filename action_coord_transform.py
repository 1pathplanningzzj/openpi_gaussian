"""
Action Coordinate Transform: World Frame → Camera Frame

For LIBERO, we need to transform action (in world frame) to camera frame
to correctly predict motion in the rendered image.
"""

import torch
import torch.nn as nn
import numpy as np


def quat_to_rotation_matrix(quat):
    """
    Convert quaternion to rotation matrix.

    Args:
        quat: [B, 4] or [4] - quaternion [w, x, y, z] or [x, y, z, w]

    Returns:
        R: [B, 3, 3] or [3, 3] - rotation matrix
    """
    if quat.dim() == 1:
        quat = quat.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    # Assume quat is [x, y, z, w] (LIBERO format)
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # Rotation matrix from quaternion
    R = torch.stack([
        torch.stack([1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)], dim=-1),
        torch.stack([2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)], dim=-1),
        torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)], dim=-1),
    ], dim=-2)

    if squeeze:
        R = R.squeeze(0)

    return R


def build_viewmatrix_from_pose(cam_pos, cam_quat):
    """
    Build view matrix from camera pose in world frame.

    Args:
        cam_pos: [B, 3] or [3] - camera position in world frame
        cam_quat: [B, 4] or [4] - camera orientation quaternion

    Returns:
        viewmatrix: [B, 4, 4] or [4, 4] - transforms world → camera
    """
    if cam_pos.dim() == 1:
        cam_pos = cam_pos.unsqueeze(0)
        cam_quat = cam_quat.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    B = cam_pos.shape[0]
    device = cam_pos.device

    # Get rotation matrix: world → camera
    R_world_to_cam = quat_to_rotation_matrix(cam_quat)  # [B, 3, 3]

    # View matrix: [R | -R @ t]
    #              [0 | 1      ]
    # where R rotates world to camera, t is camera position in world
    viewmatrix = torch.zeros(B, 4, 4, device=device, dtype=cam_pos.dtype)
    viewmatrix[:, :3, :3] = R_world_to_cam
    viewmatrix[:, :3, 3] = -torch.bmm(R_world_to_cam, cam_pos.unsqueeze(-1)).squeeze(-1)
    viewmatrix[:, 3, 3] = 1.0

    if squeeze:
        viewmatrix = viewmatrix.squeeze(0)

    return viewmatrix


def transform_action_to_camera_frame(action_world, viewmatrix):
    """
    Transform action from world frame to camera frame.

    Args:
        action_world: [B, 7] - action in world frame
                      [eef_pos(3), axis_angle(3), gripper(1)]
        viewmatrix: [B, 4, 4] - view matrix (world → camera)

    Returns:
        action_camera: [B, 7] - action in camera frame
                       [eef_pos_cam(3), axis_angle_cam(3), gripper(1)]
    """
    B = action_world.shape[0]
    device = action_world.device

    # Extract components
    eef_pos_world = action_world[:, :3]  # [B, 3]
    axis_angle_world = action_world[:, 3:6]  # [B, 3]
    gripper = action_world[:, 6:7]  # [B, 1]

    # Transform position: world → camera
    eef_pos_homo = torch.cat([eef_pos_world, torch.ones(B, 1, device=device)], dim=-1)  # [B, 4]
    eef_pos_cam_homo = torch.bmm(viewmatrix, eef_pos_homo.unsqueeze(-1)).squeeze(-1)  # [B, 4]
    eef_pos_cam = eef_pos_cam_homo[:, :3]  # [B, 3]

    # Transform rotation (axis-angle): world → camera
    # Axis-angle rotation: just rotate the axis vector
    R_world_to_cam = viewmatrix[:, :3, :3]  # [B, 3, 3]
    axis_angle_cam = torch.bmm(R_world_to_cam, axis_angle_world.unsqueeze(-1)).squeeze(-1)  # [B, 3]

    # Gripper state is invariant
    action_camera = torch.cat([eef_pos_cam, axis_angle_cam, gripper], dim=-1)  # [B, 7]

    return action_camera


class ActionMotionPriorWithCoordTransform(nn.Module):
    """
    Action Motion Prior with coordinate frame transformation.

    Transforms action from world frame to camera frame before encoding.
    """

    def __init__(self, action_dim=7, token_dim=2048, grid_size=16,
                 camera_pos=None, camera_quat=None):
        super().__init__()
        self.action_dim = action_dim
        self.token_dim = token_dim
        self.grid_size = grid_size

        # LIBERO agentview camera pose (fixed)
        if camera_pos is None:
            camera_pos = torch.tensor([1.5, 0.0, 0.9], dtype=torch.float32)
        if camera_quat is None:
            # [x, y, z, w] format
            camera_quat = torch.tensor([0.43, 0.43, 0.56, 0.56], dtype=torch.float32)

        self.register_buffer('camera_pos', camera_pos)
        self.register_buffer('camera_quat', camera_quat)

        # Build view matrix once
        self.register_buffer('viewmatrix',
                            build_viewmatrix_from_pose(camera_pos, camera_quat))

        # Motion encoders (same as before)
        self.motion_3d_encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, 256),
        )

        self.rotation_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, 128),
        )

        self.gripper_encoder = nn.Linear(1, 32)

        self.motion_fusion = nn.Sequential(
            nn.Linear(256 + 128 + 32, 512),
            nn.GELU(),
            nn.Linear(512, token_dim),
        )

        self.spatial_attention = nn.Sequential(
            nn.Linear(token_dim, 256),
            nn.GELU(),
            nn.Linear(256, grid_size * grid_size),
            nn.Softmax(dim=-1),
        )

    def forward(self, actions, future_tokens, camera_params=None):
        """
        Args:
            actions: [B, action_horizon, 7] - actions in WORLD frame
            future_tokens: [B, 256, D]
            camera_params: Optional (not used, we use fixed camera)

        Returns:
            future_tokens_with_motion: [B, 256, D]
            spatial_attn: [B, 16, 16]
        """
        B = actions.shape[0]
        device = actions.device

        # Compute motion delta in WORLD frame
        if actions.shape[1] >= 2:
            action_t_world = actions[:, 0, :]  # [B, 7]
            action_t1_world = actions[:, 1, :]  # [B, 7]
        else:
            # Fallback: use single action
            action_t_world = torch.zeros(B, 7, device=device)
            action_t1_world = actions[:, 0, :]

        # Transform both actions to CAMERA frame
        viewmatrix_batch = self.viewmatrix.unsqueeze(0).expand(B, -1, -1)  # [B, 4, 4]
        action_t_cam = transform_action_to_camera_frame(action_t_world, viewmatrix_batch)
        action_t1_cam = transform_action_to_camera_frame(action_t1_world, viewmatrix_batch)

        # Compute motion delta in CAMERA frame
        action_delta_cam = action_t1_cam - action_t_cam  # [B, 7]

        # Decompose motion (now in camera frame!)
        motion_pos = action_delta_cam[:, :3]  # [B, 3] - XYZ in camera frame
        motion_rot = action_delta_cam[:, 3:6]  # [B, 3] - rotation in camera frame
        motion_gripper = action_delta_cam[:, 6:7]  # [B, 1]

        # Encode motion
        pos_feat = self.motion_3d_encoder(motion_pos)
        rot_feat = self.rotation_encoder(motion_rot)
        gripper_feat = self.gripper_encoder(motion_gripper)

        motion_feat = torch.cat([pos_feat, rot_feat, gripper_feat], dim=-1)
        motion_token = self.motion_fusion(motion_feat)

        # Spatial attention
        spatial_attn = self.spatial_attention(motion_token)  # [B, 256]

        # Apply motion prior
        motion_prior = motion_token.unsqueeze(1) * spatial_attn.unsqueeze(-1)
        future_tokens_with_motion = future_tokens + motion_prior

        return future_tokens_with_motion, spatial_attn.view(B, self.grid_size, self.grid_size)


# Test
if __name__ == "__main__":
    B = 4
    actions = torch.randn(B, 10, 7)  # World frame actions
    future_tokens = torch.randn(B, 256, 2048)

    model = ActionMotionPriorWithCoordTransform()
    output, spatial_attn = model(actions, future_tokens)

    print(f"Output shape: {output.shape}")
    print(f"Spatial attention shape: {spatial_attn.shape}")
    print(f"Camera viewmatrix:\n{model.viewmatrix}")
