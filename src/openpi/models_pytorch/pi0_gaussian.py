# zijian
# date 2026.01.19
# Description: Adapter module for integrating frozen 3D Gaussian Splatting (DF3DGS) features into OpenPI.
import logging
import sys
from pathlib import Path
import itertools

import torch
from torch import nn
import torch.nn.functional as F

# Add AD-FFgsStudio to python path
# Assuming the file is at src/openpi/models_pytorch/pi0_gaussian.py
# We need to go up 3 levels to reach the root (src/openpi/models_pytorch -> src/openpi -> src -> root)
# that ‘s done ！
_root_path = Path(__file__).resolve().parents[3]
_ad_ffgs_path = _root_path / "third_party" / "AD-FFgsStudio"
if str(_ad_ffgs_path) not in sys.path:
    sys.path.append(str(_ad_ffgs_path))

try:
    from models.df3dgs_model_module import DF3DGS_LITModelModule
    # Mock load_official_weights to avoid error if weights are missing
    from models.df3dgs_model import DF3DGSModel
    _original_load = getattr(DF3DGSModel, "load_official_weights", None)
    DF3DGSModel.load_official_weights = lambda self: None 
except ImportError:
    logging.warning("Could not import DF3DGS_LITModelModule. 3DGS integration will fail if used.")
    DF3DGS_LITModelModule = None


class GaussianPredictor(nn.Module):
    """
    Predicts the evolution of Gaussian features conditioned on:
    1. Current State Representation (from Transformer)
    2. Predicted Action (from Action Head)
    
    This enables the "World Model" capability where the model learns:
    "Given current state H_t, and executing Action A_t, what is the Next Gaussian G_{t+1}?"
    """
    def __init__(self, feature_dim, action_dim, gaussian_dim, hidden_dim=512):
        super().__init__()
        # Condition: Feature (Transformer Output) + Action
        self.cond_proj = nn.Linear(feature_dim + action_dim, hidden_dim)
        
        # Flow Matching Network for Gaussian Tokens
        # We model the distribution of *Next Frame's Gaussian Tokens*
        # Input to Flow Net: Noisy Target Gaussian (gaussian_dim) + Condition (hidden_dim) + Time (1)
        self.net = nn.Sequential(
            nn.Linear(gaussian_dim + hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, gaussian_dim) # Output vector field
        )
        # linear + Silu v1 for model test
        # todo zijian 0121 fix and enhance it #
    def forward(self, x_t, transformer_features, actions, t):
        """
        x_t: [B, N, D] - Noisy Future Gaussian Tokens (or flattened)
        transformer_features: [B, FeatureDim] - Aggregated State Info
        actions: [B, ActionDim] - Actions
        t: [B] - Time
        """
        B = x_t.shape[0]
        
        # 1. Prepare Condition
        # Concatenate State + Action
        cond_raw = torch.cat([transformer_features, actions], dim=-1) # [B, F+A]
        cond = self.cond_proj(cond_raw) # [B, H]
        
        # 2. Expand Condition for Tokens if x_t is sequence
        # x_t is likely [B, NumTokens, GaussianDim]
        # We treat each token independently conditioned on the global context (or process them together)
        # For simplicity, we broadcast condition to all tokens
        if x_t.ndim == 3:
            num_tokens = x_t.shape[1]
            cond = cond.unsqueeze(1).repeat(1, num_tokens, 1) # [B, N, H]
            t = t.view(B, 1, 1).repeat(1, num_tokens, 1) # [B, N, 1]
        elif t.ndim == 1:
            t = t.unsqueeze(-1)

        # 3. Concatenate for Update
        # Input: [B, N, D + H + 1]
        inp = torch.cat([x_t, cond, t], dim=-1)
        
        return self.net(inp)


class GaussianAdapter(nn.Module):
    """
    Adapter for integrating 3D Gaussian Splatting Encoder into OpenPI models.
    Handles initialization, data preparation, feature extraction OR Future Prediction.
    """
    def __init__(self, use_gaussian: bool, action_expert_width: int):
        super().__init__()
        self.use_gaussian = use_gaussian
        self.encoder = None
        self.proj = None
        self.pool = None
        self.predictor = None # Future Predictor
    
        if self.use_gaussian and DF3DGS_LITModelModule is not None:
            logging.info("Initializing 3DGS Components in Adapter...")
            # Config based on AD-FFgsStudio requirements
            # Updated to match OpenPI/Libero resolution (224x224) and camera count (3)
            # DepthNetwork expects parameters unpacked from a sub-dictionary
            depth_net_sub_cfg = {
                "num_layers": 18, 
                "weights_init": True,
                "fusion_level": 2, 
                "fusion_feat_in_dim": 256,
                "use_skips": False,
                "scales": [0, 1, 2, 3],
                "num_cams": 3,
                "novel_view_mode": 'MF',
                # VFNet params must be here too as DepthNetwork passes cfg to VFNet
                "height": 224, "width": 224,
                "voxel_unit_size": [1.0, 1.0, 1.5],
                "voxel_size": [100, 100, 20],
                "voxel_str_p": [-50.0, -50.0, -15.0],
                "voxel_pre_dim": [64],
                "proj_d_bins": 50, "proj_d_str": 2, "proj_d_end": 50,
            }

            gaussian_cfg = {
                "height": 224, "width": 224, "num_cams": 3, "embed_dim": 1024,
                "learning_rate": 2e-5, "weight_decay": 0.01,
                "lr_restart_epoch": 5, "lr_restart_mult": 2, "lr_min_factor": 0.01,
                "frame_ids": [0, -1, 1], "depth_conf_thr": 0.99,
                "min_depth": 1.5, "max_depth": 80,
                "focal_length_scale": 300, "init_scale_thr": 0.02,
                "sh_degree": 4, "save_image_duration": 100,
                "lambda_project": 1.0, "lambda_edge": 0.1, "lambda_depth": 0.001,
                "lambda_gaussian": 2, "lambda_scale": 0.01, "lambda_opacity": 0.01,
                "batch_size": 1, 
                # Params that DepthNetwork needs must be inside a dict
                "depth_net_cfg_wrapper": depth_net_sub_cfg,
                # Keep top-level for DF3DGSModel just in case
                "num_layers": 18, "weights_init": True,
                "fusion_level": 2, "fusion_feat_in_dim": 256,
                "use_skips": False,
                "voxel_unit_size": [1.0, 1.0, 1.5],
                "voxel_size": [100, 100, 20],
                "voxel_str_p": [-50.0, -50.0, -15.0],
                "voxel_pre_dim": [64],
                "proj_d_bins": 50, "proj_d_str": 2, "proj_d_end": 50,
                "depth_net_cfg": {"novel_view_mode": 'MF'}
            }
            
            try:
                self.module_wrapper = DF3DGS_LITModelModule(gaussian_cfg, save_dir='/tmp')
                self.encoder = self.module_wrapper.model.depth_net
                
                # Freeze Encoder
                for param in self.encoder.parameters():
                    param.requires_grad = False
                self.encoder.eval()
                
                # Projection and Head
                # fusion_level 2 (ResNet) -> 128 channels
                self.gaussian_feat_dim = 128 
                # OpenPI models (especially Pi0) typically use an embedding width of 2048 or similar
                # Check pi0_pytorch.py: action_expert_config.width is passed here
                self.proj = nn.Linear(self.gaussian_feat_dim, action_expert_width)
                
                # Pooling for token reduction (16 tokens per view)
                self.pool = nn.AdaptiveAvgPool2d((4, 4)) 

                # Future Gaussian Predictor (World Model)
                # Inputs: TransformerFeats (action_expert_width) + Actions (assumed flattened or pooled)
                # We assume we pass in the full action sequence logic later
                # For now, let's assume we flatten the action horizon
                action_dim_flat = 14 * 6 # Horizon * Dim (Example default) - will need to be dynamic
                # To be safe, we let the projection handle a generic size or specific size.
                # Actually, let's make it dynamic in forward or init with config.
                # Hardcoding a reasonable default or relying on passed args.
                # Let's use 512 as hidden.
                self.predictor = GaussianPredictor(
                    feature_dim=action_expert_width, 
                    action_dim=32, # Placeholder, will need to match actual usage
                    gaussian_dim=self.gaussian_feat_dim,
                    hidden_dim=512
                )
                
            except Exception as e:
                logging.error(f"Failed to initialize 3DGS components: {e}")
                self.use_gaussian = False
                self.encoder = None
        else:
            self.use_gaussian = False

    def _extract_features(self, outputs):
        feats_list = []
        # Assuming 3 cameras (same logic as before)
        for cam_idx in range(3): 
            key = ('cam', cam_idx)
            if key in outputs:
                # Use scale 0 for highest res features 
                feat_maps = outputs[key].get(('img_feat', 0, 0))
                if feat_maps is not None:
                     feat = feat_maps[-1]
                     feats_list.append(feat)

        if not feats_list:
            return None
            
        feats_stacked = torch.stack(feats_list, dim=1) # [B, N, C, H, W]
        B_val, N, C, H, W = feats_stacked.shape
        
        # Reshape for pooling: [B*N, C, H, W]
        feats_flat = feats_stacked.view(B_val*N, C, H, W)
        
        # Pool: [B*N, C, h_p, w_p]
        feats_pooled = self.pool(feats_flat)
        
        # Flatten spatial: [B*N, C, K] -> [B*N, K, C]
        feats_tokens = feats_pooled.flatten(2).transpose(1, 2)
        
        # Reshape back: [B, N*K, C]
        gaussian_raw_embs = feats_tokens.contiguous().view(B_val, -1, C)
        
        return gaussian_raw_embs

    def encode(self, observation, device, batch_size):
        """Pure encoding step to get Gaussian features (no projection)."""
        inputs = self.prepare_inputs(observation, device, batch_size)
        if inputs is None:
            return None
            
        with torch.no_grad():
            outputs = self.encoder(inputs, MF_frames=[-1, 1])
            return self._extract_features(outputs)



    def _compute_camera_extrinsics(self, state):
        """Computes World-to-Camera extrinsics (C2W) from robot state (EE Pose)."""
        device = state.device
        dtype = state.dtype
        B = state.shape[0]
        
        # State: [B, 8] -> [x, y, z, euler_x, euler_y, euler_z, gripper...]
        pos = state[:, :3] # [B, 3] Position (meters)
        euler = state[:, 3:6] # [B, 3] Euler Angles (XYZ order, radians)
        
        # Pure PyTorch Euler XYZ -> Rotation Matrix
        # R = R_z @ R_y @ R_x (Extrinsic XYZ)
        x = euler[:, 0]
        y = euler[:, 1]
        z = euler[:, 2]
        
        cx, sx = torch.cos(x), torch.sin(x)
        cy, sy = torch.cos(y), torch.sin(y)
        cz, sz = torch.cos(z), torch.sin(z)
        
        # R_x:
        # [1, 0, 0]
        # [0, cx, -sx]
        # [0, sx, cx]
        rx = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
        rx[:, 1, 1] = cx
        rx[:, 1, 2] = -sx
        rx[:, 2, 1] = sx
        rx[:, 2, 2] = cx
        
        # R_y:
        # [cy, 0, sy]
        # [0, 1, 0]
        # [-sy, 0, cy]
        ry = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
        ry[:, 0, 0] = cy
        ry[:, 0, 2] = sy
        ry[:, 2, 0] = -sy
        ry[:, 2, 2] = cy
        
        # R_z:
        # [cz, -sz, 0]
        # [sz, cz, 0]
        # [0, 0, 1]
        rz = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
        rz[:, 0, 0] = cz
        rz[:, 0, 1] = -sz
        rz[:, 1, 0] = sz
        rz[:, 1, 1] = cz
        
        # R = R_z @ R_y @ R_x
        rot_mat = rz @ ry @ rx

        # C2W Pose (World-to-Camera Transform IS the Pose of Camera in World)
        # So T_c2w = [R|t]
        T_c2w = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
        T_c2w[:, :3, :3] = rot_mat
        T_c2w[:, :3, 3] = pos
        
        return T_c2w

    def prepare_inputs(self, observation, device, batch_size):
        """Helper to prepare inputs for 3DGS encoder from observation.
        
        Modified for Temporal 3DGS:
        - Accepts wrist camera images at t-1, t, t+1.
        - Computes extrinsics from robot state.
        """
        if not self.use_gaussian:
            return None
            
        images_dict = observation.images
        masks_dict = observation.image_masks
        
        # Find wrist camera
        wrist_key = None
        for name in images_dict.keys():
            if "wrist" in name or "eye" in name or "hand" in name:
                wrist_key = name
                break
                
        if wrist_key is None:
            logging.warning("GaussianAdapter: No Wrist camera found! Skipping 3DGS.")
            return None
            
        # Get Images: Expected [B, T=3, C, H, W]
        # If T dim is missing (B, C, H, W) -> unsqueeze to (B, 1, C, H, W) and handle gracefully
        imgs = images_dict[wrist_key]
        if imgs.ndim == 5:
            # [B, T, C, H, W]
            pass
        elif imgs.ndim == 4:
            # [B, C, H, W] -> [B, 1, C, H, W]
            imgs = imgs.unsqueeze(1)
        else:
             logging.error(f"Unexpected image shape: {imgs.shape}")
             return None
             
        # Normalize if uint8
        if imgs.dtype == torch.uint8:
            imgs = imgs.to(torch.float32) / 127.5 - 1.0

        # Resize if needed
        # Assuming we need 224x224
        target_h, target_w = 224, 224
        if imgs.shape[-2:] != (target_h, target_w):
             B, T, C, H, W = imgs.shape
             imgs_flat = imgs.view(B*T, C, H, W)
             imgs_flat = F.interpolate(imgs_flat, size=(target_h, target_w), mode='bilinear', align_corners=False)
             imgs = imgs_flat.view(B, T, C, target_h, target_w)
             
        # Extract slices
        # Data Loader returns [-1, 0, 1]. Size 3.
        # If size < 3, we replicate.
        T = imgs.shape[1]
        if T == 3:
            img_prev = imgs[:, 0]
            img_curr = imgs[:, 1]
            img_next = imgs[:, 2]
        else:
            # Fallback
            img_curr = imgs[:, 0]
            img_prev = img_curr
            img_next = img_curr
            
        # Extrinsics
        # Expect state: [B, T, 8]
        if not hasattr(observation, "state"):
             logging.warning("No state found for extrinsics!")
             return None
        
        state = observation.state # [B, T, 8] or [B, 8]
        # Force float32 for compatibility with AD-FFgsStudio (avoid Double vs Float matmul error)
        if state.dtype != torch.float32:
            state = state.to(dtype=torch.float32)

        if state.ndim == 2:
             state = state.unsqueeze(1)
             
        T_state = state.shape[1]
        if T_state == 3:
             state_prev = state[:, 0]
             state_curr = state[:, 1]
             state_next = state[:, 2]
        else:
             state_curr = state[:, 0]
             state_prev = state_curr
             state_next = state_curr
             
        ext_prev = self._compute_camera_extrinsics(state_prev)
        ext_curr = self._compute_camera_extrinsics(state_curr)
        ext_next = self._compute_camera_extrinsics(state_next)
        
        # Prepare Inputs Dict
        # Need to replicate to num_cams=3 for DF3DGS compatibility
        num_cams = 3
        
        def replicate(tensor, n):
             # tensor: [B, ...] -> [B, N, ...]
             return tensor.unsqueeze(1).repeat(1, n, *([1]*(tensor.ndim-1)))

        # K intrinsics (Approximate or use placeholder)
        # Wrist camera K. 
        # TODO: Read from calibration if possible. For now, Use 45 deg FOV approx/Identity.
        f = 0.5 * target_w / 0.41421356 # tan(22.5 deg)
        K_mat = torch.eye(3, device=device, dtype=torch.float32)
        K_mat[0, 0] = f
        K_mat[1, 1] = f
        K_mat[0, 2] = target_w / 2
        K_mat[1, 2] = target_h / 2
        K = replicate(K_mat.unsqueeze(0).repeat(batch_size, 1, 1), num_cams)
        
        # Masks
        masks = torch.ones(batch_size, num_cams, 1, target_h, target_w, device=device)
        
        # Stack Temporal Frames as Spatial Views (Cam0=Prev, Cam1=Curr, Cam2=Next)
        img_stack = torch.stack([img_prev, img_curr, img_next], dim=1)
        extr_stack = torch.stack([ext_prev, ext_curr, ext_next], dim=1)
        
        inputs = {
            # Provide the stacked temporal frames as the "current" observation
            ('color_aug', 0): img_stack,
            # Duplicate for temporal keys to avoid errors if MF mode is used
            ('color_aug', -1): img_stack,
            ('color_aug', 1): img_stack,
            'mask': masks,
            'K': K,
            # Use real temporal extrinsics
            'c2e_extr': extr_stack,
            'e2c_extr': torch.linalg.inv(extr_stack)
        }
        
        return inputs

    def forward(self, gaussian_inputs):
        """
        Processes gaussian inputs (from prepare_inputs) and returns embeddings.
        """
        if not self.use_gaussian or gaussian_inputs is None:
            return None, None

        with torch.no_grad():
             # Run frozen encoder with MF Frames
             outputs = self.encoder(gaussian_inputs, MF_frames=[-1, 1])
             
             gaussian_raw_embs = self._extract_features(outputs)
             
        if gaussian_raw_embs is None:
            logging.warning("No features extracted from Gaussian Encoder.")
            return None, None
        
        # Project
        gaussian_embs = self.proj(gaussian_raw_embs)
        
        # Prepare masks
        g_bs = gaussian_embs.shape[0]
        g_len = gaussian_embs.shape[1]
        g_mask = torch.ones(g_bs, g_len, dtype=torch.bool, device=gaussian_embs.device)
        
        return gaussian_embs, g_mask
