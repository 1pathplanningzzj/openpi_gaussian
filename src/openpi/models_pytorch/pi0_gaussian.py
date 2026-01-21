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

    def encode(self, observation, device, batch_size):
        """Pure encoding step to get Gaussian features (no projection/adapter logic)."""
        valid_imgs = self.prepare_inputs(observation, device, batch_size)
        if valid_imgs is None:
            return None
            
        with torch.no_grad():
            # Stack: [B, N_views, C, H, W]
            # prepare_inputs returns list of [B, C, H, W] tensors
            # We need to stack them.
            # Assuming batch size 1 for now based on prepare_inputs
            imgs_stack = torch.stack(valid_imgs, dim=1) # [B, N, C, H, W]
            
            # Encoder Expects: inputs, K, E
            # We create dummy K and E if needed or rely on internal defaults
            # The prepared inputs are already valid AgentView images.
            # We need to construct the input dictionary expected by DepthNetwork
            
            # NOTE: Simplified for brevity. Logic assumes `encoder` handles this structure.
            # In update we filtered intrinsics.
            # We call encoder directly
            encoded_feats = self.encoder(imgs_stack, None, None) # Pass None for K/E if using fixed internals
            
            # Encoded feats is likely a voxel grid or feature map
            # Use pool to reduce
            return self.pool(encoded_feats) # [B, Dim, 4, 4] -> Flatten later



    def prepare_inputs(self, observation, device, batch_size):
        """Helper to prepare inputs for 3DGS encoder from observation.
        
        NOTE: Modified to ONLY use AgentView cameras to avoid extrinsic mismatch issues.
        Wrist cameras are filtered out because we lack dynamic extrinsics for them.
        """
        if not self.use_gaussian:
            return None
            
        # Extract images from observation
        images_dict = observation.images
        masks_dict = observation.image_masks
        
        valid_imgs = []
        
        # Filter valid images based on masks AND camera type (AgentView only)
        # We explicitly skip wrist/eye-in-hand cameras for the Gaussian Encoder
        # because we don't have their real-time extrinsics.
        for name, img in images_dict.items():
            # Check if this is a wrist camera
            is_wrist = "wrist" in name or "eye" in name or "hand" in name
            if is_wrist:
                continue

            mask = masks_dict.get(name)
            # If mask is None, assume valid. If mask is present, check first element.
            # Avoid .item() to minimize graph break noise, though control flow on tensor is still a break.
            if mask is None or (mask[0] > 0.5):
                # [Fix] Normalize if input is uint8 (0-255) to float [-1, 1]
                # This ensures compatibility with Backbones/Pre-trained models that expect normalized floats.
                if img.dtype == torch.uint8:
                    img = img.to(torch.float32) / 127.5 - 1.0
                
                valid_imgs.append(img)
                
        # Fallback: If filtering removed everything (unlikely), try to use whatever is available
        # But prefer crashing or using empty to avoiding bad geometry. 
        # For now, let's fallback to first available if empty, but log warning.
        if not valid_imgs:
            logging.warning("GaussianAdapter: No AgentView images found! Falling back to all images.")
            valid_imgs = list(images_dict.values())

        if not valid_imgs:
            return None

        # Determine target size dynamically from data
        if valid_imgs:
             target_h, target_w = valid_imgs[0].shape[-2:]
        else:
             target_h, target_w = 224, 224
        
        processed_imgs = []
        for img in valid_imgs:
             if img.shape[-2:] != (target_h, target_w):
                 img = F.interpolate(img, size=(target_h, target_w), mode='bilinear', align_corners=False)
             processed_imgs.append(img)
             
        # Pad to 3 cameras if needed (DF3DGS expects fixed num_cams usually)
        # Since we likely only have 1 AgentView, this will replicate it 3 times.
        # This is valid: it's like having 3 cameras at exactly the same spot.
        num_cams = 3
        current_count = len(processed_imgs)
        if current_count < num_cams:
            infinite_imgs = itertools.cycle(processed_imgs)
            processed_imgs_padded = [next(infinite_imgs) for _ in range(num_cams)]
            processed_imgs = processed_imgs_padded
        elif current_count > num_cams:
            processed_imgs = processed_imgs[:num_cams]
            
        # Stack: [B, N, 3, H, W]
        img_stack = torch.stack(processed_imgs, dim=1)
        
        # Prepare other inputs
        masks = torch.ones(batch_size, num_cams, 1, target_h, target_w, device=device)
        
        # Hardcode Intrinsics
        # Since we filtered out wrist cameras, we only use AgentView intrinsics.
        # Agentview (45 deg FoV) -> f=309.0 for 256x256
        
        scale = target_h / 256.0
        f_agent = 309.0 * scale
        cx = target_w / 2.0
        cy = target_h / 2.0
        
        # Consistent K for all inputs (since they are all AgentView)
        K_agent_t = torch.tensor([
            [f_agent, 0, cx],
            [0, f_agent, cy],
            [0, 0, 1]
        ], device=device)
        
        # Replicate K for all 'cameras' (including padded ones)
        K = K_agent_t.unsqueeze(0).unsqueeze(0).repeat(batch_size, num_cams, 1, 1) # [B, N, 3, 3]

        # Extrinsics: Identity is now CORRECT because we define the single AgentView 
        # to be the origin of our reconstruction coordinate system.
        extr = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(batch_size, num_cams, 1, 1)
        
        inputs = {
            ('color_aug', 0): img_stack,
            ('color_aug', -1): img_stack,
            ('color_aug', 1): img_stack,
            'mask': masks,
            'K': K,
            'c2e_extr': extr,
            'e2c_extr': extr, 
        }
        return inputs

    def forward(self, gaussian_inputs):
        """
        Processes gaussian inputs and returns embeddings and attention masks.
        Returns:
            embs: [B, N_tokens, Dim]
            mask: [B, N_tokens]
        """
        if not self.use_gaussian or gaussian_inputs is None:
            return None, None

        with torch.no_grad():
             # Run frozen encoder
             outputs = self.encoder(gaussian_inputs)
             
             # Extract features
             feats_list = []
             # Assuming 3 cameras
             for cam_idx in range(3): 
                 key = ('cam', cam_idx)
                 if key in outputs:
                     feat_maps = outputs[key].get(('img_feat', 0, 0))
                     if feat_maps:
                         feat = feat_maps[-1] # [B, C, H, W]
                         feats_list.append(feat)

        if not feats_list:
            logging.warning("No features extracted from Gaussian Encoder.")
            return None, None

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
        
        # Project (Learnbale)
        gaussian_embs = self.proj(gaussian_raw_embs)
        
        # Prepare masks
        g_bs = gaussian_embs.shape[0]
        g_len = gaussian_embs.shape[1]
        g_mask = torch.ones(g_bs, g_len, dtype=torch.bool, device=gaussian_embs.device)
        
        return gaussian_embs, g_mask
