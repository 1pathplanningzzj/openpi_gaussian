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
# that ‘s done ！！⭐️ 🫡
_root_path = Path(__file__).resolve().parents[3]
_ad_ffgs_path = _root_path / "third_party" / "AD-FFgsStudio"
if str(_ad_ffgs_path) not in sys.path:
    sys.path.append(str(_ad_ffgs_path))

try:
    from models.vggt3dgs_model import VGGT3DGSModel
except ImportError:
    logging.warning("Could not import VGGT3DGSModel. 3DGS integration may fail.")
    VGGT3DGSModel = None

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
    Adapter for integrating VGGT (Transformer-based 3DGS) features into OpenPI models.
    """
    def __init__(self, use_gaussian: bool, action_expert_width: int):
        super().__init__()
        self.use_gaussian = use_gaussian
        self.encoder = None
        self.proj = None
        self.predictor = None # Future Predictor
    
        if self.use_gaussian and VGGT3DGSModel is not None:
            logging.info("Initializing VGGT 3DGS Components in Adapter...")
            
            try:
                # Initialize VGGT Model
                # Parameters based on vggt3dgs_model.py defaults or typical values
                self.encoder = VGGT3DGSModel(sh_degree=4, min_depth=1.5, max_depth=100.0)
                
                # Freeze Encoder
                for param in self.encoder.parameters():
                    param.requires_grad = False
                self.encoder.eval()
                
                # Projection and Head
                # VGGT embed_dim: The aggregator seems to return 2048 dim (concatenated? or large DINO)
                self.gaussian_feat_dim = 2048 
                self.proj = nn.Linear(self.gaussian_feat_dim, action_expert_width)
                
                # Pooling
                # Input: 2 views (Agent + Wrist)
                # Target: ~200 tokens total
                # 10x10 -> 100 tokens/view * 2 views = 200 tokens
                self.pool = nn.AdaptiveAvgPool2d((10, 10))
                
                # Future Gaussian Predictor (World Model)
                self.predictor = GaussianPredictor(
                    feature_dim=action_expert_width, 
                    action_dim=32, 
                    gaussian_dim=self.gaussian_feat_dim,
                    hidden_dim=512
                )
                
            except Exception as e:
                logging.error(f"Failed to initialize VGGT components: {e}")
                self.use_gaussian = False
                self.encoder = None
        else:
            self.use_gaussian = False

    def encode(self, observation, device, batch_size):
        """Pure encoding step to get Gaussian features."""
        # This is mainly for inference/debugging checks
        inputs = self.prepare_inputs(observation, device, batch_size)
        if inputs is None:
            return None
        return self.forward(inputs)[0] # Just return embs

    def prepare_inputs(self, observation, device, batch_size):
        """Helper to prepare inputs for VGGT encoder from observation.
        VGGT expects [Batch_size, view_num, 3, H, W]
        We select 2 views: Agent (Global) and Wrist.
        """
        if not self.use_gaussian:
            return None
            
        images_dict = observation.images
        keys = list(images_dict.keys())
        
        # 1. Find Wrist Camera
        wrist_key = next((k for k in keys if "wrist" in k or "eye" in k or "hand" in k), None)
        
        # 2. Find Agent/Global Camera
        # Heuristic: Look for 'agent', 'high', 'env', 'cam0'
        # Or simple fallback: Anything that is NOT the wrist key
        agent_key = next((k for k in keys if k != wrist_key and any(x in k for x in ["agent", "high", "env", "front", "cam0"])), None)
        
        # Fallback if no specific keyword found: pick first non-wrist key
        if not agent_key and len(keys) > (1 if wrist_key else 0):
             agent_key = next((k for k in keys if k != wrist_key), None)
        
        selected_imgs = []
        # Order: Agent, then Wrist (Subjective choice, VGGT is robust to order)
        if agent_key: selected_imgs.append(images_dict[agent_key])
        if wrist_key: selected_imgs.append(images_dict[wrist_key])
        
        if not selected_imgs:
            return None
            
        processed_imgs = []
        # Use the image size defined in the encoder if available, otherwise default to 518 (VGGT standard)
        target_size = getattr(self.encoder, "img_size", 518)
        
        for img in selected_imgs:
            # Handle Temporal Dimension
            # If 5D: [B, T, ...] -> Take last frame [B, ...]
            # Note: T is usually dim 1.
            if img.ndim == 5: 
                img = img[:, -1] # [B, ...]
            
            # Start with [B, ...] (4D)
            # Check for Channel Last [B, H, W, C] (C=3)
            if img.shape[-1] == 3:
                img = img.permute(0, 3, 1, 2) # [B, 3, H, W]
                
            # Normalize
            if img.dtype == torch.uint8:
                img = img.to(torch.float32) / 255.0 
            
            # Resize
            # F.interpolate expects [B, C, H, W]
            if img.shape[-2:] != (target_size, target_size):
                img = F.interpolate(img, size=(target_size, target_size), mode='bilinear', align_corners=False)
                
            processed_imgs.append(img)
            
        # Stack Views: [B, V=2, C, H, W]
        # If only 1 camera found, V=1
        imgs_stacked = torch.stack(processed_imgs, dim=1)
             
        return imgs_stacked.to(device)

    def forward(self, gaussian_inputs):
        """
        Processes gaussian inputs and returns embeddings.
        Input: [B, S, 3, H, W]
        """
        if not self.use_gaussian or gaussian_inputs is None:
            return None, None

        with torch.no_grad():
             outputs = self.encoder(gaussian_inputs)
             # Extract tokens from the modified return signature
             # outputs: depth, rot, scale, opacity, sh, aggregated_tokens_list, patch_start_idx
             aggregated_tokens_list = outputs[-2]
             patch_start_idx = outputs[-1]

             if isinstance(aggregated_tokens_list, (list, tuple)):
                 raw_tokens = aggregated_tokens_list[-1]
             else:
                 raw_tokens = aggregated_tokens_list
             
        if raw_tokens is None:
            logging.warning("No features extracted from Gaussian Encoder.")
            return None, None
            
        # raw_tokens shape: [B, S, N_total, D]
        # Remove register tokens
        if patch_start_idx > 0:
            raw_tokens = raw_tokens[:, :, patch_start_idx:, :]
            
        # raw_tokens shape: [B, S, N_patches, D]
        # e.g., [B, 2, 1369, 1024]
        B, S, N, D = raw_tokens.shape

        
        # Reshape to spatial for pooling
        # N = 37*37 = 1369 (assuming 518/14)
        H_feat = int(N**0.5) 
        
        # [B, S, N, D] -> [B*S, D, H_feat, W_feat]
        tokens_spatial = raw_tokens.view(B*S, H_feat, H_feat, D).permute(0, 3, 1, 2)
        
        # Pool
        # [B*S, D, 10, 10]
        pooled = self.pool(tokens_spatial)
        
        # Flatten back
        # [B, S, D, 100] -> [B, S, 100, D]
        tokens_pooled = pooled.flatten(2).transpose(1, 2).view(B, -1, D)
        
        # Project [B, S*100, D] -> [B, TotalTokens, ProjDataset]
        gaussian_embs = self.proj(tokens_pooled)
        
        # Prepare masks
        g_bs = gaussian_embs.shape[0]
        g_len = gaussian_embs.shape[1]
        g_mask = torch.ones(g_bs, g_len, dtype=torch.bool, device=gaussian_embs.device)
        
        return gaussian_embs, g_mask
