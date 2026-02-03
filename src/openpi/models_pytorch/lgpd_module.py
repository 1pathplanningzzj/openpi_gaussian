# zijian
# date 2026.01.26
# Description: Language-Gated Physical Distillation (LGPD) module for OpenPI.
# Implements semantic spectral filtering to refocus 3D tokens based on language instructions.

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Tuple, Union

# Removed explicit CLIPTextEncoder to reuse Pi0/PaliGemma's internal embeddings
# Users should pass the pooled text embedding directly to LGPD.
# zijian it is convenient to have this module here, clip or not PaliGemma. 
# todo zijian: debug and fix date 2026.0128
class LanguageGatedPhysicalDistillation(nn.Module):
    """
    Language-Gated Physical Distillation (LGPD) Module.
    
    Mechanics:
    1. Cross-Attention: Visual Tokens (Keys) <-> Text Embedding (Query) -> Attention Map.
    2. Gating: Generates a gate G in [bias, 1.0].
    3. Separation:
       - Foreground (High-Freq): Tokens * G
       - Background (Low-Freq): Tokens * (1-G) -> Global Average Pooling -> Context
    4. Fusion: Foreground + Expanded Context -> Refined Z_t
    """
    def __init__(
        self, 
        token_dim: int, 
        text_dim: int = 512, 
        num_context_tokens: int = 16, # Not strictly used in simple version, but kept for interface
        background_weight: float = 0.1, 
        use_learnable_fusion: bool = True
    ):
        super().__init__()
        self.token_dim = token_dim
        self.text_dim = text_dim
        self.background_weight = background_weight
        
        # Project text to visual dimension for attention
        self.text_proj = nn.Linear(text_dim, token_dim)
        
        # Attention scale
        self.scale = token_dim ** -0.5
        
        # Fusion layer (Optional: if we want to mix context back differently)
        if use_learnable_fusion:
            self.fusion_net = nn.Sequential(
                nn.Linear(token_dim * 2, token_dim),
                nn.LayerNorm(token_dim),
                nn.SiLU(),
                nn.Linear(token_dim, token_dim)
            )
        else:
            self.fusion_net = None
            
        # Learnable temperature for gating
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)

    def forward(
        self, 
        visual_tokens: torch.Tensor, 
        text_embedding: torch.Tensor, 
        return_gate: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            visual_tokens: [B, N, D] - The 3D/Visual tokens (e.g. from VGGT reasoning ???).
            text_embedding: [B, D_text] - Pooled text embedding.
        
        Returns:
            refined_tokens: [B, N, D]
            gate (optional): [B, N, 1]
        """
        B, N, D = visual_tokens.shape
        
        # 1. Project text: [B, D_text] -> [B, D] -> [B, 1, D]
        # Handle case where text_embedding might have temporal dimension [B, T, D_text]
        if text_embedding.ndim == 3:
            # [B, T, D_text] -> take mean over time -> [B, D_text]
            text_embedding = text_embedding.mean(dim=1)
        elif text_embedding.ndim != 2:
            raise ValueError(f"text_embedding must be 2D [B, D] or 3D [B, T, D], got shape {text_embedding.shape}")
        
        # Now text_embedding is [B, D_text]
        text_proj = self.text_proj(text_embedding)  # [B, D]
        text_query = text_proj.unsqueeze(1)  # [B, 1, D]
        
        # 2. Compute Match Score (Dot Product): [B, 1, D] @ [B, D, N] -> [B, 1, N]
        # We want to know how much each visual token matches the text visulize to confirm it.
        scores = torch.bmm(text_query, visual_tokens.transpose(1, 2))
        scores = scores * self.scale / self.temperature
        
        # 3. Compute Soft Gate
        # Use Sigmoid because we want independent probability of being "relevant" per token,
        # not a distribution summing to 1 (Softmax). 
        # But attention usually implies competition. 
        gate = torch.sigmoid(scores).permute(0, 2, 1) # [B, N, 1]
        
        # 4. Apply Background Bias
        # We don't want to zero out background completely, just suppress it.
        # G' = w_bg + (1 - w_bg) * G
        effective_gate = self.background_weight + (1.0 - self.background_weight) * gate
        
        # 5. Separate Streams
        foreground = visual_tokens * effective_gate
        
        # Background Context: simple average of weighted background features
        # Weight = (1 - gate)
        bg_weight = (1.0 - gate)
        # Avoid div by zero
        bg_sum = (visual_tokens * bg_weight).sum(dim=1, keepdim=True) # [B, 1, D]
        bg_norm = bg_weight.sum(dim=1, keepdim=True).clamp(min=1e-6)
        background_context = bg_sum / bg_norm # [B, 1, D] Global background vector
        
        # 6. Fusion
        if self.fusion_net is not None:
            # Expand context to [B, N, D]
            context_expanded = background_context.expand(-1, N, -1)
            # Concat and fuse
            combined = torch.cat([foreground, context_expanded], dim=-1) # [B, N, 2D]
            refined_tokens = self.fusion_net(combined)
            
            # Residual connection to original tokens (optional but good for stability)
            refined_tokens = refined_tokens + visual_tokens
        else:
            # Simple addition? Or just replace?
            # If no fusion net, we just add context to foreground
            refined_tokens = foreground + background_context
            
        if return_gate:
            return refined_tokens, effective_gate
            
        return refined_tokens

def visualize_lgpd_gate(gate: torch.Tensor, height: int, width: int, save_path: str = "lgpd_mask.png"):
    """
    Helper to visualize the attention gate as a heatmap.
    Args:
        gate: [B, N, 1] tensor (take first batch item usually).
        height, width: Spatial dimensions of tokens (e.g. 14x14).
    """
    import matplotlib.pyplot as plt
    try:
        # Take first item in batch
        g = gate[0].detach().cpu().squeeze(-1) # [N]
        
        # Reshape to 2D
        # Warning: Assumes tokens are spatially ordered!
        g_map = g.view(height, width).numpy()
        
        plt.figure(figsize=(5, 5))
        plt.imshow(g_map, cmap='jet', vmin=0, vmax=1)
        plt.colorbar()
        plt.title("LGPD Attention Mask")
        plt.axis('off')
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        logging.error(f"Failed to visualize LGPD gate: {e}")
