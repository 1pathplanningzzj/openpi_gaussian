import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Lightweight LoRA adapter (no peft dependency)
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """LoRA adapter wrapping an existing nn.Linear (frozen)."""

    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 32.0):
        super().__init__()
        self.original = original
        self.scaling = alpha / rank
        self.lora_A = nn.Linear(original.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, original.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.original(x) + self.lora_B(self.lora_A(x)) * self.scaling


def apply_lora_to_model(model: nn.Module, target_names=("qkv", "proj"), rank: int = 8, alpha: float = 32.0):
    """Walk *model* and replace matching nn.Linear layers with LoRALinear wrappers.

    Returns the number of LoRA-injected layers.
    """
    count = 0
    for parent_name, parent_module in list(model.named_modules()):
        for attr_name, child in list(parent_module.named_children()):
            if isinstance(child, nn.Linear) and any(t in attr_name for t in target_names):
                lora = LoRALinear(child, rank=rank, alpha=alpha)
                setattr(parent_module, attr_name, lora)
                count += 1
    return count


# ---------------------------------------------------------------------------
# Enhanced Temporal Encoding Modules (Priority 1 Improvements)
# ---------------------------------------------------------------------------

class TemporalConv3DEncoder(nn.Module):
    """3D Convolutional encoder for temporal-spatial feature extraction.

    Preserves temporal structure while downsampling spatial dimensions.
    Input: [B, C, T, H, W] where T is number of frames
    Output: [B, C_out, T, H', W'] where H', W' are downsampled
    """

    def __init__(self, in_channels=2048, out_channels=512, num_frames=3):
        super().__init__()

        # 3D Conv layers: preserve temporal dimension, downsample spatial
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 1024, kernel_size=(3, 3, 3),
                     stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.GroupNorm(32, 1024),
            nn.GELU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(1024, out_channels, kernel_size=(3, 3, 3),
                     stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
        )

        # Output: [B, 512, T, H/4, W/4]
        # For 37×37 input: [B, 512, 3, 9, 9]

    def forward(self, x):
        """
        Args:
            x: [B, C, T, H, W] - Temporal-spatial features
        Returns:
            [B, C_out, T, H', W'] - Downsampled features
        """
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class CausalTemporalAttention(nn.Module):
    """Causal temporal attention for modeling t-2, t-1 → t dependencies.

    Each frame can only attend to itself and previous frames (causal mask).
    This enforces temporal causality: future frames cannot influence past frames.
    """

    def __init__(self, embed_dim=512, num_heads=8, num_frames=3):
        super().__init__()

        self.num_frames = num_frames
        self.embed_dim = embed_dim

        # Multi-head attention
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )

        # Layer norm
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1),
        )

    def _generate_causal_mask(self, num_frames, tokens_per_frame, device):
        """Generate causal mask: frame t can only see frames 0...t.

        Returns:
            mask: [total_tokens, total_tokens] bool tensor
                  True = cannot attend, False = can attend
        """
        total_tokens = num_frames * tokens_per_frame
        mask = torch.ones(total_tokens, total_tokens, dtype=torch.bool, device=device)

        for t in range(num_frames):
            start_t = t * tokens_per_frame
            end_t = (t + 1) * tokens_per_frame

            # Frame t can see all previous frames (0...t-1) and itself
            for prev_t in range(t + 1):
                start_prev = prev_t * tokens_per_frame
                end_prev = (prev_t + 1) * tokens_per_frame
                mask[start_t:end_t, start_prev:end_prev] = False  # False = can attend

        return mask

    def forward(self, x, tokens_per_frame):
        """
        Args:
            x: [B, T*N, D] - Temporal tokens (T frames, N tokens per frame)
            tokens_per_frame: int - Number of tokens per frame
        Returns:
            [B, T*N, D] - Attended features with temporal dependencies
        """
        B, total_tokens, D = x.shape
        num_frames = total_tokens // tokens_per_frame

        # Generate causal mask
        attn_mask = self._generate_causal_mask(num_frames, tokens_per_frame, x.device)

        # Self-attention with causal mask
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = x + attn_out

        # FFN
        x = x + self.ffn(self.norm2(x))

        return x


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal frames.

    Combines fixed sinusoidal encoding with learnable scaling.
    """

    def __init__(self, num_frames, embed_dim):
        super().__init__()

        # Generate sinusoidal encoding
        position = torch.arange(num_frames).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() *
                            -(math.log(10000.0) / embed_dim))

        pe = torch.zeros(num_frames, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

        # Learnable scaling
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, frame_idx):
        """
        Args:
            frame_idx: int - Frame index (0, 1, 2, ...)
        Returns:
            [embed_dim] - Positional encoding for this frame
        """
        return self.pe[frame_idx] * self.scale


# Add AD-FFgsStudio to python path
_root_path = Path(__file__).resolve().parents[3]
_ad_ffgs_path = _root_path / "third_party" / "AD-FFgsStudio"
if str(_ad_ffgs_path) not in sys.path:
    sys.path.append(str(_ad_ffgs_path))

try:
    from models.vggt3dgs_model import VGGT3DGSModel
except ImportError:
    VGGT3DGSModel = None
    logging.warning("VGGT3DGSModel not available. Gaussian features will be disabled.")

try:
    from openpi.models_pytorch.lgpd_module import LanguageGatedPhysicalDistillation
except ImportError:
    LanguageGatedPhysicalDistillation = None
    logging.warning("LanguageGatedPhysicalDistillation not available. LGPD will be disabled.")


def visualize_vggt_and_lgpd(
    gaussian_inputs: torch.Tensor,
    gaussian_params: Optional[Dict[str, torch.Tensor]],
    lgpd_gate: Optional[torch.Tensor],
    step: int,
    save_dir: str = "./visualizations/vggt_lgpd"
):
    """
    Visualize VGGT frames and LGPD gate together.
    
    Args:
        gaussian_inputs: [B, S, 3, H, W] - Input frames to VGGT
        gaussian_params: Dict with depth_maps, rot_maps, scale_maps, opacity_maps, sh_maps
        lgpd_gate: [B, N, 1] - LGPD gate values (optional)
        step: Current training step
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Take first batch item
    if gaussian_inputs.ndim == 5:
        frames = gaussian_inputs[0].cpu().numpy()  # [S, 3, H, W]
    else:
        frames = gaussian_inputs.cpu().numpy()  # [S, 3, H, W]
    
    S = frames.shape[0]  # Number of frames (should be 3)
    H, W = frames.shape[2], frames.shape[3]
    
    # Convert from CHW to HWC and normalize
    frames_hwc = []
    for s in range(S):
        frame = frames[s].transpose(1, 2, 0)  # [H, W, 3]
        # Normalize to [0, 1] if needed
        if frame.max() > 1.0:
            frame = frame / 255.0
        frame = np.clip(frame, 0, 1)
        frames_hwc.append(frame)
    
    # Determine layout: if LGPD gate is available, add extra column
    num_cols = 5 if lgpd_gate is not None else 4
    fig, axes = plt.subplots(S, num_cols, figsize=(4 * num_cols, 4 * S))
    if S == 1:
        axes = axes[None, :]  # Ensure 2D array
    
    for s in range(S):
        col_idx = 0
        
        # Column 0: Input RGB frame
        axes[s, col_idx].imshow(frames_hwc[s])
        axes[s, col_idx].set_title(f"Frame {s+1}: Input RGB")
        axes[s, col_idx].axis('off')
        col_idx += 1
        
        # Column 1: Depth map
        if gaussian_params is not None and "depth_maps" in gaussian_params:
            depth = gaussian_params["depth_maps"][0, s].cpu().numpy()  # [H, W, 1]
            if depth.ndim == 3:
                depth = depth.squeeze(-1)  # [H, W]
            im1 = axes[s, col_idx].imshow(depth, cmap='viridis')
            axes[s, col_idx].set_title(f"Frame {s+1}: Depth")
            axes[s, col_idx].axis('off')
            plt.colorbar(im1, ax=axes[s, col_idx], fraction=0.046)
        else:
            axes[s, col_idx].text(0.5, 0.5, "No depth data", ha='center', va='center')
            axes[s, col_idx].axis('off')
        col_idx += 1
        
        # Column 2: Scale map
        if gaussian_params is not None and "scale_maps" in gaussian_params:
            scale = gaussian_params["scale_maps"][0, s].cpu().numpy()  # [H, W, 3]
            # Normalize scale values for visualization
            scale_norm = (scale - scale.min()) / (scale.max() - scale.min() + 1e-8)
            scale_norm = np.clip(scale_norm, 0, 1)
            axes[s, col_idx].imshow(scale_norm)
            axes[s, col_idx].set_title(f"Frame {s+1}: Scale")
            axes[s, col_idx].axis('off')
        else:
            axes[s, col_idx].text(0.5, 0.5, "No scale data", ha='center', va='center')
            axes[s, col_idx].axis('off')
        col_idx += 1
        
        # Column 3: Opacity map
        if gaussian_params is not None and "opacity_maps" in gaussian_params:
            opacity = gaussian_params["opacity_maps"][0, s].cpu().numpy()  # [H, W, 1]
            if opacity.ndim == 3:
                opacity = opacity.squeeze(-1)  # [H, W]
            im3 = axes[s, col_idx].imshow(opacity, cmap='gray')
            axes[s, col_idx].set_title(f"Frame {s+1}: Opacity")
            axes[s, col_idx].axis('off')
            plt.colorbar(im3, ax=axes[s, col_idx], fraction=0.046)
        else:
            axes[s, col_idx].text(0.5, 0.5, "No opacity data", ha='center', va='center')
            axes[s, col_idx].axis('off')
        # Column 4: LGPD Gate (only for first frame, as gate is per-token not per-frame)
        # Only access this column if num_cols >= 5 (i.e., lgpd_gate is not None)
        if num_cols >= 5:
            col_idx += 1
            if lgpd_gate is not None and s == 0:
                gate_np = lgpd_gate[0].detach().cpu().squeeze(-1).numpy()  # [N]
                # Reshape gate to 2D: assume tokens are in 10x10 grid (from pooling)
                N = gate_np.shape[0]
                gate_h = int(np.sqrt(N))
                if gate_h * gate_h == N:
                    gate_map = gate_np.reshape(gate_h, gate_h)
                    im4 = axes[s, col_idx].imshow(gate_map, cmap='jet', vmin=0, vmax=1)
                    axes[s, col_idx].set_title("LGPD Gate (10x10 tokens)")
                    axes[s, col_idx].axis('off')
                    plt.colorbar(im4, ax=axes[s, col_idx], fraction=0.046)
                else:
                    axes[s, col_idx].text(0.5, 0.5, f"Gate shape: {N}", ha='center', va='center')
                    axes[s, col_idx].axis('off')
            elif lgpd_gate is not None:
                axes[s, col_idx].axis('off')
            else:  # lgpd_gate is None but num_cols >= 5 (shouldn't happen, but handle gracefully)
                if s == 0:
                    axes[s, col_idx].text(0.5, 0.5, "No LGPD gate", ha='center', va='center')
                axes[s, col_idx].axis('off')
    
    plt.suptitle(f"VGGT + LGPD Visualization - Step {step}", fontsize=16)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"vggt_lgpd_step_{step:06d}.png")
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    logging.info(f"Saved VGGT + LGPD visualization to {save_path}")




class GaussianAdapter(nn.Module):
    """
    Adapter for integrating VGGT (Transformer-based 3DGS) features into OpenPI models.
    """
    def __init__(self, use_gaussian: bool, action_expert_width: int, use_lgpd: bool = True,
                 num_frames: int = 3, inference_num_frames: int = 1,
                 unfreeze_encoder: bool = False, unfreeze_decoder_only: bool = True,
                 use_lora: bool = True, lora_rank: int = 8, lora_alpha: float = 32.0,
                 lora_targets=("qkv", "proj")):
        """
        Args:
            use_gaussian: Whether to use Gaussian features
            action_expert_width: Width of action expert
            use_lgpd: Whether to use Language-Gated Physical Distillation
            num_frames: Number of consecutive frames to use from agent view during training (default: 3)
                       Uses [t-2, t-1, t] for temporal modeling. Reduced from 5 to 3 to save memory.
                       VGGT uses temporal frames to establish cross-frame geometry relationships.
                       Wrist view is excluded from VGGT encoding (but still used in 2D encoders like SigLIP).
            inference_num_frames: Number of frames to use during inference (default: 1, single frame).
                                 Single frame is more practical for real-time inference.
            unfreeze_encoder: If True, unfreeze VGGT encoder to allow end-to-end training (default: False).
                             When True, encoder will be trained with reconstruction loss.
            unfreeze_decoder_only: If True, only unfreeze decoder (gs_head) while keeping encoder frozen (default: True).
                                 This is a middle ground: train decoder with reconstruction loss while keeping encoder fixed.
        """
        super().__init__()
        self.use_gaussian = use_gaussian
        self.encoder = None
        self.proj = None
        self.lgpd = None  # Language-Gated Physical Distillation
        self.use_lgpd = use_lgpd
        self.num_frames = num_frames
        self.inference_num_frames = inference_num_frames

        if self.use_gaussian and VGGT3DGSModel is not None:
            logging.info("Initializing VGGT 3DGS Components in Adapter...")
            
            try:
                # Initialize VGGT Model
                # Parameters based on vggt3dgs_model.py defaults or typical values
                self.encoder = VGGT3DGSModel(sh_degree=4, min_depth=1.5, max_depth=100.0)
                
                # Freeze/Unfreeze Encoder based on configuration
                if unfreeze_encoder:
                    # Unfreeze entire encoder for end-to-end training
                    for param in self.encoder.parameters():
                        param.requires_grad = True
                    self.encoder.train()
                    logging.info("VGGT encoder is UNFROZEN - will be trained with reconstruction loss")
                elif unfreeze_decoder_only:
                    # Freeze encoder backbone, but unfreeze decoder (gs_head) and depth_head
                    for name, param in self.encoder.named_parameters():
                        if 'gs_head' in name or 'gs_feathead' in name or 'depth_head' in name:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                    # Set encoder to train mode so decoder/depth_head can be trained
                    self.encoder.train()
                    logging.info("VGGT encoder backbone is FROZEN, but decoder (gs_head) and depth_head are UNFROZEN")
                else:
                    # Fully freeze encoder (original behavior)
                    for param in self.encoder.parameters():
                        param.requires_grad = False
                    self.encoder.eval()
                    logging.info("VGGT encoder is FROZEN (original behavior)")

                # Apply LoRA to encoder backbone (after freezing)
                if use_lora and not unfreeze_encoder:
                    aggregator = self.encoder.aggregator
                    n_lora = apply_lora_to_model(
                        aggregator, target_names=lora_targets,
                        rank=lora_rank, alpha=lora_alpha,
                    )
                    lora_params = sum(
                        p.numel() for p in aggregator.parameters() if p.requires_grad
                    )
                    total_params = sum(p.numel() for p in aggregator.parameters())
                    logging.info(
                        f"LoRA applied: {n_lora} layers, rank={lora_rank}, "
                        f"trainable={lora_params:,} / {total_params:,} "
                        f"({100*lora_params/total_params:.2f}%)"
                    )
                
                # === Priority 1 & 2 Improvements: Enhanced Temporal Encoding + Multi-Scale Features ===

                # Configuration
                self.gaussian_feat_dim = 2048  # VGGT output dimension
                self.tokens_per_frame = 256  # 16×16 = 256 tokens per frame (was 100)
                self.use_enhanced_temporal = True  # Enable enhanced temporal encoding
                self.use_multi_scale = True  # Enable multi-scale feature extraction

                if self.use_enhanced_temporal:
                    # === Multi-Scale Feature Extraction (Priority 2) ===
                    if self.use_multi_scale:
                        # Extract features from layers [11, 17, 23] instead of just [23]
                        self.layer_indices = [11, 17, 23]

                        # Project each layer to unified dimension (512)
                        self.layer_projs = nn.ModuleList([
                            nn.Linear(2048, 512) for _ in self.layer_indices
                        ])

                        # FPN-style fusion (bottom-up)
                        self.fusion_blocks = nn.ModuleList([
                            nn.Sequential(
                                nn.Linear(512, 512),
                                nn.LayerNorm(512),
                                nn.GELU(),
                            ) for _ in range(len(self.layer_indices) - 1)
                        ])

                        # Project fused features back to 2048 for temporal_conv
                        self.fused_to_2048 = nn.Linear(512, 2048)

                        logging.info(
                            f"Multi-scale feature extraction enabled:\n"
                            f"  - Layers: {self.layer_indices}\n"
                            f"  - Fusion: FPN-style bottom-up\n"
                            f"  - Projection: 512 → 2048 for temporal conv"
                        )
                    else:
                        self.layer_indices = [23]  # Only last layer
                        self.layer_projs = None
                        self.fusion_blocks = None
                        self.fused_to_2048 = None

                    # 1. 3D Conv for temporal-spatial feature extraction
                    self.temporal_conv = TemporalConv3DEncoder(
                        in_channels=2048,
                        out_channels=512,
                        num_frames=num_frames
                    )
                    # Output: [B, 512, 3, 9, 9] for 37×37 input

                    # 2. Spatial pooling to 16×16 (256 tokens per frame)
                    self.spatial_pool = nn.AdaptiveAvgPool2d((16, 16))

                    # 3. Causal temporal attention
                    self.temporal_attn = CausalTemporalAttention(
                        embed_dim=512,
                        num_heads=8,
                        num_frames=num_frames
                    )

                    # 4. Frame positional encoding (sinusoidal + learnable)
                    self.frame_pos_encoding = nn.ModuleList([
                        SinusoidalPositionalEncoding(num_frames, 512)
                        for _ in range(num_frames)
                    ])
                    self.frame_embeddings = nn.Parameter(
                        torch.randn(num_frames, 512) * 0.02
                    )

                    # 5. Projection to action_expert_width
                    self.proj = nn.Linear(512, action_expert_width)

                    logging.info(
                        f"Enhanced Temporal Encoding enabled:\n"
                        f"  - 3D Conv: 2048 → 512 channels\n"
                        f"  - Spatial resolution: 37×37 → 16×16\n"
                        f"  - Tokens per frame: {self.tokens_per_frame}\n"
                        f"  - Total tokens: {num_frames * self.tokens_per_frame}\n"
                        f"  - Causal temporal attention: {num_frames} frames"
                    )
                else:
                    # Original implementation (fallback)
                    self.proj = nn.Linear(self.gaussian_feat_dim, action_expert_width)
                    self.pool = nn.AdaptiveAvgPool2d((10, 10))
                    self.tokens_per_frame = 100

                    # Original frame positional encoding
                    self.use_frame_pos_encoding = True
                    self.frame_embeddings = nn.Parameter(
                        torch.randn(num_frames, action_expert_width) * 0.02
                    )
                    logging.info(f"Using original temporal encoding with {num_frames} frames, 100 tokens/frame")
                
                # LGPD Module
                if self.use_lgpd:
                    logging.info("Initializing LGPD Module...")
                    self.lgpd = LanguageGatedPhysicalDistillation(
                        token_dim=action_expert_width,
                        text_dim=action_expert_width,  # Assuming pooled text emb has same dim as visual proj
                        num_context_tokens=16,
                        background_weight=0.1
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

    def prepare_inputs(self, observation, device, batch_size, is_training: bool = None):
        """Helper to prepare inputs for VGGT encoder from observation.
        VGGT expects [Batch_size, S (frames), 3, H, W]
        
        Strategy: Use consecutive frames from agent view only.
        - Agent view is more stable and suitable for temporal modeling
        - Wrist view is excluded from VGGT encoding (but still used in 2D encoders like SigLIP)
        - Multiple frames allow VGGT's global attention to establish cross-frame geometry relationships
        
        Args:
            is_training: If True, uses num_frames. If False, uses inference_num_frames.
                        If None, auto-detects from model.training.
        """
        if not self.use_gaussian:
            return None
        
        # Find agent view image key
        # Observation is a dataclass with .images attribute (dict)
        if not hasattr(observation, 'images'):
            # Fallback: try to_dict() if it's an Observation object
            if hasattr(observation, 'to_dict'):
                obs_dict = observation.to_dict()
                images_dict = obs_dict.get('image', {})
            else:
                logging.warning("Observation does not have 'images' attribute and cannot convert to dict.")
                return None
        else:
            images_dict = observation.images
        
        # Filter for agent view (exclude wrist)
        # Common agent view keys: "base_0_rgb", "agent_image", "agentview_image", etc.
        # Wrist keys: "left_wrist_0_rgb", "right_wrist_0_rgb", "wrist_image", etc.
        agent_images = {}
        for key in images_dict.keys():
            key_lower = key.lower()
            # Exclude wrist views
            if "wrist" not in key_lower:
                # Accept keys that are agent views (base, agent, etc.) or contain _image/_rgb
                if any(prefix in key_lower for prefix in ["base", "agent", "sideview"]) or \
                   "_image" in key_lower or "_rgb" in key_lower:
                    agent_images[key] = images_dict[key]
        
        if not agent_images:
            # Log available keys for debugging
            available_keys = list(images_dict.keys())
            logging.warning(f"No agent view images found for VGGT encoding. Available keys: {available_keys}")
            return None
        
        # Use the first agent view found (typically "agent_image" or "agentview_image" or "base_0_rgb")
        agent_key = list(agent_images.keys())[0]
        
        # Get agent view image
        img = agent_images[agent_key]
        logging.debug(f"Using agent view for VGGT: {agent_key}, shape: {img.shape}, ndim: {img.ndim}")
        # Determine number of frames to use
        # Auto-detect training mode if not specified
        if is_training is None:
            is_training = self.training if hasattr(self, 'training') else True
        
        if is_training:
            target_num_frames = self.num_frames
        else:
            target_num_frames = self.inference_num_frames
            
        # Use the image size defined in the encoder if available, otherwise default to 518 (VGGT standard)
        target_size = getattr(self.encoder, "img_size", 518)
        
        # Handle Temporal Dimension
        # Goal: Extract current frame + previous (target_num_frames - 1) frames
        # Strategy: Use past frames for temporal consistency (better for VGGT's global attention)
        # If dataset has [t-2, t-1, t, t+1], we want [t-2, t-1, t] (current + 2 past frames)
        if img.ndim == 5:  # [B, T, C, H, W] or [B, T, H, W, C]
            T = img.shape[1]
            
            # Check if channel last
            if img.shape[-1] == 3:  # [B, T, H, W, C]
                img = img.permute(0, 1, 4, 2, 3)  # [B, T, C, H, W]
            
            # Determine current frame index based on time dimension
            # If T == 2: [curr, next] -> use idx_curr=0 (current frame)
            # If T == 3: [prev, curr, next] -> use idx_curr=1 (current frame)
            # If T == 4: [t-2, t-1, t, t+1] -> use idx_curr=2 (current frame at index 2)
            # If T == 6: [t-4, t-3, t-2, t-1, t, t+1] -> use idx_curr=4 (current frame at index 4)
            if T == 2:
                idx_curr = 0  # First frame is current
            elif T == 3:
                idx_curr = 1  # Middle frame is current
            elif T == 4:
                idx_curr = 2  # Current frame is at index 2 in [t-2, t-1, t, t+1]
            elif T == 6:
                idx_curr = 4  # Current frame is at index 4 in [t-4, t-3, t-2, t-1, t, t+1]
            else:
                # For other T values, assume current is at T-2 (for [..., t-1, t, t+1] format)
                idx_curr = T - 2
            
            # Select frames: current frame + previous (target_num_frames - 1) frames
            # We want: [t-2, t-1, t] where t is current frame
            if T >= target_num_frames:
                # Start from (idx_curr - target_num_frames + 1) to idx_curr (inclusive)
                start_idx = max(0, idx_curr - target_num_frames + 1)
                end_idx = idx_curr + 1
                selected_frames = img[:, start_idx:end_idx]  # [B, selected_num, C, H, W]
                
                # If we don't have enough past frames, pad with the earliest available frame
                selected_num = selected_frames.shape[1]
                if selected_num < target_num_frames:
                    padding_needed = target_num_frames - selected_num
                    earliest_frame = selected_frames[:, :1]  # [B, 1, C, H, W]
                    padding = earliest_frame.repeat(1, padding_needed, 1, 1, 1)
                    img = torch.cat([padding, selected_frames], dim=1)  # [B, target_num_frames, C, H, W]
                else:
                    img = selected_frames
            else:
                # Fewer frames than requested
                if not is_training:
                    # Inference mode: use available frames (use what we have)
                    logging.debug(f"Inference: Using {T} available frames (requested {target_num_frames})")
                    img = img  # Use all available frames
                    target_num_frames = T  # Update target to match available
                else:
                    # Training mode: pad by repeating the first frame (past frames)
                    # We want current + past frames, so pad at the beginning
                    # This is normal when dataset only provides 2 frames (curr, next)
                    # We pad to get [curr, curr, curr] for 3-frame VGGT encoding
                    if T == 2:
                        logging.debug(f"Dataset has 2 frames (curr, next). Padding to get 3 frames for VGGT: [curr, curr, curr]")
                    else:
                        logging.warning(f"Only {T} frames available, requested {target_num_frames}. Padding with first frame at the beginning.")
                    first_frame = img[:, :1]  # [B, 1, C, H, W]
                    padding = first_frame.repeat(1, target_num_frames - T, 1, 1, 1)
                    img = torch.cat([padding, img], dim=1)  # [B, target_num_frames, C, H, W]
                
        elif img.ndim == 4:  # [B, C, H, W] or [B, H, W, C] - single frame
            # Check if channel last
            if img.shape[-1] == 3:  # [B, H, W, C]
                img = img.permute(0, 3, 1, 2)  # [B, C, H, W]
            
            # Single frame handling
            if not is_training:
                # Inference mode: use single frame directly (default: 1 frame)
                logging.debug(f"Inference: Using single frame (requested {target_num_frames} frames)")
                img = img.unsqueeze(1)  # [B, 1, C, H, W]
                target_num_frames = 1  # Update target to match available
            else:
                # Training mode: repeat frame to match num_frames
                # This warning indicates that the dataset might not be loading temporal frames correctly
                # or the observation was processed incorrectly before reaching prepare_inputs
                logging.warning(
                    f"Single frame input detected during training. "
                    f"Image shape: {img.shape}, Expected 5D [B, T, H, W, C] with T>=3. "
                    f"Repeating frame {target_num_frames} times for VGGT. "
                    f"Check data_loader delta_timestamps configuration."
                )
                img = img.unsqueeze(1).repeat(1, target_num_frames, 1, 1, 1)  # [B, target_num_frames, C, H, W]
        else:
            logging.error(f"Unexpected image shape: {img.shape}. Expected 4D [B, C, H, W] or 5D [B, T, C, H, W]")
            return None
        
        # Now img is [B, target_num_frames, C, H, W]
        # Process each frame: normalize and resize
        processed_frames = []
        for t in range(img.shape[1]):
            frame = img[:, t]  # [B, C, H, W]
            
            # Normalize
            if frame.dtype == torch.uint8:
                frame = frame.to(torch.float32) / 255.0
            
            # Resize
            if frame.shape[-2:] != (target_size, target_size):
                frame = F.interpolate(frame, size=(target_size, target_size), 
                                    mode='bilinear', align_corners=False)
            
            processed_frames.append(frame)
        
        # Stack frames: [B, target_num_frames, C, H, W]
        imgs_stacked = torch.stack(processed_frames, dim=1)
        
        # Fix NaN: Check and sanitize input images before passing to VGGT encoder
        # This is critical because NaN in input will propagate through the entire model
        if torch.isnan(imgs_stacked).any() or torch.isinf(imgs_stacked).any():
            import warnings
            warnings.warn("NaN/Inf detected in VGGT input images! Replacing with zeros. This may indicate data loading issues.")
            imgs_stacked = torch.where(
                torch.isnan(imgs_stacked) | torch.isinf(imgs_stacked),
                torch.zeros_like(imgs_stacked),
                imgs_stacked
            )
        
        # Clamp to valid range [0, 1] for normalized images
        imgs_stacked = torch.clamp(imgs_stacked, min=0.0, max=1.0)
        
        mode_str = "training" if is_training else "inference"
        logging.debug(f"VGGT input shape: {imgs_stacked.shape} (agent view, {target_num_frames} frames, {mode_str})")
             
        return imgs_stacked.to(device)

    def forward(self, gaussian_inputs, text_embedding=None, return_gaussian_params=False, return_raw_tokens=False, step=None, visualize=False):
        """
        Processes gaussian inputs and returns embeddings.
        Input: 
            gaussian_inputs: [B, S, 3, H, W]
            text_embedding: [B, D] Optional text embedding for LGPD.
            return_gaussian_params: If True, also return decoded Gaussian parameters from VGGT.
            return_raw_tokens: If True, also return raw tokens before pooling (for VAE supervision).
            step: Current training step (for visualization)
            visualize: If True and step % 100 == 0, save visualization
        Returns:
            gaussian_embs: [B, N, D] - Token embeddings for World Model
            g_mask: [B, N] - Mask for tokens
            gaussian_params (optional): Dict with depth, rot, scale, opacity, sh if return_gaussian_params=True
            raw_tokens (optional): [B, S, 1369, D] - Raw patch tokens before pooling if return_raw_tokens=True
        """
        if not self.use_gaussian or gaussian_inputs is None:
            return (None, None) if not return_gaussian_params else (None, None, None)

        # Fix NaN: Check input before encoding
        if torch.isnan(gaussian_inputs).any() or torch.isinf(gaussian_inputs).any():
            import warnings
            warnings.warn("NaN/Inf detected in gaussian_inputs before VGGT encoding! This will cause NaN outputs.")
            # Replace NaN/Inf with zeros (black images) as fallback
            gaussian_inputs = torch.where(
                torch.isnan(gaussian_inputs) | torch.isinf(gaussian_inputs),
                torch.zeros_like(gaussian_inputs),
                gaussian_inputs
            )
        
        # Ensure encoder is in eval mode and disable gradient computation
        self.encoder.eval()
        with torch.no_grad():
            # Clear cache before encoding to save memory
            torch.cuda.empty_cache()
            
            # Use inference mode for better memory efficiency
            with torch.inference_mode():
                outputs = self.encoder(gaussian_inputs)
                
                # Extract all outputs from VGGT
                # outputs: depth_maps, rot_maps, scale_maps, opacity_maps, sh_maps, aggregated_tokens_list, patch_start_idx
                depth_maps = outputs[0]  # [B, S, H, W, 1]
                rot_maps = outputs[1]     # [B, S, H, W, 4]
                scale_maps = outputs[2]  # [B, S, H, W, 3]
                opacity_maps = outputs[3] # [B, S, H, W, 1]
                sh_maps = outputs[4]      # [B, S, H, W, K, 3] where K = (sh_degree+1)^2
                aggregated_tokens_list = outputs[-2]
                patch_start_idx = outputs[-1]
            
            # Fix NaN: Check VGGT encoder outputs immediately after encoding
            # This helps identify if NaN comes from encoder itself
            if torch.isnan(depth_maps).any() or torch.isnan(rot_maps).any() or torch.isnan(scale_maps).any():
                import warnings
                warnings.warn("NaN detected in VGGT encoder outputs! This may be due to: 1) Input NaN, 2) Encoder weights NaN, 3) Numerical instability in attention/normalization.")
                # Log which outputs have NaN for debugging
                nan_info = {
                    "depth_maps": torch.isnan(depth_maps).any().item(),
                    "rot_maps": torch.isnan(rot_maps).any().item(),
                    "scale_maps": torch.isnan(scale_maps).any().item(),
                    "opacity_maps": torch.isnan(opacity_maps).any().item(),
                    "sh_maps": torch.isnan(sh_maps).any().item(),
                }
                logging.warning(f"NaN in VGGT outputs: {nan_info}")
            
            # Explicitly delete outputs to free memory
            del outputs
            
            # Clear cache after encoding to free memory
            torch.cuda.empty_cache()

        # Check if we got valid features
        if aggregated_tokens_list is None or (isinstance(aggregated_tokens_list, (list, tuple)) and len(aggregated_tokens_list) == 0):
            logging.warning("No features extracted from Gaussian Encoder.")
            return (None, None) if not return_gaussian_params else (None, None, None)
        
        # Store decoded Gaussian parameters if requested
        gaussian_params_dict = None
        if return_gaussian_params:
            gaussian_params_dict = {
                "depth_maps": depth_maps,      # [B, S, H, W, 1]
                "rot_maps": rot_maps,          # [B, S, H, W, 4]
                "scale_maps": scale_maps,      # [B, S, H, W, 3]
                "opacity_maps": opacity_maps,  # [B, S, H, W, 1]
                "sh_maps": sh_maps             # [B, S, H, W, K, 3]
            }
        
        # Visualize if requested and step is a multiple of 100
        # Note: LGPD gate will be available after LGPD is applied below
        visualize_gate = False
        if visualize and step is not None and step % 100 == 0:
            visualize_gate = True
        
        # Process tokens with temporal awareness
        # Strategy: Enhanced temporal encoding with 3D Conv + Causal Attention

        # === Priority 3: Multi-scale Feature Extraction ===
        if hasattr(self, 'use_multi_scale') and self.use_multi_scale:
            # Extract features from layers [11, 17, 23]
            multi_scale_features = []
            for idx, layer_idx in enumerate(self.layer_indices):
                layer_tokens = aggregated_tokens_list[layer_idx]  # [B, S, N_patches, 2048]

                # Remove camera + register tokens
                if hasattr(self.encoder, 'aggregator') and hasattr(self.encoder.aggregator, 'patch_start_idx'):
                    patch_start_idx_val = self.encoder.aggregator.patch_start_idx
                    if layer_tokens.shape[2] > 1369:
                        layer_tokens = layer_tokens[:, :, patch_start_idx_val:]

                # Project to 512 dims
                layer_tokens = self.layer_projs[idx](layer_tokens)  # [B, S, 1369, 512]
                multi_scale_features.append(layer_tokens)

            # FPN-style bottom-up fusion: [11] <- [17] <- [23]
            fused_features = multi_scale_features[-1]  # Start with layer 23
            for i in range(len(multi_scale_features) - 2, -1, -1):
                # Fuse current layer with previous
                fused_features = fused_features + multi_scale_features[i]
                if i > 0:  # Apply fusion block except for the last iteration
                    B_f, S_f, N_f, D_f = fused_features.shape
                    fused_features = self.fusion_blocks[i](
                        fused_features.reshape(B_f * S_f * N_f, D_f)
                    ).reshape(B_f, S_f, N_f, D_f)

            # Project fused features back to 2048 for temporal_conv
            B_f, S_f, N_f, D_f = fused_features.shape
            fused_features = self.fused_to_2048(
                fused_features.reshape(B_f * S_f * N_f, D_f)
            ).reshape(B_f, S_f, N_f, 2048)

            raw_tokens = fused_features  # [B, S, 1369, 2048]
            B, S, N_patches, D = raw_tokens.shape
        else:
            # Original single-layer extraction
            if isinstance(aggregated_tokens_list, (list, tuple)):
                raw_tokens = aggregated_tokens_list[-1]
            else:
                raw_tokens = aggregated_tokens_list

            B, S, N_patches, D = raw_tokens.shape

            # Remove camera + register tokens (if not already removed)
            if hasattr(self.encoder, 'aggregator') and hasattr(self.encoder.aggregator, 'patch_start_idx'):
                patch_start_idx_val = self.encoder.aggregator.patch_start_idx
                if N_patches > 1369:  # Has camera + register tokens
                    raw_tokens = raw_tokens[:, :, patch_start_idx_val:]
                    B, S, N_patches, D = raw_tokens.shape

        # Initialize variables to ensure they're defined in all code paths
        gaussian_embs = None
        tokens_flat = None  # Initialize to avoid UnboundLocalError

        # === Priority 1: Enhanced Temporal Encoding ===
        if hasattr(self, 'use_enhanced_temporal') and self.use_enhanced_temporal:
            # 1. Reshape to 3D: [B, S, 1369, D] -> [B, D, S, 37, 37]
            tokens_3d = raw_tokens.permute(0, 3, 1, 2).reshape(B, D, S, 37, 37)

            # 2. 3D Conv (temporal-spatial encoding)
            tokens_conv = self.temporal_conv(tokens_3d)  # [B, 512, S, 9, 9]

            # 3. Spatial pooling to 16×16
            B_conv, C_conv, S_conv, H_conv, W_conv = tokens_conv.shape
            tokens_conv = tokens_conv.permute(0, 2, 1, 3, 4).reshape(B*S, C_conv, H_conv, W_conv)
            tokens_pooled = self.spatial_pool(tokens_conv)  # [B*S, 512, 16, 16]
            tokens_pooled = tokens_pooled.reshape(B, S, C_conv, 16, 16)

            # 4. Flatten spatial dimensions: [B, S, 512, 16, 16] -> [B, S, 256, 512]
            tokens_flat = tokens_pooled.permute(0, 1, 3, 4, 2).reshape(B, S, 256, C_conv)

            # 5. Add frame positional encoding
            frame_tokens_list = []
            for frame_idx in range(S):
                frame_tokens = tokens_flat[:, frame_idx]  # [B, 256, 512]

                # Sinusoidal + learnable positional encoding
                frame_pos = self.frame_pos_encoding[frame_idx](frame_idx)  # [512]
                frame_pos = frame_pos + self.frame_embeddings[frame_idx]  # [512]
                frame_pos = frame_pos.unsqueeze(0).unsqueeze(0)  # [1, 1, 512]

                frame_tokens = frame_tokens + frame_pos  # [B, 256, 512]
                frame_tokens_list.append(frame_tokens)

            # 6. Concatenate frames: [B, S, 256, 512] -> [B, S*256, 512]
            temporal_tokens = torch.cat(frame_tokens_list, dim=1)  # [B, 768, 512] for S=3

            # 7. Causal temporal attention
            temporal_tokens = self.temporal_attn(temporal_tokens, tokens_per_frame=256)

            # 8. Project to action_expert_width
            gaussian_embs = self.proj(temporal_tokens)  # [B, 768, action_expert_width]

            if step is not None and step % 100 == 0:
                logging.info(
                    f"[Enhanced Temporal] Step {step}: "
                    f"tokens shape={gaussian_embs.shape}, "
                    f"frames={S}, tokens_per_frame={self.tokens_per_frame}"
                )

        # === Fallback: Original Per-frame Processing ===
        elif hasattr(self, 'use_frame_pos_encoding') and self.use_frame_pos_encoding:
            # Process each frame separately to maintain temporal structure
            frame_tokens_list = []
            for frame_idx in range(S):
                # Extract tokens for this frame: [B, N_patches, D]
                frame_tokens = raw_tokens[:, frame_idx, :, :]

                # Reshape for pooling: [B, N_patches, D] -> [B, D, sqrt(N), sqrt(N)]
                patch_h = int(np.sqrt(N_patches))
                patch_w = N_patches // patch_h if patch_h > 0 else 1

                # Reshape to 2D: [B, N_patches, D] -> [B, D, patch_h, patch_w]
                if patch_h * patch_w == N_patches:
                    tokens_2d = frame_tokens.permute(0, 2, 1).view(B, D, patch_h, patch_w)
                else:
                    # Fallback: reshape to approximate grid
                    spatial_size = int(np.sqrt(N_patches))
                    tokens_reshaped = frame_tokens.permute(0, 2, 1)  # [B, D, N_patches]
                    if spatial_size * spatial_size != N_patches:
                        spatial_size = int(np.ceil(np.sqrt(N_patches)))
                        padding = spatial_size * spatial_size - N_patches
                        tokens_reshaped = F.pad(tokens_reshaped, (0, padding), mode='constant', value=0)
                    tokens_2d = tokens_reshaped.view(B, D, spatial_size, spatial_size)

                # Pool to reduce tokens: [B, D, H, W] -> [B, D, 10, 10]
                tokens_pooled = self.pool(tokens_2d)  # [B, D, 10, 10]

                # Flatten: [B, D, 10, 10] -> [B, 100, D]
                tokens_final = tokens_pooled.view(B, -1, D)  # [B, 100, D]

                # Project to action_expert_width
                frame_embs = self.proj(tokens_final)  # [B, 100, action_expert_width]

                # Add frame positional encoding
                if hasattr(self, 'frame_embeddings'):
                    # frame_embeddings: [S, action_expert_width]
                    # Expand to [B, 100, action_expert_width] and add
                    frame_emb = self.frame_embeddings[frame_idx]  # [action_expert_width]
                    frame_emb = frame_emb.unsqueeze(0).unsqueeze(0).expand(B, 100, -1)  # [B, 100, action_expert_width]
                    frame_embs = frame_embs + frame_emb

                frame_tokens_list.append(frame_embs)

            # Concatenate frames: [B, 100, D] * S -> [B, S*100, D]
            # This preserves temporal order: tokens from frame 0, then frame 1, then frame 2
            gaussian_embs = torch.cat(frame_tokens_list, dim=1)  # [B, S*100, action_expert_width]

        else:
            # Option 2: Original approach (flatten all frames together)
            # This loses temporal structure but maintains backward compatibility
            try:
                tokens_flat = raw_tokens.view(B, S * N_patches, D)  # [B, S*N_patches, D]
            except Exception as e:
                logging.error(f"Failed to reshape raw_tokens: {e}, raw_tokens.shape={raw_tokens.shape}, expected shape=[{B}, {S}, {N_patches}, {D}]")
                raise
            
            # Reshape for pooling: [B, S*N_patches, D] -> [B, D, sqrt(N), sqrt(N)]
            spatial_size = int(np.sqrt(S * N_patches))
            if spatial_size * spatial_size != S * N_patches:
                spatial_size = int(np.ceil(np.sqrt(S * N_patches)))
                padding = spatial_size * spatial_size - S * N_patches
                tokens_reshaped = tokens_flat.permute(0, 2, 1)  # [B, D, S*N_patches]
                tokens_reshaped = F.pad(tokens_reshaped, (0, padding), mode='constant', value=0)
            else:
                tokens_reshaped = tokens_flat.permute(0, 2, 1)  # [B, D, S*N_patches]
            
            tokens_2d = tokens_reshaped.view(B, D, spatial_size, spatial_size)
            
            # Pool to reduce tokens: [B, D, spatial_size, spatial_size] -> [B, D, 10, 10]
            tokens_pooled = self.pool(tokens_2d)  # [B, D, 10, 10]
            
            # Flatten back: [B, D, 10, 10] -> [B, 100, D]
            tokens_final = tokens_pooled.view(B, -1, D)  # [B, 100, D]
            
            # Project to action_expert_width
            gaussian_embs = self.proj(tokens_final)  # [B, 100, action_expert_width]
        
        # Create mask (all tokens are valid)
        # If using frame pos encoding, we have S*100 tokens, otherwise 100 tokens
        g_mask = torch.ones(B, gaussian_embs.shape[1], dtype=torch.bool, device=gaussian_embs.device)
        
        # Apply LGPD if enabled and text embedding is provided
        lgpd_gate = None
        if self.use_lgpd and self.lgpd is not None and text_embedding is not None:
            # text_embedding: [B, D] - LGPD module expects [B, D] and will handle unsqueeze internally
            # Return gate for visualization if needed
            if visualize_gate:
                gaussian_embs, lgpd_gate = self.lgpd(gaussian_embs, text_embedding, return_gate=True)
            else:
                gaussian_embs = self.lgpd(gaussian_embs, text_embedding)
        
        # Visualize VGGT frames and LGPD gate together
        # Only visualize if LGPD is enabled and gate is available, or if we just want VGGT visualization
        # Only visualize on rank 0 to avoid NCCL timeout in distributed training
        if visualize_gate:
            try:
                import torch.distributed as dist
                is_main_process = not dist.is_initialized() or dist.get_rank() == 0
                if is_main_process:
                    # If LGPD is disabled, lgpd_gate will be None, but visualization function can handle it
                    visualize_vggt_and_lgpd(gaussian_inputs, gaussian_params_dict, lgpd_gate, step)
            except Exception as e:
                logging.warning(f"Failed to visualize VGGT and LGPD at step {step}: {e}")
        
        # Prepare return values
        if return_gaussian_params and return_raw_tokens:
            # Project raw_tokens to action_expert_width for VAE supervision
            # raw_tokens: [B, S, 1369, D] where D is VGGT embed_dim (2048)
            # VAE expects tokens in action_expert_width dimension
            raw_tokens_proj = None
            if hasattr(self, 'proj'):
                # Project each frame's tokens: [B, S, 1369, D] -> [B, S, 1369, action_expert_width]
                B, S, N, D = raw_tokens.shape
                raw_tokens_flat = raw_tokens.view(B * S, N, D)
                raw_tokens_proj_flat = self.proj(raw_tokens_flat)  # [B*S, 1369, action_expert_width]
                raw_tokens_proj = raw_tokens_proj_flat.view(B, S, N, -1)  # [B, S, 1369, action_expert_width]
            return gaussian_embs, g_mask, gaussian_params_dict, raw_tokens_proj
        elif return_gaussian_params:
            return gaussian_embs, g_mask, gaussian_params_dict
        elif return_raw_tokens:
            # Project raw_tokens to action_expert_width for VAE supervision
            raw_tokens_proj = None
            if hasattr(self, 'proj'):
                B, S, N, D = raw_tokens.shape
                raw_tokens_flat = raw_tokens.view(B * S, N, D)
                raw_tokens_proj_flat = self.proj(raw_tokens_flat)
                raw_tokens_proj = raw_tokens_proj_flat.view(B, S, N, -1)
            return gaussian_embs, g_mask, raw_tokens_proj
        else:
            return gaussian_embs, g_mask
