# zijian

import sys
import os
import torch
import torch.nn as nn
from pathlib import Path
import time

# Verify CUDA
if not torch.cuda.is_available():
    print("CUDA not available, cannot check GPU memory.")
    sys.exit(1)

device = torch.device("cuda:0")

# Setup paths
project_root = Path(__file__).resolve().parent.parent
ad_ffgs_root = project_root / "third_party" / "AD-FFgsStudio"
sys.path.append(str(ad_ffgs_root))

print(f"Added {ad_ffgs_root} to sys.path")

try:
    from models.df3dgs_model_module import DF3DGS_LITModelModule
    from models.df3dgs_model import DF3DGSModel
    DF3DGSModel.load_official_weights = lambda self: print("Skipped loading official weights.")
except ImportError as e:
    print(f"Failed to import AD-FFgsStudio modules: {e}")
    print("Please make sure you have the correct python environment.")
    sys.exit(1)

# Minimal config based on df3dgs_inference.yaml
model_cfg = {
  "height": 352,
  "width": 640,
  "batch_size": 1,
  "num_cams": 6,
  "embed_dim": 1024,
  "learning_rate": 0.00002,
  "weight_decay": 0.01,
  "lr_restart_epoch": 5,
  "lr_restart_mult": 2,
  "lr_min_factor": 0.01,
  "frame_ids": [0, -1, 1], # Typically uses prev, cur, next frames for training, but inference might differ
  "depth_conf_thr": 0.99,
  "min_depth": 1.5,
  "max_depth": 80,
  "focal_length_scale": 300,
  "init_scale_thr": 0.02,
  "sh_degree": 4,
  "save_image_duration": 100,
  "lambda_project": 1.0,
  "lambda_edge": 0.1,
  "lambda_depth": 0.001,
  "lambda_gaussian": 2,
  "lambda_scale": 0.01,
  "lambda_opacity": 0.01,
  "depth_net_cfg": {
    "height": 352,
    "width": 640,
    "scales": [0],
    "num_cams": 6,
    "novel_view_mode": 'MF',
    "num_layers": 18,     # ResNet layers
    "weights_init": True,
    "fusion_level": 2,
    "fusion_feat_in_dim": 256,
    "use_skips": False,
    "voxel_unit_size": [1.0, 1.0, 1.5],
    "voxel_size": [100, 100, 20],
    "voxel_str_p": [-50.0, -50.0, -15.0],
    "voxel_pre_dim": [64],
    "proj_d_bins": 50,
    "proj_d_str": 2,
    "proj_d_end": 50,
  }
}

def log_memory(phase):
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    print(f"[{phase}] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

print("Initializing Model...")
log_memory("Start")

try:
    # Initialize model
    model_module = DF3DGS_LITModelModule(
        cfg=model_cfg,
        save_dir='./temp_log',
        logger=None
    )
    
    # Move to GPU
    model = model_module.model.to(device)
    model.eval()
    
    # Freeze parameters (since we want to verify frozen encoder usage)
    for param in model.parameters():
        param.requires_grad = False
        
    log_memory("Model Loaded")
    
    # Create Dummy Input
    # Batch size 1, 6 cameras, 3 channels, 352 height, 640 width
    B = 1
    N = 6
    C = 3
    H = 352
    W = 640
    
    dummy_imgs = torch.randn(B, N, C, H, W, device=device)
    dummy_depths = torch.randn(B, N, 1, H, W, device=device) # Suppose we have depth
    dummy_masks = torch.ones(B, N, 1, H, W, device=device)
    dummy_K = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(B, N, 1, 1) # simple identity intrinsics
    dummy_extr = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(B, N, 1, 1) # simple identity extrinsics

    # Create dummy intrinsics/extrinsics if needed. 
    # Based on df3dgs_model.py, it likely takes a dictionary or similar structure.
    # Let's check how forward is called. 
    # Actually, usually these models take a batch dictionary.
    
    inputs = {
        ('color_aug', 0): dummy_imgs,
        ('color_aug', -1): dummy_imgs,
        ('color_aug', 1): dummy_imgs,
        'mask': dummy_masks,
        'K': dummy_K,
        'c2e_extr': dummy_extr,
        'e2c_extr': dummy_extr,
    }

    # Let's look at `model.forward` signature in source if this fails.
    # But first, let's just see memory after loading. That's the biggest part for weights.
    # Inference activation memory depends on resolution.
    
    print("Running dummy inference (forward pass)...")
    
    with torch.no_grad():
        # Based on typical depth estimation models
        # We might need to call specific sub-modules if strictly checking encoder memory
        # But let's try to run the encoder part.
        
        # model.depth_net is usually the encoder
        features = model.depth_net(inputs) 
        # Usually depth_net expects [B*N, C, H, W]
        
        input_reshaped = dummy_imgs.view(-1, C, H, W)
        print(f"Input shape: {input_reshaped.shape}")
        
        features = model.depth_net.encoder(input_reshaped)
        print("Encoder run successful.")
        log_memory("After Encoder Forward")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()

print("Done.")
