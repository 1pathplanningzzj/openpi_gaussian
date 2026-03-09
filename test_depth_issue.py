
import torch
import jax.numpy as jnp
from openpi.models.model import Observation
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch as Pi0
import numpy as np

def test_depth_slicing():
    # Create fake data
    B, T, H, W, C = 4, 4, 224, 224, 3
    DH, DW = 256, 256
    
    images = {
        "base_0_rgb": torch.randn(B, T, H, W, C)
    }
    image_masks = {
        "base_0_rgb": torch.ones(B, T, dtype=torch.bool)
    }
    state = torch.randn(B, T, 32)
    depth = torch.randn(B, T, 1, DH, DW)
    
    obs = Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        depth=depth
    )
    
    print(f"Original depth shape: {obs.depth.shape}")
    
    # Instantiate model (mocking parts if needed, but Pi0 is complex)
    # We just want to run _preprocess_observation.
    # We can perform the slicing logic manually to verify it works as expected.
    
    img_val = list(obs.images.values())[0]
    time_dim = img_val.shape[1]
    print(f"Time dim: {time_dim}")
    
    if time_dim == 4:
        idx_curr, idx_fut = 2, 3
    
    print(f"Indices: curr={idx_curr}, fut={idx_fut}")
    
    if obs.depth.ndim == 5:
        if obs.depth.shape[1] == time_dim:
            fut_depth = obs.depth[:, idx_fut]
            print(f"Sliced fut_depth shape: {fut_depth.shape}")
            
            if fut_depth.ndim == 4:
                print("Slicing works as expected.")
            else:
                print("Slicing FAILED to reduce dimension.")
    
    # Now try replace (which invokes jaxtyping)
    fut_imgs = {k: v[:, idx_fut] for k, v in obs.images.items()}
    fut_masks = {k: v[:, idx_fut] for k, v in obs.image_masks.items()}
    fut_state = obs.state[:, idx_fut]
    
    try:
        new_obs = obs.replace(
            images=fut_imgs,
            image_masks=fut_masks,
            state=fut_state,
            depth=fut_depth
        )
        print("Observation.replace successful.")
    except Exception as e:
        print(f"Observation.replace FAILED: {e}")

if __name__ == "__main__":
    test_depth_slicing()
