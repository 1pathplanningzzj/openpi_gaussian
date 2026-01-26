# zijian
# date 2026.01.24
# v1 to be debug and much test to do 
# Description: Bi-Directional World Model with Environment and Interaction Flows, Contact Gating, and Inverse Dynamics.
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# Todo zijian 2026.0123 ：to fix linear attach anything ？？
class EnvironmentFlowNet(nn.Module):
    """
    Model the natural evolution of the environment (physics, gravity etc.) independent of agent interaction.
    Input: latent tokens z_t [B, N, D] + time t [B, 1]
    Output: environment flow vector v_env [B, N, D]
    Meaning: B batch size N number of tokens D token dimension t timestamp 
    """
    def __init__(self, token_dim, hidden_dim=512):
        super().__init__()
        # Input dim: token_dim + 1 (time)
        self.net = nn.Sequential(
            nn.Linear(token_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, token_dim)
        )

    def forward(self, z_t, t):
        # z_t: [B, N, D]
        # t: [B] or [B, 1]
        
        B, N, D = z_t.shape
        
        # Expand time to [B, N, 1]
        if t.ndim == 1:
            t_expanded = t.view(B, 1, 1).expand(B, N, 1)
        elif t.ndim == 2:
            t_expanded = t.view(B, 1, 1).expand(B, N, 1)
        else:
             t_expanded = t # assume correct shape
             
        # Concatenate: [B, N, D+1]
        inp = torch.cat([z_t, t_expanded], dim=-1)
        
        v_env = self.net(inp)
        return v_env


class InteractionFlowNet(nn.Module):
    """
    Model the changes caused by agent interaction.
    Input: z_t [B, N, D] + encoded_action [B, D_act] + t [B, 1]
    Output: interaction flow vector v_int [B, N, D]
    """
    def __init__(self, token_dim, action_dim, hidden_dim=512):
        super().__init__()
        
        # Action encoder to map raw action to a useful embedding space if needed, 
        # or we concatenate directly. The prompt says "First MLP encode action".
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Flow net
        # Input: token_dim + hidden_dim (encoded action) + 1 (time)
        self.net = nn.Sequential(
            nn.Linear(token_dim + hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, token_dim)
        )

    def forward(self, z_t, action, t):
        B, N, D = z_t.shape
        
        # Encode action: [B, A] -> [B, H]
        action_emb = self.action_encoder(action)
        
        # Expand action to per-token: [B, N, H]
        action_emb_expanded = action_emb.unsqueeze(1).expand(B, N, -1)
        
        # Expand time: [B, N, 1]
        if t.ndim == 1:
            t_expanded = t.view(B, 1, 1).expand(B, N, 1)
        else:
            t_expanded = t.view(B, 1, 1).expand(B, N, 1)

        # Concatenate: [B, N, D + H + 1]
        inp = torch.cat([z_t, action_emb_expanded, t_expanded], dim=-1)
        
        v_int = self.net(inp)
        return v_int


class ContactGatingNet(nn.Module):
    """
    Predict contact probability mask to modulate interaction flow.
    Input: z_t [B, N, D] + action [B, A]
    Output: contact mask [B, N, 1] in range [0, 1]
    """
    def __init__(self, token_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(token_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, z_t, action):
        B, N, D = z_t.shape
        
        # Expand action: [B, N, A]
        action_expanded = action.unsqueeze(1).expand(B, N, -1)
        
        # Concatenate: [B, N, D+A]
        inp = torch.cat([z_t, action_expanded], dim=-1)
        
        # Output: [B, N, 1]
        gate = self.net(inp)
        return gate


class ForwardInteractionPredictor(nn.Module):
    """
    Joint Forward Model combining Environment and Interaction flows with Contact Gating.
    z_{t+1} = z_t + (v_env + I_contact * v_int) * dt
    """
    def __init__(self, token_dim, action_dim, hidden_dim=512):
        super().__init__()
        
        self.env_flow = EnvironmentFlowNet(token_dim, hidden_dim)
        self.int_flow = InteractionFlowNet(token_dim, action_dim, hidden_dim)
        self.contact_gate = ContactGatingNet(token_dim, action_dim, hidden_dim // 2)

    def forward(self, z_t, action, t, dt=1.0):
        """
        Args:
            z_t: Current latent tokens [B, N, D]
            action: Action vector [B, A]
            t: Current time scalar/vector [B] or [B, 1]
            dt: Time step size (default 1.0 for discrete step)
        """
        # 1. Environment Flow
        v_env = self.env_flow(z_t, t)
        
        # 2. Interaction Flow
        v_int = self.int_flow(z_t, action, t)
        
        # 3. Contact Gating
        i_contact = self.contact_gate(z_t, action)
        
        # 4. Integrate
        # delta = (v_env + i_contact * v_int) * dt
        delta = (v_env + i_contact * v_int) * dt
        
        z_next_pred = z_t + delta
        
        return z_next_pred, {
            "v_env": v_env,
            "v_int": v_int,
            "i_contact": i_contact,
            "delta": delta
        }


class InverseModel(nn.Module):
    """
    Inverse Dynamics Model to recover action from state transition.
    Input: z_t, z_{t+1}
    Output: predicted action
    """
    def __init__(self, token_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # Input dim: Pooled z_t (D) + Pooled z_t+1 (D) = 2*D
        self.net = nn.Sequential(
            nn.Linear(token_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.token_dim = token_dim

    def forward(self, z_t, z_next):
        # z: [B, N, D]
        
        # Pooling (Mean) -> [B, D]
        z_t_pooled = z_t.mean(dim=1)
        z_next_pooled = z_next.mean(dim=1)
        
        # Concatenate: [B, 2D]
        inp = torch.cat([z_t_pooled, z_next_pooled], dim=-1)
        
        action_pred = self.net(inp)
        return action_pred


class Privileged4DGSDecoder(nn.Module):
    """
    Decodes latent tokens into 3D/4D Gaussian parameters for supervision/rendering.
    """
    def __init__(self, token_dim):
        super().__init__()
        
        # Output dims per gaussian:
        # mu (3) + scale (3 -> expand to cov later) + quat/rot (4) + opacity (1) + sh (16 for deg 2 or similar)
        # Using simplified set from prompt:
        # mu (3), Sigma (6 -> 3x3 low cholesky or similar), SH (16), alpha (1)
        # Sum = 3 + 6 + 16 + 1 = 26
        # Update 0126: For GaussianRenderer compatibility (SH Degree 3)
        # We need (3+1)^2 * 3 = 16 * 3 = 48 SH coefficients
        self.sh_dim = 48
        self.out_dim = 3 + 6 + self.sh_dim + 1 # 58
        
        self.decoder = nn.Sequential(
            nn.Linear(token_dim, 256),
            nn.SiLU(),
            nn.Linear(256, self.out_dim)
        )

        # Initialize the last layer to produce better initial Gaussians
        # This helps avoid black images at step 0
        with torch.no_grad():
            # Initialize opacity bias to positive value (sigmoid(2.0) ≈ 0.88)
            self.decoder[-1].bias[-1] = 2.0

            # Initialize first SH coefficient (DC component) to positive value
            # This gives Gaussians a base gray color
            # SH coefficients start at index 9, first 3 are RGB DC components
            self.decoder[-1].bias[9:12] = 0.5  # Gray color

    def forward(self, z):
        # z: [B, N, D]
        B, N, D = z.shape
        raw = self.decoder(z)
        
        # Split
        mu = raw[..., 0:3]
        sigma_params = raw[..., 3:9] # 6 params for covariance (e.g. upper triangle)
        sh = raw[..., 9 : 9 + self.sh_dim]
        opacity = torch.sigmoid(raw[..., -1:])
        
        return {
            "xyz": mu,
            "sigma": sigma_params,
            "sh": sh,
            "opacity": opacity
        }


class BiDirectionalWorldModel(nn.Module):
    """
    Main Phase 2 World Model container.
    """
    def __init__(self, token_dim, action_dim):
        super().__init__()
        
        self.forward_model = ForwardInteractionPredictor(token_dim, action_dim)
        self.inverse_model = InverseModel(token_dim, action_dim)
        self.decoder = Privileged4DGSDecoder(token_dim) # Optional usage
        
    def forward(self, z_t, action, t=0.0):
        """
        Run forward prediction step.
        """
        # Ensure t is a tensor
        if isinstance(t, (int, float)):
            t = torch.tensor([t], device=z_t.device, dtype=z_t.dtype).repeat(z_t.shape[0])
            
        z_next_pred, details = self.forward_model(z_t, action, t)
        return z_next_pred, details

    def compute_cycle_consistency_score(self, z_t, z_next_pred, action_gt):
        """
        Run inverse consistency check.
        Recover action from z_t -> z_next_pred and compare with action_gt.
        """
        action_recovered = self.inverse_model(z_t, z_next_pred)
        
        # Log distance/MSE
        consistency_loss = F.mse_loss(action_recovered, action_gt)
        return consistency_loss, action_recovered

    def compute_full_loss(self, z_t, action_t, z_t1_gt, 
                          lambda_fwd=1.0, lambda_inv=0.1, lambda_render=0.0):
        """
        Compute total training loss for this step.
        """
        # 1. Forward Prediction
        # Assuming t=0 for single step training usually, or pass t if maintaining state
        z_t1_pred, _ = self.forward(z_t, action_t)
        
        loss_fwd = F.mse_loss(z_t1_pred, z_t1_gt)
        
        # 2. Inverse Consistency
        # Can be computed on (z_t, z_t1_gt) or (z_t, z_t1_pred). 
        # Typically Inverse Model should predict action from *ground truth* transitions to learn dynamics,
        # OR from predicted transitions to enforce consistency. 
        # The prompt implies "Consistency" ||a_recovered - a_gt||. 
        # Typically trained with GT transitions: Inverse(z_t, z_t1_gt) -> action
        
        action_rec_from_gt = self.inverse_model(z_t, z_t1_gt)
        loss_inv = F.mse_loss(action_rec_from_gt, action_t)
        
        # 3. Optional: Cycle consistency using predicted z (Self-supervised reinforcement)
        # action_rec_from_pred = self.inverse_model(z_t, z_t1_pred)
        # loss_cycle = F.mse_loss(action_rec_from_pred, action_t)
        
        loss_total = lambda_fwd * loss_fwd + lambda_inv * loss_inv
        
        return {
            "loss_total": loss_total,
            "loss_fwd": loss_fwd,
            "loss_inv": loss_inv,
            "z_t1_pred": z_t1_pred,
            "action_recovered": action_rec_from_gt
        }

def visualize_world_model_prediction(model: BiDirectionalWorldModel, z_t, action, z_t1_gt, batch_idx=0, step=10):
    """
    Visualization helper to compare Predicted vs Ground Truth Next State in 3D space.
    Saves the visualization to './visualizations' directory.
    
    since the latent tokens z represent 3D Gaussians, we can decode them to XYZ 
    coordinates and visualize the point cloud dynamics.
    
    Args:
        model: Trained/Loaded BiDirectionalWorldModel
        z_t: Current latent state [B, N, D]
        action: Action taken [B, A]
        z_t1_gt: Ground Truth next latent state [B, N, D]
        batch_idx: Which sample in the batch to visualize
        step: Current training step (used for filename).
    """
    try:
        import matplotlib.pyplot as plt
        import os
        # Enable 3D plotting
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("Matplotlib not found. Please install it to visualize: pip install matplotlib")
        return

    model.eval()
    with torch.no_grad():
        # 1. Run Forward Prediction
        # z_t1_pred: Predicted latent state at t+1
        z_t1_pred, details = model(z_t, action)
        
        # 2. Decode Latents to 3D Gaussian Attributes (XYZ, Opacity, etc.)
        # We assume the decoder is trained to map z -> Gaussian Params
        pred_decode = model.decoder(z_t1_pred)
        gt_decode = model.decoder(z_t1_gt)
        curr_decode = model.decoder(z_t) # Also visualize t for reference
        
        # Extract XYZ coordinates for the specific batch index
        # [B, N, 3] -> [N, 3]
        xyz_pred = pred_decode['xyz'][batch_idx].cpu().numpy()
        xyz_gt = gt_decode['xyz'][batch_idx].cpu().numpy()
        xyz_curr = curr_decode['xyz'][batch_idx].cpu().numpy()
        
        # Extract Opacity for filtering (optional, if opacity is learned)
        # [B, N, 1] -> [N]
        op_pred = pred_decode['opacity'][batch_idx].squeeze(-1).cpu().numpy()
        op_gt = gt_decode['opacity'][batch_idx].squeeze(-1).cpu().numpy()
        
        # Extract predicted flow components if available
        v_env = details['v_env'][batch_idx].norm(dim=-1).cpu().numpy() # Magnitude of env flow
        v_int = details['v_int'][batch_idx].norm(dim=-1).cpu().numpy() # Magnitude of interaction flow

    # 3. Create Visualization
    fig = plt.figure(figsize=(18, 6))
    
    # Filter points with low opacity to reduce clutter (if opacity is meaningful)
    # If opacity is not well trained yet, you might want to remove this mask.
    mask_pred = op_pred > 0.05
    mask_gt = op_gt > 0.05
    
    # --- Plot 1: Movement Overview (Current vs Pred) ---
    ax1 = fig.add_subplot(131, projection='3d')
    # Plot Current State (Blue)
    ax1.scatter(xyz_curr[:, 0], xyz_curr[:, 1], xyz_curr[:, 2], c='b', s=1, alpha=0.1, label='t (Current)')
    # Plot Prediction (Red)
    ax1.scatter(xyz_pred[mask_pred, 0], xyz_pred[mask_pred, 1], xyz_pred[mask_pred, 2], c='r', s=2, alpha=0.5, label='t+1 (Pred)')
    ax1.set_title(f"Step {step}: Dynamics (Blue->Red)")
    ax1.legend()

    # --- Plot 2: Prediction Accuracy (Pred vs GT) ---
    ax2 = fig.add_subplot(132, projection='3d')
    # Plot GT (Green)
    ax2.scatter(xyz_gt[mask_gt, 0], xyz_gt[mask_gt, 1], xyz_gt[mask_gt, 2], c='g', s=2, alpha=0.3, label='t+1 (GT)')
    # Plot Pred (Red)
    ax2.scatter(xyz_pred[mask_pred, 0], xyz_pred[mask_pred, 1], xyz_pred[mask_pred, 2], c='r', s=2, alpha=0.3, label='t+1 (Pred)')
    ax2.set_title(f"Step {step}: Accuracy (Green=GT, Red=Pred)")
    ax2.legend()
    
    # --- Plot 3: Flow Heatmaps (Where is the action?) ---
    # Visualize which parts of the scene are moving due to interaction
    ax3 = fig.add_subplot(133, projection='3d')
    p = ax3.scatter(xyz_curr[:, 0], xyz_curr[:, 1], xyz_curr[:, 2], c=v_int, cmap='plasma', s=2, alpha=0.8)
    fig.colorbar(p, ax=ax3, label='Interaction Flow Magnitude')
    ax3.set_title(f"Step {step}: Interaction Heatmap")

    plt.tight_layout()
    
    # Save to ./visualizations
    save_dir = "./visualizations/world_model_predictions"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"world_model_viz_step_{step:06d}.png")
    plt.savefig(save_path, dpi=100)
    # Explicitly close to prevent memory leak
    plt.close(fig)
    print(f"Saved visualization to {save_path}")

