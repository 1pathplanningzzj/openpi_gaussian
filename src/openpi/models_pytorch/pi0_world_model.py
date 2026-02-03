# zijian
# date 2026.01.24
# v1 to be debug and much test to do 
# Description: Bi-Directional World Model with Environment and Interaction Flows, Contact Gating, and Inverse Dynamics.
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
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


class BiDirectionalWorldModel(nn.Module):
    """
    Main Phase 2 World Model container.
    """
    def __init__(self, token_dim, action_dim, use_vggt_decoder=False, vggt_decoder=None, 
                 input_num_tokens=100, target_num_tokens=1369, vggt_embed_dim=1024):
        super().__init__()
        
        self.forward_model = ForwardInteractionPredictor(token_dim, action_dim)
        self.inverse_model = InverseModel(token_dim, action_dim)
        self.use_vggt_decoder = use_vggt_decoder
        self.vggt_decoder = vggt_decoder  # VGGT gs_head decoder
        
        # Token upsampling for VGGT decoder compatibility
        # Input: 100 tokens (10x10 grid) -> Output: 1369 tokens (37x37 grid)
        self.input_num_tokens = input_num_tokens  # 100
        self.target_num_tokens = target_num_tokens  # 1369
        self.token_dim = token_dim
        self.vggt_embed_dim = vggt_embed_dim  # VGGT encoder's embed_dim (1024)
        
        # Projection layer to match VGGT encoder's token dimension
        # z_t1_pred has dimension token_dim (from paligemma_config.width, typically 2048 for gemma_2b)
        # aggregated_tokens_list from VGGT encoder actually has 2*embed_dim (2048), not embed_dim (1024)
        # This is because VGGT uses concatenated features from multiple layers
        # We need to project z_t1_pred to 2*vggt_embed_dim (2048) to match aggregated_tokens_list
        if use_vggt_decoder and vggt_decoder is not None:
            # Project to 2*vggt_embed_dim (2048) to match aggregated_tokens_list token dimension
            # Runtime evidence shows last_layer_shape is [B, S, P, 2048], not [B, S, P, 1024]
            target_dim = 2 * vggt_embed_dim  # 2048
            self.token_proj = nn.Linear(token_dim, target_dim)
            self.vggt_decoder_expected_dim = target_dim
            # #region agent log
            import json
            import os
            try:
                with open('/home/zijianzhang/openpi/.cursor/debug.log', 'a') as f:
                    log_entry = {
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "A",
                        "location": "pi0_world_model.py:__init__",
                        "message": "Token projection layer initialized",
                        "data": {
                            "token_dim": token_dim,
                            "vggt_embed_dim": vggt_embed_dim,
                            "projection_target": target_dim,
                            "note": "Fixed: Projecting to 2*embed_dim (2048) to match aggregated_tokens_list actual dimension"
                        },
                        "timestamp": int(__import__('time').time() * 1000)
                    }
                    f.write(json.dumps(log_entry) + '\n')
            except: pass
            # #endregion
        else:
            # Legacy decoder path has been removed. World model now *requires* a VGGT decoder
            # if Gaussian supervision is used.
            self.token_proj = None
            self.vggt_decoder_expected_dim = None
        
        if use_vggt_decoder and input_num_tokens != target_num_tokens:
            # Create upsampling layer: 10x10 -> 37x37
            # Strategy: Reshape to 2D, use interpolation, then flatten
            self.token_upsampler = nn.Sequential(
                # First reshape: [B, 100, D] -> [B, D, 10, 10]
                # Then interpolate: [B, D, 10, 10] -> [B, D, 37, 37]
                # Then flatten: [B, D, 37, 37] -> [B, 1369, D]
                nn.Identity()  # Placeholder, actual upsampling in forward
            )
        else:
            self.token_upsampler = None
        
        # Legacy decoder (Privileged4DGSDecoder) has been removed.
        # All decoding to Gaussian parameters must go through the VGGT decoder path.
        self.decoder = None
        
    def forward(self, z_t, action, t=0.0):
        """
        Run forward prediction step.
        """
        # Ensure t is a tensor
        if isinstance(t, (int, float)):
            t = torch.tensor([t], device=z_t.device, dtype=z_t.dtype).repeat(z_t.shape[0])
            
        z_next_pred, details = self.forward_model(z_t, action, t)
        
        # Upsample tokens if needed for VGGT decoder
        if self.token_upsampler is not None and z_next_pred.shape[1] == self.input_num_tokens:
            z_next_pred = self._upsample_tokens(z_next_pred)
        
        return z_next_pred, details
    
    def _upsample_tokens(self, tokens):
        """
        Upsample tokens from [B, 100, D] to [B, 1369, D].
        
        Strategy:
        1. Reshape: [B, 100, D] -> [B, D, 10, 10] (assuming 10x10 grid)
        2. Interpolate: [B, D, 10, 10] -> [B, D, 37, 37] (bilinear interpolation)
        3. Flatten: [B, D, 37, 37] -> [B, 1369, D]
        """
        B, N, D = tokens.shape
        
        if N != self.input_num_tokens:
            # If input doesn't match expected size, return as-is (shouldn't happen)
            return tokens
        
        # Reshape to 2D grid: [B, 100, D] -> [B, D, 10, 10]
        tokens_2d = tokens.view(B, D, 10, 10)
        
        # Bilinear interpolation: [B, D, 10, 10] -> [B, D, 37, 37]
        import torch.nn.functional as F
        tokens_upsampled = F.interpolate(
            tokens_2d, 
            size=(37, 37), 
            mode='bilinear', 
            align_corners=False
        )  # [B, D, 37, 37]
        
        # Flatten back: [B, D, 37, 37] -> [B, 1369, D]
        tokens_final = tokens_upsampled.permute(0, 2, 3, 1).reshape(B, 37 * 37, D)  # [B, 1369, D]
        
        return tokens_final
    
    def decode(self, z, future_observation=None, gaussian_adapter=None, camera_params=None, return_2d_maps=False, step=None):
        """
        Decode latent tokens to Gaussian parameters.
        If use_vggt_decoder=True, uses VGGT decoder with future_observation.
        Otherwise, uses legacy Privileged4DGSDecoder.
        
        Args:
            z: [B, N, D] - latent tokens
            future_observation: Observation object with images (required for VGGT decoder)
            gaussian_adapter: GaussianAdapter instance (required for VGGT decoder)
            camera_params: Optional camera parameters for unprojection (dict with 'intrinsics' and 'viewmatrix')
            return_2d_maps: If True, return 2D maps directly instead of converting to 3D (for direct 2D supervision)
        
        Returns:
            gaussian_params: Dict with Gaussian parameters
                If return_2d_maps=True and use_vggt_decoder=True:
                    - rot_maps: [B, S, H, W, 4]
                    - scale_maps: [B, S, H, W, 3]
                    - opacity_maps: [B, S, H, W, 1]
                    - sh_maps: [B, S, H, W, K, 3]
                    - depth_maps: [B, S, H, W, 1]
                Otherwise:
                    - xyz: [B, N, 3]
                    - sigma: [B, N, 6]
                    - opacity: [B, N, 1]
                    - sh: [B, N, K*3]
        """
        if self.use_vggt_decoder and self.vggt_decoder is not None:
            # Use VGGT decoder
            if future_observation is None or gaussian_adapter is None:
                raise ValueError("future_observation and gaussian_adapter are required for VGGT decoder")
            
            # 1. Prepare images for VGGT
            vggt_inputs = gaussian_adapter.prepare_inputs(
                future_observation,
                z.device,
                z.shape[0],
                is_training=False
            )
            
            if vggt_inputs is None:
                raise ValueError("Failed to prepare VGGT inputs from future_observation")
            
            # 2. Get aggregated_tokens_list from VGGT encoder
            with torch.no_grad():
                aggregated_tokens_list, patch_start_idx = gaussian_adapter.encoder.aggregator(
                    vggt_inputs.to(torch.bfloat16)
                )
            
            # FIX: Ensure all tokens are [B, S, P, C] (4D)
            # This is critical because GSDPTHead slices along dim 2, expecting it to be sequence length.
            # If input is [B*S, P, C] (3D), slicing dim 2 cuts the feature dimension, causing 2043 vs 2048 mismatch.
            B_vggt, S_vggt = vggt_inputs.shape[:2]
            for i in range(len(aggregated_tokens_list)):
                if aggregated_tokens_list[i].ndim == 3:
                    _bs, _p, _c = aggregated_tokens_list[i].shape
                    # Reshape [B*S, P, C] -> [B, S, P, C]
                    aggregated_tokens_list[i] = aggregated_tokens_list[i].view(B_vggt, S_vggt, _p, _c)
            
            # 3. Replace last layer tokens with z_t1_pred (World Model prediction)
            # aggregated_tokens_list is a list of tensors from different layers [B*S, P, D]
            # where P includes camera tokens + register tokens + patch tokens
            # z_t1_pred is [B, N, D] where N is only patch tokens (from VGGT encoder)
            # We need to:
            #   1. Reshape z_t1_pred to [B*S, N, D] to match batch*sequence format
            #   2. Replace patch tokens in the last layer (keep camera + register tokens)
            if aggregated_tokens_list and len(aggregated_tokens_list) > 0:
                last_layer_tokens = aggregated_tokens_list[-1]  # Expected: [B*S, P, D]
                # #region agent log
                import json
                try:
                    with open('/home/zijianzhang/openpi/.cursor/debug.log', 'a') as f:
                        log_entry = {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "B",
                            "location": "pi0_world_model.py:decode:before_token_replacement",
                            "message": "Before token replacement - checking shapes",
                            "data": {
                                "z_shape": list(z.shape),
                                "last_layer_tokens_shape": list(last_layer_tokens.shape),
                                "last_layer_tokens_ndim": last_layer_tokens.ndim,
                                "patch_start_idx": patch_start_idx
                            },
                            "timestamp": int(__import__('time').time() * 1000)
                        }
                        f.write(json.dumps(log_entry) + '\n')
                except: pass
                # #endregion
                
                # Handle different possible shapes
                if last_layer_tokens.ndim == 2:
                    # Shape: [P, D] - single sequence, need to add batch dimension
                    # This shouldn't happen in normal flow, but handle it gracefully
                    import warnings
                    warnings.warn(
                        f"Unexpected 2D shape for last_layer_tokens: {last_layer_tokens.shape}. "
                        f"Skipping token replacement."
                    )
                elif last_layer_tokens.ndim == 3:
                    # Normal case: [B*S, P, D]
                    B_S, P, D = last_layer_tokens.shape
                    B = z.shape[0]
                    S = B_S // B
                    
                    if S == 0 or B_S % B != 0:
                        # Shape mismatch, skip replacement
                        import warnings
                        warnings.warn(
                            f"Shape mismatch: last_layer_tokens has {B_S} tokens, "
                            f"but z has batch size {B}. Cannot determine sequence length. Skipping replacement."
                        )
                    else:
                        # z_t1_pred is [B, N, D], reshape to [B*S, N, D]
                        # If S > 1, we need to expand z_t1_pred to match sequence length
                        if S > 1:
                            z_t1_pred_expanded = z.unsqueeze(1).expand(B, S, -1, -1)  # [B, S, N, D]
                            z_t1_pred_expanded = z_t1_pred_expanded.reshape(B * S, -1, D)  # [B*S, N, D]
                        else:
                            z_t1_pred_expanded = z  # [B, N, D] = [B*S, N, D] if S=1
                        
                        # Project z_t1_pred to match VGGT encoder's token dimension
                        # z_t1_pred has dimension token_dim (e.g., 2043 or 2048), but VGGT encoder uses 2*embed_dim (2048)
                        # CRITICAL: We must project to 2048 to match aggregated_tokens_list dimension
                        if self.token_proj is not None:
                            z_t1_pred_projected = self.token_proj(z_t1_pred_expanded)  # [B*S, N, 2048]
                            # #region agent log
                            import json
                            try:
                                with open('/home/zijianzhang/openpi/.cursor/debug.log', 'a') as f:
                                    log_entry = {
                                        "sessionId": "debug-session",
                                        "runId": "run1",
                                        "hypothesisId": "B",
                                        "location": "pi0_world_model.py:decode:after_projection",
                                        "message": "After token projection",
                                        "data": {
                                            "z_t1_pred_expanded_shape": list(z_t1_pred_expanded.shape),
                                            "z_t1_pred_projected_shape": list(z_t1_pred_projected.shape),
                                            "token_proj_input_dim": self.token_proj.in_features,
                                            "token_proj_output_dim": self.token_proj.out_features,
                                            "last_layer_tokens_last_dim": D
                                        },
                                        "timestamp": int(__import__('time').time() * 1000)
                                    }
                                    f.write(json.dumps(log_entry) + '\n')
                            except: pass
                            # #endregion
                        else:
                            z_t1_pred_projected = z_t1_pred_expanded
                        
                        # #region agent log
                        import json
                        import os
                        try:
                            with open('/home/zijianzhang/openpi/.cursor/debug.log', 'a') as f:
                                log_entry = {
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "B",
                                    "location": "pi0_world_model.py:decode:token_replacement",
                                    "message": "Token replacement dimensions check",
                                    "data": {
                                        "z_t1_pred_expanded_shape": list(z_t1_pred_expanded.shape),
                                        "z_t1_pred_projected_shape": list(z_t1_pred_projected.shape),
                                        "last_layer_tokens_shape": list(last_layer_tokens.shape),
                                        "D": D,
                                        "P": P,
                                        "patch_start_idx": patch_start_idx,
                                        "expected_patch_tokens": P - patch_start_idx,
                                        "actual_patch_tokens": z_t1_pred_projected.shape[1],
                                        "token_proj_exists": self.token_proj is not None
                                    },
                                    "timestamp": int(__import__('time').time() * 1000)
                                }
                                f.write(json.dumps(log_entry) + '\n')
                        except: pass
                        # #endregion
                        
                        # Replace patch tokens (after patch_start_idx) with z_t1_pred
                        # Keep camera tokens and register tokens (before patch_start_idx)
                        if z_t1_pred_projected.shape[1] == P - patch_start_idx:
                            # Check dimension compatibility before concatenation
                            if z_t1_pred_projected.shape[-1] != D:
                                # #region agent log
                                import json
                                import os
                                try:
                                    with open('/home/zijianzhang/openpi/.cursor/debug.log', 'a') as f:
                                        log_entry = {
                                            "sessionId": "debug-session",
                                            "runId": "run1",
                                            "hypothesisId": "C",
                                            "location": "pi0_world_model.py:decode:dimension_mismatch",
                                            "message": "Dimension mismatch detected before token replacement",
                                            "data": {
                                                "z_t1_pred_projected_last_dim": z_t1_pred_projected.shape[-1],
                                                "last_layer_tokens_last_dim": D,
                                                "mismatch": True
                                            },
                                            "timestamp": int(__import__('time').time() * 1000)
                                        }
                                        f.write(json.dumps(log_entry) + '\n')
                                except: pass
                                # #endregion
                                import warnings
                                warnings.warn(
                                    f"Dimension mismatch: z_t1_pred_projected has last dim {z_t1_pred_projected.shape[-1]}, "
                                    f"but last_layer_tokens has last dim {D}. Skipping token replacement."
                                )
                            else:
                                # Shape matches, replace patch tokens
                                new_last_layer = torch.cat([
                                    last_layer_tokens[:, :patch_start_idx],  # Keep camera + register tokens
                                    z_t1_pred_projected  # Replace with World Model prediction (projected to match VGGT dim)
                                ], dim=1)  # [B*S, P, D]
                                aggregated_tokens_list[-1] = new_last_layer
                                # #region agent log
                                import json
                                import os
                                try:
                                    with open('/home/zijianzhang/openpi/.cursor/debug.log', 'a') as f:
                                        log_entry = {
                                            "sessionId": "debug-session",
                                            "runId": "run1",
                                            "hypothesisId": "D",
                                            "location": "pi0_world_model.py:decode:token_replaced",
                                            "message": "Token replacement successful",
                                            "data": {
                                                "new_last_layer_shape": list(new_last_layer.shape),
                                                "replacement_successful": True
                                            },
                                            "timestamp": int(__import__('time').time() * 1000)
                                        }
                                        f.write(json.dumps(log_entry) + '\n')
                                except: pass
                                # #endregion
                        else:
                            # Shape mismatch, log warning but continue with original tokens
                            import warnings
                            warnings.warn(
                                f"Shape mismatch: z_t1_pred has {z_t1_pred_expanded.shape[1]} tokens, "
                                f"expected {P - patch_start_idx} patch tokens. Using original tokens."
                            )
                elif last_layer_tokens.ndim == 4:
                    # Shape: [B, S, P, D] - need to reshape to [B*S, P, D]
                    B_tokens, S_tokens, P, D = last_layer_tokens.shape
                    last_layer_tokens = last_layer_tokens.reshape(B_tokens * S_tokens, P, D)  # [B*S, P, D]
                    
                    # Now process as 3D case
                    B_S, P, D = last_layer_tokens.shape
                    B = z.shape[0]
                    S = B_S // B
                    
                    if S == 0 or B_S % B != 0:
                        # Shape mismatch, skip replacement
                        import warnings
                        warnings.warn(
                            f"Shape mismatch after reshape: last_layer_tokens has {B_S} tokens, "
                            f"but z has batch size {B}. Cannot determine sequence length. Skipping replacement."
                        )
                    else:
                        # z_t1_pred is [B, N, D], reshape to [B*S, N, D]
                        if S > 1:
                            z_t1_pred_expanded = z.unsqueeze(1).expand(B, S, -1, -1)  # [B, S, N, D]
                            z_t1_pred_expanded = z_t1_pred_expanded.reshape(B * S, -1, D)  # [B*S, N, D]
                        else:
                            z_t1_pred_expanded = z  # [B, N, D] = [B*S, N, D] if S=1
                        
                        # Project z_t1_pred to match VGGT encoder's token dimension
                        if self.token_proj is not None:
                            z_t1_pred_projected = self.token_proj(z_t1_pred_expanded)  # [B*S, N, vggt_embed_dim]
                        else:
                            z_t1_pred_projected = z_t1_pred_expanded
                        
                        # Replace patch tokens (after patch_start_idx) with z_t1_pred
                        expected_patch_tokens = P - patch_start_idx
                        actual_patch_tokens = z_t1_pred_projected.shape[1]
                        
                        if step is not None and step % 40 == 0:
                            print(f"[DEBUG] Token replacement check:")
                            print(f"  z_t1_pred_projected.shape={z_t1_pred_projected.shape}")
                            print(f"  last_layer_tokens.shape={last_layer_tokens.shape}")
                            print(f"  P={P}, patch_start_idx={patch_start_idx}")
                            print(f"  expected_patch_tokens={expected_patch_tokens}, actual_patch_tokens={actual_patch_tokens}")
                        
                        if actual_patch_tokens == expected_patch_tokens:
                            # Check dimension compatibility before concatenation
                            if z_t1_pred_projected.shape[-1] != D:
                                import warnings
                                warnings.warn(
                                    f"Dimension mismatch: z_t1_pred_projected has last dim {z_t1_pred_projected.shape[-1]}, "
                                    f"but last_layer_tokens has last dim {D}. Skipping token replacement."
                                )
                                if step is not None and step % 40 == 0:
                                    print(f"[WARNING] Dimension mismatch! z_t1_pred_projected.dim={z_t1_pred_projected.shape[-1]}, D={D}")
                            else:
                                new_last_layer = torch.cat([
                                    last_layer_tokens[:, :patch_start_idx],  # Keep camera + register tokens
                                    z_t1_pred_projected  # Replace with World Model prediction (projected to match VGGT dim)
                                ], dim=1)  # [B*S, P, D]
                                
                                # FIX: Reshape back to [B, S, P, D] because we enforce 4D for all tokens
                                aggregated_tokens_list[-1] = new_last_layer.view(B_tokens, S_tokens, P, D)
                                
                                if step is not None and step % 40 == 0:
                                    print(f"[DEBUG] Token replacement SUCCESS! Replaced {actual_patch_tokens} patch tokens.")
                        else:
                            import warnings
                            warnings.warn(
                                f"Shape mismatch: z_t1_pred has {actual_patch_tokens} tokens, "
                                f"expected {expected_patch_tokens} patch tokens. Using original tokens."
                            )
                            if step is not None and step % 40 == 0:
                                print(f"[WARNING] Token replacement FAILED! This means decoder will use GT tokens instead of predicted tokens!")
                                print(f"  This is likely why rendering shows GT-like results instead of predictions.")
                else:
                    # Unexpected shape (5D or more)
                    import warnings
                    warnings.warn(
                        f"Unexpected shape for last_layer_tokens: {last_layer_tokens.shape} "
                        f"(ndim={last_layer_tokens.ndim}). Skipping token replacement."
                    )
            
            # 4. Call VGGT decoder
            # Note: VGGT decoder outputs 2D maps [B, S, H, W, D], not 3D point cloud
            # We'll need to convert 2D maps to 3D Gaussians later
            # CRITICAL: Check that aggregated_tokens_list[-1] has the correct dimension (2048) before calling decoder
            if aggregated_tokens_list and len(aggregated_tokens_list) > 0:
                last_layer_before_decode = aggregated_tokens_list[-1]
                if last_layer_before_decode.shape[-1] != 2048:
                    # Force projection to 2048 if dimension mismatch
                    import warnings
                    warnings.warn(
                        f"CRITICAL: aggregated_tokens_list[-1] has last dim {last_layer_before_decode.shape[-1]}, "
                        f"but VGGT decoder expects 2048. This will cause a RuntimeError. "
                        f"Attempting to fix by projecting to 2048..."
                    )
                    # Project the last layer to 2048 if we have a projection layer
                    if self.token_proj is not None and last_layer_before_decode.shape[-1] == self.token_proj.in_features:
                        # Reshape to [B*S*P, D_in] for projection
                        B_S, P, D_in = last_layer_before_decode.shape
                        last_layer_flat = last_layer_before_decode.reshape(B_S * P, D_in)
                        last_layer_projected = self.token_proj(last_layer_flat)  # [B*S*P, 2048]
                        aggregated_tokens_list[-1] = last_layer_projected.reshape(B_S, P, 2048)
                    else:
                        # If we can't fix it, raise an error with helpful message
                        raise RuntimeError(
                            f"Cannot fix dimension mismatch: aggregated_tokens_list[-1] has shape {last_layer_before_decode.shape}, "
                            f"but VGGT decoder expects last dim 2048. "
                            f"token_proj exists: {self.token_proj is not None}, "
                            f"token_proj input dim: {self.token_proj.in_features if self.token_proj is not None else None}"
                        )
            # #region agent log
            import json
            import os
            try:
                with open('/home/zijianzhang/openpi/.cursor/debug.log', 'a') as f:
                    log_entry = {
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "E",
                        "location": "pi0_world_model.py:decode:before_vggt_decoder",
                        "message": "Before calling VGGT decoder",
                        "data": {
                            "aggregated_tokens_list_length": len(aggregated_tokens_list),
                            "last_layer_shape": list(aggregated_tokens_list[-1].shape) if aggregated_tokens_list else None,
                            "last_layer_last_dim": aggregated_tokens_list[-1].shape[-1] if aggregated_tokens_list else None,
                            "vggt_inputs_shape": list(vggt_inputs.shape),
                            "patch_start_idx": patch_start_idx,
                            "expected_dim": 2048
                        },
                        "timestamp": int(__import__('time').time() * 1000)
                    }
                    f.write(json.dumps(log_entry) + '\n')
            except: pass
            # #endregion
            raw_gaussian = self.vggt_decoder(
                aggregated_tokens_list,
                images=vggt_inputs,
                patch_start_idx=patch_start_idx
            )  # [B, S, H, W, D]
            # #region agent log
            import json
            import os
            try:
                with open('/home/zijianzhang/openpi/.cursor/debug.log', 'a') as f:
                    log_entry = {
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "E",
                        "location": "pi0_world_model.py:decode:after_vggt_decoder",
                        "message": "After calling VGGT decoder",
                        "data": {
                            "raw_gaussian_shape": list(raw_gaussian.shape),
                            "decoder_successful": True
                        },
                        "timestamp": int(__import__('time').time() * 1000)
                    }
                    f.write(json.dumps(log_entry) + '\n')
            except: pass
            # #endregion
            
            # 5. Get depth maps from VGGT encoder
            with torch.no_grad():
                depth_maps, depth_conf = gaussian_adapter.encoder.depth_head(
                    aggregated_tokens_list,
                    images=vggt_inputs,
                    patch_start_idx=patch_start_idx
                )
                # Process depth maps (same as VGGT forward)
                depth_maps = torch.nn.functional.sigmoid(torch.log(depth_maps))
                min_depth = gaussian_adapter.encoder.min_depth
                max_depth = gaussian_adapter.encoder.max_depth
                
                # DEBUG: Check depth stats
                if torch.isnan(depth_maps).any():
                     import warnings
                     warnings.warn("VGGT depth_maps head output contains NaNs!")
                # print(f"[DEBUG] Depth info: min_depth={min_depth}, max_depth={max_depth}, depth_map_range=[{depth_maps.min():.4f}, {depth_maps.max():.4f}]")
                
                depth_range = max_depth - min_depth
                depth_maps = min_depth + depth_range * depth_maps  # [B, S, H, W, 1]
            
            # 6. Parse VGGT output
            d_sh = gaussian_adapter.encoder.d_sh
            rot_maps, scale_maps, opacity_maps, sh_maps = raw_gaussian.split((4, 3, 1, 3 * d_sh), dim=-1)
            
            # Process maps (same as VGGT forward)
            rot_maps = rot_maps / (rot_maps.norm(dim=-1, keepdim=True) + 1e-8)
            
            # Debug: Check raw scale_maps before processing
            if step is not None and step % 40 == 0:
                print(f"[DEBUG] Raw scale_maps (before softplus): min={scale_maps.min():.6f}, max={scale_maps.max():.6f}, mean={scale_maps.mean():.6f}")
            
            scale_maps = torch.nn.functional.softplus(scale_maps, beta=1) * 0.001
            
            # Debug: Check processed scale_maps
            if step is not None and step % 40 == 0:
                print(f"[DEBUG] Processed scale_maps (after softplus*0.001): min={scale_maps.min():.6f}, max={scale_maps.max():.6f}, mean={scale_maps.mean():.6f}")
            
            opacity_maps = torch.sigmoid(opacity_maps)
            
            # Reshape sh_maps from [B, S, H, W, 3 * d_sh] to [B, S, H, W, 3, d_sh]
            # This matches VGGT's processing: rearrange(sh_maps, "b n h w (i c) -> b n h w i c", i=3)
            from einops import rearrange
            sh_maps = rearrange(sh_maps, "b s h w (i c) -> b s h w i c", i=3, c=d_sh)
            
            # Apply SH mask if available
            if hasattr(gaussian_adapter.encoder, 'sh_mask'):
                sh_mask = gaussian_adapter.encoder.sh_mask  # [d_sh]
                # sh_mask shape: [d_sh], sh_maps shape: [B, S, H, W, 3, d_sh]
                # Broadcasting: [1, 1, 1, 1, 1, d_sh] * [B, S, H, W, 3, d_sh]
                sh_maps = sh_maps * sh_mask.view(1, 1, 1, 1, 1, -1)
            
            # 7. Return 2D maps directly if requested (for direct 2D supervision)
            if return_2d_maps:
                return {
                    "rot_maps": rot_maps,        # [B, S, H, W, 4]
                    "scale_maps": scale_maps,     # [B, S, H, W, 3]
                    "opacity_maps": opacity_maps, # [B, S, H, W, 1]
                    "sh_maps": sh_maps,          # [B, S, H, W, K, 3]
                    "depth_maps": depth_maps,    # [B, S, H, W, 1]
                    "is_2d_maps": True
                }
            
            # 8. Convert 2D maps to 3D point cloud (for rendering)
            gaussian_params = self._convert_2d_maps_to_3d_gaussians(
                depth_maps=depth_maps,           # [B, S, H, W, 1]
                rot_maps=rot_maps,              # [B, S, H, W, 4]
                scale_maps=scale_maps,           # [B, S, H, W, 3]
                opacity_maps=opacity_maps,       # [B, S, H, W, 1]
                sh_maps=sh_maps,                 # [B, S, H, W, K, 3] where K = d_sh
                vggt_inputs=vggt_inputs,         # [B, S, 3, H, W] - for getting image size
                camera_params=camera_params  # Pass camera params for proper unprojection
            )
            
            return gaussian_params
        else:
            # Legacy decoder path has been removed to simplify and align with VGGT-based decoding.
            raise ValueError(
                "BiDirectionalWorldModel.decode called without VGGT decoder. "
                "Legacy Privileged4DGSDecoder has been removed. "
                "Please ensure `use_vggt_decoder=True` and a valid `vggt_decoder` is provided."
            )
    
    def _convert_2d_maps_to_3d_gaussians(self, depth_maps, rot_maps, scale_maps, opacity_maps, sh_maps, 
                                         vggt_inputs, camera_params=None, downsample_factor=4, step=None):
        """
        Convert VGGT decoder's 2D maps to 3D Gaussian point cloud.
        
        Args:
            depth_maps: [B, S, H, W, 1] - depth values
            rot_maps: [B, S, H, W, 4] - rotation quaternions
            scale_maps: [B, S, H, W, 3] - scale parameters
            opacity_maps: [B, S, H, W, 1] - opacity values
            sh_maps: [B, S, H, W, K, 3] - spherical harmonics coefficients (K = d_sh)
            vggt_inputs: [B, S, 3, H, W] - input images (for getting H, W)
            camera_params: Optional camera parameters for unprojection
            downsample_factor: Factor to downsample for efficiency (default: 4, so 224x224 -> 56x56)
        
        Returns:
            gaussian_params: Dict with keys:
                - xyz: [B, N, 3] - 3D point positions
                - sigma: [B, N, 6] - covariance parameters
                - opacity: [B, N, 1] - opacity values
                - sh: [B, N, K*3] - spherical harmonics coefficients
        """
        import torch.nn.functional as F
        
        B, S, H, W, _ = depth_maps.shape
        device = depth_maps.device
        dtype = depth_maps.dtype
        
        # Use last frame (S-1) for single frame prediction
        # If S > 1, we'll use the last frame
        frame_idx = S - 1
        
        # Extract single frame
        depth = depth_maps[:, frame_idx, :, :, 0]  # [B, H, W]
        
        # Debug: Check depth values
        if step is not None and step % 40 == 0:
            print(f"[DEBUG] Depth map stats: min={depth.min():.4f}, max={depth.max():.4f}, mean={depth.mean():.4f}, "
                  f"shape={depth.shape}, valid_pixels={(depth > 0.01).sum().item()}/{depth.numel()}")
        rot = rot_maps[:, frame_idx]  # [B, H, W, 4]
        scale = scale_maps[:, frame_idx]  # [B, H, W, 3]
        opacity = opacity_maps[:, frame_idx, :, :, 0]  # [B, H, W]
        sh = sh_maps[:, frame_idx]  # [B, H, W, K, 3]
        
        # Downsample for efficiency (optional)
        if downsample_factor > 1:
            H_ds = H // downsample_factor
            W_ds = W // downsample_factor
            depth = F.interpolate(depth.unsqueeze(1), size=(H_ds, W_ds), mode='bilinear', align_corners=False).squeeze(1)
            rot = F.interpolate(rot.permute(0, 3, 1, 2), size=(H_ds, W_ds), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            scale = F.interpolate(scale.permute(0, 3, 1, 2), size=(H_ds, W_ds), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            opacity = F.interpolate(opacity.unsqueeze(1), size=(H_ds, W_ds), mode='bilinear', align_corners=False).squeeze(1)
            # For SH, interpolate each channel separately
            sh_reshaped = sh.permute(0, 4, 1, 2, 3).reshape(B, -1, H, W)  # [B, K*3, H, W]
            sh_reshaped = F.interpolate(sh_reshaped, size=(H_ds, W_ds), mode='bilinear', align_corners=False)
            sh = sh_reshaped.reshape(B, -1, H_ds, W_ds).permute(0, 2, 3, 1).reshape(B, H_ds, W_ds, sh.shape[-2], sh.shape[-1])
            H, W = H_ds, W_ds
        
        # Normalize rotation quaternions
        rot = rot / (rot.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Process scale (same as VGGT)
        scale = F.softplus(scale, beta=1) * 0.001  # [B, H, W, 3]
        
        # Process opacity
        opacity = torch.sigmoid(opacity)  # [B, H, W]
        
        # Process SH (apply mask if available) - Note: gaussian_adapter is not available here
        # We'll skip the mask for now, or pass it as a parameter if needed
        
        # Unproject depth to 3D points
        # Create pixel grid
        u = torch.arange(W, device=device, dtype=dtype).view(1, 1, -1).expand(B, H, -1)  # [B, H, W]
        v = torch.arange(H, device=device, dtype=dtype).view(1, -1, 1).expand(B, -1, W)  # [B, H, W]
        
        # Get camera intrinsics (from camera_params or use defaults)
        if camera_params is not None and "intrinsics" in camera_params:
            intrinsics = camera_params["intrinsics"]  # [B, 3, 3] or [3, 3]
            if intrinsics.ndim == 3:
                fx = intrinsics[:, 0, 0].view(B, 1, 1)  # [B, 1, 1]
                fy = intrinsics[:, 1, 1].view(B, 1, 1)
                cx = intrinsics[:, 0, 2].view(B, 1, 1)
                cy = intrinsics[:, 1, 2].view(B, 1, 1)
            else:
                fx = intrinsics[0, 0]
                fy = intrinsics[1, 1]
                cx = intrinsics[0, 2]
                cy = intrinsics[1, 2]
        else:
            # Use default intrinsics (assuming 224x224 image, FOV=60)
            # fx = fy = 224 / (2 * tan(30°)) ≈ 194
            fx = fy = 194.0
            cx = cy = 112.0  # H/2, W/2
        
        # Unproject: X = (u - cx) * depth / fx, Y = (v - cy) * depth / fy, Z = depth
        x_cam = (u - cx) * depth / fx  # [B, H, W]
        y_cam = (v - cy) * depth / fy  # [B, H, W]
        z_cam = depth  # [B, H, W]
        
        # Debug: Check unprojection results
        if step is not None and step % 40 == 0:
            print(f"[DEBUG] Unprojection stats:")
            print(f"  Camera intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
            print(f"  x_cam range: [{x_cam.min():.4f}, {x_cam.max():.4f}], mean={x_cam.mean():.4f}")
            print(f"  y_cam range: [{y_cam.min():.4f}, {y_cam.max():.4f}], mean={y_cam.mean():.4f}")
            print(f"  z_cam range: [{z_cam.min():.4f}, {z_cam.max():.4f}], mean={z_cam.mean():.4f}")
            print(f"  Pixel grid: u range=[{u.min():.1f}, {u.max():.1f}], v range=[{v.min():.1f}, {v.max():.1f}]")
        
        # Stack to get camera coordinates
        xyz_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # [B, H, W, 3]
        
        # Transform to world coordinates if camera_params provided
        if camera_params is not None and "viewmatrix" in camera_params:
            viewmatrix = camera_params["viewmatrix"]  # [B, 4, 4]
            # Inverse viewmatrix to get world coordinates
            # xyz_world = xyz_cam @ viewmatrix_inv[:3, :3].T + viewmatrix_inv[:3, 3]
            viewmatrix_inv = torch.inverse(viewmatrix)  # [B, 4, 4]
            R_inv = viewmatrix_inv[:, :3, :3]  # [B, 3, 3]
            t_inv = viewmatrix_inv[:, :3, 3]  # [B, 3]
            
            xyz_cam_flat = xyz_cam.reshape(B, -1, 3)  # [B, H*W, 3]
            xyz_world_flat = torch.matmul(xyz_cam_flat, R_inv.transpose(-1, -2)) + t_inv.unsqueeze(1)  # [B, H*W, 3]
            xyz = xyz_world_flat  # [B, H*W, 3]
            
            # Debug: Check world coordinates
            if step is not None and step % 40 == 0:
                print(f"[DEBUG] World coordinates after transform:")
                print(f"  xyz range: [{xyz.min():.4f}, {xyz.max():.4f}], mean={xyz.mean(dim=1).mean():.4f}")
        else:
            # Use camera coordinates directly
            xyz = xyz_cam.reshape(B, -1, 3)  # [B, H*W, 3]
            
            # Debug: Using camera coordinates directly
            if step is not None and step % 40 == 0:
                print(f"[DEBUG] Using camera coordinates directly (no viewmatrix transform)")
                print(f"  xyz_cam range: [{xyz.min():.4f}, {xyz.max():.4f}], mean={xyz.mean(dim=1).mean():.4f}")
        
        # Flatten all parameters
        N = H * W
        rot_flat = rot.reshape(B, N, 4)  # [B, N, 4]
        scale_flat = scale.reshape(B, N, 3)  # [B, N, 3]
        opacity_flat = opacity.reshape(B, N, 1)  # [B, N, 1]
        sh_flat = sh.reshape(B, N, -1)  # [B, N, K*3]
        
        # Convert rotation quaternion and scale to covariance (sigma)
        # For simplicity, we'll create a diagonal covariance from scale
        # sigma = scale^2 (diagonal)
        # But GaussianRenderer expects 6D covariance parameters
        # We'll use a simple conversion: create 6D params from scale
        # #region agent log
        import json
        log_path = "/home/zijianzhang/openpi/.cursor/debug.log"
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "A",
                    "location": "pi0_world_model.py:640",
                    "message": "Before creating sigma_params",
                    "data": {
                        "scale_flat_shape": list(scale_flat.shape),
                        "scale_flat_min": [float(x) for x in scale_flat.min(dim=1)[0].cpu().tolist()],
                        "scale_flat_max": [float(x) for x in scale_flat.max(dim=1)[0].cpu().tolist()]
                    },
                    "timestamp": 0
                }) + "\n")
        except: pass
        # #endregion
        
        # sigma_params = [s11, s12, s13, s22, s23, s33] (upper triangle of 3x3 matrix)
        # Clamp scale to reasonable range to avoid numerical issues
        # Debug: Check scale_flat before clamping
        if torch.isnan(scale_flat).any() or torch.isinf(scale_flat).any():
            import warnings
            warnings.warn("NaN/Inf detected in scale_flat before clamping. Replacing with default scale.")
            scale_flat = torch.where(
                torch.isnan(scale_flat) | torch.isinf(scale_flat),
                torch.full_like(scale_flat, 0.01),  # Default scale: 0.01 (reasonable for rendering)
                scale_flat
            )
        
        scale_flat_clamped = torch.clamp(scale_flat, min=1e-6, max=1.0)
        
        # Debug: Check if scale_flat_clamped is too small (would cause rendering issues)
        if scale_flat_clamped.max() < 0.01:
            import warnings
            warnings.warn(
                f"Scale values are very small (max={scale_flat_clamped.max():.6f}). "
                f"This may cause rendering issues. Consider checking VGGT decoder output."
            )
        
        sigma_params = torch.zeros(B, N, 6, device=device, dtype=dtype)
        # Diagonal terms: s11, s22, s33
        # IMPORTANT: sigma is the variance (scale^2), not the standard deviation
        # For very small scales, we need to ensure sigma is not too small to avoid numerical issues
        # Use a minimum sigma value to prevent rendering issues
        min_sigma = 1e-8  # Minimum variance to ensure numerical stability
        sigma_params[:, :, 0] = torch.clamp(scale_flat_clamped[:, :, 0] ** 2, min=min_sigma)  # s11 = sx^2
        sigma_params[:, :, 3] = torch.clamp(scale_flat_clamped[:, :, 1] ** 2, min=min_sigma)  # s22 = sy^2
        sigma_params[:, :, 5] = torch.clamp(scale_flat_clamped[:, :, 2] ** 2, min=min_sigma)  # s33 = sz^2
        # Off-diagonal terms are 0 (axis-aligned Gaussians in current implementation)
        # s12, s13, s23 are already 0 from zeros initialization
        
        # #region agent log
        try:
            with open(log_path, "a") as f:
                offdiag = sigma_params[:, :, [1, 2, 4]]
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "F",
                    "location": "pi0_world_model.py:_convert_2d_maps_to_3d_gaussians",
                    "message": "After creating sigma_params (checking axis alignment)",
                    "data": {
                        "sigma_params_shape": list(sigma_params.shape),
                        "sigma_params_min": float(sigma_params.min().item()),
                        "sigma_params_max": float(sigma_params.max().item()),
                        "sigma_params_mean": float(sigma_params.mean().item()),
                        "sigma_diag_0_min": float(sigma_params[:, :, 0].min().item()),
                        "sigma_diag_3_min": float(sigma_params[:, :, 3].min().item()),
                        "sigma_diag_5_min": float(sigma_params[:, :, 5].min().item()),
                        "sigma_offdiag_abs_max": float(offdiag.abs().max().item())
                    },
                    "timestamp": 0
                }) + "\n")
        except: pass
        # #endregion
        
        # Clamp xyz to reasonable range to avoid CUDA memory access errors
        # Large coordinates can cause issues in rasterization
        # Also check for NaN/Inf values
        xyz = torch.clamp(xyz, min=-100.0, max=100.0)
        if torch.isnan(xyz).any() or torch.isinf(xyz).any():
            import warnings
            warnings.warn("NaN or Inf detected in xyz coordinates. Replacing with zeros.")
            xyz = torch.where(torch.isnan(xyz) | torch.isinf(xyz), torch.zeros_like(xyz), xyz)
        
        return {
            "xyz": xyz,  # [B, N, 3]
            "sigma": sigma_params,  # [B, N, 6]
            "opacity": opacity_flat,  # [B, N, 1]
            "sh": sh_flat,  # [B, N, K*3]
            "rotations": rot_flat  # [B, N, 4] - quaternions (for reference)
        }
    
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
        # DEBUG: Check for NaNs in inputs
        if torch.isnan(z_t).any():
            print("[compute_full_loss] WARNING: z_t contains NaNs!")
        if torch.isnan(z_t1_gt).any():
            print("[compute_full_loss] WARNING: z_t1_gt contains NaNs!")
            
        # 1. Forward Prediction
        # Assuming t=0 for single step training usually, or pass t if maintaining state
        # Get raw prediction (100 tokens)
        # Ensure t is a tensor (forward_model expects tensor, not float)
        t_tensor = torch.tensor([0.0], device=z_t.device, dtype=z_t.dtype).repeat(z_t.shape[0])
        z_t1_pred_raw, _ = self.forward_model(z_t, action_t, t=t_tensor)
        
        # DEBUG: Check output of forward model
        if torch.isnan(z_t1_pred_raw).any():
            print("[compute_full_loss] WARNING: z_t1_pred_raw contains NaNs! Forward model unstable?")
        
        # Compute latent loss on original 100 tokens (before upsampling)
        # This ensures the loss is computed on the same scale as input
        loss_fwd = F.mse_loss(z_t1_pred_raw, z_t1_gt)
        
        # Upsample z_t1_pred for VGGT decoder (if needed)
        # This allows z_t1_pred to match VGGT decoder's expected 1369 tokens
        if self.token_upsampler is not None and z_t1_pred_raw.shape[1] == self.input_num_tokens:
            z_t1_pred = self._upsample_tokens(z_t1_pred_raw)
        else:
            z_t1_pred = z_t1_pred_raw
        
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
        
        # FIX: Check if decoder exists (it might be None if using VGGT decoder)
        if model.decoder is None:
            print(f"[World Model Viz] Step {step} - Skipping visualization: Legacy decoder is None (using VGGT decoder?)")
            return

        pred_decode = model.decoder(z_t1_pred)
        gt_decode = model.decoder(z_t1_gt)
        curr_decode = model.decoder(z_t) # Also visualize t for reference
        
        # Extract XYZ coordinates for the specific batch index
        # [B, N, 3] -> [N, 3]
        xyz_pred = pred_decode['xyz'][batch_idx].cpu().numpy()
        xyz_gt = gt_decode['xyz'][batch_idx].cpu().numpy()
        xyz_curr = curr_decode['xyz'][batch_idx].cpu().numpy()
        
        # Debug: Print XYZ coordinate statistics
        print(f"[World Model Viz] Step {step} - XYZ Statistics:")
        print(f"  Current: X range=[{xyz_curr[:, 0].min():.3f}, {xyz_curr[:, 0].max():.3f}], "
              f"Y range=[{xyz_curr[:, 1].min():.3f}, {xyz_curr[:, 1].max():.3f}], "
              f"Z range=[{xyz_curr[:, 2].min():.3f}, {xyz_curr[:, 2].max():.3f}]")
        print(f"  Pred:    X range=[{xyz_pred[:, 0].min():.3f}, {xyz_pred[:, 0].max():.3f}], "
              f"Y range=[{xyz_pred[:, 1].min():.3f}, {xyz_pred[:, 1].max():.3f}], "
              f"Z range=[{xyz_pred[:, 2].min():.3f}, {xyz_pred[:, 2].max():.3f}]")
        print(f"  GT:      X range=[{xyz_gt[:, 0].min():.3f}, {xyz_gt[:, 0].max():.3f}], "
              f"Y range=[{xyz_gt[:, 1].min():.3f}, {xyz_gt[:, 1].max():.3f}], "
              f"Z range=[{xyz_gt[:, 2].min():.3f}, {xyz_gt[:, 2].max():.3f}]")
        
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
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    # Set equal aspect ratio to better see 3D structure
    all_x = np.concatenate([xyz_curr[:, 0], xyz_pred[:, 0]])
    all_y = np.concatenate([xyz_curr[:, 1], xyz_pred[:, 1]])
    all_z = np.concatenate([xyz_curr[:, 2], xyz_pred[:, 2]])
    max_range = np.array([all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()]).max() / 2.0
    mid_x = (all_x.max()+all_x.min()) * 0.5
    mid_y = (all_y.max()+all_y.min()) * 0.5
    mid_z = (all_z.max()+all_z.min()) * 0.5
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)
    ax1.legend()

    # --- Plot 2: Prediction Accuracy (Pred vs GT) ---
    ax2 = fig.add_subplot(132, projection='3d')
    # Plot GT (Green)
    ax2.scatter(xyz_gt[mask_gt, 0], xyz_gt[mask_gt, 1], xyz_gt[mask_gt, 2], c='g', s=2, alpha=0.3, label='t+1 (GT)')
    # Plot Pred (Red)
    ax2.scatter(xyz_pred[mask_pred, 0], xyz_pred[mask_pred, 1], xyz_pred[mask_pred, 2], c='r', s=2, alpha=0.3, label='t+1 (Pred)')
    ax2.set_title(f"Step {step}: Accuracy (Green=GT, Red=Pred)")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    # Set equal aspect ratio
    all_x2 = np.concatenate([xyz_gt[:, 0], xyz_pred[:, 0]])
    all_y2 = np.concatenate([xyz_gt[:, 1], xyz_pred[:, 1]])
    all_z2 = np.concatenate([xyz_gt[:, 2], xyz_pred[:, 2]])
    max_range2 = np.array([all_x2.max()-all_x2.min(), all_y2.max()-all_y2.min(), all_z2.max()-all_z2.min()]).max() / 2.0
    mid_x2 = (all_x2.max()+all_x2.min()) * 0.5
    mid_y2 = (all_y2.max()+all_y2.min()) * 0.5
    mid_z2 = (all_z2.max()+all_z2.min()) * 0.5
    ax2.set_xlim(mid_x2 - max_range2, mid_x2 + max_range2)
    ax2.set_ylim(mid_y2 - max_range2, mid_y2 + max_range2)
    ax2.set_zlim(mid_z2 - max_range2, mid_z2 + max_range2)
    ax2.legend()
    
    # --- Plot 3: Flow Heatmaps (Where is the action?) ---
    # Visualize which parts of the scene are moving due to interaction
    ax3 = fig.add_subplot(133, projection='3d')
    p = ax3.scatter(xyz_curr[:, 0], xyz_curr[:, 1], xyz_curr[:, 2], c=v_int, cmap='plasma', s=2, alpha=0.8)
    fig.colorbar(p, ax=ax3, label='Interaction Flow Magnitude')
    ax3.set_title(f"Step {step}: Interaction Heatmap")
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    # Set equal aspect ratio
    max_range3 = np.array([xyz_curr[:, 0].max()-xyz_curr[:, 0].min(), 
                           xyz_curr[:, 1].max()-xyz_curr[:, 1].min(), 
                           xyz_curr[:, 2].max()-xyz_curr[:, 2].min()]).max() / 2.0
    mid_x3 = (xyz_curr[:, 0].max()+xyz_curr[:, 0].min()) * 0.5
    mid_y3 = (xyz_curr[:, 1].max()+xyz_curr[:, 1].min()) * 0.5
    mid_z3 = (xyz_curr[:, 2].max()+xyz_curr[:, 2].min()) * 0.5
    ax3.set_xlim(mid_x3 - max_range3, mid_x3 + max_range3)
    ax3.set_ylim(mid_y3 - max_range3, mid_y3 + max_range3)
    ax3.set_zlim(mid_z3 - max_range3, mid_z3 + max_range3)

    plt.tight_layout()
    
    # Save to ./visualizations
    save_dir = "./visualizations/world_model_predictions"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"world_model_viz_step_{step:06d}.png")
    plt.savefig(save_path, dpi=100)
    # Explicitly close to prevent memory leak
    plt.close(fig)
    print(f"Saved visualization to {save_path}")

