# zijian
# date 2026.02.03
# v1 to be debug and much test to do 
# Description: VAE-based Token Compressor for VGGT tokens.
# Gaussian adaptor vae or not considering .. todo(zijian)
"""
VAE-based Token Compressor for VGGT tokens.

This module implements a Variational Autoencoder (VAE) to compress
1369 patch tokens to 100 latent tokens, which is better than simple
pooling as it learns to preserve important information.

Architecture:
- Encoder: 1369 tokens -> (mean, logvar) -> 100 latent tokens (via reparameterization)
- Decoder: 100 latent tokens -> 1369 tokens (reconstruction)
- Loss: Reconstruction loss + KL divergence loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

logger = logging.getLogger(__name__)


class VAETokenEncoder(nn.Module):
    """
    VAE Encoder: Compresses 1369 tokens to 100 latent tokens.
    
    Input: [B, 1369, D] (patch tokens from VGGT)
    Output: (mean, logvar) for reparameterization
    """
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 512):
        """
        Args:
            input_dim: Dimension of input tokens (D)
            latent_dim: Dimension of latent space (100 tokens * D)
            hidden_dim: Hidden dimension for MLP
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Reshape tokens to 2D for spatial processing
        # Input: [B, 1369, D] -> [B, D, 37, 37]
        self.input_tokens = 1369  # 37 * 37
        self.spatial_size = 37
        
        # Encoder: Process spatial tokens
        # Strategy: Use CNN to compress spatial dimensions, then MLP for latent
        self.encoder_conv = nn.Sequential(
            # [B, D, 37, 37] -> [B, hidden_dim, 10, 10]
            nn.Conv2d(input_dim, hidden_dim, kernel_size=4, stride=3, padding=1),  # 37 -> 13
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),  # 13 -> 13
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((10, 10))  # [B, hidden_dim, 10, 10]
        )
        
        # Flatten: [B, hidden_dim, 10, 10] -> [B, hidden_dim * 100]
        # Then project to latent mean and logvar
        self.fc_mean = nn.Linear(hidden_dim * 100, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 100, latent_dim)
        
    def forward(self, tokens):
        """
        Args:
            tokens: [B, 1369, D] - patch tokens from VGGT
        
        Returns:
            mean: [B, latent_dim] - mean of latent distribution
            logvar: [B, latent_dim] - log variance of latent distribution
        """
        B, N, D = tokens.shape
        assert N == self.input_tokens, f"Expected {self.input_tokens} tokens, got {N}"
        
        # Reshape to 2D: [B, 1369, D] -> [B, D, 37, 37]
        tokens_2d = tokens.view(B, D, self.spatial_size, self.spatial_size)
        
        # Encode: [B, D, 37, 37] -> [B, hidden_dim, 10, 10]
        encoded = self.encoder_conv(tokens_2d)
        
        # Flatten: [B, hidden_dim, 10, 10] -> [B, hidden_dim * 100]
        encoded_flat = encoded.view(B, -1)
        
        # Project to latent parameters
        mean = self.fc_mean(encoded_flat)  # [B, latent_dim]
        logvar = self.fc_logvar(encoded_flat)  # [B, latent_dim]
        
        return mean, logvar


class VAETokenDecoder(nn.Module):
    """
    VAE Decoder: Reconstructs 1369 tokens from 100 latent tokens.
    
    Input: [B, latent_dim] (latent tokens)
    Output: [B, 1369, D] (reconstructed patch tokens)
    """
    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int = 512):
        """
        Args:
            latent_dim: Dimension of latent space (100 tokens * D)
            output_dim: Dimension of output tokens (D)
            hidden_dim: Hidden dimension for MLP
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        self.output_tokens = 1369  # 37 * 37
        self.spatial_size = 37
        
        # Decoder: Expand latent to spatial tokens
        # Strategy: MLP to expand, then transposed conv to upsample
        self.fc_expand = nn.Linear(latent_dim, hidden_dim * 100)  # [B, latent_dim] -> [B, hidden_dim * 100]
        
        # Reshape and upsample: [B, hidden_dim, 10, 10] -> [B, D, 37, 37]
        self.decoder_conv = nn.Sequential(
            # [B, hidden_dim, 10, 10] -> [B, hidden_dim, 13, 13]
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=3, padding=1),  # 10 -> 13
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            # [B, hidden_dim, 13, 13] -> [B, hidden_dim, 37, 37]
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, stride=3, padding=0),  # 13 -> 37
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            # [B, hidden_dim, 37, 37] -> [B, D, 37, 37]
            nn.Conv2d(hidden_dim, output_dim, kernel_size=1)
        )
        
    def forward(self, latent):
        """
        Args:
            latent: [B, latent_dim] - latent tokens
        
        Returns:
            tokens: [B, 1369, D] - reconstructed patch tokens
        """
        B = latent.shape[0]
        
        # Expand: [B, latent_dim] -> [B, hidden_dim * 100]
        expanded = self.fc_expand(latent)
        
        # Reshape: [B, hidden_dim * 100] -> [B, hidden_dim, 10, 10]
        expanded_2d = expanded.view(B, self.hidden_dim, 10, 10)
        
        # Decode: [B, hidden_dim, 10, 10] -> [B, D, 37, 37]
        decoded = self.decoder_conv(expanded_2d)
        
        # Reshape: [B, D, 37, 37] -> [B, 1369, D]
        tokens = decoded.permute(0, 2, 3, 1).reshape(B, self.output_tokens, self.output_dim)
        
        return tokens


class VAETokenCompressor(nn.Module):
    """
    Complete VAE for token compression.
    
    Compresses 1369 tokens to 100 latent tokens using VAE,
    which learns to preserve important information better than pooling.
    """
    def __init__(
        self,
        token_dim: int,
        latent_tokens: int = 100,
        hidden_dim: int = 512,
        beta: float = 0.01,  # KL divergence weight
    ):
        """
        Args:
            token_dim: Dimension of tokens (D)
            latent_tokens: Number of latent tokens (100)
            hidden_dim: Hidden dimension for encoder/decoder
            beta: Weight for KL divergence loss (beta-VAE)
        """
        super().__init__()
        self.token_dim = token_dim
        self.latent_tokens = latent_tokens
        self.latent_dim = latent_tokens * token_dim  # Total latent dimension
        self.hidden_dim = hidden_dim
        self.beta = beta
        
        # VAE components
        self.encoder = VAETokenEncoder(
            input_dim=token_dim,
            latent_dim=self.latent_dim,
            hidden_dim=hidden_dim
        )
        
        self.decoder = VAETokenDecoder(
            latent_dim=self.latent_dim,
            output_dim=token_dim,
            hidden_dim=hidden_dim
        )
        
    def reparameterize(self, mean, logvar):
        """
        Reparameterization trick: z = mean + std * epsilon
        
        Args:
            mean: [B, latent_dim]
            logvar: [B, latent_dim]
        
        Returns:
            z: [B, latent_dim] - sampled latent
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z
    
    def encode(self, tokens):
        """
        Encode tokens to latent representation.
        
        Args:
            tokens: [B, 1369, D] - patch tokens
        
        Returns:
            latent: [B, latent_tokens, D] - latent tokens (reshaped from [B, latent_dim])
            mean: [B, latent_dim] - mean of latent distribution
            logvar: [B, latent_dim] - log variance of latent distribution
        """
        mean, logvar = self.encoder(tokens)
        latent_flat = self.reparameterize(mean, logvar)
        
        # Reshape: [B, latent_dim] -> [B, latent_tokens, D]
        latent = latent_flat.view(-1, self.latent_tokens, self.token_dim)
        
        return latent, mean, logvar
    
    def decode(self, latent):
        """
        Decode latent tokens back to patch tokens.
        
        Args:
            latent: [B, latent_tokens, D] or [B, latent_dim]
        
        Returns:
            tokens: [B, 1369, D] - reconstructed patch tokens
        """
        # Handle both shapes
        if latent.ndim == 3:
            # [B, latent_tokens, D] -> [B, latent_dim]
            latent_flat = latent.view(-1, self.latent_dim)
        else:
            latent_flat = latent
        
        tokens = self.decoder(latent_flat)
        return tokens
    
    def forward(self, tokens, return_loss=True):
        """
        Forward pass: encode and decode tokens.
        
        Args:
            tokens: [B, 1369, D] - patch tokens
            return_loss: If True, compute and return VAE loss
        
        Returns:
            latent: [B, latent_tokens, D] - latent tokens
            reconstructed: [B, 1369, D] - reconstructed tokens
            loss_dict: Dict with reconstruction and KL losses (if return_loss=True)
        """
        # Encode
        latent, mean, logvar = self.encode(tokens)
        
        # Decode
        reconstructed = self.decode(latent)
        
        if return_loss:
            # Reconstruction loss (MSE)
            recon_loss = F.mse_loss(reconstructed, tokens, reduction='mean')
            
            # KL divergence loss (regularization)
            # KL(q(z|x) || p(z)) where p(z) = N(0, I)
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
            kl_loss = kl_loss.mean()
            
            # Total loss (beta-VAE)
            total_loss = recon_loss + self.beta * kl_loss
            
            loss_dict = {
                'recon_loss': recon_loss,
                'kl_loss': kl_loss,
                'total_loss': total_loss
            }
            
            return latent, reconstructed, loss_dict
        else:
            return latent, reconstructed
    
    def compute_kl_loss(self, mean, logvar):
        """
        Compute KL divergence loss separately (for training flexibility).
        
        Args:
            mean: [B, latent_dim]
            logvar: [B, latent_dim]
        
        Returns:
            kl_loss: Scalar KL divergence loss
        """
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
        return kl_loss.mean()
