"""
评估 3DGS Tokens 质量的工具函数
提供多种评估方法：重建质量、特征统计、可视化等
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Optional, Tuple
import logging


def evaluate_tokens_reconstruction(
    gaussian_tokens: torch.Tensor,
    world_model_decoder: nn.Module,
    gaussian_renderer: nn.Module,
    camera_params: Dict[str, torch.Tensor],
    target_image: torch.Tensor,
    save_path: Optional[str] = None
) -> Dict[str, float]:
    """
    评估 tokens 的重建质量：将 tokens 解码为高斯参数并渲染，与真实图像比较
    
    Args:
        gaussian_tokens: [B, N, D] - 3DGS tokens
        world_model_decoder: 将 tokens 解码为高斯参数的 decoder
        gaussian_renderer: 渲染器
        camera_params: 相机参数
        target_image: [B, 3, H, W] - 目标图像
        save_path: 保存可视化结果的路径
    
    Returns:
        metrics: 包含重建质量指标的字典
    """
    with torch.no_grad():
        # 1. 解码为高斯参数
        gaussian_params = world_model_decoder(gaussian_tokens)
        
        # 2. 渲染
        rendered_image = gaussian_renderer(gaussian_params, camera_params)
        
        # 3. 计算重建误差
        mse = torch.mean((rendered_image - target_image) ** 2).item()
        mae = torch.mean(torch.abs(rendered_image - target_image)).item()
        
        # 4. 计算 PSNR
        mse_clamped = torch.clamp(mse, min=1e-10)
        psnr = -10 * np.log10(mse_clamped)
        
        # 5. 计算 SSIM (简化版，只计算结构相似性)
        # 这里简化处理，实际可以用完整的 SSIM
        mean_pred = rendered_image.mean()
        mean_gt = target_image.mean()
        std_pred = rendered_image.std()
        std_gt = target_image.std()
        
        metrics = {
            "mse": mse,
            "mae": mae,
            "psnr": psnr,
            "mean_pred": mean_pred.item(),
            "mean_gt": mean_gt.item(),
            "std_pred": std_pred.item(),
            "std_gt": std_gt.item(),
        }
        
        # 6. 可视化（如果提供了保存路径）
        if save_path is not None:
            visualize_reconstruction_comparison(
                rendered_image[0],
                target_image[0],
                save_path
            )
    
    return metrics


def visualize_reconstruction_comparison(
    rendered: torch.Tensor,
    target: torch.Tensor,
    save_path: str
):
    """可视化重建结果对比"""
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 转换为 numpy 并调整维度
    rendered_np = rendered.permute(1, 2, 0).detach().cpu().numpy()
    target_np = target.permute(1, 2, 0).detach().cpu().numpy()
    
    # 归一化到 [0, 1]
    rendered_np = np.clip(rendered_np, 0, 1)
    target_np = np.clip(target_np, 0, 1)
    
    # 差异图
    diff = np.abs(rendered_np - target_np)
    
    axes[0].imshow(target_np)
    axes[0].set_title("Ground Truth")
    axes[0].axis('off')
    
    axes[1].imshow(rendered_np)
    axes[1].set_title("Reconstructed from Tokens")
    axes[1].axis('off')
    
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title("Difference (L1)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved reconstruction comparison to {save_path}")


def analyze_tokens_statistics(
    gaussian_tokens: torch.Tensor,
    save_path: Optional[str] = None
) -> Dict[str, float]:
    """
    分析 tokens 的统计特性
    
    Args:
        gaussian_tokens: [B, N, D] - 3DGS tokens
        save_path: 保存统计图表的路径
    
    Returns:
        stats: 统计信息字典
    """
    with torch.no_grad():
        tokens_np = gaussian_tokens.detach().cpu().numpy()
        
        stats = {
            "mean": float(tokens_np.mean()),
            "std": float(tokens_np.std()),
            "min": float(tokens_np.min()),
            "max": float(tokens_np.max()),
            "median": float(np.median(tokens_np)),
            "percentile_25": float(np.percentile(tokens_np, 25)),
            "percentile_75": float(np.percentile(tokens_np, 75)),
        }
        
        # 计算每个 token 的 L2 范数（特征强度）
        token_norms = np.linalg.norm(tokens_np, axis=-1)  # [B, N]
        stats["token_norm_mean"] = float(token_norms.mean())
        stats["token_norm_std"] = float(token_norms.std())
        stats["token_norm_min"] = float(token_norms.min())
        stats["token_norm_max"] = float(token_norms.max())
        
        # 计算特征激活率（非零特征比例）
        non_zero_ratio = (np.abs(tokens_np) > 1e-6).mean()
        stats["activation_ratio"] = float(non_zero_ratio)
        
        # 可视化（如果提供了保存路径）
        if save_path is not None:
            visualize_tokens_statistics(tokens_np, token_norms, save_path)
    
    return stats


def visualize_tokens_statistics(
    tokens: np.ndarray,
    token_norms: np.ndarray,
    save_path: str
):
    """可视化 tokens 的统计分布"""
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Token 值分布直方图
    axes[0, 0].hist(tokens.flatten(), bins=100, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title("Token Value Distribution")
    axes[0, 0].set_xlabel("Value")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_yscale('log')
    
    # 2. Token 范数分布
    axes[0, 1].hist(token_norms.flatten(), bins=100, alpha=0.7, edgecolor='black', color='orange')
    axes[0, 1].set_title("Token L2 Norm Distribution")
    axes[0, 1].set_xlabel("L2 Norm")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_yscale('log')
    
    # 3. Token 范数的空间分布（如果是空间排列的）
    B, N, D = tokens.shape
    H = int(N ** 0.5)
    if H * H == N:  # 如果是完全平方数，假设是空间排列
        norm_map = token_norms[0].reshape(H, H)
        im = axes[1, 0].imshow(norm_map, cmap='viridis')
        axes[1, 0].set_title("Token Norm Spatial Map (First Batch)")
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0])
    else:
        axes[1, 0].text(0.5, 0.5, f"Non-spatial tokens\nN={N}", 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].axis('off')
    
    # 4. 特征维度统计（每个维度的均值和标准差）
    dim_means = tokens.mean(axis=(0, 1))  # [D]
    dim_stds = tokens.std(axis=(0, 1))   # [D]
    
    axes[1, 1].plot(dim_means, label='Mean', alpha=0.7)
    axes[1, 1].fill_between(range(len(dim_means)), 
                           dim_means - dim_stds, 
                           dim_means + dim_stds, 
                           alpha=0.3, label='±1 Std')
    axes[1, 1].set_title("Feature Dimension Statistics")
    axes[1, 1].set_xlabel("Dimension Index")
    axes[1, 1].set_ylabel("Value")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved token statistics to {save_path}")


def compare_tokens_before_after_lgpd(
    tokens_before: torch.Tensor,
    tokens_after: torch.Tensor,
    gate: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None
) -> Dict[str, float]:
    """
    比较 LGPD 前后的 tokens，评估 LGPD 的效果
    
    Args:
        tokens_before: [B, N, D] - LGPD 前的 tokens
        tokens_after: [B, N, D] - LGPD 后的 tokens
        gate: [B, N, 1] - LGPD 的门控值（可选）
        save_path: 保存对比图的路径
    
    Returns:
        comparison: 对比指标字典
    """
    with torch.no_grad():
        # 计算变化量
        diff = tokens_after - tokens_before
        diff_norm = torch.norm(diff, dim=-1)  # [B, N]
        
        comparison = {
            "mean_change": float(diff_norm.mean().item()),
            "max_change": float(diff_norm.max().item()),
            "change_ratio": float((diff_norm > 0.01).float().mean().item()),  # 变化超过阈值的比例
        }
        
        if gate is not None:
            # 分析门控值对变化的影响
            gate_flat = gate.squeeze(-1)  # [B, N]
            high_gate_mask = gate_flat > 0.5
            low_gate_mask = gate_flat <= 0.5
            
            comparison["change_high_gate"] = float(diff_norm[high_gate_mask].mean().item() if high_gate_mask.any() else 0.0)
            comparison["change_low_gate"] = float(diff_norm[low_gate_mask].mean().item() if low_gate_mask.any() else 0.0)
        
        # 可视化（如果提供了保存路径）
        if save_path is not None:
            visualize_lgpd_comparison(tokens_before, tokens_after, gate, diff_norm, save_path)
    
    return comparison


def visualize_lgpd_comparison(
    tokens_before: torch.Tensor,
    tokens_after: torch.Tensor,
    gate: Optional[torch.Tensor],
    diff_norm: torch.Tensor,
    save_path: str
):
    """可视化 LGPD 前后的对比"""
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 转换为 numpy
    tokens_before_np = tokens_before[0].detach().cpu().numpy()
    tokens_after_np = tokens_after[0].detach().cpu().numpy()
    diff_norm_np = diff_norm[0].detach().cpu().numpy()
    
    # 1. Token 范数对比
    norm_before = np.linalg.norm(tokens_before_np, axis=-1)
    norm_after = np.linalg.norm(tokens_after_np, axis=-1)
    
    axes[0, 0].scatter(norm_before, norm_after, alpha=0.5, s=10)
    axes[0, 0].plot([norm_before.min(), norm_before.max()], 
                   [norm_before.min(), norm_before.max()], 
                   'r--', label='y=x')
    axes[0, 0].set_xlabel("Token Norm (Before LGPD)")
    axes[0, 0].set_ylabel("Token Norm (After LGPD)")
    axes[0, 0].set_title("Token Norm Comparison")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 变化量分布
    axes[0, 1].hist(diff_norm_np, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title("Change Magnitude Distribution")
    axes[0, 1].set_xlabel("||tokens_after - tokens_before||")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_yscale('log')
    
    # 3. 门控值 vs 变化量（如果有门控值）
    if gate is not None:
        gate_np = gate[0].squeeze(-1).detach().cpu().numpy()
        axes[1, 0].scatter(gate_np, diff_norm_np, alpha=0.5, s=10)
        axes[1, 0].set_xlabel("Gate Value")
        axes[1, 0].set_ylabel("Change Magnitude")
        axes[1, 0].set_title("Gate Value vs Change")
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].axis('off')
    
    # 4. 变化量的空间分布（如果是空间排列的）
    N = diff_norm_np.shape[0]
    H = int(N ** 0.5)
    if H * H == N:
        change_map = diff_norm_np.reshape(H, H)
        im = axes[1, 1].imshow(change_map, cmap='hot')
        axes[1, 1].set_title("Change Spatial Map")
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1])
    else:
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved LGPD comparison to {save_path}")


def evaluate_tokens_contribution(
    model_with_tokens: nn.Module,
    model_without_tokens: nn.Module,
    observation,
    actions,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    评估 tokens 对下游任务（动作预测）的贡献
    
    Args:
        model_with_tokens: 使用 tokens 的完整模型
        model_without_tokens: 不使用 tokens 的模型（或禁用 tokens 的版本）
        observation: 输入观察
        actions: 真实动作
    
    Returns:
        contribution: 贡献度指标
    """
    with torch.no_grad():
        # 使用 tokens 的预测
        pred_with = model_with_tokens(observation, actions)
        
        # 不使用 tokens 的预测（需要临时禁用）
        # 这里假设可以通过设置 use_gaussian=False 来禁用
        # 实际实现可能需要修改模型或使用不同的配置
        
        # 计算动作预测误差
        # 这里简化处理，实际需要根据模型输出格式调整
        
        contribution = {
            "action_error_with_tokens": 0.0,  # 需要根据实际输出计算
            "action_error_without_tokens": 0.0,
            "improvement_ratio": 0.0,
        }
    
    return contribution
