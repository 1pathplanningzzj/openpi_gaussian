"""
评估 3DGS Tokens 效果的示例脚本

使用方法：
1. 在训练过程中评估 tokens 质量
2. 可视化 tokens 的统计特性
3. 比较 LGPD 前后的 tokens
4. 评估重建质量
"""

import torch
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.openpi.models_pytorch.evaluate_gaussian_tokens import (
    evaluate_tokens_reconstruction,
    analyze_tokens_statistics,
    compare_tokens_before_after_lgpd
)
from src.openpi.models_pytorch.pi0_pytorch import PI0Pytorch


def evaluate_tokens_during_training(
    model: PI0Pytorch,
    observation,
    future_observation,
    actions,
    step: int,
    save_dir: str = "./evaluations/gaussian_tokens"
):
    """
    在训练过程中评估 3DGS tokens 的质量
    
    Args:
        model: 训练中的模型
        observation: 当前观察
        future_observation: 未来观察（用于重建评估）
        actions: 动作
        step: 当前训练步数
        save_dir: 保存评估结果的目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if not model.gaussian_adapter.use_gaussian:
        print("Gaussian adapter is not enabled. Skipping token evaluation.")
        return
    
    device = next(model.parameters()).device
    
    # 1. 获取 3DGS tokens
    gaussian_inputs = model._prepare_gaussian_inputs(observation, device, observation.images[list(observation.images.keys())[0]].shape[0])
    
    with torch.no_grad():
        # 获取 tokens（不经过 LGPD）
        # 注意：这里需要访问内部状态，可能需要修改 GaussianAdapter 来暴露中间结果
        gaussian_embs, _ = model.gaussian_adapter(gaussian_inputs, text_embedding=None)
        
        if gaussian_embs is None:
            print("No Gaussian tokens extracted. Skipping evaluation.")
            return
        
        # 2. 分析 tokens 统计特性
        stats = analyze_tokens_statistics(
            gaussian_embs,
            save_path=os.path.join(save_dir, f"token_stats_step_{step:06d}.png")
        )
        
        print(f"\n[Step {step}] Token Statistics:")
        print(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
        print(f"  Token Norm - Mean: {stats['token_norm_mean']:.4f}, Max: {stats['token_norm_max']:.4f}")
        print(f"  Activation Ratio: {stats['activation_ratio']:.4f}")
        
        # 3. 如果有 World Model，评估重建质量
        if model.world_model is not None and model.gaussian_renderer is not None:
            # 准备相机参数
            cam_params_dict = {}
            for view_name in ["agent"]:  # 只评估 agent view
                cam_params_dict[view_name] = model._get_camera_params_for_view(
                    view_name, device, gaussian_embs.shape[0]
                )
            
            # 准备目标图像
            target_obs = {}
            for k, v in future_observation.images.items():
                if "agent" in k or "high" in k:
                    img_tensor = v
                    if img_tensor.shape[-1] == 3:
                        img_tensor = img_tensor.permute(0, 3, 1, 2)
                    img_tensor = (img_tensor + 1.0) / 2.0  # [-1, 1] -> [0, 1]
                    target_obs["agent_image"] = img_tensor
                    break
            
            if "agent_image" in target_obs:
                # 评估重建质量
                metrics = evaluate_tokens_reconstruction(
                    gaussian_embs,
                    model.world_model.decoder,
                    model.gaussian_renderer,
                    cam_params_dict["agent"],
                    target_obs["agent_image"],
                    save_path=os.path.join(save_dir, f"reconstruction_step_{step:06d}.png")
                )
                
                print(f"\n[Step {step}] Reconstruction Quality:")
                print(f"  MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}")
                print(f"  PSNR: {metrics['psnr']:.2f} dB")
        
        # 4. 如果有 LGPD，比较前后的 tokens
        if model.gaussian_adapter.use_lgpd and model.gaussian_adapter.lgpd is not None:
            # 获取语言嵌入
            lang_tokens = observation.language_tokens
            lang_masks = observation.language_masks
            
            # 这里需要重新计算以获取 LGPD 前后的 tokens
            # 由于 GaussianAdapter 的内部实现，可能需要修改代码来暴露中间结果
            # 这里提供一个示例框架
            
            print(f"\n[Step {step}] LGPD Analysis:")
            print("  Note: LGPD comparison requires access to intermediate tokens.")
            print("  Consider modifying GaussianAdapter.forward() to return tokens before/after LGPD.")


def evaluate_tokens_in_inference(
    model: PI0Pytorch,
    observation,
    save_dir: str = "./evaluations/gaussian_tokens_inference"
):
    """
    在推理时评估 tokens（不需要 future_observation）
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if not model.gaussian_adapter.use_gaussian:
        print("Gaussian adapter is not enabled.")
        return
    
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        gaussian_inputs = model._prepare_gaussian_inputs(observation, device, 1)
        gaussian_embs, _ = model.gaussian_adapter(gaussian_inputs, text_embedding=None)
        
        if gaussian_embs is None:
            print("No Gaussian tokens extracted.")
            return
        
        # 分析统计特性
        stats = analyze_tokens_statistics(
            gaussian_embs,
            save_path=os.path.join(save_dir, "token_stats_inference.png")
        )
        
        print("Token Statistics (Inference):")
        for key, value in stats.items():
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    print("This is an example script for evaluating 3DGS tokens.")
    print("Import these functions in your training script to use them.")
    print("\nExample usage:")
    print("  from examples.evaluate_gaussian_tokens_example import evaluate_tokens_during_training")
    print("  evaluate_tokens_during_training(model, observation, future_obs, actions, step=100)")
