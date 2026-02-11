"""
最小改动版 z -> 3DGS 重建 / 渲染脚本
================================================

用途：
- 在不跑训练循环的前提下，直接从数据集中取一批样本；
- 用当前的 `GaussianAdapter` 生成未来帧的 3DGS tokens `z_{t+1}^{GT}`；
- 用 `world_model.decode` 将这些 tokens 解码成 3D 高斯参数；
- 用 `GaussianRenderer` 渲染，并和真实图像对比，可视化 + 简单指标。

典型用法（单机单卡）：

```bash
cd /home/zijianzhang/openpi

CUDA_VISIBLE_DEVICES=0 uv run --active python examples/libero/reconstruct_z_to_3dgs.py \
  --config-name pi05_libero \
  --exp-name gaussian_world_model_exp0210 \
  --checkpoint-base-dir /data/zijianzhang/train_ckpts \
  --step 2000 \
  --num-batches 1 \
  --output-dir visualizations/z_to_3dgs_reconstruction
```
"""

import argparse
import dataclasses
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_model  # noqa: E402


# ---------------------------------------------------------------------------
# 项目路径 & 导入
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import openpi.training.config as _config  # noqa: E402
import openpi.training.data_loader as _data  # noqa: E402
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch  # noqa: E402
from openpi.models_pytorch.pi0_world_model import BiDirectionalWorldModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="z -> 3DGS 重建 / 渲染调试脚本（Libero，pi05_libero）"
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="pi05_libero",
        help="TrainConfig 名称（默认 pi05_libero）",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        required=True,
        help="实验名，对应训练时的 --exp-name，例如 gaussian_world_model_exp0210",
    )
    parser.add_argument(
        "--checkpoint-base-dir",
        type=str,
        default="/data/zijianzhang/train_ckpts",
        help="checkpoint_base_dir（默认为本机训练路径）",
    )
    parser.add_argument(
        "--step",
        type=int,
        required=True,
        help="要加载的 checkpoint step，例如 5000",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=1,
        help="从数据集中评估多少个 batch（每个 batch 只取第 0 个样本可视化）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualizations/z_to_3dgs_reconstruction",
        help="可视化保存目录",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备（cuda / cpu），默认 cuda",
    )
    parser.add_argument(
        "--depth-scales",
        type=str,
        default="1.0",
        help="逗号分隔的全局 depth 缩放因子列表，例如 '1.0,0.5,0.2'",
    )
    parser.add_argument(
        "--scale-gains",
        type=str,
        default="1.0",
        help="逗号分隔的全局 scale 增益列表，例如 '1.0,2.0,5.0'",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> _config.TrainConfig:
    """
    基于已有的 pi05_libero 配置，构造一个新的 TrainConfig。
    注意：TrainConfig 是 frozen dataclass，必须用 dataclasses.replace，而不能直接赋值。
    """
    base_cfg = _config.get_config(args.config_name)

    # 为了评估方便，可以把 batch_size 调小一点
    new_batch_size = min(base_cfg.batch_size, 8)
    new_num_workers = max(1, base_cfg.num_workers)

    cfg = dataclasses.replace(
        base_cfg,
        exp_name=args.exp_name,
        checkpoint_base_dir=args.checkpoint_base_dir,
        batch_size=new_batch_size,
        num_workers=new_num_workers,
        wandb_enabled=False,
        resume=False,
        overwrite=False,
    )

    return cfg


def load_model_from_checkpoint(
    config: _config.TrainConfig, device: torch.device, step: int
) -> Tuple[PI0Pytorch, Path]:
    # 构建模型（与 scripts/train_pytorch.py 中逻辑一致）
    model_cfg = config.model
    # 确保 dtype 与 pytorch_training_precision 一致
    object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    model = PI0Pytorch(model_cfg).to(device)
    model.eval()

    # checkpoint 路径：checkpoint_base_dir / name / exp_name / step
    ckpt_root = Path(config.checkpoint_base_dir) / config.name / config.exp_name
    ckpt_dir = ckpt_root / f"{step}"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint 目录不存在: {ckpt_dir}")

    safetensors_path = ckpt_dir / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"找不到模型权重: {safetensors_path}")

    load_model(model, safetensors_path, device=str(device))
    print(f"[INFO] Loaded model weights from {safetensors_path}")

    return model, ckpt_dir


def get_agent_view_from_future_observation(future_observation) -> torch.Tensor:
    """
    从 future_observation.images 中选出 agent 视角的图像，返回 [B, 3, H, W]，范围 [0,1]。
    逻辑基本复用 PI0Pytorch.forward 里处理 future_observation 的那段。
    """
    target_img = None
    for k, v in future_observation.images.items():
        view_name = None
        if (
            "agent" in k
            or "high" in k
            or "cam_high" in k
            or "exterior" in k
            or "base" in k
        ):
            view_name = "agent"
        if view_name == "agent":
            img_tensor = v  # 可能是 [B, H, W, C] 或 [B, C, H, W]
            if img_tensor.ndim == 4 and img_tensor.shape[-1] == 3:
                # [B, H, W, C] -> [B, C, H, W]
                img_tensor = img_tensor.permute(0, 3, 1, 2)
            # 训练中图像在 [-1,1]，这里统一映射回 [0,1]
            img_tensor = (img_tensor + 1.0) / 2.0
            img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
            target_img = img_tensor
            break

    if target_img is None:
        # fallback：随便拿一个视角
        k, v = next(iter(future_observation.images.items()))
        img_tensor = v
        if img_tensor.ndim == 4 and img_tensor.shape[-1] == 3:
            img_tensor = img_tensor.permute(0, 3, 1, 2)
        target_img = torch.clamp((img_tensor + 1.0) / 2.0, 0.0, 1.0)

    return target_img


def visualize_single_sample(
    gt_img: torch.Tensor,
    rendered_from_gt_tokens: torch.Tensor,
    rendered_from_z: torch.Tensor,
    rendered_bright: torch.Tensor,
    save_path: Path,
) -> None:
    """
    保存单个样本的对比图：
    - GT 图像
    - Encoder+Renderer: 直接用 gaussian_params_t1_gt 渲染
    - Decode(z)+Renderer: 先用 world_model.decode，再渲染
    - Diff: Decode(z) 渲染结果 vs GT 的差异

    额外增加一列：
    - Bright Render (GT 3DGS, bright)：在相同几何下，把 opacity/DC SH 调亮后的渲染结果，用于区分“几何/相机问题”和“radiance 太暗”。
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    gt_np = gt_img.permute(1, 2, 0).detach().cpu().numpy()
    recon_gt_np = rendered_from_gt_tokens.permute(1, 2, 0).detach().cpu().numpy()
    recon_z_np = rendered_from_z.permute(1, 2, 0).detach().cpu().numpy()
    bright_np = rendered_bright.permute(1, 2, 0).detach().cpu().numpy()
    gt_np = np.clip(gt_np, 0.0, 1.0)
    recon_gt_np = np.clip(recon_gt_np, 0.0, 1.0)
    recon_z_np = np.clip(recon_z_np, 0.0, 1.0)
    bright_np = np.clip(bright_np, 0.0, 1.0)
    diff_np = np.abs(recon_z_np - gt_np)

    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    axes[0].imshow(gt_np)
    axes[0].set_title("GT (t+1)")
    axes[0].axis("off")

    axes[1].imshow(recon_gt_np)
    axes[1].set_title("Render (GT 3DGS)")
    axes[1].axis("off")

    axes[2].imshow(recon_z_np)
    axes[2].set_title("Render (decode z)")
    axes[2].axis("off")

    axes[3].imshow(bright_np)
    axes[3].set_title("Bright Render (GT 3DGS)")
    axes[3].axis("off")

    axes[4].imshow(diff_np, cmap="hot")
    axes[4].set_title("Diff (decode z vs GT)")
    axes[4].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved reconstruction comparison to {save_path}")


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = build_config(args)

    # 构建数据加载器（与训练一致，只是不开 shuffle）
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)

    # 构建模型并加载 checkpoint
    model, ckpt_dir = load_model_from_checkpoint(config, device, step=args.step)

    if not hasattr(model, "world_model") or model.world_model is None:
        raise RuntimeError("当前 PI0Pytorch 实例未启用 world_model，无法进行 z->3DGS 解码调试。")
    if not hasattr(model, "gaussian_renderer") or model.gaussian_renderer is None:
        raise RuntimeError("当前 PI0Pytorch 实例未启用 GaussianRenderer。")

    model.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_iter = iter(data_loader)

    import jax  # 与训练保持一致，使用 jax.tree.map 搬运 Observation

    with torch.no_grad():
        for batch_idx in range(args.num_batches):
            try:
                observation, actions = next(batch_iter)
            except StopIteration:
                print("[INFO] 数据不足，提前结束。")
                break

            # 与训练保持一致的迁移到 device 的方式：用 jax.tree.map 把 Observation 内部的 tensor 搬到 GPU
            observation = jax.tree.map(lambda x: x.to(device), observation)  # type: ignore[arg-type]
            actions = actions.to(torch.float32).to(device)

            # 预处理 observation，拿到 future_observation
            (
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                future_observation,
                preprocessed_observation,
            ) = model._preprocess_observation(observation, train=False)

            B = actions.shape[0]

            # 1) 用 GaussianAdapter 在未来帧上得到 z_{t+1}^{GT}
            gaussian_inputs_t1 = model._prepare_gaussian_inputs(
                future_observation, actions.device, B, is_training=False
            )

            adapter_output_t1 = model.gaussian_adapter(
                gaussian_inputs_t1,
                return_gaussian_params=True,
                return_raw_tokens=False,
            )
            z_t1_gt, _, gaussian_params_t1_gt = adapter_output_t1  # z_t1_gt: [B, N, D]

            # 2) 相机参数（两条路径复用）
            camera_params_agent = model._get_camera_params_for_view("agent", actions.device, B)

            # 2a) Encoder+Renderer: 直接将 VGGT 的 2D maps 转成 3DGS，再渲染
            depth_maps = gaussian_params_t1_gt["depth_maps"]
            rot_maps = gaussian_params_t1_gt["rot_maps"]
            scale_maps = gaussian_params_t1_gt["scale_maps"]
            opacity_maps = gaussian_params_t1_gt["opacity_maps"]
            sh_maps = gaussian_params_t1_gt["sh_maps"]

            # 构造一个最小的 BiDirectionalWorldModel 实例，只用它的 _convert_2d_maps_to_3d_gaussians 工具函数
            temp_wm = BiDirectionalWorldModel(
                token_dim=model.world_model.token_dim,
                action_dim=model.world_model.action_dim,
                use_vggt_decoder=model.world_model.use_vggt_decoder,
                vggt_decoder=model.world_model.vggt_decoder,
                input_num_tokens=model.world_model.input_num_tokens,
                target_num_tokens=model.world_model.target_num_tokens,
                vggt_embed_dim=getattr(model.world_model, "vggt_embed_dim", 1024),
            ).to(actions.device)
            temp_wm.eval()

            z_t1_gt_fp32 = z_t1_gt.to(dtype=torch.float32)
            gaussian_params_from_z = model.world_model.decode(
                z_t1_gt_fp32,
                future_observation=future_observation,
                gaussian_adapter=model.gaussian_adapter,
                camera_params=camera_params_agent,
                return_2d_maps=False,
                step=args.step,
            )

            # 解析 sweep 参数
            depth_scales = [float(x) for x in args.depth_scales.split(",")]
            scale_gains = [float(x) for x in args.scale_gains.split(",")]

            gt_agent = get_agent_view_from_future_observation(future_observation)[0]  # [3, H, W]

            for ds in depth_scales:
                for sg in scale_gains:
                    # 3) 使用 GaussianRenderer 渲染（Encoder+Renderer 路径，带 depth/scale sweep）
                    gaussian_params_gt_3d = temp_wm._convert_2d_maps_to_3d_gaussians(
                        depth_maps,
                        rot_maps,
                        scale_maps,
                        opacity_maps,
                        sh_maps,
                        gaussian_inputs_t1,
                        camera_params=camera_params_agent,
                        downsample_factor=4,
                        depth_scale=ds,
                        scale_gain=sg,
                        step=args.step,
                    )

                    rendered_from_gt_tokens = model.gaussian_renderer(
                        gaussian_params_gt_3d,
                        camera_params_agent,
                        step=args.step,
                    )  # [B, 3, H, W]

                    rendered_from_z = model.gaussian_renderer(
                        gaussian_params_from_z,
                        camera_params_agent,
                        step=args.step,
                    )  # [B, 3, H, W]

                    # Bright 渲染测试：只改 radiance，保持几何与相机不变
                    gaussian_params_bright = {
                        "xyz": gaussian_params_gt_3d["xyz"],
                        "sigma": gaussian_params_gt_3d["sigma"],
                        "opacity": torch.ones_like(gaussian_params_gt_3d["opacity"]),
                        "sh": torch.zeros_like(gaussian_params_gt_3d["sh"]),
                    }
                    gaussian_params_bright["sh"][:, :, :3] = 0.7  # RGB DC term

                    rendered_bright = model.gaussian_renderer(
                        gaussian_params_bright,
                        camera_params_agent,
                        step=args.step,
                    )  # [B, 3, H, W]

                    # 取 batch 中第 0 个样本做可视化
                    recon_from_gt = rendered_from_gt_tokens[0].detach().clamp(0.0, 1.0)
                    recon_from_z = rendered_from_z[0].detach().clamp(0.0, 1.0)
                    recon_bright = rendered_bright[0].detach().clamp(0.0, 1.0)

                    # 简单指标：两条路径各算一份
                    mse_gt = torch.mean((recon_from_gt - gt_agent) ** 2).item()
                    mae_gt = torch.mean(torch.abs(recon_from_gt - gt_agent)).item()
                    mse_gt_clamped = max(mse_gt, 1e-10)
                    psnr_gt = -10.0 * np.log10(mse_gt_clamped)

                    mse_z = torch.mean((recon_from_z - gt_agent) ** 2).item()
                    mae_z = torch.mean(torch.abs(recon_from_z - gt_agent)).item()
                    mse_z_clamped = max(mse_z, 1e-10)
                    psnr_z = -10.0 * np.log10(mse_z_clamped)

                    print(
                        f"[Batch {batch_idx}] ds={ds}, sg={sg} | "
                        f"Encoder+Renderer: MSE={mse_gt:.6f}, MAE={mae_gt:.6f}, PSNR={psnr_gt:.2f} dB | "
                        f"Decode(z)+Renderer: MSE={mse_z:.6f}, MAE={mae_z:.6f}, PSNR={psnr_z:.2f} dB "
                        f"(checkpoint step={args.step}, ckpt_dir={ckpt_dir})"
                    )

                    save_path = out_dir / f"recon_step_{args.step:06d}_ds_{ds:g}_sg_{sg:g}_batch_{batch_idx:03d}.png"
                    visualize_single_sample(
                        gt_agent, recon_from_gt, recon_from_z, recon_bright, save_path
                    )

            # 4) 额外：人造高斯球云测试 renderer（与真实相机参数配合）
            # 只在第一个 batch 上做一次即可
            if batch_idx == 0:
                # 取 batch 中第 0 个相机参数
                cam_params_single = {}
                for k, v in camera_params_agent.items():
                    if isinstance(v, torch.Tensor):
                        cam_params_single[k] = v[:1].to(device)
                    else:
                        cam_params_single[k] = v

                viewmatrix = cam_params_single["viewmatrix"][0]  # [4,4]
                world_from_cam = torch.inverse(viewmatrix)  # [4,4]
                R = world_from_cam[:3, :3]  # [3,3]
                t = world_from_cam[:3, 3]   # [3]

                # 在相机坐标系下构造一个 z≈5 的小平面点云 [-1,1]^2
                grid_size = 32
                xs = torch.linspace(-1.0, 1.0, grid_size, device=device)
                ys = torch.linspace(-1.0, 1.0, grid_size, device=device)
                xv, yv = torch.meshgrid(xs, ys, indexing="xy")
                zv = torch.full_like(xv, 5.0)
                cam_xyz = torch.stack([xv, yv, zv], dim=-1).view(-1, 3)  # [N,3]

                # 转到世界坐标：x_world = R * x_cam + t
                world_xyz = cam_xyz @ R.transpose(0, 1) + t.unsqueeze(0)  # [N,3]
                world_xyz = world_xyz.unsqueeze(0)  # [1, N, 3]

                N_synth = world_xyz.shape[1]
                # 设一个中等半径 r
                r = 0.3
                sigma = torch.zeros(1, N_synth, 6, device=device, dtype=world_xyz.dtype)
                sigma[:, :, 0] = r * r
                sigma[:, :, 3] = r * r
                sigma[:, :, 5] = r * r

                opacity = torch.ones(1, N_synth, 1, device=device, dtype=world_xyz.dtype)

                sh_degree = model.gaussian_renderer.sh_degree
                num_coeffs = (sh_degree + 1) ** 2
                sh = torch.zeros(1, N_synth, num_coeffs * 3, device=device, dtype=world_xyz.dtype)
                # 明亮的白色 DC 项
                sh[:, :, :3] = 0.8

                gaussian_params_synth = {
                    "xyz": world_xyz,
                    "sigma": sigma,
                    "opacity": opacity,
                    "sh": sh,
                }

                rendered_synth = model.gaussian_renderer(
                    gaussian_params_synth,
                    cam_params_single,
                    step=args.step,
                )  # [1,3,H,W]

                synth_img = rendered_synth[0].detach().clamp(0.0, 1.0)
                synth_np = synth_img.permute(1, 2, 0).cpu().numpy()

                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                ax.imshow(synth_np)
                ax.set_title("Synthetic Gaussians (camera space z≈5)")
                ax.axis("off")
                synth_path = out_dir / f"synth_gaussians_step_{args.step:06d}_batch_{batch_idx:03d}.png"
                plt.tight_layout()
                plt.savefig(synth_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"[INFO] Saved synthetic Gaussian render to {synth_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)

