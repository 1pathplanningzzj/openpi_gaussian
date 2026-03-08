#!/usr/bin/env python3
"""
诊断 3D Gaussian 渲染质量问题
分析 Gaussian 参数分布，找出渲染模糊的根本原因
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_gaussian_params(gaussian_params, step=0, save_dir="./visualizations/diagnostics"):
    """
    详细分析 Gaussian 参数分布

    Args:
        gaussian_params: Dict with keys: xyz, scales, opacity, sh, rotations
        step: Training step number
        save_dir: Directory to save diagnostic plots
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Extract parameters
    xyz = gaussian_params["xyz"]  # [B, N, 3]
    scales = gaussian_params["scales"]  # [B, N, 3]
    opacity = gaussian_params["opacity"]  # [B, N, 1]
    sh = gaussian_params["sh"]  # [B, N, 9]
    rotations = gaussian_params["rotations"]  # [B, N, 4]

    B, N, _ = xyz.shape

    # Move to CPU for analysis
    xyz_np = xyz.detach().cpu().numpy()
    scales_np = scales.detach().cpu().numpy()
    opacity_np = opacity.detach().cpu().numpy()
    sh_np = sh.detach().cpu().numpy()

    print(f"\n{'='*80}")
    print(f"Gaussian Parameters Diagnostic Report - Step {step}")
    print(f"{'='*80}")
    print(f"Batch size: {B}, Number of Gaussians: {N}")

    # Analyze each batch
    for b in range(min(B, 2)):  # Only analyze first 2 batches
        print(f"\n--- Batch {b} ---")

        # 1. XYZ Distribution
        xyz_b = xyz_np[b]
        print(f"\n1. XYZ (3D Position):")
        print(f"   Range: X=[{xyz_b[:, 0].min():.3f}, {xyz_b[:, 0].max():.3f}], "
              f"Y=[{xyz_b[:, 1].min():.3f}, {xyz_b[:, 1].max():.3f}], "
              f"Z=[{xyz_b[:, 2].min():.3f}, {xyz_b[:, 2].max():.3f}]")
        print(f"   Mean: X={xyz_b[:, 0].mean():.3f}, Y={xyz_b[:, 1].mean():.3f}, Z={xyz_b[:, 2].mean():.3f}")
        print(f"   Std: X={xyz_b[:, 0].std():.3f}, Y={xyz_b[:, 1].std():.3f}, Z={xyz_b[:, 2].std():.3f}")

        # Check for degenerate positions
        zero_xyz = np.all(xyz_b == 0, axis=1).sum()
        print(f"   Degenerate (all zeros): {zero_xyz}/{N} ({100*zero_xyz/N:.1f}%)")

        # 2. Scales Distribution (CRITICAL for rendering quality)
        scales_b = scales_np[b]
        print(f"\n2. Scales (Gaussian Size) - CRITICAL:")
        print(f"   Range: [{scales_b.min():.6f}, {scales_b.max():.6f}]")
        print(f"   Mean: {scales_b.mean():.6f}, Median: {np.median(scales_b):.6f}")
        print(f"   Std: {scales_b.std():.6f}")

        # Categorize scales
        tiny_scales = (scales_b < 0.001).sum()
        small_scales = ((scales_b >= 0.001) & (scales_b < 0.01)).sum()
        medium_scales = ((scales_b >= 0.01) & (scales_b < 0.1)).sum()
        large_scales = (scales_b >= 0.1).sum()

        print(f"   Distribution:")
        print(f"     Tiny (<0.001):   {tiny_scales}/{N*3} ({100*tiny_scales/(N*3):.1f}%)")
        print(f"     Small [0.001-0.01): {small_scales}/{N*3} ({100*small_scales/(N*3):.1f}%)")
        print(f"     Medium [0.01-0.1): {medium_scales}/{N*3} ({100*medium_scales/(N*3):.1f}%)")
        print(f"     Large (>=0.1):   {large_scales}/{N*3} ({100*large_scales/(N*3):.1f}%)")

        if scales_b.mean() < 0.01:
            print(f"   ⚠️  WARNING: Scales are very small (mean={scales_b.mean():.6f})")
            print(f"       This will cause poor rendering coverage!")

        # 3. Opacity Distribution (CRITICAL for visibility)
        opacity_b = opacity_np[b]
        print(f"\n3. Opacity (Visibility) - CRITICAL:")
        print(f"   Range: [{opacity_b.min():.6f}, {opacity_b.max():.6f}]")
        print(f"   Mean: {opacity_b.mean():.6f}, Median: {np.median(opacity_b):.6f}")

        # Categorize opacity
        invisible = (opacity_b < 0.01).sum()
        faint = ((opacity_b >= 0.01) & (opacity_b < 0.1)).sum()
        visible = ((opacity_b >= 0.1) & (opacity_b < 0.5)).sum()
        opaque = (opacity_b >= 0.5).sum()

        print(f"   Distribution:")
        print(f"     Invisible (<0.01):  {invisible}/{N} ({100*invisible/N:.1f}%)")
        print(f"     Faint [0.01-0.1):   {faint}/{N} ({100*faint/N:.1f}%)")
        print(f"     Visible [0.1-0.5):  {visible}/{N} ({100*visible/N:.1f}%)")
        print(f"     Opaque (>=0.5):     {opaque}/{N} ({100*opaque/N:.1f}%)")

        if opacity_b.mean() < 0.1:
            print(f"   ⚠️  WARNING: Opacity is very low (mean={opacity_b.mean():.6f})")
            print(f"       Most Gaussians are nearly transparent!")

        # 4. Spherical Harmonics (Color)
        sh_b = sh_np[b]
        sh_dc = sh_b[:, :3]  # DC coefficients (base color)
        sh_higher = sh_b[:, 3:]  # Higher order coefficients

        print(f"\n4. Spherical Harmonics (Appearance):")
        print(f"   DC (base color): Range=[{sh_dc.min():.6f}, {sh_dc.max():.6f}], Mean={sh_dc.mean():.6f}")
        print(f"   Higher order: Range=[{sh_higher.min():.6f}, {sh_higher.max():.6f}], Mean={sh_higher.mean():.6f}")

        if abs(sh_dc.mean()) < 0.01:
            print(f"   ⚠️  WARNING: SH DC coefficients are very small (mean={sh_dc.mean():.6f})")
            print(f"       This will result in dark/gray rendering!")

        # 5. Effective Gaussians (visible + reasonable size)
        effective_mask = (opacity_b.squeeze() > 0.05) & (scales_b.min(axis=1) > 0.001)
        effective_count = effective_mask.sum()
        print(f"\n5. Effective Gaussians (opacity>0.05 AND min_scale>0.001):")
        print(f"   Count: {effective_count}/{N} ({100*effective_count/N:.1f}%)")

        if effective_count < N * 0.1:
            print(f"   ⚠️  WARNING: Very few effective Gaussians ({effective_count}/{N})!")
            print(f"       Most Gaussians are too small or too transparent to render!")

    # Create diagnostic plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Gaussian Parameters Distribution - Step {step}", fontsize=16)

    b = 0  # Plot first batch

    # Plot 1: Scales histogram
    ax = axes[0, 0]
    scales_flat = scales_np[b].flatten()
    ax.hist(scales_flat, bins=100, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Scale Value')
    ax.set_ylabel('Count')
    ax.set_title(f'Scales Distribution (mean={scales_flat.mean():.6f})')
    ax.axvline(scales_flat.mean(), color='r', linestyle='--', label=f'Mean={scales_flat.mean():.6f}')
    ax.legend()
    ax.set_yscale('log')

    # Plot 2: Opacity histogram
    ax = axes[0, 1]
    opacity_flat = opacity_np[b].flatten()
    ax.hist(opacity_flat, bins=100, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Opacity Value')
    ax.set_ylabel('Count')
    ax.set_title(f'Opacity Distribution (mean={opacity_flat.mean():.6f})')
    ax.axvline(opacity_flat.mean(), color='r', linestyle='--', label=f'Mean={opacity_flat.mean():.6f}')
    ax.legend()

    # Plot 3: SH DC histogram
    ax = axes[0, 2]
    sh_dc_flat = sh_np[b, :, :3].flatten()
    ax.hist(sh_dc_flat, bins=100, edgecolor='black', alpha=0.7)
    ax.set_xlabel('SH DC Value')
    ax.set_ylabel('Count')
    ax.set_title(f'SH DC Distribution (mean={sh_dc_flat.mean():.6f})')
    ax.axvline(sh_dc_flat.mean(), color='r', linestyle='--', label=f'Mean={sh_dc_flat.mean():.6f}')
    ax.legend()

    # Plot 4: XYZ scatter (top view)
    ax = axes[1, 0]
    xyz_b = xyz_np[b]
    scatter = ax.scatter(xyz_b[:, 0], xyz_b[:, 1], c=opacity_np[b].flatten(),
                        s=1, cmap='viridis', alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('XYZ Top View (colored by opacity)')
    plt.colorbar(scatter, ax=ax, label='Opacity')

    # Plot 5: Scale vs Opacity scatter
    ax = axes[1, 1]
    mean_scales = scales_np[b].mean(axis=1)
    ax.scatter(mean_scales, opacity_np[b].flatten(), s=1, alpha=0.3)
    ax.set_xlabel('Mean Scale')
    ax.set_ylabel('Opacity')
    ax.set_title('Scale vs Opacity')
    ax.set_xscale('log')

    # Plot 6: Effective Gaussians
    ax = axes[1, 2]
    categories = ['Invisible\n(<0.01)', 'Faint\n(0.01-0.1)', 'Visible\n(0.1-0.5)', 'Opaque\n(>=0.5)']
    counts = [invisible, faint, visible, opaque]
    colors = ['red', 'orange', 'yellow', 'green']
    ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Count')
    ax.set_title('Opacity Categories')
    for i, (cat, count) in enumerate(zip(categories, counts)):
        ax.text(i, count, f'{count}\n({100*count/N:.1f}%)',
               ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    save_path = Path(save_dir) / f"gaussian_diagnostic_step_{step:06d}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n{'='*80}")
    print(f"Diagnostic plot saved to: {save_path}")
    print(f"{'='*80}\n")

    return {
        "mean_scale": float(scales_np.mean()),
        "mean_opacity": float(opacity_np.mean()),
        "mean_sh_dc": float(sh_np[:, :, :3].mean()),
        "effective_ratio": float(effective_count / N),
    }


if __name__ == "__main__":
    print("This is a diagnostic utility module.")
    print("Import and call analyze_gaussian_params(gaussian_params, step) in your training code.")
