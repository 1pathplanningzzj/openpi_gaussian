"""Analysis script: Verify hypothesis that 2D images cause inaccurate 3D spatial information

Usage: 
python examples/libero/analyze_spatial_hypothesis.py --csv data/libero_spatial_vis_3d_aware/eval_metrics.csv
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pathlib

def analyze_spatial_alignment_hypothesis(csv_path):
    """Quick verification of spatial alignment hypothesis"""
    df = pd.read_csv(csv_path)
    
    # Filter out invalid data
    df = df[df['success'].notna()]
    
    success = df[df['success'] == 1]
    failure = df[df['success'] == 0]
    
    print("=" * 80)
    print("Spatial Alignment Hypothesis Verification Report")
    print("=" * 80)
    print(f"\nTotal trials: {len(df)}")
    print(f"Success cases: {len(success)} ({len(success)/len(df)*100:.1f}%)")
    print(f"Failure cases: {len(failure)} ({len(failure)/len(df)*100:.1f}%)")
    
    # 1. Alignment comparison
    if 'avg_alignment' in df.columns:
        success_align = success['avg_alignment'].dropna()
        failure_align = failure['avg_alignment'].dropna()
        
        if len(success_align) > 0 and len(failure_align) > 0:
            t_stat, p_val = stats.ttest_ind(success_align, failure_align)
            print(f"\n[Hypothesis 1] Alignment Difference:")
            print(f"  Success avg alignment: {success_align.mean():.3f} ± {success_align.std():.3f}")
            print(f"  Failure avg alignment: {failure_align.mean():.3f} ± {failure_align.std():.3f}")
            print(f"  T-test p-value: {p_val:.4f}")
            print(f"  Conclusion: {'✓ Hypothesis supported - failures have significantly lower alignment' if p_val < 0.05 else '✗ No significant difference found'}")
    
    # 2. Alignment when close to target
    if 'avg_alignment_close' in df.columns:
        success_close = success['avg_alignment_close'].dropna()
        failure_close = failure['avg_alignment_close'].dropna()
        
        if len(success_close) > 0 and len(failure_close) > 0:
            t_stat, p_val = stats.ttest_ind(success_close, failure_close)
            print(f"\n[Hypothesis 2] Alignment when close to target (<10cm):")
            print(f"  Success: {success_close.mean():.3f} ± {success_close.std():.3f}")
            print(f"  Failure: {failure_close.mean():.3f} ± {failure_close.std():.3f}")
            print(f"  T-test p-value: {p_val:.4f}")
            print(f"  Conclusion: {'✓ Hypothesis supported' if p_val < 0.05 else '✗ No significant difference found'}")
    
    # 3. Warning frequency comparison
    if 'num_misalignment_warnings' in df.columns:
        print(f"\n[Hypothesis 3] Misalignment Warning Frequency:")
        print(f"  Success avg warnings: {success['num_misalignment_warnings'].mean():.2f}")
        print(f"  Failure avg warnings: {failure['num_misalignment_warnings'].mean():.2f}")
        t_stat, p_val = stats.ttest_ind(
            success['num_misalignment_warnings'],
            failure['num_misalignment_warnings']
        )
        print(f"  T-test p-value: {p_val:.4f}")
        print(f"  Conclusion: {'✓ Failures have more warnings' if p_val < 0.05 and failure['num_misalignment_warnings'].mean() > success['num_misalignment_warnings'].mean() else '✗ No significant difference found'}")
    
    # 4. Angle error comparison
    if 'avg_angle_error_deg' in df.columns:
        success_angle = success['avg_angle_error_deg'].dropna()
        failure_angle = failure['avg_angle_error_deg'].dropna()
        
        if len(success_angle) > 0 and len(failure_angle) > 0:
            t_stat, p_val = stats.ttest_ind(success_angle, failure_angle)
            print(f"\n[Hypothesis 4] Angle Error (degrees):")
            print(f"  Success: {success_angle.mean():.2f}° ± {success_angle.std():.2f}°")
            print(f"  Failure: {failure_angle.mean():.2f}° ± {failure_angle.std():.2f}°")
            print(f"  T-test p-value: {p_val:.4f}")
            print(f"  Conclusion: {'✓ Failures have larger angle error' if p_val < 0.05 and failure_angle.mean() > success_angle.mean() else '✗ No significant difference found'}")
    
    # 5. Target visibility ratio
    if 'target_in_view_ratio' in df.columns:
        success_view = success['target_in_view_ratio'].dropna()
        failure_view = failure['target_in_view_ratio'].dropna()
        
        if len(success_view) > 0 and len(failure_view) > 0:
            print(f"\n[Hypothesis 5] Target in View Ratio:")
            print(f"  Success: {success_view.mean():.3f} ± {success_view.std():.3f}")
            print(f"  Failure: {failure_view.mean():.3f} ± {failure_view.std():.3f}")
            t_stat, p_val = stats.ttest_ind(success_view, failure_view)
            print(f"  T-test p-value: {p_val:.4f}")
            print(f"  Conclusion: {'✓ Significant difference found' if p_val < 0.05 else '✗ No significant difference found'}")
    
    print("\n" + "=" * 80)
    
    return df, success, failure


def plot_analysis(df, success, failure, output_dir):
    """Generate visualization plots"""
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # 1. Alignment distribution comparison
    if 'avg_alignment' in df.columns:
        ax = axes[0]
        success_align = success['avg_alignment'].dropna()
        failure_align = failure['avg_alignment'].dropna()
        if len(success_align) > 0 and len(failure_align) > 0:
            ax.hist(success_align, alpha=0.5, label='Success', bins=20, color='green')
            ax.hist(failure_align, alpha=0.5, label='Failure', bins=20, color='red')
            ax.set_xlabel('Average Alignment (Cosine Similarity)')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.set_title('Alignment Distribution Comparison')
            ax.axvline(success_align.mean(), color='green', linestyle='--', alpha=0.7, label=f'Success mean: {success_align.mean():.3f}')
            ax.axvline(failure_align.mean(), color='red', linestyle='--', alpha=0.7, label=f'Failure mean: {failure_align.mean():.3f}')
            ax.legend()
    
    # 2. Alignment when close to target
    if 'avg_alignment_close' in df.columns:
        ax = axes[1]
        success_close = success['avg_alignment_close'].dropna()
        failure_close = failure['avg_alignment_close'].dropna()
        if len(success_close) > 0 and len(failure_close) > 0:
            ax.hist(success_close, alpha=0.5, label='Success', bins=20, color='green')
            ax.hist(failure_close, alpha=0.5, label='Failure', bins=20, color='red')
            ax.set_xlabel('Alignment when Close to Target')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.set_title('Alignment Comparison when Close (<10cm)')
    
    # 3. Misalignment warnings
    if 'num_misalignment_warnings' in df.columns:
        ax = axes[2]
        ax.hist(success['num_misalignment_warnings'], alpha=0.5, label='Success', bins=20, color='green')
        ax.hist(failure['num_misalignment_warnings'], alpha=0.5, label='Failure', bins=20, color='red')
        ax.set_xlabel('Number of Misalignment Warnings')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.set_title('Misalignment Warning Distribution')
    
    # 4. Angle error
    if 'avg_angle_error_deg' in df.columns:
        ax = axes[3]
        success_angle = success['avg_angle_error_deg'].dropna()
        failure_angle = failure['avg_angle_error_deg'].dropna()
        if len(success_angle) > 0 and len(failure_angle) > 0:
            ax.hist(success_angle, alpha=0.5, label='Success', bins=20, color='green')
            ax.hist(failure_angle, alpha=0.5, label='Failure', bins=20, color='red')
            ax.set_xlabel('Average Angle Error (degrees)')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.set_title('Angle Error Distribution Comparison')
    
    # 5. Alignment vs Success rate scatter plot
    if 'avg_alignment' in df.columns:
        ax = axes[4]
        ax.scatter(success['avg_alignment'], [1]*len(success), alpha=0.3, label='Success', color='green', s=50)
        ax.scatter(failure['avg_alignment'], [0]*len(failure), alpha=0.3, label='Failure', color='red', s=50)
        ax.set_xlabel('Average Alignment')
        ax.set_ylabel('Success (1) / Failure (0)')
        ax.legend()
        ax.set_title('Alignment vs Success Rate')
        ax.set_ylim(-0.1, 1.1)
    
    # 6. Distance vs Alignment (if data available)
    if 'avg_true_distance' in df.columns and 'avg_alignment' in df.columns:
        ax = axes[5]
        success_dist = success['avg_true_distance'].dropna()
        failure_dist = failure['avg_true_distance'].dropna()
        success_align = success['avg_alignment'].dropna()
        failure_align = failure['avg_alignment'].dropna()
        
        if len(success_dist) > 0 and len(success_align) > 0:
            # Ensure length matches
            min_len = min(len(success_dist), len(success_align))
            ax.scatter(success_dist[:min_len], success_align[:min_len], alpha=0.3, label='Success', color='green', s=50)
        
        if len(failure_dist) > 0 and len(failure_align) > 0:
            min_len = min(len(failure_dist), len(failure_align))
            ax.scatter(failure_dist[:min_len], failure_align[:min_len], alpha=0.3, label='Failure', color='red', s=50)
        
        ax.set_xlabel('Average True Distance (m)')
        ax.set_ylabel('Average Alignment')
        ax.legend()
        ax.set_title('Distance vs Alignment Relationship')
        ax.axvline(0.10, color='gray', linestyle='--', alpha=0.5, label='10cm threshold')
        ax.legend()
    
    plt.tight_layout()
    output_file = output_path / 'spatial_alignment_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze spatial alignment hypothesis')
    parser.add_argument('--csv', type=str, required=True, help='CSV file path')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: same as CSV directory)')
    args = parser.parse_args()
    
    csv_path = pathlib.Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV file does not exist: {csv_path}")
        return
    
    output_dir = args.output_dir if args.output_dir else csv_path.parent
    
    df, success, failure = analyze_spatial_alignment_hypothesis(csv_path)
    plot_analysis(df, success, failure, output_dir)


if __name__ == '__main__':
    main()
