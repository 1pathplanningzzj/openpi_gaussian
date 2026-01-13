"""对比3D接管版本和未启用版本的效果

Usage: 
    python examples/libero/compare_3d_takeover.py \
        --csv-no-takeover data/libero_spatial_vis_3d_aware/videos/spatial_alignment_analysis.csv \
        --csv-with-takeover data/libero_spatial_active_vis_3d_aware/videos/3d_takeover_active.csv \
        --task-id 9
"""
import argparse
import pandas as pd
import pathlib
import numpy as np

def analyze_csv(csv_path, task_id=None):
    """分析CSV文件，返回统计信息"""
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
    except:
        # 如果标准读取失败，尝试手动解析
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        
        # 解析header
        header = lines[0].strip().split(',')
        data = []
        for line in lines[1:]:
            if not line.strip():
                continue
            try:
                parts = line.strip().split(',')
                if len(parts) >= len(header):
                    # 只取前len(header)个字段
                    parts = parts[:len(header)]
                    data.append(parts)
            except:
                continue
        
        df = pd.DataFrame(data, columns=header)
        # 转换数据类型
        for col in df.columns:
            if col in ['task_id', 'success', 'steps', 'num_takeovers', 'num_misalignment_warnings', 
                      'use_3d_guard', 'active_3d_takeover']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif col in ['avg_alignment', 'avg_alignment_close', 'avg_correction', 
                        'avg_policy_action_norm', 'avg_takeover_action_norm', 'avg_angle_error_deg',
                        'avg_distance_error', 'target_in_view_ratio', 'avg_true_distance', 
                        'avg_predicted_distance']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 过滤任务
    if task_id is not None:
        df = df[df['task_id'] == task_id]
    
    if len(df) == 0:
        return None
    
    # 统计
    total = len(df)
    success = len(df[df['success'] == 1])
    failure = len(df[df['success'] == 0])
    success_rate = (success / total * 100) if total > 0 else 0
    
    # 对齐度统计
    success_align = df[df['success'] == 1]['avg_alignment'].dropna()
    failure_align = df[df['success'] == 0]['avg_alignment'].dropna()
    
    # 接管次数统计
    avg_takeovers = df['num_takeovers'].mean() if 'num_takeovers' in df.columns else 0
    
    # 错位警告统计
    success_warnings = df[df['success'] == 1]['num_misalignment_warnings'].mean() if len(df[df['success'] == 1]) > 0 else 0
    failure_warnings = df[df['success'] == 0]['num_misalignment_warnings'].mean() if len(df[df['success'] == 0]) > 0 else 0
    
    return {
        'total': total,
        'success': success,
        'failure': failure,
        'success_rate': success_rate,
        'success_align_mean': success_align.mean() if len(success_align) > 0 else 0,
        'success_align_std': success_align.std() if len(success_align) > 0 else 0,
        'failure_align_mean': failure_align.mean() if len(failure_align) > 0 else 0,
        'failure_align_std': failure_align.std() if len(failure_align) > 0 else 0,
        'avg_takeovers': avg_takeovers,
        'success_warnings': success_warnings,
        'failure_warnings': failure_warnings,
    }


def main():
    parser = argparse.ArgumentParser(description='对比3D接管版本和未启用版本')
    parser.add_argument('--csv-no-takeover', type=str, required=True, help='未启用接管的CSV文件')
    parser.add_argument('--csv-with-takeover', type=str, required=True, help='启用接管的CSV文件')
    parser.add_argument('--task-id', type=int, default=None, help='任务ID（可选）')
    args = parser.parse_args()
    
    # 分析两个文件
    stats_no_takeover = analyze_csv(args.csv_no_takeover, args.task_id)
    stats_with_takeover = analyze_csv(args.csv_with_takeover, args.task_id)
    
    if stats_no_takeover is None:
        print(f"错误: 无法分析文件 {args.csv_no_takeover}")
        return
    
    if stats_with_takeover is None:
        print(f"错误: 无法分析文件 {args.csv_with_takeover}")
        return
    
    # 打印对比结果
    print("\n" + "=" * 80)
    print("3D接管效果对比分析")
    print("=" * 80)
    
    print(f"\n任务ID: {args.task_id if args.task_id is not None else '所有任务'}")
    
    print("\n【未启用3D接管】")
    print(f"  总试验次数: {stats_no_takeover['total']}")
    print(f"  成功次数: {stats_no_takeover['success']} ({stats_no_takeover['success']/stats_no_takeover['total']*100:.1f}%)")
    print(f"  失败次数: {stats_no_takeover['failure']} ({stats_no_takeover['failure']/stats_no_takeover['total']*100:.1f}%)")
    print(f"  成功率: {stats_no_takeover['success_rate']:.2f}%")
    print(f"  平均接管次数: {stats_no_takeover['avg_takeovers']:.2f}")
    print(f"  成功案例平均对齐度: {stats_no_takeover['success_align_mean']:.3f} ± {stats_no_takeover['success_align_std']:.3f}")
    print(f"  失败案例平均对齐度: {stats_no_takeover['failure_align_mean']:.3f} ± {stats_no_takeover['failure_align_std']:.3f}")
    
    print("\n【启用3D接管】")
    print(f"  总试验次数: {stats_with_takeover['total']}")
    print(f"  成功次数: {stats_with_takeover['success']} ({stats_with_takeover['success']/stats_with_takeover['total']*100:.1f}%)")
    print(f"  失败次数: {stats_with_takeover['failure']} ({stats_with_takeover['failure']/stats_with_takeover['total']*100:.1f}%)")
    print(f"  成功率: {stats_with_takeover['success_rate']:.2f}%")
    print(f"  平均接管次数: {stats_with_takeover['avg_takeovers']:.2f}")
    print(f"  成功案例平均对齐度: {stats_with_takeover['success_align_mean']:.3f} ± {stats_with_takeover['success_align_std']:.3f}")
    print(f"  失败案例平均对齐度: {stats_with_takeover['failure_align_mean']:.3f} ± {stats_with_takeover['failure_align_std']:.3f}")
    
    print("\n【对比分析】")
    success_rate_diff = stats_with_takeover['success_rate'] - stats_no_takeover['success_rate']
    if success_rate_diff > 0:
        print(f"  ✅ 成功率提升: +{success_rate_diff:.2f}%")
    elif success_rate_diff < 0:
        print(f"  ❌ 成功率下降: {success_rate_diff:.2f}%")
    else:
        print(f"  ➡️  成功率无变化")
    
    takeover_diff = stats_with_takeover['avg_takeovers'] - stats_no_takeover['avg_takeovers']
    print(f"  接管次数差异: {takeover_diff:.2f} (启用版本平均 {stats_with_takeover['avg_takeovers']:.2f} 次)")
    
    align_diff_success = stats_with_takeover['success_align_mean'] - stats_no_takeover['success_align_mean']
    print(f"  成功案例对齐度差异: {align_diff_success:+.3f}")
    
    align_diff_failure = stats_with_takeover['failure_align_mean'] - stats_no_takeover['failure_align_mean']
    print(f"  失败案例对齐度差异: {align_diff_failure:+.3f}")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
