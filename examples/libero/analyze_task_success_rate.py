"""分析CSV文件中特定任务的成功率

Usage: python examples/libero/analyze_task_success_rate.py --csv data/libero_spatial_vis_3d_aware/videos/spatial_alignment_analysis.csv --task-id 9
"""
import argparse
import pandas as pd
import pathlib

def analyze_task_success_rate(csv_path, task_id=None, task_description_filter=None):
    """分析特定任务的成功率"""
    df = pd.read_csv(csv_path)
    
    # 过滤数据
    if task_id is not None:
        df_filtered = df[df['task_id'] == task_id]
        task_info = f"Task ID: {task_id}"
    elif task_description_filter:
        df_filtered = df[df['task_description'].str.contains(task_description_filter, case=False, na=False)]
        task_info = f"Task Description contains: {task_description_filter}"
    else:
        df_filtered = df
        task_info = "All tasks"
    
    if len(df_filtered) == 0:
        print(f"没有找到匹配的数据")
        return
    
    # 统计成功和失败
    success_count = len(df_filtered[df_filtered['success'] == 1])
    failure_count = len(df_filtered[df_filtered['success'] == 0])
    total_count = len(df_filtered)
    success_rate = (success_count / total_count * 100) if total_count > 0 else 0
    
    # 显示任务信息
    if len(df_filtered) > 0:
        sample_task_desc = df_filtered.iloc[0]['task_description']
        print(f"\n任务信息: {task_info}")
        print(f"任务描述: {sample_task_desc}")
    
    # 打印统计结果
    print("\n" + "=" * 80)
    print("成功率统计")
    print("=" * 80)
    print(f"总试验次数: {total_count}")
    print(f"成功次数: {success_count} ({success_count/total_count*100:.1f}%)")
    print(f"失败次数: {failure_count} ({failure_count/total_count*100:.1f}%)")
    print(f"成功率: {success_rate:.2f}%")
    
    # 如果有3D guard相关的数据，也统计一下
    if 'use_3d_guard' in df_filtered.columns:
        use_3d_guard = df_filtered['use_3d_guard'].iloc[0] if len(df_filtered) > 0 else 0
        active_3d_takeover = df_filtered['active_3d_takeover'].iloc[0] if len(df_filtered) > 0 else 0
        print(f"\n3D Guard设置:")
        print(f"  use_3d_guard: {bool(use_3d_guard)}")
        print(f"  active_3d_takeover: {bool(active_3d_takeover)}")
    
    # 如果有对齐度数据，显示成功和失败案例的平均对齐度
    if 'avg_alignment' in df_filtered.columns:
        success_align = df_filtered[df_filtered['success'] == 1]['avg_alignment'].dropna()
        failure_align = df_filtered[df_filtered['success'] == 0]['avg_alignment'].dropna()
        
        if len(success_align) > 0 and len(failure_align) > 0:
            print(f"\n对齐度分析:")
            print(f"  成功案例平均对齐度: {success_align.mean():.3f} ± {success_align.std():.3f}")
            print(f"  失败案例平均对齐度: {failure_align.mean():.3f} ± {failure_align.std():.3f}")
    
    # 如果有错位警告数据，也显示一下
    if 'num_misalignment_warnings' in df_filtered.columns:
        success_warnings = df_filtered[df_filtered['success'] == 1]['num_misalignment_warnings'].mean()
        failure_warnings = df_filtered[df_filtered['success'] == 0]['num_misalignment_warnings'].mean()
        print(f"\n错位警告分析:")
        print(f"  成功案例平均警告数: {success_warnings:.2f}")
        print(f"  失败案例平均警告数: {failure_warnings:.2f}")
    
    print("\n" + "=" * 80)
    
    return {
        'total': total_count,
        'success': success_count,
        'failure': failure_count,
        'success_rate': success_rate
    }


def main():
    parser = argparse.ArgumentParser(description='分析任务成功率')
    parser.add_argument('--csv', type=str, required=True, help='CSV文件路径')
    parser.add_argument('--task-id', type=int, default=None, help='任务ID（可选）')
    parser.add_argument('--task-description', type=str, default=None, help='任务描述关键词（可选）')
    args = parser.parse_args()
    
    csv_path = pathlib.Path(args.csv)
    if not csv_path.exists():
        print(f"错误: CSV文件不存在: {csv_path}")
        return
    
    analyze_task_success_rate(
        csv_path, 
        task_id=args.task_id,
        task_description_filter=args.task_description
    )


if __name__ == '__main__':
    main()
