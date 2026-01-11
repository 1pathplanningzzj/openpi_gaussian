"""手动分析3D接管版本的数据"""
import csv

# 读取未启用接管版本
no_takeover_file = "data/libero_spatial_vis_3d_aware/videos/spatial_alignment_analysis.csv"
with_takeover_file = "data/libero_spatial_active_vis_3d_aware/videos/3d_takeover_active.csv"

# 分析未启用接管版本
print("=" * 80)
print("未启用3D接管版本 (Task 9)")
print("=" * 80)

no_takeover_success = 0
no_takeover_failure = 0
no_takeover_success_align = []
no_takeover_failure_align = []

with open(no_takeover_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row['task_id']) == 9:
            if int(row['success']) == 1:
                no_takeover_success += 1
                if row['avg_alignment'] and row['avg_alignment'] != 'nan':
                    no_takeover_success_align.append(float(row['avg_alignment']))
            else:
                no_takeover_failure += 1
                if row['avg_alignment'] and row['avg_alignment'] != 'nan':
                    no_takeover_failure_align.append(float(row['avg_alignment']))

no_takeover_total = no_takeover_success + no_takeover_failure
print(f"总试验次数: {no_takeover_total}")
print(f"成功: {no_takeover_success} ({no_takeover_success/no_takeover_total*100:.1f}%)")
print(f"失败: {no_takeover_failure} ({no_takeover_failure/no_takeover_total*100:.1f}%)")
print(f"成功率: {no_takeover_success/no_takeover_total*100:.2f}%")
if no_takeover_success_align:
    print(f"成功案例平均对齐度: {sum(no_takeover_success_align)/len(no_takeover_success_align):.3f}")
if no_takeover_failure_align:
    print(f"失败案例平均对齐度: {sum(no_takeover_failure_align)/len(no_takeover_failure_align):.3f}")

# 分析启用接管版本
print("\n" + "=" * 80)
print("启用3D接管版本 (Task 9)")
print("=" * 80)

with_takeover_success = 0
with_takeover_failure = 0
with_takeover_success_align = []
with_takeover_failure_align = []
with_takeover_takeovers = []

with open(with_takeover_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row['task_id']) == 9:
            if int(row['success']) == 1:
                with_takeover_success += 1
                if row['avg_alignment'] and row['avg_alignment'] != 'nan':
                    try:
                        with_takeover_success_align.append(float(row['avg_alignment']))
                    except:
                        pass
            else:
                with_takeover_failure += 1
                if row['avg_alignment'] and row['avg_alignment'] != 'nan':
                    try:
                        with_takeover_failure_align.append(float(row['avg_alignment']))
                    except:
                        pass
            if row['num_takeovers'] and row['num_takeovers'] != 'nan':
                try:
                    with_takeover_takeovers.append(int(row['num_takeovers']))
                except:
                    pass

with_takeover_total = with_takeover_success + with_takeover_failure
print(f"总试验次数: {with_takeover_total}")
print(f"成功: {with_takeover_success} ({with_takeover_success/with_takeover_total*100:.1f}%)")
print(f"失败: {with_takeover_failure} ({with_takeover_failure/with_takeover_total*100:.1f}%)")
print(f"成功率: {with_takeover_success/with_takeover_total*100:.2f}%")
if with_takeover_success_align:
    print(f"成功案例平均对齐度: {sum(with_takeover_success_align)/len(with_takeover_success_align):.3f}")
if with_takeover_failure_align:
    print(f"失败案例平均对齐度: {sum(with_takeover_failure_align)/len(with_takeover_failure_align):.3f}")
if with_takeover_takeovers:
    print(f"平均接管次数: {sum(with_takeover_takeovers)/len(with_takeover_takeovers):.2f}")

# 对比
print("\n" + "=" * 80)
print("对比分析")
print("=" * 80)
success_rate_diff = (with_takeover_success/with_takeover_total*100) - (no_takeover_success/no_takeover_total*100)
print(f"成功率变化: {success_rate_diff:+.2f}%")
if success_rate_diff < 0:
    print("  ⚠️  警告: 启用3D接管后成功率反而下降了！")
    print("  可能的原因:")
    print("  1. 接管逻辑有问题（速度、方向等）")
    print("  2. 接管触发条件不合适")
    print("  3. 接管动作与policy动作冲突")
    print("  4. 接管速度过小（0.08）导致动作太慢")
