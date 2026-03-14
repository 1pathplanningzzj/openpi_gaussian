#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict

def parse_log(log_file):
    """Parse evaluation log and extract success rates per task."""
    task_results = defaultdict(lambda: {"successes": 0, "total": 0})
    current_task = None

    with open(log_file, 'r') as f:
        for line in f:
            task_match = re.search(r'Task: (.+)', line)
            if task_match:
                current_task = task_match.group(1).strip()

            success_match = re.search(r'Success: (True|False)', line)
            if success_match and current_task:
                task_results[current_task]["total"] += 1
                if success_match.group(1) == "True":
                    task_results[current_task]["successes"] += 1

    results = {}
    for task, data in task_results.items():
        if data["total"] > 0:
            results[task] = {
                "successes": data["successes"],
                "total": data["total"],
                "rate": data["successes"] / data["total"] * 100
            }
    return results

# Parse logs
pi05_results = parse_log("/home/zijianzhang/openpi/evalpi05_test.log")
gwm_results = parse_log("/home/zijianzhang/openpi/goal_run_0314_goal_replan_10_2_12000step.log")

# Get all tasks (sorted)
all_tasks = sorted(set(pi05_results.keys()) | set(gwm_results.keys()))

# Prepare data
pi05_rates = [pi05_results.get(task, {"rate": 0})["rate"] for task in all_tasks]
gwm_rates = [gwm_results.get(task, {"rate": 0})["rate"] for task in all_tasks]

# Use full task names with automatic wrapping
task_labels = []
for task in all_tasks:
    # Add line breaks for long task names (wrap at ~40 chars)
    if len(task) > 40:
        words = task.split()
        lines = []
        current_line = []
        current_length = 0
        for word in words:
            if current_length + len(word) + 1 > 40:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                current_line.append(word)
                current_length += len(word) + 1
        if current_line:
            lines.append(' '.join(current_line))
        task = '\n'.join(lines)
    task_labels.append(task)

# Create figure with larger size for better readability
fig, ax = plt.subplots(figsize=(18, 10))

x = np.arange(len(all_tasks))
width = 0.35

# Create bars
bars1 = ax.bar(x - width/2, pi05_rates, width, label='Pi0.5 Baseline',
               color='#4A90E2', alpha=0.8, edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, gwm_rates, width, label='GaussianVLA (12000 steps)',
               color='#E94B3C', alpha=0.8, edgecolor='black', linewidth=1.2)

# Add value labels on bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

add_value_labels(bars1)
add_value_labels(bars2)

# Customize plot
ax.set_xlabel('Task', fontsize=13, fontweight='bold')
ax.set_ylabel('Success Rate (%)', fontsize=13, fontweight='bold')
ax.set_title('Success Rate Comparison: Pi0.5 Baseline vs GaussianVLA\nLibero Goal Benchmark (50 episodes per task)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(task_labels, rotation=30, ha='right', fontsize=9)
ax.legend(fontsize=11, loc='lower right')
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add average success rate
pi05_avg = np.mean(pi05_rates)
gwm_avg = np.mean(gwm_rates)

ax.axhline(y=pi05_avg, color='#4A90E2', linestyle='--', linewidth=2, alpha=0.5,
           label=f'Pi0.5 Avg: {pi05_avg:.1f}%')
ax.axhline(y=gwm_avg, color='#E94B3C', linestyle='--', linewidth=2, alpha=0.5,
           label=f'GaussianVLA Avg: {gwm_avg:.1f}%')

# Update legend to include averages
ax.legend(fontsize=11, loc='lower right')

plt.tight_layout()
plt.savefig('/home/zijianzhang/openpi/success_rate_comparison_goal_0315.png', dpi=300, bbox_inches='tight')
print(f"Saved comparison plot to: /home/zijianzhang/openpi/success_rate_comparison_goal_0315.png")

# Print summary statistics
print(f"\n=== Summary Statistics ===")
print(f"Pi0.5 Baseline:")
print(f"  Average: {pi05_avg:.1f}%")
print(f"  Total: {sum([pi05_results[t]['successes'] for t in all_tasks])}/{sum([pi05_results[t]['total'] for t in all_tasks])}")

print(f"\nGaussianVLA (12000 steps):")
print(f"  Average: {gwm_avg:.1f}%")
print(f"  Total: {sum([gwm_results[t]['successes'] for t in all_tasks])}/{sum([gwm_results[t]['total'] for t in all_tasks])}")

print(f"\nDifference: {gwm_avg - pi05_avg:+.1f}%")
