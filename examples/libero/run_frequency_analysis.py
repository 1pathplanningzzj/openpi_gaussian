import subprocess
import re
import matplotlib.pyplot as plt
import os
import sys

# Configuration
REPLAN_STEPS_LIST = [1, 2, 5, 10]
# Use a small number of trials per setting to get results quickly. 
# Increase this if you have time usually 20+ is better for stats.
NUM_TRIALS = 5 
PYTHON_EXECUTABLE = sys.executable 
MAIN_SCRIPT = "examples/libero/main.py"

# Setup Environment
env_vars = os.environ.copy()
env_vars["MUJOCO_GL"] = "egl"
cwd = os.getcwd()
# Ensure third_party libs are in pythonpath
current_path = env_vars.get("PYTHONPATH", "")
env_vars["PYTHONPATH"] = f"{current_path}:{cwd}/third_party/libero:{cwd}/third_party/robosuite"

results = {
    "steps": [],
    "success_rate": [],
    "avg_correction": [],
    "max_correction": []
}

def parse_output(output):
    success_count = 0
    episode_count = 0
    total_avg_correction = 0
    total_max_correction = 0
    correction_data_points = 0

    lines = output.split('\n')
    for line in lines:
        # Metrics parsing from our previous modification
        if "Episode Analysis" in line:
            # Info:root:Episode Analysis - Avg Correction: 0.1067, Max Correction: 0.5369
            match = re.search(r"Avg Correction: ([\d\.]+), Max Correction: ([\d\.]+)", line)
            if match:
                avg_c = float(match.group(1))
                max_c = float(match.group(2))
                total_avg_correction += avg_c
                total_max_correction = max(total_max_correction, max_c) 
                correction_data_points += 1
        
        # Success parsing
        # Info:root:Success: True
        if "Success: True" in line:
            success_count += 1
            episode_count += 1
        elif "Success: False" in line:
            episode_count += 1
            
    success_rate = (success_count / episode_count) * 100 if episode_count > 0 else 0
    avg_correction_metric = (total_avg_correction / correction_data_points) if correction_data_points > 0 else 0
    
    return success_rate, avg_correction_metric, total_max_correction

print(f"Starting Frequency Analysis [Trials per setting: {NUM_TRIALS}]...")
print("This may take a few minutes depending on simulation speed.")

for steps in REPLAN_STEPS_LIST:
    print(f"\n--- Testing Replan Steps = {steps} (Control Freq: ~{20/steps:.1f} Hz) ---")
    
    # We use arguments compatible with tyro/argparse as defined in main.py
    # Note: Tyro uses the parameter name 'args' as a prefix because eval_libero(args: Args)
    cmd = [
        PYTHON_EXECUTABLE, 
        MAIN_SCRIPT, 
        "--args.replan-steps", str(steps),
        "--args.num-trials-per-task", str(NUM_TRIALS),
        "--args.video-out-path", f"data/analysis_steps_{steps}/videos" 
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        # Run process and capture output
        # Using subprocess.run to block until finished
        result = subprocess.run(
            cmd, 
            env=env_vars,
            capture_output=True, 
            text=True
        )
        
        if result.returncode != 0:
            print(f"Error running for steps={steps}:")
            print(result.stderr[-500:]) # Print last 500 chars of error
            continue
            
        # Parse logs
        full_log = result.stderr + result.stdout
        success_rate, avg_corr, max_corr = parse_output(full_log)
        
        print(f"  -> Success Rate: {success_rate:.1f}%")
        print(f"  -> Avg Correction: {avg_corr:.4f}")
        print(f"  -> Max Correction: {max_corr:.4f}")
        
        results["steps"].append(steps)
        results["success_rate"].append(success_rate)
        results["avg_correction"].append(avg_corr)
        results["max_correction"].append(max_corr)
        
    except Exception as e:
        print(f"Exception during execution: {e}")

# Plotting
if not results["steps"]:
    print("No results collected.")
    sys.exit(1)

print("\nGenerating Plots...")

plt.style.use('ggplot')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Success Rate
# X-axis: Replan Steps. We might want to label X as Frequency (Hz) approx.
freqs = [20/s for s in results["steps"]]

ax1.plot(results["steps"], results["success_rate"], 'o-', color='green', linewidth=2, markersize=8)
ax1.set_title('Impact of Control Frequency on Success Rate')
ax1.set_xlabel('Replan Steps (1 = 20Hz, 10 = 2Hz)')
ax1.set_ylabel('Success Rate (%)')
ax1.set_xticks(results["steps"])
ax1.set_ylim(-5, 105)
for i, txt in enumerate(results["success_rate"]):
    ax1.annotate(f"{txt:.1f}%", (results["steps"][i], results["success_rate"][i]), textcoords="offset points", xytext=(0,10), ha='center')

# Plot 2: Correction Magnitude
# This shows how much the policy "changed its mind" when given the chance to replan.
ax2.plot(results["steps"], results["avg_correction"], 'o-', color='#1f77b4', label='Avg Correction per Step', linewidth=2)
ax2.set_title('Policy Correction Magnitude')
ax2.set_xlabel('Replan Steps')
ax2.set_ylabel('L2 Norm of Action Difference')
ax2.set_xticks(results["steps"])
ax2.legend()

plt.tight_layout()
output_file = 'frequency_analysis_results.png'
plt.savefig(output_file)
print(f"Analysis complete! Saved plot to '{os.path.abspath(output_file)}'")
