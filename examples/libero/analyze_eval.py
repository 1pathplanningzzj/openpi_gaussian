"""Simple analysis script for Libero eval CSVs.
Usage: python examples/libero/analyze_eval.py --dir data/libero_spatial_vis_3d_aware/videos
It will read any CSV files in the directory, combine them, and print/group simple statistics by takeover flag.
"""
import argparse
import csv
import pathlib
import statistics
import sys

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def read_csvs_in_dir(d: pathlib.Path):
    rows = []
    for p in d.glob("*.csv"):
        with open(p, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append({k: (float(v) if v not in ("", "nan") and is_number(v) else v) for k, v in r.items()})
    return rows


def is_number(s):
    try:
        float(s)
        return True
    except Exception:
        return False


def summarize(rows):
    # Group by active_3d_takeover
    groups = {0: [], 1: []}
    for r in rows:
        key = int(r.get("active_3d_takeover", 0))
        groups[key].append(r)

    summary = {}
    for k, vals in groups.items():
        if not vals:
            summary[k] = None
            continue
        successes = [int(v.get("success", 0)) for v in vals]
        success_rate = sum(successes) / len(successes)
        avg_takeovers = statistics.mean([float(v.get("num_takeovers", 0)) for v in vals])
        avg_alignment = statistics.mean([float(v.get("avg_alignment", float('nan'))) for v in vals if v.get("avg_alignment") not in (None, '', 'nan')]) if any(v.get("avg_alignment") not in (None, '', 'nan') for v in vals) else float('nan')
        summary[k] = {
            "n": len(vals),
            "success_rate": success_rate,
            "avg_takeovers": avg_takeovers,
            "avg_alignment": avg_alignment,
        }
    return summary


def plot_summary(summary, out_path: pathlib.Path):
    if plt is None:
        print("matplotlib not available, skipping plots")
        return
    labels = ["baseline", "takeover"]
    rates = [summary.get(0, {}).get("success_rate", 0) if summary.get(0) else 0, summary.get(1, {}).get("success_rate", 0) if summary.get(1) else 0]
    plt.figure(figsize=(4, 3))
    plt.bar(labels, rates, color=["#2b8cbe", "#7b3294"])
    plt.ylim(0, 1)
    plt.ylabel("Success Rate")
    plt.title("Libero Eval: Baseline vs 3D Takeover")
    out_file = out_path / "summary_success_rate.png"
    plt.savefig(out_file)
    print(f"Saved plot to {out_file}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=str, required=True)
    args = p.parse_args()

    d = pathlib.Path(args.dir)
    if not d.exists():
        print(f"Dir not found: {d}")
        sys.exit(1)

    rows = read_csvs_in_dir(d)
    if not rows:
        print("No CSV rows found in dir")
        sys.exit(0)

    summary = summarize(rows)
    print("Summary:")
    for k in (0, 1):
        print(f"--- {'takeover' if k==1 else 'baseline'} ---")
        s = summary.get(k)
        if s is None:
            print("  no data")
        else:
            print(f"  n={s['n']}, success_rate={s['success_rate']:.3f}, avg_takeovers={s['avg_takeovers']:.2f}, avg_alignment={s['avg_alignment']:.3f}")

    plot_summary(summary, d)

if __name__ == '__main__':
    main()
