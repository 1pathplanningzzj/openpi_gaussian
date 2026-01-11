"""Plot distribution of relative correction (avg_correction normalized by avg_policy_action_norm).
Usage: python examples/libero/plot_relative_correction.py --csv data/libero_spatial_vis_3d_aware/videos/spatial_alignment_analysis.csv
"""
import argparse
import pathlib
import sys
import numpy as np
import matplotlib.pyplot as plt

try:
    import pandas as pd
except Exception:
    pd = None


def load_csv(p: pathlib.Path):
    if pd is not None:
        df = pd.read_csv(p)
    else:
        import csv
        rows = []
        with open(p, 'r') as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        df = None
        print("Pandas not available; please install pandas for best experience.")
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=str, required=True)
    args = p.parse_args()

    pth = pathlib.Path(args.csv)
    if not pth.exists():
        print(f"CSV not found: {pth}")
        sys.exit(1)

    df = load_csv(pth)
    if df is None:
        sys.exit(1)

    # Required columns
    if 'avg_correction' not in df.columns or 'avg_policy_action_norm' not in df.columns:
        print('CSV missing required columns: avg_correction or avg_policy_action_norm')
        print('Available columns:', df.columns.tolist())
        sys.exit(1)

    eps = 1e-6
    df['relative_correction'] = df['avg_correction'] / (df['avg_policy_action_norm'] + eps)

    # Filter out NaNs and infs
    vals = df['relative_correction'].replace([np.inf, -np.inf], np.nan).dropna()
    abs_vals = df['avg_correction'].replace([np.inf, -np.inf], np.nan).dropna()

    # Summary stats
    def summarize(series):
        return {
            'n': int(series.count()),
            'mean': float(series.mean()),
            'median': float(series.median()),
            'p25': float(series.quantile(0.25)),
            'p75': float(series.quantile(0.75)),
            'std': float(series.std()),
        }

    rel_summary = summarize(vals)
    abs_summary = summarize(abs_vals)

    print('Relative Correction Summary:', rel_summary)
    print('Absolute Avg Correction Summary:', abs_summary)

    out_dir = pth.parent
    # Plot histogram + KDE
    try:
        import seaborn as sns
        sns.set_style('darkgrid')
    except Exception:
        plt.style.use('ggplot')

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].hist(vals, bins=50, density=True, alpha=0.7)
    axs[0].set_title('Relative Correction (avg_correction / avg_policy_action_norm)')
    axs[0].set_xlabel('Relative Correction')

    axs[1].boxplot(vals, vert=False)
    axs[1].set_title('Relative Correction Distribution (boxplot)')

    plt.tight_layout()
    out_file = out_dir / 'relative_correction_distribution.png'
    fig.savefig(out_file)
    print(f'Saved plot to {out_file}')

    # Save summary to text
    summary_file = out_dir / 'relative_correction_summary.txt'
    with open(summary_file, 'w') as f:
        f.write('Relative Correction Summary:\n')
        for k, v in rel_summary.items():
            f.write(f'{k}: {v}\n')
        f.write('\nAbsolute Avg Correction Summary:\n')
        for k, v in abs_summary.items():
            f.write(f'{k}: {v}\n')
    print(f'Saved summary to {summary_file}')


if __name__ == '__main__':
    main()
