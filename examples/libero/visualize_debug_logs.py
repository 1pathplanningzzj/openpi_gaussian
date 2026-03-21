"""Visualize debug logs from a single evaluation run.

For each episode, generates:
  1. A side-by-side video (agentview + wrist) of inference-time observations
  2. An action trajectory plot (7-dim: dx, dy, dz, droll, dpitch, dyaw, gripper)

Usage:
    python examples/libero/visualize_debug_logs.py --log_dir debug_logs_run1
    python examples/libero/visualize_debug_logs.py --log_dir debug_logs_run1 --task_id 8
    python examples/libero/visualize_debug_logs.py --log_dir debug_logs_run1 --plot_chunk_steps 5
"""

import argparse
import pathlib
import pickle

import imageio
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ACTION_LABELS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


def load_episodes(log_dir: pathlib.Path, task_id: int | None = None):
    """Load episode pkl files, optionally filtered by task_id."""
    episodes = []
    for f in sorted(log_dir.glob("task_*_ep_*.pkl")):
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        if task_id is not None and d["task_id"] != task_id:
            continue
        episodes.append(d)
    return episodes


def make_side_by_side_video(episode: dict, out_path: pathlib.Path):
    """Create a video with agentview and wrist view side by side."""
    records = episode["records"]
    if not records or "image" not in records[0]:
        print(f"  Skipping video: no image data in records")
        return

    frames = []
    success_str = "SUCCESS" if episode["success"] else "FAILURE"
    color = (0, 255, 0) if episode["success"] else (255, 0, 0)
    label = f"Task {episode['task_id']} Ep {episode['episode_idx']} - {success_str}"

    for idx, rec in enumerate(records):
        img = rec["image"]
        wrist = rec["wrist_image"]
        combined = np.concatenate([img, wrist], axis=1)

        # Draw label on frame
        pil_img = Image.fromarray(combined)
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except (IOError, OSError):
            font = ImageFont.load_default()
        draw.text((5, 5), f"{label}  t={rec['timestep']}", fill=color, font=font)
        frames.append(np.array(pil_img))

    imageio.mimwrite(str(out_path), frames, fps=5)


def _expand_chunk_actions(records: list[dict], plot_chunk_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Expand the first N actions in each predicted chunk onto environment timesteps."""
    expanded_timesteps = []
    expanded_actions = []

    for rec in records:
        chunk = np.asarray(rec["action_chunk"])
        if chunk.ndim != 2:
            raise ValueError(f"Expected action_chunk to have shape (chunk, action_dim), got {chunk.shape}")

        num_steps = min(plot_chunk_steps, chunk.shape[0])
        base_timestep = rec["timestep"]
        for step_idx in range(num_steps):
            expanded_timesteps.append(base_timestep + step_idx)
            expanded_actions.append(chunk[step_idx])

    return np.asarray(expanded_timesteps), np.asarray(expanded_actions)


def plot_action_trajectory(episode: dict, out_path: pathlib.Path, plot_chunk_steps: int = 1):
    """Plot the first N actions from each chunk over time."""
    records = episode["records"]
    if not records:
        return

    timesteps, actions = _expand_chunk_actions(records, plot_chunk_steps)
    n_dims = actions.shape[1]

    fig, axes = plt.subplots(n_dims, 1, figsize=(12, 2 * n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        label = ACTION_LABELS[i] if i < len(ACTION_LABELS) else f"dim_{i}"
        ax.plot(timesteps, actions[:, i], marker=".", markersize=3, linewidth=1)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Timestep")
    success_str = "SUCCESS" if episode["success"] else "FAILURE"
    fig.suptitle(
        f"Task {episode['task_id']} Ep {episode['episode_idx']} - {success_str}\n"
        f"{episode['task_description']}\n"
        f"Showing first {plot_chunk_steps} action(s) from each predicted chunk",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)


def plot_overlay_action_trajectories(
    episodes_a: list[dict],
    episodes_b: list[dict],
    out_path: pathlib.Path,
    plot_chunk_steps: int,
    label_a: str,
    label_b: str,
):
    """Overlay two runs on one figure. Solid=success, dashed=failure."""
    common_episode_keys = {
        (ep["task_id"], ep["episode_idx"]) for ep in episodes_a
    } & {
        (ep["task_id"], ep["episode_idx"]) for ep in episodes_b
    }
    if common_episode_keys:
        episodes_a = [ep for ep in episodes_a if (ep["task_id"], ep["episode_idx"]) in common_episode_keys]
        episodes_b = [ep for ep in episodes_b if (ep["task_id"], ep["episode_idx"]) in common_episode_keys]

    fig, axes = plt.subplots(7, 1, figsize=(15, 19), sharex=True)

    groups = [
        (episodes_a, "tab:blue", label_a),
        (episodes_b, "tab:orange", label_b),
    ]

    legend_elements = []
    for episodes, color, label in groups:
        success_eps = [ep for ep in episodes if ep["success"]]
        fail_eps = [ep for ep in episodes if not ep["success"]]

        for ep in success_eps:
            if not ep["records"]:
                continue
            timesteps, actions = _expand_chunk_actions(ep["records"], plot_chunk_steps)
            for i, ax in enumerate(axes):
                ax.plot(timesteps, actions[:, i], color=color, linestyle="-", alpha=0.8, linewidth=1.6)

        for ep in fail_eps:
            if not ep["records"]:
                continue
            timesteps, actions = _expand_chunk_actions(ep["records"], plot_chunk_steps)
            for i, ax in enumerate(axes):
                ax.plot(timesteps, actions[:, i], color=color, linestyle="--", alpha=0.35, linewidth=1.0)

        legend_elements.extend([
            Line2D([0], [0], color=color, linestyle="-", linewidth=1.6, alpha=0.8, label=f"{label} success ({len(success_eps)})"),
            Line2D([0], [0], color=color, linestyle="--", linewidth=1.0, alpha=0.35, label=f"{label} failure ({len(fail_eps)})"),
        ])

    for i, ax in enumerate(axes):
        label = ACTION_LABELS[i] if i < len(ACTION_LABELS) else f"dim_{i}"
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Timestep")
    axes[0].legend(handles=legend_elements, loc="upper right", fontsize=10)

    task_desc = None
    for ep in episodes_a + episodes_b:
        task_desc = ep.get("task_description")
        if task_desc:
            break

    fig.suptitle(
        f"Action Trajectories Overlay\n{label_a} vs {label_b}\n"
        f"{task_desc if task_desc else ''}\n"
        f"Showing first {plot_chunk_steps} action(s) from each predicted chunk",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--compare_log_dir", type=str, default=None, help="Optional second log dir for overlay comparison")
    parser.add_argument("--label_a", type=str, default="run_a", help="Label for --log_dir in overlay mode")
    parser.add_argument("--label_b", type=str, default="run_b", help="Label for --compare_log_dir in overlay mode")
    parser.add_argument("--task_id", type=int, default=None, help="Only visualize this task")
    parser.add_argument("--out_dir", type=str, default=None, help="Output dir (default: <log_dir>/vis)")
    parser.add_argument(
        "--plot_chunk_steps",
        type=int,
        default=1,
        help="How many actions to plot from each predicted chunk. Use 5 to match replan_steps=5.",
    )
    args = parser.parse_args()

    log_dir = pathlib.Path(args.log_dir)
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else log_dir / "vis"
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = load_episodes(log_dir, args.task_id)
    print(f"Loaded {len(episodes)} episodes from {log_dir}")

    if args.compare_log_dir:
        compare_log_dir = pathlib.Path(args.compare_log_dir)
        compare_episodes = load_episodes(compare_log_dir, args.task_id)
        print(f"Loaded {len(compare_episodes)} episodes from {compare_log_dir}")
        overlay_name = (
            f"task_{args.task_id}_overlay_{args.label_a}_vs_{args.label_b}.png"
            if args.task_id is not None
            else f"overlay_{args.label_a}_vs_{args.label_b}.png"
        )
        plot_overlay_action_trajectories(
            episodes,
            compare_episodes,
            out_dir / overlay_name,
            plot_chunk_steps=args.plot_chunk_steps,
            label_a=args.label_a,
            label_b=args.label_b,
        )
        print(f"Done. Output saved to {out_dir}")
        return

    for ep in episodes:
        tid, eidx = ep["task_id"], ep["episode_idx"]
        tag = f"task_{tid}_ep_{eidx}"
        print(f"  Visualizing {tag} ...")
        make_side_by_side_video(ep, out_dir / f"{tag}_views.mp4")
        plot_action_trajectory(ep, out_dir / f"{tag}_actions.png", plot_chunk_steps=args.plot_chunk_steps)

    print(f"Done. Output saved to {out_dir}")



if __name__ == "__main__":
    main()
