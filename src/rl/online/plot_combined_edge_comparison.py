from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.rl.config import RLConfig
from src.rl.utils.io import ensure_dirs


def _parse_args() -> argparse.Namespace:
    config = RLConfig()
    default_dir = config.project_root / "outputs" / "results" / "online_evaluation"
    parser = argparse.ArgumentParser(
        description="Plot RL-only vs VLM+RL metrics on the same combined edge-case scenario."
    )
    parser.add_argument(
        "--rl-steps",
        type=Path,
        default=default_dir / "combined_edge_best_trained_steps.csv",
    )
    parser.add_argument(
        "--rl-summary",
        type=Path,
        default=default_dir / "combined_edge_best_trained_summary.csv",
    )
    parser.add_argument(
        "--vlm-steps",
        type=Path,
        default=default_dir / "combined_edge_vlm_rl_best_trained_steps.csv",
    )
    parser.add_argument(
        "--vlm-summary",
        type=Path,
        default=default_dir / "combined_edge_vlm_rl_best_trained_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_dir / "plots",
    )
    return parser.parse_args()


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")


def _mean_by_step(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce").groupby(df["step"]).mean()


def _plot_step_curves(rl_steps: pd.DataFrame, vlm_steps: pd.DataFrame, output_dir: Path) -> None:
    curves = [
        ("total_reward", "Cumulative Reward", "combined_edge_reward_curve.png"),
        ("speed", "Speed (m/s)", "combined_edge_speed_curve.png"),
        ("collision", "Collision Indicator", "combined_edge_collision_curve.png"),
        ("lane_invasion", "Lane Invasion Indicator", "combined_edge_lane_curve.png"),
    ]

    for column, ylabel, filename in curves:
        plt.figure(figsize=(9, 5))
        _mean_by_step(rl_steps, column).plot(label="RL-only", linewidth=2)
        _mean_by_step(vlm_steps, column).plot(label="VLM+RL", linewidth=2)
        plt.title(f"Combined Edge Case - {ylabel}")
        plt.xlabel("Step")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=160)
        plt.close()


def _plot_episode_bars(rl_summary: pd.DataFrame, vlm_summary: pd.DataFrame, output_dir: Path) -> None:
    metrics = {
        "total_reward": "Mean Total Reward",
        "survival_time_seconds": "Mean Survival Time (s)",
        "collision_count": "Mean Collision Count",
        "lane_invasion_count": "Mean Lane Invasion Count",
        "defensive_action_rate": "Mean Defensive Action Rate",
        "average_speed_kmh": "Mean Speed (km/h)",
    }

    values = []
    labels = []
    for column, label in metrics.items():
        labels.append(label)
        values.append([
            pd.to_numeric(rl_summary[column], errors="coerce").mean(),
            pd.to_numeric(vlm_summary[column], errors="coerce").mean(),
        ])

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()
    for ax, label, pair in zip(axes, labels, values):
        ax.bar(["RL-only", "VLM+RL"], pair, color=["#4c78a8", "#f58518"])
        ax.set_title(label)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Combined Edge Case - Aggregate Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "combined_edge_aggregate_comparison.png", dpi=160)
    plt.close(fig)


def _plot_action_distribution(rl_steps: pd.DataFrame, vlm_steps: pd.DataFrame, output_dir: Path) -> None:
    rl_counts = rl_steps["action_name"].value_counts(normalize=True).sort_index()
    vlm_counts = vlm_steps["action_name"].value_counts(normalize=True).sort_index()
    actions = sorted(set(rl_counts.index) | set(vlm_counts.index))
    data = pd.DataFrame({
        "RL-only": [rl_counts.get(action, 0.0) for action in actions],
        "VLM+RL": [vlm_counts.get(action, 0.0) for action in actions],
    }, index=actions)

    ax = data.plot(kind="bar", figsize=(10, 5), color=["#4c78a8", "#f58518"])
    ax.set_title("Combined Edge Case - Action Distribution")
    ax.set_xlabel("Action")
    ax.set_ylabel("Fraction of Steps")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "combined_edge_action_distribution.png", dpi=160)
    plt.close()


def main() -> None:
    args = _parse_args()
    for path in [args.rl_steps, args.rl_summary, args.vlm_steps, args.vlm_summary]:
        _require(path)

    ensure_dirs([args.output_dir])
    rl_steps = pd.read_csv(args.rl_steps)
    rl_summary = pd.read_csv(args.rl_summary)
    vlm_steps = pd.read_csv(args.vlm_steps)
    vlm_summary = pd.read_csv(args.vlm_summary)

    _plot_step_curves(rl_steps, vlm_steps, args.output_dir)
    _plot_episode_bars(rl_summary, vlm_summary, args.output_dir)
    _plot_action_distribution(rl_steps, vlm_steps, args.output_dir)

    print(f"PLOTS SAVED | {args.output_dir}")


if __name__ == "__main__":
    main()
