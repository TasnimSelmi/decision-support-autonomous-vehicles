from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.rl.config import RLConfig
from src.rl.online.evaluation_metrics import EvaluationPaths
from src.rl.utils.io import ensure_dirs

DPI = 300


def _parse_args():
    config = RLConfig()
    base_dir = config.project_root / "outputs" / "results" / "online_evaluation"
    paths = EvaluationPaths.from_root(base_dir)

    parser = argparse.ArgumentParser(
        description="Plot summary mean metrics for RL-only vs VLM+RL."
    )
    parser.add_argument(
        "--rl-episodes",
        type=Path,
        default=paths.episodes_dir / "rl_only_combined_edge_episodes.csv",
    )
    parser.add_argument(
        "--vlm-episodes",
        type=Path,
        default=paths.episodes_dir / "vlm_rl_combined_edge_episodes.csv",
    )
    parser.add_argument(
        "--rl-aggregate",
        type=Path,
        default=paths.aggregate_dir / "rl_only_combined_edge_aggregate.csv",
    )
    parser.add_argument(
        "--vlm-aggregate",
        type=Path,
        default=paths.aggregate_dir / "vlm_rl_combined_edge_aggregate.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.figures_dir,
    )
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"SUMMARY PLOT SKIP | missing file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _num(df: pd.DataFrame, column: str) -> pd.Series:
    if df.empty or column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _first_value(df: pd.DataFrame, column: str) -> float:
    values = _num(df, column)
    if values.empty:
        return float("nan")
    return float(values.iloc[0])


def main() -> None:
    args = _parse_args()
    ensure_dirs([args.output_dir])

    rl_ep = _read_csv(args.rl_episodes)
    vlm_ep = _read_csv(args.vlm_episodes)
    rl_agg = _read_csv(args.rl_aggregate)
    vlm_agg = _read_csv(args.vlm_aggregate)

    if rl_ep.empty or vlm_ep.empty:
        print("SUMMARY PLOT SKIP | both episode CSV files are required")
        return

    metrics = {
        "Mean Reward": (
            _num(rl_ep, "total_reward").mean(),
            _num(vlm_ep, "total_reward").mean(),
        ),
        "Mean Lane Invasions": (
            _num(rl_ep, "lane_invasion_count").mean(),
            _num(vlm_ep, "lane_invasion_count").mean(),
        ),
        "Collision Rate": (
            _first_value(rl_agg, "collision_rate"),
            _first_value(vlm_agg, "collision_rate"),
        ),
        "Mean Survival Steps": (
            _num(rl_ep, "episode_steps").mean(),
            _num(vlm_ep, "episode_steps").mean(),
        ),
    }

    labels = list(metrics.keys())
    rl_values = [metrics[label][0] for label in labels]
    vlm_values = [metrics[label][1] for label in labels]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.bar([i - width / 2 for i in x], rl_values, width, label="RL-only")
    ax.bar([i + width / 2 for i in x], vlm_values, width, label="VLM+RL")

    ax.set_title("Summary Mean Metrics on Combined Edge Case")
    ax.set_ylabel("Metric Value")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    for i, value in enumerate(rl_values):
        if pd.notna(value):
            ax.text(i - width / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    for i, value in enumerate(vlm_values):
        if pd.notna(value):
            ax.text(i + width / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()

    output_path = args.output_dir / "summary_mean_metrics_comparison.png"
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)

    print(f"SUMMARY PLOT SAVED | {output_path}")


if __name__ == "__main__":
    main()