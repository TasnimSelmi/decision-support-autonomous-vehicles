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
        description="Plot RL-only and VLM+RL survival length over evaluation episodes."
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
        "--output-dir",
        type=Path,
        default=paths.figures_dir,
    )
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"SURVIVAL PLOT SKIP | missing file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["episode_id"] = pd.to_numeric(df["episode_id"], errors="coerce")
    df["episode_steps"] = pd.to_numeric(df["episode_steps"], errors="coerce")
    return df.dropna(subset=["episode_id", "episode_steps"])


def main() -> None:
    args = _parse_args()
    ensure_dirs([args.output_dir])

    rl = _prepare(_read_csv(args.rl_episodes))
    vlm = _prepare(_read_csv(args.vlm_episodes))

    if rl.empty or vlm.empty:
        print("SURVIVAL PLOT SKIP | both episode CSV files are required")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        rl["episode_id"],
        rl["episode_steps"],
        marker="o",
        linewidth=2,
        label="RL-only",
    )

    ax.plot(
        vlm["episode_id"],
        vlm["episode_steps"],
        marker="o",
        linewidth=2,
        label="VLM+RL",
    )

    ax.set_title("Episode Survival Length on Combined Edge Case")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Survival Length (Steps)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    output_path = args.output_dir / "survival_steps_comparison_over_episodes.png"
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)

    print(f"SURVIVAL PLOT SAVED | {output_path}")


if __name__ == "__main__":
    main()