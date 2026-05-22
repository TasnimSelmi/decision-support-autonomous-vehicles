from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.rl.config import RLConfig
from src.rl.online.evaluation_metrics import EvaluationPaths
from src.rl.utils.io import ensure_dirs

DPI = 300
COLORS = {"RL-only": "#4c78a8", "VLM+RL": "#f58518"}


def _parse_args() -> argparse.Namespace:
    config = RLConfig()
    base_dir = config.project_root / "outputs" / "results" / "online_evaluation"
    paths = EvaluationPaths.from_root(base_dir)
    parser = argparse.ArgumentParser(
        description="Plot RL-only vs VLM+RL combined-edge evaluation metrics."
    )
    parser.add_argument("--base-dir", type=Path, default=base_dir)
    parser.add_argument("--rl-steps", type=Path, default=paths.raw_dir / "rl_only_combined_edge_steps.csv")
    parser.add_argument("--rl-episodes", type=Path, default=paths.episodes_dir / "rl_only_combined_edge_episodes.csv")
    parser.add_argument("--rl-aggregate", type=Path, default=paths.aggregate_dir / "rl_only_combined_edge_aggregate.csv")
    parser.add_argument("--vlm-steps", type=Path, default=paths.raw_dir / "vlm_rl_combined_edge_steps.csv")
    parser.add_argument("--vlm-episodes", type=Path, default=paths.episodes_dir / "vlm_rl_combined_edge_episodes.csv")
    parser.add_argument("--vlm-aggregate", type=Path, default=paths.aggregate_dir / "vlm_rl_combined_edge_aggregate.csv")
    parser.add_argument("--output-dir", type=Path, default=paths.figures_dir)
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"PLOT SKIP | missing input: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _value(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return math.nan
    return float(pd.to_numeric(df[column], errors="coerce").iloc[0])


def _plot_bar(ax, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    clean = [0.0 if pd.isna(value) else value for value in values]
    bars = ax.bar(labels, clean, color=[COLORS.get(label, "#777777") for label in labels])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        label = "NaN" if pd.isna(value) else f"{value:.3g}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom", fontsize=8)


def _mean_by_step(df: pd.DataFrame, column: str) -> pd.Series:
    if df.empty or "step" not in df.columns or column not in df.columns:
        return pd.Series(dtype=float)
    return _num(df[column]).groupby(df["step"]).mean()


def _plot_metric_over_time(
    rl_steps: pd.DataFrame,
    vlm_steps: pd.DataFrame,
    column: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for label, df in [("RL-only", rl_steps), ("VLM+RL", vlm_steps)]:
        series = _mean_by_step(df, column)
        if not series.empty and series.notna().any():
            ax.plot(series.index, series.values, label=label, linewidth=2, color=COLORS[label])
            plotted = True
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend()
    else:
        ax.text(0.5, 0.5, f"No available {ylabel} data", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)


def _plot_success_route(rl_agg: pd.DataFrame, vlm_agg: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    _plot_bar(
        axes[0],
        ["RL-only", "VLM+RL"],
        [_value(rl_agg, "success_rate"), _value(vlm_agg, "success_rate")],
        "Success Rate",
        "Rate",
    )
    _plot_bar(
        axes[1],
        ["RL-only", "VLM+RL"],
        [_value(rl_agg, "mean_route_completion"), _value(vlm_agg, "mean_route_completion")],
        "Route Completion",
        "Completion",
    )
    fig.suptitle("Combined Edge Case - Reliability and Progress")
    fig.tight_layout()
    fig.savefig(output_dir / "success_route_completion.png", dpi=DPI)
    plt.close(fig)


def _plot_collision_safety(rl_agg: pd.DataFrame, vlm_agg: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("collision_rate", "Collision Rate", "Rate"),
        ("mean_collision_speed_kmh", "Collision Speed", "km/h"),
        ("mean_dcf_collisions_per_km", "DCF", "Collisions/km"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (column, title, ylabel) in zip(axes, metrics):
        _plot_bar(
            ax,
            ["RL-only", "VLM+RL"],
            [_value(rl_agg, column), _value(vlm_agg, column)],
            title,
            ylabel,
        )
    fig.suptitle("Combined Edge Case - Collision Safety")
    fig.tight_layout()
    fig.savefig(output_dir / "collision_safety_comparison.png", dpi=DPI)
    plt.close(fig)


def _plot_action_distribution(rl_steps: pd.DataFrame, vlm_steps: pd.DataFrame, output_dir: Path) -> None:
    if "selected_action_name" not in rl_steps.columns or "selected_action_name" not in vlm_steps.columns:
        print("PLOT SKIP | action_distribution.png requires selected_action_name")
        return
    rl_counts = rl_steps["selected_action_name"].value_counts(normalize=True).sort_index()
    vlm_counts = vlm_steps["selected_action_name"].value_counts(normalize=True).sort_index()
    actions = sorted(set(rl_counts.index) | set(vlm_counts.index))
    data = pd.DataFrame({
        "RL-only": [rl_counts.get(action, 0.0) for action in actions],
        "VLM+RL": [vlm_counts.get(action, 0.0) for action in actions],
    }, index=actions)
    ax = data.plot(kind="bar", figsize=(10, 5), color=[COLORS["RL-only"], COLORS["VLM+RL"]])
    ax.set_title("Combined Edge Case - Action Distribution")
    ax.set_xlabel("Action")
    ax.set_ylabel("Fraction of Steps")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "action_distribution.png", dpi=DPI)
    plt.close()


def _encode_vlm_signal(series: pd.Series, mapping: dict[str, float]) -> pd.Series:
    return series.astype(str).str.lower().map(mapping)


def _plot_vlm_semantics(vlm_steps: pd.DataFrame, output_dir: Path) -> None:
    required = [
        "vlm_hazard_level",
        "vlm_collision_risk",
        "vlm_pedestrian_presence",
        "vlm_lane_safety",
        "vlm_action_urgency",
    ]
    if vlm_steps.empty or any(column not in vlm_steps.columns for column in required):
        print("PLOT SKIP | vlm_semantic_signals.png requires VLM semantic columns")
        return

    df = vlm_steps.copy()
    encoded = pd.DataFrame({
        "hazard": _encode_vlm_signal(df["vlm_hazard_level"], {"low": 0.0, "medium": 1.0, "high": 2.0}),
        "collision risk": _encode_vlm_signal(df["vlm_collision_risk"], {"no": 0.0, "yes": 1.0}),
        "pedestrian": _encode_vlm_signal(df["vlm_pedestrian_presence"], {"no": 0.0, "yes": 1.0}),
        "unsafe lane": _encode_vlm_signal(df["vlm_lane_safety"], {"safe": 0.0, "unsafe": 1.0}),
        "urgency": _encode_vlm_signal(df["vlm_action_urgency"], {"continue": 0.0, "slow": 1.0, "slow_down": 1.0, "brake": 2.0}),
        "step": df["step"],
    })

    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False
    for column in ["hazard", "collision risk", "pedestrian", "unsafe lane", "urgency"]:
        series = encoded[column].groupby(encoded["step"]).mean()
        if series.notna().any():
            ax.plot(series.index, series.values, label=column, linewidth=2)
            plotted = True
    ax.set_title("VLM Semantic Signals Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Encoded Signal")
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No available VLM signal data", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(output_dir / "vlm_semantic_signals.png", dpi=DPI)
    plt.close(fig)


def _plot_linkedin_summary(
    rl_steps: pd.DataFrame,
    vlm_steps: pd.DataFrame,
    rl_agg: pd.DataFrame,
    vlm_agg: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    _plot_bar(
        axes[0, 0],
        ["RL-only", "VLM+RL"],
        [_value(rl_agg, "success_rate"), _value(vlm_agg, "success_rate")],
        "Success Rate",
        "Rate",
    )
    _plot_bar(
        axes[0, 1],
        ["RL-only", "VLM+RL"],
        [_value(rl_agg, "collision_rate"), _value(vlm_agg, "collision_rate")],
        "Collision Rate",
        "Rate",
    )
    for ax, column, title, ylabel in [
        (axes[1, 0], "ttc_seconds", "TTC Over Time", "Seconds"),
        (axes[1, 1], "min_hazard_distance_m", "Minimum Hazard Distance", "Meters"),
    ]:
        plotted = False
        for label, df in [("RL-only", rl_steps), ("VLM+RL", vlm_steps)]:
            series = _mean_by_step(df, column)
            if not series.empty and series.notna().any():
                ax.plot(series.index, series.values, label=label, linewidth=2, color=COLORS[label])
                plotted = True
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend()
        else:
            ax.text(0.5, 0.5, "Data not available", ha="center", va="center", transform=ax.transAxes)
    fig.suptitle("RL-only vs VLM+RL on Combined CARLA Edge Case", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "linkedin_summary_figure.png", dpi=DPI)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    ensure_dirs([args.output_dir])

    rl_steps = _read_csv(args.rl_steps)
    vlm_steps = _read_csv(args.vlm_steps)
    rl_episodes = _read_csv(args.rl_episodes)
    vlm_episodes = _read_csv(args.vlm_episodes)
    rl_agg = _read_csv(args.rl_aggregate)
    vlm_agg = _read_csv(args.vlm_aggregate)

    if rl_agg.empty or vlm_agg.empty:
        print("PLOT SKIP | aggregate plots require both aggregate CSVs")
    else:
        _plot_success_route(rl_agg, vlm_agg, args.output_dir)
        _plot_collision_safety(rl_agg, vlm_agg, args.output_dir)

    if not rl_steps.empty and not vlm_steps.empty:
        _plot_metric_over_time(
            rl_steps,
            vlm_steps,
            "ttc_seconds",
            "TTC (seconds)",
            "Combined Edge Case - TTC Over Time",
            args.output_dir / "ttc_over_time.png",
        )
        _plot_metric_over_time(
            rl_steps,
            vlm_steps,
            "min_hazard_distance_m",
            "Minimum Hazard Distance (m)",
            "Combined Edge Case - Hazard Distance Over Time",
            args.output_dir / "hazard_distance_over_time.png",
        )
        _plot_metric_over_time(
            rl_steps,
            vlm_steps,
            "speed_kmh",
            "Speed (km/h)",
            "Combined Edge Case - Speed Over Time",
            args.output_dir / "speed_over_time.png",
        )
        _plot_action_distribution(rl_steps, vlm_steps, args.output_dir)
        _plot_vlm_semantics(vlm_steps, args.output_dir)
        if not rl_agg.empty and not vlm_agg.empty:
            _plot_linkedin_summary(rl_steps, vlm_steps, rl_agg, vlm_agg, args.output_dir)
    else:
        print("PLOT SKIP | time-series plots require both raw step CSVs")

    # Keep these reads visible to catch malformed episode files without making plots depend on them.
    if rl_episodes.empty or vlm_episodes.empty:
        print("PLOT NOTE | one or both episode CSVs are missing or empty")

    print(f"PLOTS SAVED | {args.output_dir}")


if __name__ == "__main__":
    main()
