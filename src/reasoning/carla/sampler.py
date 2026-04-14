from __future__ import annotations

import pandas as pd


def initialize_sampling_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sampling columns if they do not already exist.
    """
    df = df.copy()

    if "sampled" not in df.columns:
        df["sampled"] = 0

    if "sample_reason" not in df.columns:
        df["sample_reason"] = ""

    return df


def sample_normal_scenario(
    df: pd.DataFrame,
    scenario_id: str,
    step: int = 50
) -> pd.DataFrame:
    """
    Sparse sampling for normal scenarios:
    keep every `step`-th frame.
    """
    df = df.copy()

    scenario_mask = df["scenario_id"] == scenario_id
    scenario_df = df[scenario_mask].sort_values("timestep")

    selected_indices = scenario_df.iloc[::step].index

    df.loc[selected_indices, "sampled"] = 1
    df.loc[selected_indices, "sample_reason"] = "normal_sparse"

    # Make sure the last frame is also included
    if len(scenario_df) > 0:
        last_idx = scenario_df.index[-1]
        df.loc[last_idx, "sampled"] = 1
        df.loc[last_idx, "sample_reason"] = "normal_last_frame"

    return df


def sample_event_window(
    df: pd.DataFrame,
    scenario_id: str,
    before: int = 10,
    after: int = 5
) -> pd.DataFrame:
    """
    Event-centered sampling:
    for each event frame (collision or lane_invasion),
    keep a window of frames around it.
    """
    df = df.copy()

    scenario_mask = df["scenario_id"] == scenario_id
    scenario_df = df[scenario_mask].sort_values("timestep")

    event_df = scenario_df[
        (scenario_df["collision"] == 1) | (scenario_df["lane_invasion"] == 1)
    ]

    if event_df.empty:
        return df

    scenario_indices = scenario_df.index.tolist()

    for event_idx in event_df.index:
        pos = scenario_indices.index(event_idx)

        start_pos = max(0, pos - before)
        end_pos = min(len(scenario_indices) - 1, pos + after)

        selected_window = scenario_indices[start_pos:end_pos + 1]

        for idx in selected_window:
            if df.loc[idx, "sampled"] == 0:
                df.loc[idx, "sampled"] = 1
                df.loc[idx, "sample_reason"] = "event_window"

        # Mark the exact event frame explicitly
        df.loc[event_idx, "sampled"] = 1
        df.loc[event_idx, "sample_reason"] = "event_frame"

    return df


def apply_sampling_strategy(
    df: pd.DataFrame,
    normal_step: int = 50,
    event_before: int = 10,
    event_after: int = 5
) -> pd.DataFrame:
    """
    Apply scenario-aware sampling strategy to the full master dataframe.

    Strategy:
    - normal scenarios -> sparse sampling
    - hazard/collision scenarios -> event-centered sampling
    """
    df = initialize_sampling_columns(df)

    for scenario_id in df["scenario_id"].unique():
        scenario_df = df[df["scenario_id"] == scenario_id]

        if scenario_df.empty:
            continue

        scenario_type = scenario_df["scenario_type"].iloc[0]

        if scenario_type == "normal":
            df = sample_normal_scenario(
                df,
                scenario_id=scenario_id,
                step=normal_step
            )
        else:
            df = sample_event_window(
                df,
                scenario_id=scenario_id,
                before=event_before,
                after=event_after
            )

    return df


def get_sampled_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only sampled rows, sorted for clean inference.
    """
    sampled_df = df[df["sampled"] == 1].copy()
    sampled_df = sampled_df.sort_values(["scenario_id", "timestep"]).reset_index(drop=True)
    return sampled_df