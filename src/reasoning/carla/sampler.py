from __future__ import annotations

from pathlib import Path
import pandas as pd


EVENT_COLUMNS = ["collision", "lane_invasion"]


def read_carla_csv(csv_path: Path) -> pd.DataFrame:
    """
    Read a CARLA CSV file safely.
    Automatically detects comma or semicolon separators.
    """
    return pd.read_csv(csv_path, sep=None, engine="python")

def initialize_sampling_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sampling columns used to mark selected frames.
    """
    df = df.copy()
    df["sampled"] = 0
    df["sample_reason"] = ""
    return df


def add_timestep_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add timestep if it does not already exist.
    """
    df = df.copy()

    if "timestep" not in df.columns:
        df["timestep"] = range(len(df))

    return df


def has_event(df: pd.DataFrame) -> bool:
    """
    Check whether the scenario contains a collision or lane invasion event.
    """
    for col in EVENT_COLUMNS:
        if col in df.columns and (df[col] == 1).any():
            return True

    return False


def sample_sparse(df: pd.DataFrame, step: int = 25) -> pd.DataFrame:
    """
    Sample every `step` frames for normal scenarios.
    Always keeps the last frame.
    """
    df = initialize_sampling_columns(df)
    df = add_timestep_column(df)

    df = df.sort_values("timestep").reset_index(drop=True)

    selected_indices = df.iloc[::step].index

    df.loc[selected_indices, "sampled"] = 1
    df.loc[selected_indices, "sample_reason"] = "sparse"

    if len(df) > 0:
        last_idx = df.index[-1]
        df.loc[last_idx, "sampled"] = 1
        df.loc[last_idx, "sample_reason"] = "last_frame"

    return df


def sample_event_window(
    df: pd.DataFrame,
    before: int = 10,
    after: int = 5,
    fallback_step: int = 25,
) -> pd.DataFrame:
    """
    Sample frames around collision/lane invasion events.
    If no event exists, fallback to sparse sampling.
    """
    df = initialize_sampling_columns(df)
    df = add_timestep_column(df)

    df = df.sort_values("timestep").reset_index(drop=True)

    event_mask = pd.Series(False, index=df.index)

    for col in EVENT_COLUMNS:
        if col in df.columns:
            event_mask = event_mask | (df[col] == 1)

    event_indices = df[event_mask].index.tolist()

    if not event_indices:
        return sample_sparse(df, step=fallback_step)

    for event_idx in event_indices:
        start_idx = max(0, event_idx - before)
        end_idx = min(len(df) - 1, event_idx + after)

        window_indices = list(range(start_idx, end_idx + 1))

        df.loc[window_indices, "sampled"] = 1
        df.loc[window_indices, "sample_reason"] = "event_window"

        df.loc[event_idx, "sampled"] = 1
        df.loc[event_idx, "sample_reason"] = "event_frame"

    return df


def sample_scenario_csv(
    csv_path: str | Path,
    output_csv_path: str | Path | None = None,
    normal_step: int = 25,
    event_before: int = 10,
    event_after: int = 5,
) -> pd.DataFrame:
    """
    Apply sampling to one independent CARLA scenario CSV.

    If the scenario contains events, event-centered sampling is used.
    Otherwise, sparse sampling is used.
    """
    csv_path = Path(csv_path)
    df = read_carla_csv(csv_path)

    if has_event(df):
        sampled_df = sample_event_window(
            df,
            before=event_before,
            after=event_after,
            fallback_step=normal_step,
        )
    else:
        sampled_df = sample_sparse(
            df,
            step=normal_step,
        )

    if output_csv_path is not None:
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        sampled_df.to_csv(output_csv_path, index=False)

    return sampled_df


def get_sampled_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only sampled rows.
    """
    if "sampled" not in df.columns:
        raise ValueError("DataFrame does not contain a 'sampled' column.")

    return df[df["sampled"] == 1].copy().reset_index(drop=True)