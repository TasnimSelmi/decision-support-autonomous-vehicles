from __future__ import annotations

from typing import Any

import pandas as pd

from src.rl.schemas import DrivingState


def encode_hazard(value: Any) -> int:
    value = str(value).strip().lower()

    mapping = {
        "low": 0,
        "medium": 1,
        "high": 2,
        "unclear": 1,
        "unknown": 1,
        "nan": 1,
        "": 1,
    }

    return mapping.get(value, 1)


def encode_binary(value: Any) -> int:
    value = str(value).strip().lower()

    mapping = {
        "no": 0,
        "yes": 1,
        "false": 0,
        "true": 1,
        "0": 0,
        "1": 1,
        "unclear": 1,
        "unknown": 1,
        "nan": 1,
        "": 1,
    }

    return mapping.get(value, 1)


def encode_lane_safety(value: Any) -> int:
    value = str(value).strip().lower()

    mapping = {
        "safe": 0,
        "unsafe": 1,
        "unclear": 1,
        "unknown": 1,
        "nan": 1,
        "": 1,
    }

    return mapping.get(value, 1)


def discretize_speed(speed: Any) -> int:
    """
    CARLA speed in your dataset is approximately:
    - 0 to 10 in normal motion
    - almost 0 after collision

    We discretize it:
    0 = stopped / very slow
    1 = normal
    2 = fast
    """
    try:
        speed_value = float(speed)
    except (TypeError, ValueError):
        return 1

    if pd.isna(speed_value):
        return 1

    if speed_value < 1.0:
        return 0

    if speed_value < 8.0:
        return 1

    return 2


def encode_vlm_window_row(row: dict[str, Any]) -> DrivingState:
    """
    Convert one row from per_window_outputs.csv into an RL DrivingState.

    Expected columns:
    - hazard_level
    - obstacle_presence
    - collision_risk
    - lane_safety
    - crossing_pedestrian_presence
    - mean_speed
    """

    return DrivingState(
        hazard_level=encode_hazard(row.get("hazard_level", "unclear")),
        obstacle_presence=encode_binary(row.get("obstacle_presence", "unclear")),
        collision_risk=encode_binary(row.get("collision_risk", "unclear")),
        lane_unsafe=encode_lane_safety(row.get("lane_safety", "unclear")),
        pedestrian_presence=encode_binary(
            row.get("crossing_pedestrian_presence", "unclear")
        ),
        speed_level=discretize_speed(row.get("mean_speed", None)),
    )


def encode_vlm_row(row: dict[str, Any]) -> DrivingState:
    """
    Backward-compatible wrapper.

    If the row comes from per-window VLM output, use mean_speed.
    If it comes from per-frame output, use speed.
    """
    if "mean_speed" in row:
        return encode_vlm_window_row(row)

    row = dict(row)
    row["mean_speed"] = row.get("speed", None)
    return encode_vlm_window_row(row)