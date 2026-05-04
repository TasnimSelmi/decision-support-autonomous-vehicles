from __future__ import annotations

import json
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.vlm.config import (
    SCENARIOS,
    SAMPLED_CSV_DIR,
    RAW_SCENARIOS_DIR,
    WINDOW_SIZE,
)

from src.vlm.utils_io import (
    sort_scenario_df,
    build_image_path,
    validate_image_path,
    create_windows_from_df,
)

from src.vlm.model_gpu import QwenVLInference


# =========================
# VALID VLM OUTPUT VALUES
# =========================

ALLOWED_VALUES = {
    "hazard_level": {"low", "medium", "high"},
    "obstacle_presence": {"yes", "no"},
    "collision_risk": {"yes", "no"},
    "lane_safety": {"safe", "unsafe"},
    "crossing_pedestrian_presence": {"yes", "no"},
    "action_urgency": {"continue", "slow", "brake"},
}

REQUIRED_KEYS = list(ALLOWED_VALUES.keys())


# =========================
# PRIORITIES FOR WINDOW AGGREGATION
# =========================

HAZARD_PRIORITY = {"low": 0, "medium": 1, "high": 2}
ACTION_URGENCY_PRIORITY = {"continue": 0, "slow": 1, "brake": 2}
BINARY_PRIORITY = {"no": 0, "yes": 1}
LANE_PRIORITY = {"safe": 0, "unsafe": 1}


# =========================
# DATA LOADING
# =========================

def load_all_sampled_scenarios() -> pd.DataFrame:
    """
    Load the sampled CSV files for all CARLA scenarios.

    Each sampled CSV does not originally contain scenario_name,
    so we add it here to preserve scenario identity.
    """
    all_dfs: list[pd.DataFrame] = []

    for scenario_name in SCENARIOS:
        csv_name = f"{scenario_name.replace(' ', '_')}.csv"
        csv_path = SAMPLED_CSV_DIR / csv_name

        if not csv_path.exists():
            raise FileNotFoundError(f"Sampled CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)

        if "frame" not in df.columns:
            raise KeyError(
                f"Missing 'frame' column in {csv_path}. "
                f"Available columns: {list(df.columns)}"
            )

        df["scenario_name"] = scenario_name
        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


# =========================
# JSON PARSING
# =========================

def extract_json_object(raw_text: str) -> dict[str, Any]:
    """
    Extract JSON object from raw VLM response.
    Handles cases where the model accidentally adds markdown fences.
    """
    raw_text = raw_text.strip()
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    start = raw_text.find("{")
    end = raw_text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No valid JSON object found in model output: {raw_text}")

    json_str = raw_text[start:end + 1]
    return json.loads(json_str)


def normalize_value(key: str, value: str) -> str:
    """
    Normalize small variations in model output.
    Example: 'slow down' -> 'slow'
    """
    value = str(value).strip().lower()

    replacements = {
        "hazard_level": {
            "low risk": "low",
            "medium risk": "medium",
            "moderate": "medium",
            "high risk": "high",
        },
        "obstacle_presence": {
            "present": "yes",
            "true": "yes",
            "false": "no",
        },
        "collision_risk": {
            "present": "yes",
            "true": "yes",
            "false": "no",
        },
        "lane_safety": {
            "safe lane": "safe",
            "unsafe lane": "unsafe",
        },
        "crossing_pedestrian_presence": {
            "present": "yes",
            "true": "yes",
            "false": "no",
        },
        "action_urgency": {
            "slow down": "slow",
            "braking": "brake",
            "continue driving": "continue",
        },
    }

    return replacements.get(key, {}).get(value, value)


def validate_output(data: dict[str, Any]) -> dict[str, str]:
    """
    Ensure the VLM output has exactly the values expected by RL.
    """
    validated: dict[str, str] = {}

    for key in REQUIRED_KEYS:
        if key not in data:
            raise ValueError(f"Missing key: {key}")

        value = normalize_value(key, data[key])

        if value not in ALLOWED_VALUES[key]:
            raise ValueError(f"Invalid value for {key}: {value}")

        validated[key] = value

    return validated


# =========================
# WINDOW AGGREGATION
# =========================

def aggregate_window_outputs(outputs: list[dict[str, str]]) -> dict[str, str]:
    """
    Aggregate frame-level VLM outputs into one window-level output.

    Conservative logic:
    - highest hazard wins
    - yes beats no for obstacle/collision/pedestrian
    - unsafe beats safe
    - brake beats slow beats continue
    """
    if not outputs:
        raise ValueError("Cannot aggregate an empty window.")

    return {
        "hazard_level": max(
            outputs,
            key=lambda x: HAZARD_PRIORITY[x["hazard_level"]],
        )["hazard_level"],

        "obstacle_presence": max(
            outputs,
            key=lambda x: BINARY_PRIORITY[x["obstacle_presence"]],
        )["obstacle_presence"],

        "collision_risk": max(
            outputs,
            key=lambda x: BINARY_PRIORITY[x["collision_risk"]],
        )["collision_risk"],

        "lane_safety": max(
            outputs,
            key=lambda x: LANE_PRIORITY[x["lane_safety"]],
        )["lane_safety"],

        "crossing_pedestrian_presence": max(
            outputs,
            key=lambda x: BINARY_PRIORITY[x["crossing_pedestrian_presence"]],
        )["crossing_pedestrian_presence"],

        "action_urgency": max(
            outputs,
            key=lambda x: ACTION_URGENCY_PRIORITY[x["action_urgency"]],
        )["action_urgency"],
    }


# =========================
# MAIN INFERENCE
# =========================

def run_full_inference() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Run VLM inference on all sampled CARLA scenario frames.

    Returns:
    - per_frame_records: one row per sampled frame
    - per_window_records: one row per temporal window
    """
    model = QwenVLInference()

    df = load_all_sampled_scenarios()

    per_frame_records: list[dict[str, Any]] = []
    per_window_records: list[dict[str, Any]] = []

    for scenario_name, scenario_df in df.groupby("scenario_name", sort=False):
        scenario_df = sort_scenario_df(scenario_df)
        windows = create_windows_from_df(scenario_df, WINDOW_SIZE)

        for window_idx, window_df in enumerate(
            tqdm(windows, desc=f"VLM inference | {scenario_name}")
        ):
            window_id = f"{scenario_name}_window_{window_idx:04d}"
            window_outputs: list[dict[str, str]] = []

            for _, row in window_df.iterrows():
                image_path = None
                raw_response = ""

                try:
                    image_path = build_image_path(row, RAW_SCENARIOS_DIR)
                    validate_image_path(image_path)

                    raw_response = model.infer_image(image_path)

                    parsed = extract_json_object(raw_response)
                    validated = validate_output(parsed)

                    parse_status = "ok"
                    error_message = ""

                except Exception as exc:
                    validated = {key: "" for key in REQUIRED_KEYS}
                    parse_status = "error"
                    error_message = str(exc)

                    print("\n[FRAME ERROR]")
                    print("Scenario:", scenario_name)
                    print("Window:", window_id)
                    print("Frame:", row.get("frame", "N/A"))
                    print("Resolved path:", image_path if image_path is not None else "N/A")
                    print("Raw response:", raw_response if raw_response else "N/A")
                    print("Error:", error_message)

                frame_record = {
                    "scenario_name": scenario_name,
                    "window_id": window_id,
                    "frame": str(row.get("frame", "")),
                    "image_path": str(image_path) if image_path is not None else "",
                    "raw_model_output": raw_response,
                    "parse_status": parse_status,
                    "error_message": error_message,
                    **validated,
                }

                # Keep original CARLA metadata for RL:
                # speed, steering, throttle, brake, collision, lane_invasion,
                # sampled, sample_reason, timestep, etc.
                for col in row.index:
                    if col not in frame_record:
                        frame_record[col] = row[col]

                per_frame_records.append(frame_record)

                if parse_status == "ok":
                    window_outputs.append(validated)

            if len(window_outputs) == len(window_df):
                aggregated = aggregate_window_outputs(window_outputs)
                window_status = "ok"
                window_error = ""
            else:
                aggregated = {key: "" for key in REQUIRED_KEYS}
                window_status = "error"
                window_error = "One or more frame outputs failed."

            window_record = {
                "scenario_name": scenario_name,
                "window_id": window_id,
                "num_frames": len(window_df),
                "start_frame": str(window_df.iloc[0]["frame"]),
                "end_frame": str(window_df.iloc[-1]["frame"]),
                "window_status": window_status,
                "window_error": window_error,
                **aggregated,
            }

            # Keep useful physical metadata at window level too
            if "speed" in window_df.columns:
                window_record["mean_speed"] = window_df["speed"].mean()
                window_record["max_speed"] = window_df["speed"].max()

            if "collision" in window_df.columns:
                window_record["collision"] = int(window_df["collision"].max())

            if "lane_invasion" in window_df.columns:
                window_record["lane_invasion"] = int(window_df["lane_invasion"].max())

            if "sample_reason" in window_df.columns:
                window_record["sample_reasons"] = ",".join(
                    sorted(set(window_df["sample_reason"].astype(str)))
                )

            per_window_records.append(window_record)

    return per_frame_records, per_window_records