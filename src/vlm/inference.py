from __future__ import annotations

import json
from typing import Dict, List, Any, Tuple

from tqdm import tqdm

from config import (
    SAMPLED_CSV_PATH,
    RAW_SCENARIOS_DIR,
    WINDOW_SIZE,
    IMAGE_NAME_COLUMN,
)
from utils_io import (
    load_sampled_dataset,
    sort_scenario_df,
    build_image_path,
    validate_image_path,
    group_by_scenario,
    create_windows_from_df,
)
from model_cpu import QwenVLInference


ALLOWED_VALUES = {
    "hazard_level": {"low", "medium", "high"},
    "obstacle_presence": {"yes", "no"},
    "collision_risk": {"yes", "no"},
    "lane_safety": {"safe", "unsafe"},
    "crossing_pedestrian_presence": {"yes", "no"},
    "action_urgency": {"continue", "slow", "brake"},
}

REQUIRED_KEYS = list(ALLOWED_VALUES.keys())

HAZARD_PRIORITY = {"low": 0, "medium": 1, "high": 2}
ACTION_URGENCY_PRIORITY = {"continue": 0, "slow": 1, "brake": 2}
BINARY_PRIORITY = {"no": 0, "yes": 1}
LANE_PRIORITY = {"safe": 0, "unsafe": 1}


def extract_json_object(raw_text: str) -> Dict[str, Any]:
    raw_text = raw_text.strip()

    # tolerate markdown fences if model outputs them
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    start = raw_text.find("{")
    end = raw_text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No valid JSON object found in model output: {raw_text}")

    json_str = raw_text[start:end + 1]
    return json.loads(json_str)


def normalize_value(key: str, value: str) -> str:
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


def validate_output(data: Dict[str, Any]) -> Dict[str, str]:
    validated: Dict[str, str] = {}

    for key in REQUIRED_KEYS:
        if key not in data:
            raise ValueError(f"Missing key: {key}")

        value = normalize_value(key, data[key])

        if value not in ALLOWED_VALUES[key]:
            raise ValueError(f"Invalid value for {key}: {value}")

        validated[key] = value

    return validated


def aggregate_window_outputs(outputs: List[Dict[str, str]]) -> Dict[str, str]:
    if not outputs:
        raise ValueError("Cannot aggregate an empty window")

    hazard = max(outputs, key=lambda x: HAZARD_PRIORITY[x["hazard_level"]])["hazard_level"]
    obstacle = max(outputs, key=lambda x: BINARY_PRIORITY[x["obstacle_presence"]])["obstacle_presence"]
    collision = max(outputs, key=lambda x: BINARY_PRIORITY[x["collision_risk"]])["collision_risk"]
    lane = max(outputs, key=lambda x: LANE_PRIORITY[x["lane_safety"]])["lane_safety"]
    pedestrian = max(
        outputs,
        key=lambda x: BINARY_PRIORITY[x["crossing_pedestrian_presence"]],
    )["crossing_pedestrian_presence"]
    urgency = max(outputs, key=lambda x: ACTION_URGENCY_PRIORITY[x["action_urgency"]])["action_urgency"]

    return {
        "hazard_level": hazard,
        "obstacle_presence": obstacle,
        "collision_risk": collision,
        "lane_safety": lane,
        "crossing_pedestrian_presence": pedestrian,
        "action_urgency": urgency,
    }


def run_full_inference() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    model = QwenVLInference()

    df = load_sampled_dataset(SAMPLED_CSV_PATH)

    per_frame_records: List[Dict[str, Any]] = []
    per_window_records: List[Dict[str, Any]] = []

    grouped = group_by_scenario(df)

    for scenario_name, scenario_df in grouped:
        scenario_df = sort_scenario_df(scenario_df)
        windows = create_windows_from_df(scenario_df, WINDOW_SIZE)

        for window_idx, window_df in enumerate(
            tqdm(windows, desc=f"Scenario {scenario_name}")
        ):
            window_outputs: List[Dict[str, str]] = []

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

                except Exception as e:
                    validated = {k: "" for k in REQUIRED_KEYS}
                    parse_status = "error"
                    error_message = str(e)

                    print("\n[FRAME ERROR]")
                    print("Scenario:", scenario_name)
                    print("Window:", f"{scenario_name}_window_{window_idx:04d}")
                    print("Row image:", row.get(IMAGE_NAME_COLUMN, ""))
                    print("Resolved path:", image_path if image_path is not None else "N/A")
                    print("Raw response:", raw_response if raw_response else "N/A")
                    print("Error:", error_message)

                frame_record = {
                    "scenario_name": scenario_name,
                    "window_id": f"{scenario_name}_window_{window_idx:04d}",
                    "image_name": str(row.get(IMAGE_NAME_COLUMN, "")),
                    "image_path": str(image_path) if image_path is not None else "",
                    "raw_model_output": raw_response,
                    "parse_status": parse_status,
                    "error_message": error_message,
                    **validated,
                }

                # preserve original CSV metadata
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
                aggregated = {k: "" for k in REQUIRED_KEYS}
                window_status = "error"
                window_error = "One or more frame outputs failed to parse"

            window_record = {
                "scenario_name": scenario_name,
                "window_id": f"{scenario_name}_window_{window_idx:04d}",
                "num_frames": len(window_df),
                "start_image": str(window_df.iloc[0][IMAGE_NAME_COLUMN]),
                "end_image": str(window_df.iloc[-1][IMAGE_NAME_COLUMN]),
                "window_status": window_status,
                "window_error": window_error,
                **aggregated,
            }

            per_window_records.append(window_record)

    return per_frame_records, per_window_records