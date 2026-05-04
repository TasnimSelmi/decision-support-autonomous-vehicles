from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Dict, Any

import pandas as pd

from src.vlm.config import (
    POSSIBLE_ORDER_COLUMNS,
    SUPPORTED_IMAGE_EXTENSIONS,
)


def load_sampled_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "frame" not in df.columns:
        raise KeyError("Missing 'frame' column")

    return df

def sort_scenario_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort scenario rows using the first available ordering column.
    If none exist, keep the original CSV order.
    """
    for col in POSSIBLE_ORDER_COLUMNS:
        if col in df.columns:
            return df.sort_values(by=col).reset_index(drop=True)

    return df.reset_index(drop=True)


def build_image_path(row: pd.Series, raw_scenarios_dir: Path) -> Path:
    from pathlib import Path

    scenario_name = row["scenario_name"]
    frame_value = row["frame"]

    frame_path = Path(frame_value)

    # try original path
    image_path = raw_scenarios_dir / scenario_name / frame_path

    if image_path.exists():
        return image_path

    # fallback (remove output/)
    fallback = raw_scenarios_dir / scenario_name / frame_path.name

    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Image not found for scenario={scenario_name}, frame={frame_value}"
    )


def validate_image_path(image_path: Path) -> None:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {image_path}")


def group_by_scenario(df: pd.DataFrame):
    return df.groupby(SCENARIO_COLUMN, sort=False)


def create_windows_from_df(df: pd.DataFrame, window_size: int) -> List[pd.DataFrame]:
    """
    Non-overlapping windows of fixed size.
    """
    windows: List[pd.DataFrame] = []
    rows = df.reset_index(drop=True)

    for i in range(0, len(rows), window_size):
        chunk = rows.iloc[i:i + window_size].copy()
        if len(chunk) == window_size:
            windows.append(chunk)

    return windows


def save_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_csv(records: List[Dict[str, Any]], path: Path) -> None:
    df = pd.DataFrame(records)
    df.to_csv(path, index=False)