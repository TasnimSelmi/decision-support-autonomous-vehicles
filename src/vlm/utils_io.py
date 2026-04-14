from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Dict, Any

import pandas as pd

from config import (
    SCENARIO_COLUMN,
    IMAGE_NAME_COLUMN,
    POSSIBLE_ORDER_COLUMNS,
    SUPPORTED_IMAGE_EXTENSIONS,
)


def load_sampled_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if SCENARIO_COLUMN not in df.columns:
        raise KeyError(
            f"Missing required scenario column '{SCENARIO_COLUMN}'. "
            f"Available columns: {list(df.columns)}"
        )

    if IMAGE_NAME_COLUMN not in df.columns:
        raise KeyError(
            f"Missing required image column '{IMAGE_NAME_COLUMN}'. "
            f"Available columns: {list(df.columns)}"
        )

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
    from config import (
        SCENARIO_ID_COLUMN,
        IMAGE_NAME_COLUMN,
        SCENARIO_FOLDER_MAP,
    )

    scenario_id = str(row[SCENARIO_ID_COLUMN])
    frame_name = str(row[IMAGE_NAME_COLUMN])

    if scenario_id not in SCENARIO_FOLDER_MAP:
        raise KeyError(f"Unknown scenario_id: {scenario_id}")

    folder_name = SCENARIO_FOLDER_MAP[scenario_id]

    # remove fake folder prefixes like output/ or output_headon/
    clean_name = Path(frame_name).name

    return raw_scenarios_dir / folder_name / folder_name / clean_name


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