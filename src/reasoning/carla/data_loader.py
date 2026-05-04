from pathlib import Path
import json
import pandas as pd


CANONICAL_COLUMNS = [
    "scenario_id",
    "scenario_name",
    "scenario_type",
    "primary_event",
    "timestep",
    "frame_name",
    "image_path",
    "speed",
    "steering",
    "throttle",
    "brake",
    "collision",
    "lane_invasion",
    "event_flag",
    "done"
]


def load_single_scenario(scenario_dir):
    scenario_dir = Path(scenario_dir)

    # Handle accidental extra nested folder with same scenario contents
    if not (scenario_dir / "meta.json").exists():
        subdirs = [p for p in scenario_dir.iterdir() if p.is_dir()]
        if len(subdirs) == 1 and (subdirs[0] / "meta.json").exists():
            scenario_dir = subdirs[0]

    # read metadata
    with open(scenario_dir / "meta.json", "r") as f:
        meta = json.load(f)

    # read csv
    df = pd.read_csv(scenario_dir / "data.csv", sep=";")
    # normalize frame column name if needed
    if "frame" not in df.columns:
        raise ValueError(f"'frame' column not found in {scenario_dir / 'data.csv'}")

    # build canonical dataframe row by row
    rows = []
    last_index = len(df) - 1

    for idx, row in df.iterrows():
        frame_name = str(row["frame"])
        image_path = scenario_dir / frame_name
        # timestep
        try:
            timestep = int(Path(frame_name).stem)
        except ValueError:
            timestep = idx

        collision = int(row["collision"])
        lane_invasion = int(row["lane_invasion"])

        event_flag = 1 if (collision == 1 or lane_invasion == 1) else 0
        done = 1 if (collision == 1 or idx == last_index) else 0

        rows.append({
            "scenario_id": meta["scenario_id"],
            "scenario_name": meta["scenario_name"],
            "scenario_type": meta["scenario_type"],
            "primary_event": meta["primary_event"],
            "timestep": timestep,
            "frame_name": frame_name,
            "image_path": str(image_path),
            "speed": float(row["speed"]),
            "steering": float(row["steering"]),
            "throttle": float(row["throttle"]),
            "brake": float(row["brake"]),
            "collision": collision,
            "lane_invasion": lane_invasion,
            "event_flag": event_flag,
            "done": done
        })

    out_df = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    return out_df

def load_all_scenarios(dataset_root):
    dataset_root = Path(dataset_root)

    all_dfs = []
    for scenario_dir in sorted(dataset_root.iterdir()):
        if scenario_dir.is_dir():
            try:
                scenario_df = load_single_scenario(scenario_dir)
                all_dfs.append(scenario_df)
                print(f"Loaded {scenario_dir.name}: {len(scenario_df)} rows")
            except Exception as e:
                print(f"Skipping {scenario_dir.name}: {e}")

    if not all_dfs:
        raise ValueError("No valid scenario folders found")

    master_df = pd.concat(all_dfs, ignore_index=True)
    return master_df