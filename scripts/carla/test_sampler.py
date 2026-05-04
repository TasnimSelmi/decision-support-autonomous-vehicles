from pathlib import Path
from src.reasoning.carla.sampler import (
    sample_scenario_csv,
    get_sampled_only
)

PROJECT_ROOT = Path("/home/sameh/VLM_RL project")

data_root = PROJECT_ROOT / "data" / "raw" / "train"
samples_root = PROJECT_ROOT / "data" / "samples" / "carla"

samples_root.mkdir(parents=True, exist_ok=True)

scenarios = [
    "Night-mode",
    "Rainy night mode",
    "Collision",
    "carCollision",
]

for scenario in scenarios:
    csv_path = data_root / scenario / "data.csv"

    output_csv = samples_root / f"{scenario.replace(' ', '_')}.csv"

    # 1. Apply sampling
    sampled_df = sample_scenario_csv(
        csv_path=csv_path,
        output_csv_path=None,  
        normal_step=25,      
        event_before=10,
        event_after=5,
    )

    # 2. Keep only sampled rows
    vlm_input_df = get_sampled_only(sampled_df)

    # 3. Save ONLY sampled rows
    vlm_input_df.to_csv(output_csv, index=False)

    print(f"Sampled ONLY: {scenario} → {output_csv}")
    print(f"Shape: {vlm_input_df.shape}")
    print(vlm_input_df["sample_reason"].value_counts())
    print("------")