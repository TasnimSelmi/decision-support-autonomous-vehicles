import sys
from pathlib import Path

import pandas as pd
from src.reasoning.carla.sampler import apply_sampling_strategy, get_sampled_only

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


def main():
    input_dir = PROJECT_ROOT / "outputs" / "carla" / "sampled_frames"
    output_dir = PROJECT_ROOT / "outputs" / "carla" / "evaluation"
    master_df = pd.read_csv(input_dir / "master_dataset.csv")

    sampled_df = apply_sampling_strategy(
        master_df,
        normal_step=50,
        event_before=10,
        event_after=5
    )

    vlm_input_df = get_sampled_only(sampled_df)

    print("Master shape:", master_df.shape)
    print("Sampled shape:", vlm_input_df.shape)
    print(vlm_input_df.head(20))
    print(vlm_input_df["scenario_name"].value_counts())
    print(vlm_input_df["sample_reason"].value_counts())

    save_path = output_dir / "sampled_dataset.csv"
    vlm_input_df.to_csv(save_path, index=False)

    print(f"\nSaved sampled dataset to: {save_path}")


if __name__ == "__main__":
    main()