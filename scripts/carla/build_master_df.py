import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.reasoning.carla.data_loader import load_all_scenarios


def main():
    dataset_root = PROJECT_ROOT / "data" / "raw" / "carla_scenarios"
    output_dir = PROJECT_ROOT / "outputs" / "evaluation" / "carla"
    output_dir.mkdir(parents=True, exist_ok=True)

    master_df = load_all_scenarios(dataset_root)

    print("\nMaster dataset preview:")
    print(master_df.head())
    print("\nShape:", master_df.shape)
    print("\nColumns:", master_df.columns.tolist())

    save_path = output_dir / "master_dataset.csv"
    master_df.to_csv(save_path, index=False)

    print(f"\nSaved master dataset to: {save_path}")


if __name__ == "__main__":
    main()