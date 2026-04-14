from config import RAW_OUTPUTS_DIR, TABLES_DIR
from utils_io import save_jsonl, save_csv
from inference import run_full_inference


def main() -> None:
    per_frame_records, per_window_records = run_full_inference()

    save_jsonl(per_frame_records, RAW_OUTPUTS_DIR / "per_frame_outputs.jsonl")
    save_jsonl(per_window_records, RAW_OUTPUTS_DIR / "per_window_outputs.jsonl")

    save_csv(per_frame_records, TABLES_DIR / "per_frame_outputs.csv")
    save_csv(per_window_records, TABLES_DIR / "per_window_outputs.csv")

    print("Done.")
    print(f"Per-frame JSONL: {RAW_OUTPUTS_DIR / 'per_frame_outputs.jsonl'}")
    print(f"Per-window JSONL: {RAW_OUTPUTS_DIR / 'per_window_outputs.jsonl'}")
    print(f"Per-frame CSV: {TABLES_DIR / 'per_frame_outputs.csv'}")
    print(f"Per-window CSV: {TABLES_DIR / 'per_window_outputs.csv'}")


if __name__ == "__main__":
    main()