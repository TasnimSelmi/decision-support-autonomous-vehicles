from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SAMPLED_CSV_PATH = PROJECT_ROOT / "data" / "samples" / "carla" / "sampled_dataset.csv"
RAW_SCENARIOS_DIR = PROJECT_ROOT / "data" / "raw" / "carla_scenarios"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "carla" / "vlm_results"
RAW_OUTPUTS_DIR = OUTPUT_DIR / "raw"
TABLES_DIR = OUTPUT_DIR / "tables"

RAW_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

SCENARIO_COLUMN = "scenario_name"
SCENARIO_ID_COLUMN = "scenario_id"
IMAGE_NAME_COLUMN = "frame_name"

SCENARIO_FOLDER_MAP = {
    "scenario_01": "scenario1-Normal",
    "scenario_02": "scenario2-LaneInvasion+collision",
    "scenario_03": "scenario3-collision",
    "scenario_04": "scenario4-carCollision",
}
POSSIBLE_ORDER_COLUMNS = [
    "timestep",
    "frame_id",
    "frame_index",
    "image_index",
    "timestamp",
]

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
TORCH_DTYPE = torch.float32
DEVICE = "cpu"

MAX_NEW_TOKENS = 96
DO_SAMPLE = False
TEMPERATURE = 0.0

WINDOW_SIZE = 5

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}