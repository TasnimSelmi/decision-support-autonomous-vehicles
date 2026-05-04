from pathlib import Path

try:
    import torch
except Exception:
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# =========================
# DATA PATHS
# =========================

RAW_SCENARIOS_DIR = PROJECT_ROOT / "data" / "raw" / "train"
SAMPLED_CSV_DIR = PROJECT_ROOT / "data" / "samples" / "carla"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "carla" / "vlm_results"
RAW_OUTPUTS_DIR = OUTPUT_DIR / "raw"
TABLES_DIR = OUTPUT_DIR / "tables"

RAW_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# SCENARIOS
# =========================

SCENARIOS = [
    "Night-mode",
    "Rainy night mode",
    "Collision",
    "carCollision",
]

# =========================
# COLUMNS
# =========================

FRAME_COLUMN = "frame"

POSSIBLE_ORDER_COLUMNS = [
    "timestep",
    "frame_id",
    "frame_index",
    "image_index",
    "timestamp",
]
# =========================
# MODEL
# =========================

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch is not None and DEVICE == "cuda" else None

MAX_NEW_TOKENS = 128
DO_SAMPLE = False
TEMPERATURE = 0.0

WINDOW_SIZE = 5

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}