from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RLConfig:
    project_root: Path = Path(__file__).resolve().parents[2]

    # CARLA connection
    carla_host: str = "localhost"  # use host machine IP if CARLA runs outside VM
    carla_port: int = 2000
    tm_port: int = 8000
    fixed_delta_seconds: float = 0.05

    # Online execution
    online_episodes: int = 1
    max_steps_per_episode: int = 500

    # General RL defaults
    seed: int = 42
    num_episodes: int = 50
    learning_rate: float = 1e-3
    gamma: float = 0.99
    replay_capacity: int = 10000
    batch_size: int = 64
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5000
    target_update_frequency: int = 250

    # Camera
    image_width: int = 640
    image_height: int = 360
    camera_fov: int = 90

    # VLM latency/windowing
    vlm_interval_steps: int = 2
    vlm_window_size: int = 5

    # Scenarios
    train_scenarios: list[str] = field(default_factory=lambda: [
        "rain",
        "night",
        "car_collision",
        "pedestrian_collision",
    ])

    test_scenarios: list[str] = field(default_factory=lambda: [
        "combined_edge_case",
    ])

    # Outputs
    online_output_dir: Path = project_root / "outputs" / "carla" / "online"

    # Models
    rl_only_model_path: Path = project_root / "models" / "rl_only" / "dqn_model.pth"
    vlm_rl_model_path: Path = project_root / "models" / "vlm_rl" / "dqn_model.pth"

    # Dimensions
    state_dim: int = 6
    num_actions: int = 6
    rl_only_state_dim: int = 6
    vlm_rl_state_dim: int = 11
    action_dim: int = 6

    # Offline training/evaluation outputs
    model_dir: Path = project_root / "models"
    log_dir: Path = project_root / "outputs" / "logs"
    result_dir: Path = project_root / "outputs" / "results"
    dqn_model_path: Path = project_root / "models" / "dqn_model.pth"
    training_log_path: Path = project_root / "outputs" / "logs" / "dqn_training.csv"
    evaluation_path: Path = project_root / "outputs" / "results" / "dqn_evaluation.csv"
