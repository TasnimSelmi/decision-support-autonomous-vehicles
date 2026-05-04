from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RLConfig:
    seed: int = 42

    state_dim: int = 6

    num_actions: int = 6

    num_episodes: int = 1000
    max_steps_per_episode: int = 50

    gamma: float = 0.99
    learning_rate: float = 5e-4
    batch_size: int = 64
    replay_capacity: int = 50_000
    min_replay_size: int = 500

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5_000

    target_update_frequency: int = 200

    project_root: Path = Path(__file__).resolve().parents[2]

    output_dir: Path = project_root / "outputs" / "rl"
    model_dir: Path = output_dir / "models"
    log_dir: Path = output_dir / "logs"
    result_dir: Path = output_dir / "results"

    dqn_model_path: Path = model_dir / "dqn_agent.pt"
    training_log_path: Path = log_dir / "dqn_training_log.csv"
    evaluation_path: Path = result_dir / "dqn_evaluation_results.csv"