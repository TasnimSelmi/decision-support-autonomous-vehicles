from __future__ import annotations

import numpy as np
import pandas as pd

from src.rl.action_space import action_to_name
from src.rl.agents.dqn import DQNAgent
from src.rl.config import RLConfig
from src.rl.state_encoder import encode_vlm_window_row
from src.rl.utils.io import ensure_dirs


def evaluate() -> None:
    config = RLConfig()
    ensure_dirs([config.result_dir])

    vlm_csv_path = (
        config.project_root
        / "outputs"
        / "carla"
        / "vlm_results"
        / "tables"
        / "per_window_outputs.csv"
    )

    df = pd.read_csv(vlm_csv_path)

    if "window_status" in df.columns:
        df = df[df["window_status"] == "ok"].copy()

    agent = DQNAgent(
        state_dim=config.state_dim,
        num_actions=config.num_actions,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        replay_capacity=config.replay_capacity,
    )

    agent.load(config.dqn_model_path)

    rows = []

    for _, row in df.iterrows():
        state = encode_vlm_window_row(row.to_dict())
        state_vector = np.array(state.as_vector(), dtype=np.float32)

        action = agent.select_action(
            state_vector=state_vector,
            epsilon=0.0,
        )

        rows.append(
            {
                "scenario_name": row.get("scenario_name", ""),
                "window_id": row.get("window_id", ""),
                "start_frame": row.get("start_frame", ""),
                "end_frame": row.get("end_frame", ""),
                "hazard_level": row.get("hazard_level", ""),
                "obstacle_presence": row.get("obstacle_presence", ""),
                "collision_risk": row.get("collision_risk", ""),
                "lane_safety": row.get("lane_safety", ""),
                "crossing_pedestrian_presence": row.get(
                    "crossing_pedestrian_presence",
                    "",
                ),
                "mean_speed": row.get("mean_speed", ""),
                "collision": row.get("collision", 0),
                "lane_invasion": row.get("lane_invasion", 0),
                "vlm_action_urgency": row.get("action_urgency", ""),
                "rl_action_id": int(action),
                "rl_action": action_to_name(action),
            }
        )

    result_df = pd.DataFrame(rows)

    output_path = config.result_dir / "vlm_carla_dqn_evaluation.csv"
    result_df.to_csv(output_path, index=False)

    print(result_df.head(20))
    print(f"\nSaved VLM+CARLA DQN evaluation to: {output_path}")


if __name__ == "__main__":
    evaluate()