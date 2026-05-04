import numpy as np
import pandas as pd

from src.rl.action_space import action_to_name
from src.rl.agents.dqn import DQNAgent
from src.rl.config import RLConfig
from src.rl.schemas import DrivingState
from src.rl.utils.io import ensure_dirs


def evaluate() -> None:
    config = RLConfig()

    ensure_dirs([config.result_dir])

    agent = DQNAgent(
        state_dim=config.state_dim,
        num_actions=config.num_actions,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        replay_capacity=config.replay_capacity,
    )

    agent.load(config.dqn_model_path)

    test_cases = [
        {
            "case": "safe road",
            "state": DrivingState(0, 0, 0, 0, 0, 1),
            "expected": "continue",
        },
        {
            "case": "high hazard",
            "state": DrivingState(2, 1, 0, 0, 0, 2),
            "expected": "slow_down or stop",
        },
        {
            "case": "collision risk",
            "state": DrivingState(2, 1, 1, 0, 0, 2),
            "expected": "stop",
        },
        {
            "case": "pedestrian crossing",
            "state": DrivingState(2, 1, 0, 0, 1, 1),
            "expected": "stop",
        },
        {
            "case": "lane unsafe",
            "state": DrivingState(1, 0, 0, 1, 0, 1),
            "expected": "correct_lane",
        },
        {
            "case": "fast medium hazard",
            "state": DrivingState(1, 0, 0, 0, 0, 2),
            "expected": "slow_down",
        },
    ]

    rows = []

    for item in test_cases:
        state = item["state"]
        state_vector = np.array(state.as_vector(), dtype=np.float32)

        action = agent.select_action(
            state_vector=state_vector,
            epsilon=0.0,
        )

        rows.append(
            {
                "case": item["case"],
                "state": state.as_tuple(),
                "predicted_action_id": int(action),
                "predicted_action": action_to_name(action),
                "expected_behavior": item["expected"],
            }
        )

    result_df = pd.DataFrame(rows)
    result_df.to_csv(config.evaluation_path, index=False)

    print(result_df)
    print(f"\nSaved evaluation to: {config.evaluation_path}")


if __name__ == "__main__":
    evaluate()