from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from src.rl.agents.replay_buffer import ReplayBuffer

try:
    from src.rl.offline.agents.dqn import DQNAgent as _OfflineDQNAgent
except Exception:
    _OfflineDQNAgent = None


if _OfflineDQNAgent is not None:

    class DQNAgent(_OfflineDQNAgent):
        def __init__(self, state_dim: int, action_dim: int, config) -> None:
            super().__init__(
                state_dim=state_dim,
                num_actions=action_dim,
                learning_rate=config.learning_rate,
                gamma=config.gamma,
                replay_capacity=config.replay_capacity,
            )

        def act(self, state_vector) -> int:
            action = self.select_action(np.asarray(state_vector, dtype=np.float32), epsilon=0.0)
            return int(action)

        def load(self, path: Path) -> None:
            if Path(path).exists():
                super().load(path)

else:

    class DQNAgent:
        def __init__(self, state_dim: int, action_dim: int, config) -> None:
            self.state_dim = state_dim
            self.num_actions = action_dim
            self.config = config
            self.replay_buffer = ReplayBuffer(config.replay_capacity)

        def _heuristic_policy(self, state_vector: np.ndarray) -> int:
            values = np.asarray(state_vector, dtype=np.float32).tolist()
            lane_unsafe = values[3] if len(values) > 3 else 0.0
            collision_risk = values[2] if len(values) > 2 else 0.0
            hazard_level = values[0] if len(values) > 0 else 0.0

            if collision_risk >= 1.0 or hazard_level >= 2.0:
                return 2
            if lane_unsafe >= 1.0:
                return 5
            if hazard_level >= 1.0:
                return 1
            return 0

        def select_action(self, state_vector: np.ndarray, epsilon: float):
            if random.random() < epsilon:
                return random.randrange(self.num_actions)
            return self._heuristic_policy(state_vector)

        def act(self, state_vector) -> int:
            return int(self.select_action(np.asarray(state_vector, dtype=np.float32), epsilon=0.0))

        def optimize(self, batch_size: int):
            return None

        def update_target_network(self) -> None:
            return None

        def save(self, path: Path) -> None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"state_dim": self.state_dim, "num_actions": self.num_actions}))

        def load(self, path: Path) -> None:
            path = Path(path)
            if path.exists():
                return None
