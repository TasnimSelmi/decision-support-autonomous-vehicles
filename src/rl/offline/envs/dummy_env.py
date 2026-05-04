import random

from src.rl.action_space import DrivingAction
from src.rl.envs.base_env import BaseDrivingEnv
from src.rl.reward import compute_reward
from src.rl.schemas import DrivingState


class DummyDrivingEnv(BaseDrivingEnv):
    def __init__(self, max_steps: int = 50):
        self.max_steps = max_steps
        self.current_step = 0
        self.state: DrivingState | None = None

    def reset(self) -> DrivingState:
        self.current_step = 0
        self.state = self._sample_state()
        return self.state

    def step(self, action: DrivingAction) -> tuple[DrivingState, float, bool, dict]:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        reward = compute_reward(self.state, action)

        self.current_step += 1
        done = self.current_step >= self.max_steps

        next_state = self._sample_state()
        self.state = next_state

        info = {
            "step": self.current_step,
            "action": action.name,
            "reward": reward,
        }

        return next_state, reward, done, info

    @staticmethod
    def _sample_state() -> DrivingState:
        return DrivingState(
            hazard_level=random.choice([0, 1, 2]),
            obstacle_presence=random.choice([0, 1]),
            collision_risk=random.choice([0, 1]),
            lane_unsafe=random.choice([0, 1]),
            pedestrian_presence=random.choice([0, 1]),
            speed_level=random.choice([0, 1, 2]),
        )