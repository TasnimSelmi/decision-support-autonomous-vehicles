from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.rl.action_space import DrivingAction
from src.rl.envs.base_env import BaseDrivingEnv
from src.rl.reward import compute_reward
from src.rl.schemas import DrivingState
from src.rl.state_encoder import encode_vlm_window_row


class OfflineVLMCarlaEnv(BaseDrivingEnv):
    """
    Offline RL environment based on VLM + CARLA window-level outputs.

    One row = one temporal window.

    The environment returns:
        state      = encoded VLM semantic state + speed level
        reward     = reward_function(state, action) + optional CARLA penalties
        next_state = next window in the same scenario
        done       = scenario ended, collision happened, or max steps reached

    Important:
    In offline mode, the action does not change the next CARLA frame.
    The next_state comes from the recorded sequence.
    """

    def __init__(
        self,
        csv_path: str | Path,
        max_steps_per_episode: int = 50,
        use_only_valid_windows: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.max_steps_per_episode = max_steps_per_episode
        self.use_only_valid_windows = use_only_valid_windows

        self.df = self._load_dataset(self.csv_path)
        self.scenario_names = self.df["scenario_name"].dropna().unique().tolist()

        if not self.scenario_names:
            raise ValueError("No scenarios found in dataset.")

        self.current_scenario_index = -1
        self.current_df: pd.DataFrame | None = None
        self.current_row_index = 0
        self.current_step = 0
        self.state: DrivingState | None = None

    def reset(self) -> DrivingState:
        """
        Start a new episode.

        Each episode uses one scenario sequence.
        We cycle through scenarios instead of always starting from the first one.
        """
        self.current_scenario_index = (
            self.current_scenario_index + 1
        ) % len(self.scenario_names)

        scenario_name = self.scenario_names[self.current_scenario_index]

        self.current_df = (
            self.df[self.df["scenario_name"] == scenario_name]
            .sort_values(by=self._get_sort_column())
            .reset_index(drop=True)
        )

        if self.current_df.empty:
            raise RuntimeError(f"Scenario has no rows: {scenario_name}")

        self.current_row_index = 0
        self.current_step = 0

        row = self.current_df.iloc[self.current_row_index].to_dict()
        self.state = encode_vlm_window_row(row)

        return self.state

    def step(self, action: DrivingAction) -> tuple[DrivingState, float, bool, dict]:
        if self.current_df is None or self.state is None:
            raise RuntimeError("Call reset() before step().")

        current_row = self.current_df.iloc[self.current_row_index].to_dict()

        reward = compute_reward(self.state, action)
        reward += self._carla_event_reward_adjustment(current_row, action)

        self.current_step += 1

        done = self._is_done(current_row)

        if not done:
            self.current_row_index += 1

            if self.current_row_index >= len(self.current_df):
                self.current_row_index = len(self.current_df) - 1
                done = True

        next_row = self.current_df.iloc[self.current_row_index].to_dict()
        next_state = encode_vlm_window_row(next_row)

        self.state = next_state

        info = {
            "scenario_name": current_row.get("scenario_name", ""),
            "window_id": current_row.get("window_id", ""),
            "action": action.name,
            "reward": reward,
            "collision": int(current_row.get("collision", 0)),
            "lane_invasion": int(current_row.get("lane_invasion", 0)),
            "mean_speed": float(current_row.get("mean_speed", 0.0)),
            "hazard_level": current_row.get("hazard_level", ""),
            "collision_risk": current_row.get("collision_risk", ""),
            "lane_safety": current_row.get("lane_safety", ""),
        }

        return next_state, reward, done, info

    def _load_dataset(self, csv_path: Path) -> pd.DataFrame:
        if not csv_path.exists():
            raise FileNotFoundError(f"VLM-RL dataset not found: {csv_path}")

        df = pd.read_csv(csv_path)

        required_columns = [
            "scenario_name",
            "window_id",
            "hazard_level",
            "obstacle_presence",
            "collision_risk",
            "lane_safety",
            "crossing_pedestrian_presence",
            "mean_speed",
            "collision",
            "lane_invasion",
        ]

        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            raise KeyError(
                f"Missing columns in {csv_path}: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        if self.use_only_valid_windows and "window_status" in df.columns:
            df = df[df["window_status"] == "ok"].copy()

        if df.empty:
            raise ValueError("Dataset is empty after filtering valid windows.")

        df["collision"] = pd.to_numeric(df["collision"], errors="coerce").fillna(0)
        df["lane_invasion"] = pd.to_numeric(
            df["lane_invasion"],
            errors="coerce",
        ).fillna(0)

        return df.reset_index(drop=True)

    def _get_sort_column(self) -> str:
        if self.current_df is not None:
            if "start_frame" in self.current_df.columns:
                return "start_frame"
            if "window_id" in self.current_df.columns:
                return "window_id"

        if "start_frame" in self.df.columns:
            return "start_frame"

        return "window_id"

    def _is_done(self, row: dict[str, Any]) -> bool:
        if self.current_step >= self.max_steps_per_episode:
            return True

        if self.current_df is None:
            return True

        if self.current_row_index >= len(self.current_df) - 1:
            return True

        # End episode when an actual CARLA collision occurred.
        if int(row.get("collision", 0)) == 1:
            return True

        return False

    @staticmethod
    def _carla_event_reward_adjustment(
        row: dict[str, Any],
        action: DrivingAction,
    ) -> float:
        """
        Add simulator-based reward corrections.

        This links VLM interpretation with CARLA ground-truth events:
        - If CARLA says collision happened, unsafe actions receive penalty.
        - If lane invasion happened, lane correction receives reward.
        """
        reward = 0.0

        collision = int(row.get("collision", 0))
        lane_invasion = int(row.get("lane_invasion", 0))

        if collision == 1:
            if action == DrivingAction.STOP:
                reward += 10.0
            else:
                reward -= 20.0

        if lane_invasion == 1:
            if action == DrivingAction.CORRECT_LANE:
                reward += 8.0
            elif action == DrivingAction.CONTINUE:
                reward -= 8.0

        return reward