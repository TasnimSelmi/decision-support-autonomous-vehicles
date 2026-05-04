from src.rl.action_space import DrivingAction


def compute_reward(obs, action_id: int, vlm_output: dict | None = None) -> float:
    reward = 1.0

    def _flag(*names: str) -> int:
        for name in names:
            if hasattr(obs, name):
                value = getattr(obs, name)
                try:
                    return 1 if float(value) != 0.0 else 0
                except Exception:
                    return 1 if bool(value) else 0
        return 0

    collision = _flag("collision", "collision_risk")
    lane_invasion = _flag("lane_invasion", "lane_unsafe")

    if collision:
        reward -= 100.0

    if lane_invasion:
        reward -= 20.0

    action = DrivingAction(action_id)

    if vlm_output is not None:
        hazard = vlm_output.get("hazard_level", "low")
        obstacle = vlm_output.get("obstacle_presence", "no")
        collision_risk = vlm_output.get("collision_risk", "no")

        if hazard == "high" and action == DrivingAction.BRAKE:
            reward += 20.0

        if hazard == "high" and action == DrivingAction.CONTINUE:
            reward -= 30.0

        if obstacle == "yes" and action in [DrivingAction.SLOW_DOWN, DrivingAction.BRAKE]:
            reward += 10.0

        if collision_risk == "yes" and action == DrivingAction.BRAKE:
            reward += 15.0

    return reward