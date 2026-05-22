from __future__ import annotations

from src.rl.action_space import DrivingAction


def compute_reward(obs, action_id: int, vlm_output: dict | None = None) -> float:
    reward = 0.0
    action = DrivingAction(action_id)

    def _get(name: str, default: float = 0.0) -> float:
        if hasattr(obs, name):
            try:
                return float(getattr(obs, name))
            except Exception:
                return default
        return default

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

    if hasattr(obs, "speed_kmh"):
        speed_kmh = _get("speed_kmh", 0.0)
    else:
        speed_kmh = _get("speed", 0.0) * 3.6
    steer = abs(_get("steer", 0.0))
    brake = _get("brake", 0.0)

    # 1) Basic survival / progress
    reward += 0.2

    # 2) Rain driving: prefer controlled moderate speed
    if 10.0 <= speed_kmh <= 35.0:
        reward += 1.0
    elif speed_kmh < 3.0:
        reward -= 1.0
    elif speed_kmh > 45.0:
        reward -= 1.5

    # 3) Lane guidance
    if lane_invasion:
        reward -= 10.0
    else:
        reward += 0.5

    # 4) Smooth control under rain
    reward -= 0.2 * steer
    reward -= 0.1 * brake

    # 5) Action-specific shaping
    if action == DrivingAction.CONTINUE and not lane_invasion and speed_kmh >= 10.0:
        reward += 0.5

    if action == DrivingAction.SLOW_DOWN and speed_kmh > 35.0:
        reward += 0.8

    if action == DrivingAction.BRAKE and speed_kmh < 15.0 and not collision:
        reward -= 0.8  # avoid useless braking

    # 6) Hard safety penalties
    if collision:
        reward -= 100.0

    # 7) VLM semantic guidance, if available
    if vlm_output is not None:
        hazard = vlm_output.get("hazard_level", "low")
        obstacle = vlm_output.get("obstacle_presence", "no")
        collision_risk = vlm_output.get("collision_risk", "no")
        lane_safety = vlm_output.get("lane_safety", "safe")

        if lane_safety == "unsafe" and action in [DrivingAction.SLOW_DOWN, DrivingAction.BRAKE]:
            reward += 3.0

        if lane_safety == "unsafe" and action == DrivingAction.CONTINUE:
            reward -= 5.0

        if hazard == "high" and action == DrivingAction.BRAKE:
            reward += 8.0

        if hazard == "high" and action == DrivingAction.CONTINUE:
            reward -= 8.0

        if obstacle == "yes" and action in [DrivingAction.SLOW_DOWN, DrivingAction.BRAKE]:
            reward += 4.0

        if collision_risk == "yes" and action == DrivingAction.BRAKE:
            reward += 6.0

    return reward
