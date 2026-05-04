import numpy as np
from src.rl.schemas import CarlaObservation, DrivingState


def encode_yes_no(value: str) -> float:
    return 1.0 if str(value).lower() == "yes" else 0.0


def encode_hazard(value: str) -> float:
    return {
        "low": 0.0,
        "medium": 1.0,
        "high": 2.0,
        "unclear": 1.0,
    }.get(str(value).lower(), 0.0)


def encode_lane_safety(value: str) -> float:
    return 0.0 if str(value).lower() == "safe" else 1.0


def encode_action_urgency(value: str) -> float:
    return {
        "continue": 0.0,
        "slow": 1.0,
        "slow_down": 1.0,
        "brake": 2.0,
        "stop": 2.0,
    }.get(str(value).lower(), 0.0)


def build_rl_only_state(obs: CarlaObservation) -> np.ndarray:
    return np.array([
        obs.speed,
        obs.steering,
        obs.throttle,
        obs.brake,
        float(obs.collision),
        float(obs.lane_invasion),
    ], dtype=np.float32)


def build_vlm_rl_state(obs: CarlaObservation, vlm_output: dict) -> np.ndarray:
    return np.array([
        obs.speed,
        obs.steering,
        obs.throttle,
        obs.brake,
        float(obs.collision),
        float(obs.lane_invasion),
        encode_hazard(vlm_output.get("hazard_level", "low")),
        encode_yes_no(vlm_output.get("obstacle_presence", "no")),
        encode_yes_no(vlm_output.get("collision_risk", "no")),
        encode_lane_safety(vlm_output.get("lane_safety", "safe")),
        encode_action_urgency(vlm_output.get("action_urgency", "continue")),
    ], dtype=np.float32)


def encode_vlm_window_row(row: dict) -> DrivingState:
    def _as_float(value, default=0.0):
        try:
            return float(value)
        except Exception:
            return float(default)

    hazard_level = {
        "low": 0.0,
        "medium": 1.0,
        "high": 2.0,
    }.get(str(row.get("hazard_level", "low")).lower(), 0.0)

    obstacle_presence = encode_yes_no(row.get("obstacle_presence", "no"))
    collision_risk = encode_yes_no(row.get("collision_risk", "no"))
    lane_unsafe = 0.0 if str(row.get("lane_safety", "safe")).lower() == "safe" else 1.0
    pedestrian_presence = encode_yes_no(
        row.get("crossing_pedestrian_presence", row.get("pedestrian_presence", "no"))
    )
    speed_level = _as_float(row.get("mean_speed", row.get("speed", 0.0)))

    return DrivingState(
        hazard_level=hazard_level,
        obstacle_presence=obstacle_presence,
        collision_risk=collision_risk,
        lane_unsafe=lane_unsafe,
        pedestrian_presence=pedestrian_presence,
        speed_level=speed_level,
    )