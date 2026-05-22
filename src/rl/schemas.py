from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple


@dataclass
class CarlaObservation:
    frame_path: str
    speed: float
    steering: float
    throttle: float
    brake: float
    collision: int
    lane_invasion: int


@dataclass
class DrivingState:
    hazard_level: float
    obstacle_presence: float
    collision_risk: float
    lane_unsafe: float
    pedestrian_presence: float
    speed_level: float

    def as_vector(self) -> list[float]:
        return [
            float(self.hazard_level),
            float(self.obstacle_presence),
            float(self.collision_risk),
            float(self.lane_unsafe),
            float(self.pedestrian_presence),
            float(self.speed_level),
        ]

    def as_tuple(self) -> Tuple[float, float, float, float, float, float]:
        return tuple(self.as_vector())


@dataclass
class StepResult:
    observation: CarlaObservation
    reward: float
    done: bool
    info: Dict[str, Any]
