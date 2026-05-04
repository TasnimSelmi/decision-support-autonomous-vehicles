from dataclasses import dataclass


@dataclass(frozen=True)
class DrivingState:
    hazard_level: int
    obstacle_presence: int
    collision_risk: int
    lane_unsafe: int
    pedestrian_presence: int
    speed_level: int

    def as_tuple(self) -> tuple[int, int, int, int, int, int]:
        return (
            self.hazard_level,
            self.obstacle_presence,
            self.collision_risk,
            self.lane_unsafe,
            self.pedestrian_presence,
            self.speed_level,
        )

    def as_vector(self) -> list[float]:
        return [
            self.hazard_level / 2.0,
            float(self.obstacle_presence),
            float(self.collision_risk),
            float(self.lane_unsafe),
            float(self.pedestrian_presence),
            self.speed_level / 2.0,
        ]