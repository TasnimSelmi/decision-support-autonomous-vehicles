from .rain import apply_scenario as apply_rain
from .night import apply_scenario as apply_night
from .car_collision import apply_scenario as apply_car_collision
from .pedestrian_collision import apply_scenario as apply_pedestrian_collision
from .combined_edge_case import apply_scenario as apply_combined_edge_case


SCENARIOS = {
    "rain": apply_rain,
    "night": apply_night,
    "car_collision": apply_car_collision,
    "pedestrian_collision": apply_pedestrian_collision,
    "combined_edge_case": apply_combined_edge_case,
}
