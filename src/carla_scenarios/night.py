from __future__ import annotations

from src.carla_scenarios.common import build_weather, set_dense_traffic


def apply_scenario(world, traffic_manager, blueprint_library, config):
    world.set_weather(
        build_weather(
            sun_altitude=-75.0,
            cloudiness=65.0,
            precipitation=0.0,
            wetness=0.0,
            fog_density=12.0,
        )
    )
    set_dense_traffic(traffic_manager, speed_difference=8.0)
