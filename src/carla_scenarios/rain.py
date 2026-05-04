from __future__ import annotations

from src.carla_scenarios.common import build_weather, set_dense_traffic


def apply_scenario(world, traffic_manager, blueprint_library, config):
    world.set_weather(
        build_weather(
            sun_altitude=35.0,
            cloudiness=100.0,
            precipitation=85.0,
            wetness=95.0,
            fog_density=25.0,
        )
    )
    set_dense_traffic(traffic_manager, speed_difference=15.0)
