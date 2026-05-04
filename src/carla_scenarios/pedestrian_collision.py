from __future__ import annotations

from src.carla_scenarios.common import build_weather, set_dense_traffic, spawn_pedestrians_nearby


def apply_scenario(world, traffic_manager, blueprint_library, config):
    world.set_weather(build_weather(sun_altitude=55.0, cloudiness=20.0))
    set_dense_traffic(traffic_manager, speed_difference=10.0)
    spawn_pedestrians_nearby(world, blueprint_library, count=5)
