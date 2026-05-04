from __future__ import annotations

from src.carla_scenarios.common import build_weather, set_dense_traffic, spawn_vehicle_ahead


def apply_scenario(world, traffic_manager, blueprint_library, config):
    world.set_weather(build_weather(sun_altitude=55.0, cloudiness=25.0))
    set_dense_traffic(traffic_manager, speed_difference=12.0)

    ego_transforms = world.get_map().get_spawn_points()
    if ego_transforms:
        spawn_vehicle_ahead(world, blueprint_library, ego_transforms[0], offset_meters=15.0)
