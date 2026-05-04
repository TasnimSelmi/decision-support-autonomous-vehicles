from __future__ import annotations

from src.carla_scenarios.common import (
    build_weather,
    set_dense_traffic,
    spawn_pedestrians_nearby,
    spawn_vehicle_ahead,
)


def apply_scenario(world, traffic_manager, blueprint_library, config):
    world.set_weather(
        build_weather(
            sun_altitude=-80.0,
            cloudiness=100.0,
            precipitation=95.0,
            wetness=100.0,
            fog_density=30.0,
        )
    )
    set_dense_traffic(traffic_manager, speed_difference=18.0)
    spawn_pedestrians_nearby(world, blueprint_library, count=6)

    ego_transforms = world.get_map().get_spawn_points()
    if ego_transforms:
        spawn_vehicle_ahead(world, blueprint_library, ego_transforms[0], offset_meters=10.0)
