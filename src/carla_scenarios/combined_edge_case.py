from __future__ import annotations

from src.carla_scenarios.common import (
    build_weather,
    set_dense_traffic,
    spawn_pedestrians_nearby,
    spawn_vehicle_ahead,
)


def apply_scenario(world, traffic_manager, blueprint_library, config):
    """
    Combined edge-case scenario:
    - Night
    - Heavy rain
    - Wet road
    - Fog
    - Dense traffic
    - Pedestrians nearby
    - Vehicle ahead of the ego vehicle
    """

    # 1) Extreme weather: night + rain + wet road + fog
    world.set_weather(
        build_weather(
            sun_altitude=-80.0,
            cloudiness=100.0,
            precipitation=95.0,
            wetness=100.0,
            fog_density=30.0,
        )
    )

    # 2) Dense / slow traffic
    set_dense_traffic(
        traffic_manager,
        speed_difference=18.0,
    )

    # 3) Pedestrians around the scene
    spawn_pedestrians_nearby(
        world,
        blueprint_library,
        count=6,
    )

    # 4) Spawn a vehicle ahead of the ego vehicle
    ego_transform = getattr(config, "ego_transform", None)

    if ego_transform is None:
        ego_transforms = world.get_map().get_spawn_points()
        ego_transform = ego_transforms[0] if ego_transforms else None

    if ego_transform is not None:
        spawn_vehicle_ahead(
            world,
            blueprint_library,
            ego_transform,
            offset_meters=10.0,
        )