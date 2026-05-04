from __future__ import annotations

import random

import carla


def build_weather(
    sun_altitude: float,
    cloudiness: float = 50.0,
    precipitation: float = 0.0,
    wetness: float = 0.0,
    fog_density: float = 0.0,
) -> carla.WeatherParameters:
    weather = carla.WeatherParameters()
    weather.sun_altitude_angle = sun_altitude
    weather.cloudiness = cloudiness
    weather.precipitation = precipitation
    weather.precipitation_deposits = precipitation
    weather.wetness = wetness
    weather.fog_density = fog_density
    weather.wind_intensity = 0.2
    return weather


def set_dense_traffic(traffic_manager, speed_difference: float = 10.0) -> None:
    traffic_manager.set_global_distance_to_leading_vehicle(2.0)
    traffic_manager.global_percentage_speed_difference(speed_difference)


def spawn_vehicle_ahead(world, blueprint_library, reference_transform: carla.Transform, offset_meters: float = 12.0):
    vehicle_blueprints = blueprint_library.filter("vehicle.*")
    if not vehicle_blueprints:
        return None

    vehicle_bp = random.choice(vehicle_blueprints)
    location = reference_transform.location
    hazard_transform = carla.Transform(
        carla.Location(
            x=location.x + offset_meters,
            y=location.y,
            z=location.z,
        ),
        reference_transform.rotation,
    )

    try:
        vehicle = world.spawn_actor(vehicle_bp, hazard_transform)
        vehicle.set_autopilot(False)
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        return vehicle
    except Exception:
        return None


def spawn_pedestrians_nearby(world, blueprint_library, count: int = 3):
    walker_bps = blueprint_library.filter("walker.pedestrian.*")
    if not walker_bps:
        return []

    pedestrians = []
    for _ in range(count):
        location = world.get_random_location_from_navigation()
        if location is None:
            continue

        walker_bp = random.choice(walker_bps)
        try:
            walker = world.spawn_actor(walker_bp, carla.Transform(location))
            pedestrians.append(walker)
        except Exception:
            continue

    return pedestrians