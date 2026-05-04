import carla
import random


def apply_scenario(world, traffic_manager, bp, config):
    """
    CAR COLLISION SCENARIO

    - Ego is controlled by RL
    - A front vehicle creates a sudden hazard
    - RL must brake / avoid collision
    """

    # -----------------------------
    # NORMAL WEATHER (keep focus on behavior)
    # -----------------------------
    weather = carla.WeatherParameters()

    weather.sun_altitude_angle = 60.0
    weather.cloudiness = 20.0
    weather.precipitation = 0.0
    weather.wetness = 0.0

    world.set_weather(weather)

    # -----------------------------
    # TRAFFIC BEHAVIOR
    # -----------------------------
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)

    # -----------------------------
    # SPAWN DANGEROUS VEHICLE
    # -----------------------------
    spawn_points = world.get_map().get_spawn_points()

    try:
        # Choose a spawn point (ego is handled in env)
        base_transform = random.choice(spawn_points)

        # Place vehicle slightly ahead
        danger_transform = carla.Transform(
            carla.Location(
                x=base_transform.location.x + 10,
                y=base_transform.location.y,
                z=base_transform.location.z
            ),
            base_transform.rotation
        )

        vehicle_bp = bp.filter("vehicle.*")[0]

        danger_vehicle = world.spawn_actor(vehicle_bp, danger_transform)

        # Disable autopilot → manual control
        danger_vehicle.set_autopilot(False)

        # Force sudden braking behavior
        danger_vehicle.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                brake=1.0
            )
        )

    except Exception as e:
        print("Collision scenario spawn failed:", e)