import carla


def apply_scenario(world, traffic_manager, bp, config):
    """
    DAY MODE SCENARIO

    - Clear weather
    - Normal traffic behavior
    - Vehicles and pedestrians handled by env
    - Ego vehicle controlled by RL (not autopilot)
    """

    # -----------------------------
    # WEATHER (clear day)
    # -----------------------------
    weather = carla.WeatherParameters()

    weather.sun_altitude_angle = 75.0   # bright day
    weather.cloudiness = 10.0
    weather.precipitation = 0.0
    weather.precipitation_deposits = 0.0
    weather.wetness = 0.0
    weather.fog_density = 0.0

    world.set_weather(weather)

    # -----------------------------
    # TRAFFIC BEHAVIOR
    # -----------------------------
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)

    # Optional: make traffic slightly dynamic
    traffic_manager.global_percentage_speed_difference(0.0)