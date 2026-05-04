import carla


def apply_scenario(world, traffic_manager, bp, config):
    """
    NIGHT MODE SCENARIO

    - Low visibility (dark)
    - Fog presence
    - Normal traffic
    - Harder perception for VLM
    """

    # -----------------------------
    # WEATHER (night conditions)
    # -----------------------------
    weather = carla.WeatherParameters()

    weather.sun_altitude_angle = -90.0   # full night
    weather.sun_azimuth_angle = 0.0

    weather.cloudiness = 100.0
    weather.precipitation = 0.0
    weather.precipitation_deposits = 0.0

    weather.wind_intensity = 0.35

    weather.fog_density = 0.1
    weather.fog_distance = 20.0   # improved from 0

    weather.wetness = 0.0

    world.set_weather(weather)

    # -----------------------------
    # TRAFFIC BEHAVIOR
    # -----------------------------
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)

    # Optional: slightly slower driving at night
    traffic_manager.global_percentage_speed_difference(10.0)