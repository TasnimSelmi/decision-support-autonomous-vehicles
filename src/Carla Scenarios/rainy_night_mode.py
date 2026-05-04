import carla


def apply_scenario(world, traffic_manager, bp, config):
    """
    RAINY NIGHT SCENARIO (HARD MODE)

    - Night + heavy rain
    - Low visibility
    - Wet/slippery roads
    - Continuous traffic flow (no red lights)
    """

    # -----------------------------
    # WEATHER (extreme conditions)
    # -----------------------------
    weather = carla.WeatherParameters()

    weather.sun_altitude_angle = -90.0
    weather.sun_azimuth_angle = 0.0

    weather.cloudiness = 100.0

    weather.precipitation = 100.0
    weather.precipitation_deposits = 100.0
    weather.wetness = 100.0

    weather.wind_intensity = 0.35

    weather.fog_density = 0.1
    weather.fog_distance = 15.0  # fixed (was 0)

    world.set_weather(weather)

    # -----------------------------
    # TRAFFIC BEHAVIOR
    # -----------------------------
    traffic_manager.set_global_distance_to_leading_vehicle(2.0)

    # slower driving due to rain
    traffic_manager.global_percentage_speed_difference(20.0)

    # -----------------------------
    # TRAFFIC LIGHTS (keep flow)
    # -----------------------------
    traffic_lights = world.get_actors().filter("traffic.traffic_light")

    for tl in traffic_lights:
        tl.set_state(carla.TrafficLightState.Green)
        tl.set_green_time(100.0)
        tl.set_yellow_time(0.0)
        tl.set_red_time(0.0)