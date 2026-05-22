import carla

try:
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    world = client.get_world()

    print("✅ Connected to CARLA")
    print("Map:", world.get_map().name)
    print("Number of actors:", len(world.get_actors()))

except Exception as e:
    print("❌ Error:", e)
