import carla
import random
import queue

from src.carla_integration.scenario_loader import load_scenario


class CarlaEnv:

    def __init__(self, scenario_name="day"):

        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(20.0)

        self.world = self.client.get_world()
        self.bp = self.world.get_blueprint_library()

        self.traffic_manager = self.client.get_trafficmanager()

        # sync mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        # scenario
        self.scenario_fn = load_scenario(scenario_name)

        # placeholders
        self.vehicle = None
        self.camera = None
        self.image_queue = queue.Queue()

    # -----------------------------
    # RESET
    # -----------------------------
    def reset(self):

        self._spawn_ego()
        self._setup_camera()

        # apply scenario AFTER spawn
        self.scenario_fn(self.world, self.traffic_manager, self.bp, {})

        self.world.tick()

        image = self.image_queue.get()

        return image

    # -----------------------------
    # STEP
    # -----------------------------
    def step(self, action):

        # action = [steer, throttle, brake]
        control = carla.VehicleControl(
            steer=float(action[0]),
            throttle=float(action[1]),
            brake=float(action[2])
        )

        self.vehicle.apply_control(control)

        self.world.tick()

        image = self.image_queue.get()

        reward = self._compute_reward()
        done = self._check_done()

        return image, reward, done, {}

    # -----------------------------
    # SPAWN EGO VEHICLE
    # -----------------------------
    def _spawn_ego(self):

        spawn_points = self.world.get_map().get_spawn_points()

        vehicle_bp = self.bp.filter("vehicle.mercedes.coupe_2020")[0]

        spawn = random.choice(spawn_points)

        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn)

    # -----------------------------
    # CAMERA
    # -----------------------------
    def _setup_camera(self):

        cam_bp = self.bp.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "640")
        cam_bp.set_attribute("image_size_y", "360")
        cam_bp.set_attribute("fov", "90")

        transform = carla.Transform(carla.Location(x=4, z=1.6))

        self.camera = self.world.spawn_actor(
            cam_bp,
            transform,
            attach_to=self.vehicle
        )

        self.camera.listen(lambda image: self.image_queue.put(image))

    # -----------------------------
    # REWARD FUNCTION
    # -----------------------------
    def _compute_reward(self):

        velocity = self.vehicle.get_velocity()
        speed = (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5

        return speed  # simple for now

    # -----------------------------
    # DONE CONDITION
    # -----------------------------
    def _check_done(self):

        # basic condition (you can improve later)
        return False