import queue
import random

import carla

from src.carla_scenarios import SCENARIOS
from src.rl.action_space import action_to_carla_control
from src.rl.base_env import BaseRLEnv
from src.rl.config import RLConfig
from src.rl.reward import compute_reward
from src.rl.schemas import CarlaObservation, StepResult


class OnlineCarlaEnv(BaseRLEnv):
    def __init__(self, config: RLConfig, scenario_name: str):
        self.config = config
        self.scenario_name = scenario_name

        self.client = None
        self.world = None
        self.bp = None
        self.traffic_manager = None

        self.vehicle = None
        self.vehicles = []
        self.walkers = []
        self.walker_controllers = []

        self.camera = None
        self.collision_sensor = None
        self.lane_sensor = None

        self.image_queue = queue.Queue()
        self.collision_flag = {"value": 0}
        self.lane_flag = {"value": 0}

        self.step_count = 0

        self.output_dir = (
            self.config.online_output_dir
            / self.scenario_name
            / "frames"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def reset(self):
        self._destroy_actors()
        self._connect()
        self._enable_sync_mode()
        self._spawn_world()
        self._apply_scenario()

        self.step_count = 0

        self.world.tick()
        image = self.image_queue.get(timeout=10)

        return self._build_observation(image)

    def step(self, action_id: int, vlm_output: dict | None = None):
        control = action_to_carla_control(action_id)
        self.vehicle.apply_control(control)

        self.world.tick()
        image = self.image_queue.get(timeout=10)

        obs = self._build_observation(image)
        reward = compute_reward(obs, action_id, vlm_output)

        self.step_count += 1

        done = (
            self.step_count >= self.config.max_steps_per_episode
            or obs.collision == 1
        )

        info = {
            "scenario_name": self.scenario_name,
            "step": self.step_count,
            "action_id": action_id,
        }

        self.collision_flag["value"] = 0
        self.lane_flag["value"] = 0

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info=info,
        )

    def close(self):
        print("Cleaning up CARLA actors...")

        self._destroy_actors()

        if self.world is not None:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            settings.no_rendering_mode = False
            self.world.apply_settings(settings)

        if self.traffic_manager is not None:
            try:
                self.traffic_manager.set_synchronous_mode(False)
            except Exception:
                pass

    def _connect(self):
        self.client = carla.Client(self.config.carla_host, self.config.carla_port)
        self.client.set_timeout(50.0)

        self.world = self.client.get_world()
        self.bp = self.world.get_blueprint_library()

        self.traffic_manager = self.client.get_trafficmanager(self.config.tm_port)
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)

    def _enable_sync_mode(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.config.fixed_delta_seconds
        settings.no_rendering_mode = True
        self.world.apply_settings(settings)

        try:
            self.traffic_manager.set_synchronous_mode(True)
        except Exception:
            pass

    def _apply_scenario(self):
        if self.scenario_name not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {self.scenario_name}")

        SCENARIOS[self.scenario_name](
            world=self.world,
            traffic_manager=self.traffic_manager,
            blueprint_library=self.bp,
            config=self.config,
        )

        traffic_lights = self.world.get_actors().filter("traffic.traffic_light")

        for tl in traffic_lights:
            tl.set_state(carla.TrafficLightState.Green)
            tl.set_green_time(100.0)
            tl.set_yellow_time(0.0)
            tl.set_red_time(0.0)

    def _spawn_world(self):
        existing_locations = []
        spawn_points = self.world.get_map().get_spawn_points()

        def is_location_safe(location, min_distance):
            for loc in existing_locations:
                if location.distance(loc) < min_distance:
                    return False
            return True

        def try_spawn_vehicle(bp_vehicle):
            for _ in range(100):
                spawn = random.choice(spawn_points)
                if is_location_safe(spawn.location, 8.0):
                    try:
                        vehicle = self.world.spawn_actor(bp_vehicle, spawn)
                        existing_locations.append(spawn.location)
                        return vehicle
                    except Exception:
                        continue
            return None

        ego_bp = self.bp.filter("vehicle.mercedes.coupe_2020")[0]
        self.vehicle = try_spawn_vehicle(ego_bp)

        if self.vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle.")

        self.vehicle.set_autopilot(False)

        for _ in range(50):
            vehicle_bp = random.choice(self.bp.filter("vehicle.*"))
            vehicle = try_spawn_vehicle(vehicle_bp)

            if vehicle:
                vehicle.set_autopilot(True, self.traffic_manager.get_port())
                self.vehicles.append(vehicle)

        self._spawn_walkers(existing_locations)
        self._attach_sensors()

    def _spawn_walkers(self, existing_locations):
        walker_bps = self.bp.filter("walker.pedestrian.*")

        def is_location_safe(location, min_distance):
            for loc in existing_locations:
                if location.distance(loc) < min_distance:
                    return False
            return True

        for _ in range(50):
            for _ in range(100):
                location = self.world.get_random_location_from_navigation()

                if location and is_location_safe(location, 4.0):
                    transform = carla.Transform(location)
                    walker_bp = random.choice(walker_bps)

                    try:
                        walker = self.world.spawn_actor(walker_bp, transform)
                        existing_locations.append(location)

                        controller_bp = self.bp.find("controller.ai.walker")
                        controller = self.world.spawn_actor(
                            controller_bp,
                            carla.Transform(),
                            attach_to=walker,
                        )

                        controller.start()
                        controller.go_to_location(
                            self.world.get_random_location_from_navigation()
                        )
                        controller.set_max_speed(1 + random.random())

                        self.walkers.append(walker)
                        self.walker_controllers.append(controller)
                        break

                    except Exception:
                        continue

    def _attach_sensors(self):
        collision_bp = self.bp.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle,
        )
        self.collision_sensor.listen(self._collision_callback)

        lane_bp = self.bp.find("sensor.other.lane_invasion")
        self.lane_sensor = self.world.spawn_actor(
            lane_bp,
            carla.Transform(),
            attach_to=self.vehicle,
        )
        self.lane_sensor.listen(self._lane_callback)

        rgb_cam = self.bp.find("sensor.camera.rgb")
        rgb_cam.set_attribute("image_size_x", str(self.config.image_width))
        rgb_cam.set_attribute("image_size_y", str(self.config.image_height))
        rgb_cam.set_attribute("fov", str(self.config.camera_fov))

        cam_transform = carla.Transform(carla.Location(x=4, z=1.6))

        self.camera = self.world.spawn_actor(
            rgb_cam,
            cam_transform,
            attach_to=self.vehicle,
        )
        self.camera.listen(self._camera_callback)

    def _camera_callback(self, image):
        self.image_queue.put(image)

    def _collision_callback(self, event):
        self.collision_flag["value"] = 1

    def _lane_callback(self, event):
        self.lane_flag["value"] = 1

    def _destroy_actors(self):
        for actor in [self.camera, self.collision_sensor, self.lane_sensor, self.vehicle]:
            if actor is not None:
                try:
                    actor.stop()
                except Exception:
                    pass

                try:
                    actor.destroy()
                except Exception:
                    pass

        for actor in self.vehicles:
            try:
                actor.destroy()
            except Exception:
                pass

        for controller in self.walker_controllers:
            try:
                controller.stop()
            except Exception:
                pass
            try:
                controller.destroy()
            except Exception:
                pass

        for walker in self.walkers:
            try:
                walker.destroy()
            except Exception:
                pass

        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.lane_sensor = None
        self.vehicles = []
        self.walkers = []
        self.walker_controllers = []
        self.image_queue = queue.Queue()
        self.collision_flag["value"] = 0
        self.lane_flag["value"] = 0

    def _build_observation(self, image):
        frame_path = self.output_dir / f"{image.frame:06d}.jpg"
        image.save_to_disk(str(frame_path))

        velocity = self.vehicle.get_velocity()
        speed = (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5

        control = self.vehicle.get_control()

        return CarlaObservation(
            frame_path=str(frame_path),
            speed=speed,
            steering=control.steer,
            throttle=control.throttle,
            brake=control.brake,
            collision=self.collision_flag["value"],
            lane_invasion=self.lane_flag["value"],
        )