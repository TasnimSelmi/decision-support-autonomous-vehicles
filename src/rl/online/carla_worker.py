from __future__ import annotations

import contextlib
import json
import queue
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

from src.rl.action_space import action_to_carla_control
from src.rl.online.carla_env import OnlineCarlaEnv

_PROTOCOL_STDOUT = sys.stdout


@dataclass
class CarlaRuntimeConfig:
    carla_host: str
    carla_port: int
    tm_port: int
    fixed_delta_seconds: float
    max_steps_per_episode: int
    image_width: int
    image_height: int
    camera_fov: int
    online_output_dir: Path


def _send(payload: dict) -> None:
    _PROTOCOL_STDOUT.write(json.dumps(payload) + "\n")
    _PROTOCOL_STDOUT.flush()


def _read_message() -> dict:
    line = sys.stdin.readline()
    if not line:
        raise EOFError("stdin closed")
    return json.loads(line)


def _runtime_config(payload: dict) -> CarlaRuntimeConfig:
    return CarlaRuntimeConfig(
        carla_host=str(payload["carla_host"]),
        carla_port=int(payload["carla_port"]),
        tm_port=int(payload["tm_port"]),
        fixed_delta_seconds=float(payload["fixed_delta_seconds"]),
        max_steps_per_episode=int(payload["max_steps_per_episode"]),
        image_width=int(payload["image_width"]),
        image_height=int(payload["image_height"]),
        camera_fov=int(payload["camera_fov"]),
        online_output_dir=Path(payload["online_output_dir"]),
    )


def _step_without_reward(env: OnlineCarlaEnv, action_id: int) -> dict:
    control = action_to_carla_control(action_id)
    env.vehicle.apply_control(control)

    env.world.tick()
    image = env.image_queue.get(timeout=10)

    obs = env._build_observation(image)

    env.step_count += 1

    done = (
        env.step_count >= env.config.max_steps_per_episode
        or obs.collision == 1
    )

    info = {
        "scenario_name": env.scenario_name,
        "step": env.step_count,
        "action_id": action_id,
    }

    env.collision_flag["value"] = 0
    env.lane_flag["value"] = 0

    return {
        "observation": asdict(obs),
        "done": done,
        "info": info,
    }


def main() -> None:
    init_message = _read_message()
    scenario_name = str(init_message["scenario_name"])
    config = _runtime_config(init_message["config"])
    env = OnlineCarlaEnv(config=config, scenario_name=scenario_name)

    try:
        _send({"status": "ready"})

        while True:
            message = _read_message()
            command = message.get("command")

            try:
                with contextlib.redirect_stdout(sys.stderr):
                    if command == "reset":
                        obs = env.reset()
                        _send({"status": "ok", "observation": asdict(obs)})
                    elif command == "step":
                        result = _step_without_reward(
                            env=env,
                            action_id=int(message["action_id"]),
                        )
                        _send({"status": "ok", "result": result})
                    elif command == "close":
                        env.close()
                        _send({"status": "ok"})
                        break
                    else:
                        raise ValueError("Unknown CARLA worker command: {}".format(command))
            except queue.Empty:
                _send({
                    "status": "error",
                    "error": "Timed out waiting for a CARLA camera frame.",
                    "traceback": traceback.format_exc(),
                })
            except Exception as exc:
                _send({
                    "status": "error",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                })
    finally:
        with contextlib.suppress(Exception), contextlib.redirect_stdout(sys.stderr):
            env.close()


if __name__ == "__main__":
    main()
