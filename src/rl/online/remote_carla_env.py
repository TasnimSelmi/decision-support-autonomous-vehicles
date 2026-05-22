from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.rl.config import RLConfig
from src.rl.reward import compute_reward
from src.rl.schemas import CarlaObservation, StepResult


class RemoteCarlaEnv:
    def __init__(
        self,
        config: RLConfig,
        scenario_name: str,
        carla_python: str | Path | None = None,
    ) -> None:
        self.config = config
        self.scenario_name = scenario_name
        self.carla_python = Path(
            carla_python or config.project_root / "carla_env_37" / "bin" / "python"
        )
        self.process = subprocess.Popen(
            [
                str(self.carla_python),
                "-u",
                "-m",
                "src.rl.online.carla_worker",
            ],
            cwd=str(config.project_root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
        )
        self._send({
            "scenario_name": scenario_name,
            "config": {
                "carla_host": config.carla_host,
                "carla_port": config.carla_port,
                "tm_port": config.tm_port,
                "fixed_delta_seconds": config.fixed_delta_seconds,
                "max_steps_per_episode": config.max_steps_per_episode,
                "image_width": config.image_width,
                "image_height": config.image_height,
                "camera_fov": config.camera_fov,
                "online_output_dir": str(config.online_output_dir),
            },
        })
        self._expect_ready()

    def reset(self) -> CarlaObservation:
        response = self._request({"command": "reset"})
        return CarlaObservation(**response["observation"])

    def step(self, action_id: int, vlm_output: dict | None = None) -> StepResult:
        response = self._request({
            "command": "step",
            "action_id": int(action_id),
        })
        result = response["result"]
        observation = CarlaObservation(**result["observation"])
        reward = compute_reward(
            obs=observation,
            action_id=int(action_id),
            vlm_output=vlm_output,
        )

        return StepResult(
            observation=observation,
            reward=reward,
            done=bool(result["done"]),
            info=dict(result["info"]),
        )

    def close(self) -> None:
        if self.process.poll() is not None:
            return

        try:
            self._request({"command": "close"})
        finally:
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)

    def _expect_ready(self) -> None:
        response = self._read_response()
        if response.get("status") != "ready":
            raise RuntimeError("CARLA worker did not become ready: {}".format(response))

    def _request(self, payload: dict) -> dict:
        if self.process.poll() is not None:
            raise RuntimeError(
                "CARLA worker exited with code {}.".format(self.process.returncode)
            )

        self._send(payload)
        response = self._read_response()
        if response.get("status") == "error":
            raise RuntimeError(
                "CARLA worker error: {}\n{}".format(
                    response.get("error"),
                    response.get("traceback", ""),
                )
            )
        return response

    def _send(self, payload: dict) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()

    def _read_response(self) -> dict:
        assert self.process.stdout is not None
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(
                "CARLA worker closed stdout with code {}.".format(self.process.poll())
            )
        return json.loads(line)
