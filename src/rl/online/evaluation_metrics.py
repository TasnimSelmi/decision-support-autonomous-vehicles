from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from src.rl.action_space import ACTION_NAMES, DrivingAction


NAN = float("nan")

TTC_CRITICAL_THRESHOLD_SECONDS = 3.0
HAZARD_DISTANCE_CRITICAL_THRESHOLD_M = 5.0
EMERGENCY_BRAKE_THRESHOLD = 0.7
HARD_STEER_THRESHOLD = 0.7
SMALL_DISTANCE_KM = 1e-6

RAW_COLUMNS = [
    "episode_id",
    "step",
    "agent_type",
    "scenario",
    "selected_action_id",
    "selected_action_name",
    "reward",
    "done",
    "termination_reason",
    "speed_kmh",
    "throttle",
    "brake",
    "steer",
    "collision_flag",
    "collision_type",
    "lane_invasion_flag",
    "distance_traveled_m",
    "route_completion",
    "nearest_vehicle_distance_m",
    "nearest_pedestrian_distance_m",
    "min_hazard_distance_m",
    "ttc_seconds",
    "vlm_hazard_level",
    "vlm_collision_risk",
    "vlm_pedestrian_presence",
    "vlm_lane_safety",
    "vlm_action_urgency",
    "vlm_recommended_action",
    "vlm_latency_ms",
    "critical_moment",
    "unsafe_action",
]

EPISODE_COLUMNS = [
    "agent_type",
    "episode_id",
    "scenario",
    "model_path",
    "total_reward",
    "episode_steps",
    "success",
    "termination_reason",
    "collision_occurred",
    "collision_count",
    "collision_speed_kmh_mean",
    "collision_speed_kmh_max",
    "average_speed_kmh",
    "max_speed_kmh",
    "total_distance_m",
    "route_completion_final",
    "lane_invasion_count",
    "emergency_brake_count",
    "hard_steer_count",
    "min_ttc_seconds",
    "mean_ttc_seconds",
    "min_hazard_distance_m",
    "mean_hazard_distance_m",
    "tcf_collisions_per_1000_steps",
    "dcf_collisions_per_km",
    "unsafe_action_count",
    "unsafe_action_rate",
    "mean_vlm_latency_ms",
    "mean_vlm_hazard_level",
]

AGGREGATE_COLUMNS = [
    "agent_type",
    "scenario",
    "model_path",
    "episodes",
    "success_rate",
    "collision_rate",
    "mean_route_completion",
    "std_route_completion",
    "mean_total_reward",
    "std_total_reward",
    "mean_average_speed_kmh",
    "std_average_speed_kmh",
    "mean_total_distance_m",
    "std_total_distance_m",
    "mean_collision_speed_kmh",
    "mean_tcf_collisions_per_1000_steps",
    "mean_dcf_collisions_per_km",
    "mean_min_ttc_seconds",
    "mean_min_hazard_distance_m",
    "mean_unsafe_action_rate",
    "mean_vlm_latency_ms",
]

COMPARISON_COLUMNS = [
    "scenario",
    "rl_only_episodes",
    "vlm_rl_episodes",
    "success_rate_delta",
    "route_completion_delta",
    "collision_rate_reduction",
    "dcf_reduction",
    "collision_speed_reduction",
    "min_ttc_improvement",
    "min_hazard_distance_improvement",
]


@dataclass
class EvaluationPaths:
    raw_dir: Path
    episodes_dir: Path
    aggregate_dir: Path
    figures_dir: Path

    @classmethod
    def from_root(cls, output_dir: Path) -> "EvaluationPaths":
        return cls(
            raw_dir=output_dir / "raw",
            episodes_dir=output_dir / "episodes",
            aggregate_dir=output_dir / "aggregate",
            figures_dir=output_dir / "figures",
        )

    def ensure(self) -> None:
        for path in [self.raw_dir, self.episodes_dir, self.aggregate_dir, self.figures_dir]:
            path.mkdir(parents=True, exist_ok=True)


def safe_float(value: Any, default: float = NAN) -> float:
    try:
        if value == "":
            return default
        result = float(value)
    except Exception:
        return default
    return result if math.isfinite(result) else default


def finite_values(values: list[Any]) -> list[float]:
    result = []
    for value in values:
        number = safe_float(value)
        if math.isfinite(number):
            result.append(number)
    return result


def safe_mean(values: list[Any]) -> float:
    numbers = finite_values(values)
    return mean(numbers) if numbers else NAN


def safe_std(values: list[Any]) -> float:
    numbers = finite_values(values)
    return stdev(numbers) if len(numbers) > 1 else NAN


def action_name(action_id: int) -> str:
    return ACTION_NAMES.get(int(action_id), str(action_id))


def termination_reason(done: bool, step: int, max_steps: int, collision: int, lane_invasion: int) -> str:
    if not done:
        return ""
    if int(collision) == 1:
        return "collision"
    if step + 1 >= max_steps:
        return "max_steps"
    if int(lane_invasion) == 1:
        return "lane_invasion"
    return "done"


def critical_moment(ttc_seconds: float, min_hazard_distance_m: float) -> bool | float:
    has_ttc = math.isfinite(ttc_seconds)
    has_distance = math.isfinite(min_hazard_distance_m)
    if not has_ttc and not has_distance:
        return NAN
    return (
        (has_ttc and ttc_seconds < TTC_CRITICAL_THRESHOLD_SECONDS)
        or (
            has_distance
            and min_hazard_distance_m < HAZARD_DISTANCE_CRITICAL_THRESHOLD_M
        )
    )


def unsafe_action_value(
    selected_action_name: str,
    brake: float,
    ttc_seconds: float,
    min_hazard_distance_m: float,
) -> int | float:
    is_critical = critical_moment(ttc_seconds, min_hazard_distance_m)
    if isinstance(is_critical, float) and math.isnan(is_critical):
        return NAN
    if not is_critical:
        return 0
    if selected_action_name in {"continue", "accelerate"} and brake < EMERGENCY_BRAKE_THRESHOLD:
        return 1
    return 0


def build_step_row(
    *,
    agent_type: str,
    episode_id: int,
    step: int,
    scenario: str,
    action_id: int,
    reward: float,
    done: bool,
    obs: Any,
    next_obs: Any,
    fixed_delta_seconds: float,
    max_steps: int,
    cumulative_distance_m: float,
    vlm_output: dict[str, Any] | None = None,
    vlm_latency_ms: float = NAN,
) -> dict[str, Any]:
    selected_action_name = action_name(action_id)
    speed_mps = safe_float(getattr(obs, "speed", NAN))
    speed_kmh = speed_mps * 3.6 if math.isfinite(speed_mps) else NAN
    distance_traveled_m = (
        cumulative_distance_m + speed_mps * fixed_delta_seconds
        if math.isfinite(speed_mps)
        else cumulative_distance_m
    )
    collision_flag = int(safe_float(getattr(next_obs, "collision", 0), 0))
    lane_invasion_flag = int(safe_float(getattr(next_obs, "lane_invasion", 0), 0))
    ttc_seconds = NAN
    min_hazard_distance_m = NAN
    brake = safe_float(getattr(obs, "brake", NAN))

    vlm_output = vlm_output or {}
    row = {
        "episode_id": episode_id,
        "step": step,
        "agent_type": agent_type,
        "scenario": scenario,
        "selected_action_id": int(action_id),
        "selected_action_name": selected_action_name,
        "reward": float(reward),
        "done": bool(done),
        "termination_reason": termination_reason(
            done=done,
            step=step,
            max_steps=max_steps,
            collision=collision_flag,
            lane_invasion=lane_invasion_flag,
        ),
        "speed_kmh": speed_kmh,
        "throttle": safe_float(getattr(obs, "throttle", NAN)),
        "brake": brake,
        "steer": safe_float(getattr(obs, "steering", NAN)),
        "collision_flag": collision_flag,
        "collision_type": "",
        "lane_invasion_flag": lane_invasion_flag,
        "distance_traveled_m": distance_traveled_m,
        "route_completion": NAN,
        "nearest_vehicle_distance_m": NAN,
        "nearest_pedestrian_distance_m": NAN,
        "min_hazard_distance_m": min_hazard_distance_m,
        "ttc_seconds": ttc_seconds,
        "vlm_hazard_level": vlm_output.get("hazard_level", ""),
        "vlm_collision_risk": vlm_output.get("collision_risk", ""),
        "vlm_pedestrian_presence": vlm_output.get("crossing_pedestrian_presence", ""),
        "vlm_lane_safety": vlm_output.get("lane_safety", ""),
        "vlm_action_urgency": vlm_output.get("action_urgency", ""),
        "vlm_recommended_action": vlm_output.get("action_urgency", ""),
        "vlm_latency_ms": vlm_latency_ms,
        "critical_moment": critical_moment(ttc_seconds, min_hazard_distance_m),
        "unsafe_action": unsafe_action_value(
            selected_action_name=selected_action_name,
            brake=brake,
            ttc_seconds=ttc_seconds,
            min_hazard_distance_m=min_hazard_distance_m,
        ),
    }
    return {column: row.get(column, NAN) for column in RAW_COLUMNS}


def hazard_level_to_numeric(value: Any) -> float:
    return {
        "low": 0.0,
        "medium": 1.0,
        "high": 2.0,
    }.get(str(value).lower(), NAN)


def compute_episode_metrics(
    *,
    agent_type: str,
    episode_id: int,
    scenario: str,
    model_path: Path,
    step_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    total_reward = sum(finite_values([row.get("reward", NAN) for row in step_rows]))
    episode_steps = len(step_rows)
    collision_rows = [row for row in step_rows if int(safe_float(row.get("collision_flag", 0), 0)) == 1]
    lane_rows = [row for row in step_rows if int(safe_float(row.get("lane_invasion_flag", 0), 0)) == 1]
    collision_speeds = finite_values([row.get("speed_kmh", NAN) for row in collision_rows])
    ttc_values = finite_values([row.get("ttc_seconds", NAN) for row in step_rows])
    hazard_values = finite_values([row.get("min_hazard_distance_m", NAN) for row in step_rows])
    unsafe_values = finite_values([row.get("unsafe_action", NAN) for row in step_rows])
    total_distance_m = max(finite_values([row.get("distance_traveled_m", NAN) for row in step_rows]) or [0.0])
    last_row = step_rows[-1] if step_rows else {}
    termination = str(last_row.get("termination_reason", ""))
    route_completion_final = safe_float(last_row.get("route_completion", NAN))

    collision_count = len(collision_rows)
    lane_invasion_count = len(lane_rows)
    success = (
        collision_count == 0
        and termination not in {"lane_invasion", "timeout", "stuck"}
        and (
            not math.isfinite(route_completion_final)
            or route_completion_final >= 1.0
        )
    )

    emergency_brake_count = 0
    hard_steer_count = 0
    for row in step_rows:
        brake = safe_float(row.get("brake", NAN))
        steer = abs(safe_float(row.get("steer", NAN)))
        selected_action_name = str(row.get("selected_action_name", ""))
        if (
            math.isfinite(brake)
            and brake >= EMERGENCY_BRAKE_THRESHOLD
        ) or selected_action_name == action_name(DrivingAction.BRAKE):
            emergency_brake_count += 1
        if math.isfinite(steer) and steer >= HARD_STEER_THRESHOLD:
            hard_steer_count += 1

    total_distance_km = max(total_distance_m / 1000.0, SMALL_DISTANCE_KM)
    mean_vlm_latency_ms = safe_mean([row.get("vlm_latency_ms", NAN) for row in step_rows])
    mean_vlm_hazard_level = safe_mean(
        [hazard_level_to_numeric(row.get("vlm_hazard_level", "")) for row in step_rows]
    )

    row = {
        "agent_type": agent_type,
        "episode_id": episode_id,
        "scenario": scenario,
        "model_path": str(model_path),
        "total_reward": total_reward,
        "episode_steps": episode_steps,
        "success": success,
        "termination_reason": termination,
        "collision_occurred": collision_count > 0,
        "collision_count": collision_count,
        "collision_speed_kmh_mean": safe_mean(collision_speeds),
        "collision_speed_kmh_max": max(collision_speeds) if collision_speeds else NAN,
        "average_speed_kmh": safe_mean([row.get("speed_kmh", NAN) for row in step_rows]),
        "max_speed_kmh": max(finite_values([row.get("speed_kmh", NAN) for row in step_rows]) or [NAN]),
        "total_distance_m": total_distance_m,
        "route_completion_final": route_completion_final,
        "lane_invasion_count": lane_invasion_count,
        "emergency_brake_count": emergency_brake_count,
        "hard_steer_count": hard_steer_count,
        "min_ttc_seconds": min(ttc_values) if ttc_values else NAN,
        "mean_ttc_seconds": safe_mean(ttc_values),
        "min_hazard_distance_m": min(hazard_values) if hazard_values else NAN,
        "mean_hazard_distance_m": safe_mean(hazard_values),
        "tcf_collisions_per_1000_steps": collision_count / max(episode_steps, 1) * 1000.0,
        "dcf_collisions_per_km": collision_count / total_distance_km,
        "unsafe_action_count": sum(unsafe_values) if unsafe_values else NAN,
        "unsafe_action_rate": safe_mean(unsafe_values),
        "mean_vlm_latency_ms": mean_vlm_latency_ms,
        "mean_vlm_hazard_level": mean_vlm_hazard_level,
    }
    return {column: row.get(column, NAN) for column in EPISODE_COLUMNS}


def compute_aggregate_metrics(episode_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not episode_rows:
        return {column: NAN for column in AGGREGATE_COLUMNS}

    first = episode_rows[0]
    row = {
        "agent_type": first.get("agent_type", ""),
        "scenario": first.get("scenario", ""),
        "model_path": first.get("model_path", ""),
        "episodes": len(episode_rows),
        "success_rate": safe_mean([1.0 if str(row.get("success")) == "True" or row.get("success") is True else 0.0 for row in episode_rows]),
        "collision_rate": safe_mean([1.0 if str(row.get("collision_occurred")) == "True" or row.get("collision_occurred") is True else 0.0 for row in episode_rows]),
        "mean_route_completion": safe_mean([row.get("route_completion_final", NAN) for row in episode_rows]),
        "std_route_completion": safe_std([row.get("route_completion_final", NAN) for row in episode_rows]),
        "mean_total_reward": safe_mean([row.get("total_reward", NAN) for row in episode_rows]),
        "std_total_reward": safe_std([row.get("total_reward", NAN) for row in episode_rows]),
        "mean_average_speed_kmh": safe_mean([row.get("average_speed_kmh", NAN) for row in episode_rows]),
        "std_average_speed_kmh": safe_std([row.get("average_speed_kmh", NAN) for row in episode_rows]),
        "mean_total_distance_m": safe_mean([row.get("total_distance_m", NAN) for row in episode_rows]),
        "std_total_distance_m": safe_std([row.get("total_distance_m", NAN) for row in episode_rows]),
        "mean_collision_speed_kmh": safe_mean([row.get("collision_speed_kmh_mean", NAN) for row in episode_rows]),
        "mean_tcf_collisions_per_1000_steps": safe_mean([row.get("tcf_collisions_per_1000_steps", NAN) for row in episode_rows]),
        "mean_dcf_collisions_per_km": safe_mean([row.get("dcf_collisions_per_km", NAN) for row in episode_rows]),
        "mean_min_ttc_seconds": safe_mean([row.get("min_ttc_seconds", NAN) for row in episode_rows]),
        "mean_min_hazard_distance_m": safe_mean([row.get("min_hazard_distance_m", NAN) for row in episode_rows]),
        "mean_unsafe_action_rate": safe_mean([row.get("unsafe_action_rate", NAN) for row in episode_rows]),
        "mean_vlm_latency_ms": safe_mean([row.get("mean_vlm_latency_ms", NAN) for row in episode_rows]),
    }
    return {column: row.get(column, NAN) for column in AGGREGATE_COLUMNS}


def write_rows(path: Path, rows: list[dict[str, Any]], columns: list[str], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.exists() else "w"
    with path.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_evaluation_outputs(
    *,
    output_dir: Path,
    output_prefix: str,
    step_rows: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
    append: bool = False,
) -> tuple[Path, Path, Path]:
    paths = EvaluationPaths.from_root(output_dir)
    paths.ensure()

    step_path = paths.raw_dir / f"{output_prefix}_steps.csv"
    episode_path = paths.episodes_dir / f"{output_prefix}_episodes.csv"
    aggregate_path = paths.aggregate_dir / f"{output_prefix}_aggregate.csv"

    write_rows(step_path, step_rows, RAW_COLUMNS, append=append)
    write_rows(episode_path, episode_rows, EPISODE_COLUMNS, append=append)

    aggregate_source = read_rows(episode_path) if append else episode_rows
    aggregate_row = compute_aggregate_metrics(aggregate_source)
    write_rows(aggregate_path, [aggregate_row], AGGREGATE_COLUMNS, append=False)

    return step_path, episode_path, aggregate_path


def write_comparison_if_available(output_dir: Path) -> Path | None:
    paths = EvaluationPaths.from_root(output_dir)
    rl_path = paths.aggregate_dir / "rl_only_combined_edge_aggregate.csv"
    vlm_path = paths.aggregate_dir / "vlm_rl_combined_edge_aggregate.csv"
    if not rl_path.exists() or not vlm_path.exists():
        return None

    rl_rows = read_rows(rl_path)
    vlm_rows = read_rows(vlm_path)
    if not rl_rows or not vlm_rows:
        return None

    rl = rl_rows[0]
    vlm = vlm_rows[0]

    row = {
        "scenario": vlm.get("scenario") or rl.get("scenario", "combined_edge_case"),
        "rl_only_episodes": rl.get("episodes", ""),
        "vlm_rl_episodes": vlm.get("episodes", ""),
        "success_rate_delta": safe_float(vlm.get("success_rate")) - safe_float(rl.get("success_rate")),
        "route_completion_delta": safe_float(vlm.get("mean_route_completion")) - safe_float(rl.get("mean_route_completion")),
        "collision_rate_reduction": safe_float(rl.get("collision_rate")) - safe_float(vlm.get("collision_rate")),
        "dcf_reduction": safe_float(rl.get("mean_dcf_collisions_per_km")) - safe_float(vlm.get("mean_dcf_collisions_per_km")),
        "collision_speed_reduction": safe_float(rl.get("mean_collision_speed_kmh")) - safe_float(vlm.get("mean_collision_speed_kmh")),
        "min_ttc_improvement": safe_float(vlm.get("mean_min_ttc_seconds")) - safe_float(rl.get("mean_min_ttc_seconds")),
        "min_hazard_distance_improvement": safe_float(vlm.get("mean_min_hazard_distance_m")) - safe_float(rl.get("mean_min_hazard_distance_m")),
    }
    comparison_path = paths.aggregate_dir / "combined_edge_comparison_aggregate.csv"
    write_rows(comparison_path, [row], COMPARISON_COLUMNS, append=False)
    return comparison_path
