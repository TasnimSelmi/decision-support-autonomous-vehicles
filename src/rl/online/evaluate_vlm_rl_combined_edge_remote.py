from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from src.rl.action_space import ACTION_NAMES
from src.rl.config import RLConfig
from src.rl.offline.agents.dqn import DQNAgent
from src.rl.online.remote_carla_env import RemoteCarlaEnv
from src.rl.online.vlm_window import VLMWindowAggregator
from src.rl.state_encoder import build_vlm_rl_state
from src.rl.utils.io import ensure_dirs
from src.vlm.vlm_interface import QwenVLMClient, run_vlm_inference


def _parse_args() -> argparse.Namespace:
    config = RLConfig()
    default_model = config.project_root / "models" / "vlm_rl" / "dqn_model.pth"

    parser = argparse.ArgumentParser(
        description="Evaluate the best trained VLM+RL DQN on the combined CARLA edge case."
    )
    parser.add_argument("--model", type=Path, default=default_model)
    parser.add_argument("--episodes", type=int, default=max(config.online_episodes, 1))
    parser.add_argument("--scenario", default="combined_edge_case")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.project_root / "outputs" / "results" / "online_evaluation",
    )
    parser.add_argument("--prefix", default="combined_edge_vlm_rl_best_trained")
    return parser.parse_args()


def _safe_mean(values: list[float]) -> float | str:
    return mean(values) if values else ""


def main() -> None:
    args = _parse_args()
    config = RLConfig()
    ensure_dirs([config.result_dir, args.output_dir])

    if not args.model.exists() or args.model.stat().st_size < 1000:
        raise FileNotFoundError(f"Model checkpoint is missing or invalid: {args.model}")

    agent = DQNAgent(
        state_dim=config.vlm_rl_state_dim,
        num_actions=config.action_dim,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        replay_capacity=config.replay_capacity,
    )
    agent.load(args.model)
    agent.policy_net.eval()
    agent.target_net.eval()

    vlm_client = QwenVLMClient()
    step_rows: list[dict] = []
    episode_rows: list[dict] = []
    defensive_actions = {"slow_down", "brake", "correct_lane"}
    evasive_actions = defensive_actions | {"turn_left", "turn_right"}

    print(f"VLM+RL EVAL | model={args.model}")
    print(f"VLM+RL EVAL | scenario={args.scenario} | episodes={args.episodes}")

    for episode_id in range(args.episodes):
        env = RemoteCarlaEnv(config=config, scenario_name=args.scenario)
        aggregator = VLMWindowAggregator(window_size=config.vlm_window_size)
        total_reward = 0.0
        action_counts: Counter[str] = Counter()
        vlm_counts: Counter[str] = Counter()
        collision_count = 0
        lane_invasion_count = 0
        speed_sum = 0.0
        steps_completed = 0
        first_defensive_action_step = None
        first_brake_step = None
        first_slow_down_step = None
        first_correct_lane_step = None
        first_high_hazard_step = None
        first_collision_risk_step = None
        first_unsafe_lane_step = None
        lane_event_step = None
        lane_recovery_steps: list[int] = []
        done = False

        try:
            obs = env.reset()

            for step in range(config.max_steps_per_episode):
                raw_vlm_output = None
                vlm_fresh = 0

                if step % config.vlm_interval_steps == 0:
                    raw_vlm_output = run_vlm_inference(obs.frame_path, client=vlm_client)
                    aggregator.add(raw_vlm_output)
                    vlm_fresh = 1

                vlm_output = aggregator.aggregate()
                state = build_vlm_rl_state(obs, vlm_output)
                action_id = int(agent.select_action(state_vector=state, epsilon=0.0))
                result = env.step(action_id=action_id, vlm_output=vlm_output)

                action_name = ACTION_NAMES.get(action_id, str(action_id))
                action_counts[action_name] += 1
                vlm_counts[f"hazard_{vlm_output.get('hazard_level', 'unknown')}"] += 1
                vlm_counts[f"urgency_{vlm_output.get('action_urgency', 'unknown')}"] += 1

                if vlm_output.get("hazard_level") == "high" and first_high_hazard_step is None:
                    first_high_hazard_step = step
                if vlm_output.get("collision_risk") == "yes" and first_collision_risk_step is None:
                    first_collision_risk_step = step
                if vlm_output.get("lane_safety") == "unsafe" and first_unsafe_lane_step is None:
                    first_unsafe_lane_step = step
                if action_name in defensive_actions and first_defensive_action_step is None:
                    first_defensive_action_step = step
                if action_name == "brake" and first_brake_step is None:
                    first_brake_step = step
                if action_name == "slow_down" and first_slow_down_step is None:
                    first_slow_down_step = step
                if action_name == "correct_lane" and first_correct_lane_step is None:
                    first_correct_lane_step = step
                if int(obs.lane_invasion) and lane_event_step is None:
                    lane_event_step = step
                if lane_event_step is not None and action_name == "correct_lane":
                    lane_recovery_steps.append(step - lane_event_step)
                    lane_event_step = None

                total_reward += float(result.reward)
                collision_count += int(result.observation.collision)
                lane_invasion_count += int(result.observation.lane_invasion)
                speed_sum += float(obs.speed)
                steps_completed = step + 1
                done = bool(result.done)

                step_rows.append({
                    "agent_type": "vlm_rl",
                    "episode_id": episode_id,
                    "scenario_name": args.scenario,
                    "step": step,
                    "action_id": action_id,
                    "action_name": action_name,
                    "reward": float(result.reward),
                    "total_reward": total_reward,
                    "speed": float(obs.speed),
                    "steering": float(obs.steering),
                    "throttle": float(obs.throttle),
                    "brake": float(obs.brake),
                    "collision": int(obs.collision),
                    "lane_invasion": int(obs.lane_invasion),
                    "done": done,
                    "next_speed": float(result.observation.speed),
                    "next_collision": int(result.observation.collision),
                    "next_lane_invasion": int(result.observation.lane_invasion),
                    "frame_path": obs.frame_path,
                    "next_frame_path": result.observation.frame_path,
                    "is_defensive_action": action_name in defensive_actions,
                    "is_evasive_action": action_name in evasive_actions,
                    "vlm_fresh": vlm_fresh,
                    "hazard_level": vlm_output.get("hazard_level", ""),
                    "obstacle_presence": vlm_output.get("obstacle_presence", ""),
                    "collision_risk": vlm_output.get("collision_risk", ""),
                    "lane_safety": vlm_output.get("lane_safety", ""),
                    "crossing_pedestrian_presence": vlm_output.get(
                        "crossing_pedestrian_presence", ""
                    ),
                    "action_urgency": vlm_output.get("action_urgency", ""),
                    "raw_vlm_output": json.dumps(raw_vlm_output) if raw_vlm_output else "",
                    "aggregated_vlm_output": json.dumps(vlm_output),
                })

                obs = result.observation
                if done:
                    break
        finally:
            env.close()

        average_speed = speed_sum / max(steps_completed, 1)
        defensive_action_count = sum(action_counts[action] for action in defensive_actions)
        evasive_action_count = sum(action_counts[action] for action in evasive_actions)
        collision_occurred = collision_count > 0
        lane_invasion_occurred = lane_invasion_count > 0
        completed_max_steps = steps_completed >= config.max_steps_per_episode
        success = completed_max_steps and not collision_occurred

        episode_rows.append({
            "agent_type": "vlm_rl",
            "episode_id": episode_id,
            "scenario_name": args.scenario,
            "model_path": str(args.model),
            "total_reward": total_reward,
            "steps_completed": steps_completed,
            "survival_time_seconds": steps_completed * config.fixed_delta_seconds,
            "completed_max_steps": completed_max_steps,
            "success": success,
            "done": done,
            "collision_occurred": collision_occurred,
            "lane_invasion_occurred": lane_invasion_occurred,
            "collision_count": collision_count,
            "lane_invasion_count": lane_invasion_count,
            "collision_rate": collision_count / max(steps_completed, 1),
            "lane_invasion_rate": lane_invasion_count / max(steps_completed, 1),
            "average_speed": average_speed,
            "average_speed_kmh": average_speed * 3.6,
            "defensive_action_count": defensive_action_count,
            "defensive_action_rate": defensive_action_count / max(steps_completed, 1),
            "evasive_action_count": evasive_action_count,
            "evasive_action_rate": evasive_action_count / max(steps_completed, 1),
            "first_defensive_action_step": (
                first_defensive_action_step
                if first_defensive_action_step is not None
                else ""
            ),
            "first_defensive_action_time_seconds": (
                first_defensive_action_step * config.fixed_delta_seconds
                if first_defensive_action_step is not None
                else ""
            ),
            "first_brake_step": first_brake_step if first_brake_step is not None else "",
            "first_brake_time_seconds": (
                first_brake_step * config.fixed_delta_seconds
                if first_brake_step is not None
                else ""
            ),
            "first_slow_down_step": (
                first_slow_down_step if first_slow_down_step is not None else ""
            ),
            "first_correct_lane_step": (
                first_correct_lane_step if first_correct_lane_step is not None else ""
            ),
            "mean_lane_recovery_steps": _safe_mean(lane_recovery_steps),
            "mean_lane_recovery_time_seconds": (
                mean(lane_recovery_steps) * config.fixed_delta_seconds
                if lane_recovery_steps
                else ""
            ),
            "first_high_hazard_step": (
                first_high_hazard_step if first_high_hazard_step is not None else ""
            ),
            "first_collision_risk_step": (
                first_collision_risk_step
                if first_collision_risk_step is not None
                else ""
            ),
            "first_unsafe_lane_step": (
                first_unsafe_lane_step if first_unsafe_lane_step is not None else ""
            ),
            "high_hazard_rate": vlm_counts["hazard_high"] / max(steps_completed, 1),
            "collision_risk_yes_rate": (
                sum(
                    1
                    for row in step_rows
                    if row["episode_id"] == episode_id
                    and row.get("collision_risk") == "yes"
                )
                / max(steps_completed, 1)
            ),
            "unsafe_lane_vlm_rate": (
                sum(
                    1
                    for row in step_rows
                    if row["episode_id"] == episode_id
                    and row.get("lane_safety") == "unsafe"
                )
                / max(steps_completed, 1)
            ),
            "continue_count": action_counts["continue"],
            "slow_down_count": action_counts["slow_down"],
            "brake_count": action_counts["brake"],
            "turn_left_count": action_counts["turn_left"],
            "turn_right_count": action_counts["turn_right"],
            "correct_lane_count": action_counts["correct_lane"],
        })

        print(
            f"VLM+RL EVAL EPISODE | episode={episode_id} | reward={total_reward:.2f} | "
            f"steps={steps_completed} | collisions={collision_count} | "
            f"lane_invasions={lane_invasion_count}"
        )

    step_path = args.output_dir / f"{args.prefix}_steps.csv"
    summary_path = args.output_dir / f"{args.prefix}_summary.csv"
    aggregate_path = args.output_dir / f"{args.prefix}_aggregate.csv"

    if step_rows:
        with step_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(step_rows[0].keys()))
            writer.writeheader()
            writer.writerows(step_rows)

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()))
        writer.writeheader()
        writer.writerows(episode_rows)

    numeric_metrics = [
        "total_reward",
        "steps_completed",
        "survival_time_seconds",
        "collision_count",
        "lane_invasion_count",
        "collision_rate",
        "lane_invasion_rate",
        "average_speed",
        "average_speed_kmh",
        "defensive_action_rate",
        "evasive_action_rate",
        "high_hazard_rate",
        "collision_risk_yes_rate",
        "unsafe_lane_vlm_rate",
    ]
    aggregate_row = {
        "agent_type": "vlm_rl",
        "scenario_name": args.scenario,
        "model_path": str(args.model),
        "episodes": len(episode_rows),
        "success_rate": mean(1.0 if row["success"] else 0.0 for row in episode_rows),
        "collision_episode_rate": mean(
            1.0 if row["collision_occurred"] else 0.0 for row in episode_rows
        ),
        "lane_invasion_episode_rate": mean(
            1.0 if row["lane_invasion_occurred"] else 0.0 for row in episode_rows
        ),
        "max_step_completion_rate": mean(
            1.0 if row["completed_max_steps"] else 0.0 for row in episode_rows
        ),
    }
    for metric in numeric_metrics:
        aggregate_row[f"mean_{metric}"] = mean(float(row[metric]) for row in episode_rows)

    for source, target in [
        ("first_defensive_action_time_seconds", "mean_first_defensive_action_time_seconds"),
        ("first_brake_time_seconds", "mean_first_brake_time_seconds"),
        ("mean_lane_recovery_time_seconds", "mean_lane_recovery_time_seconds"),
    ]:
        values = [float(row[source]) for row in episode_rows if row[source] != ""]
        aggregate_row[target] = _safe_mean(values)

    with aggregate_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate_row.keys()))
        writer.writeheader()
        writer.writerow(aggregate_row)

    print(f"VLM+RL EVAL SAVED | steps={step_path}")
    print(f"VLM+RL EVAL SAVED | summary={summary_path}")
    print(f"VLM+RL EVAL SAVED | aggregate={aggregate_path}")


if __name__ == "__main__":
    main()
