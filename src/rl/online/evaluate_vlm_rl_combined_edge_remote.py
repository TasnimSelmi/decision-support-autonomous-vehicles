from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.rl.config import RLConfig
from src.rl.offline.agents.dqn import DQNAgent
from src.rl.online.evaluation_metrics import (
    NAN,
    build_step_row,
    compute_episode_metrics,
    write_comparison_if_available,
    write_evaluation_outputs,
)
from src.rl.online.remote_carla_env import RemoteCarlaEnv
from src.rl.online.vlm_window import VLMWindowAggregator
from src.rl.state_encoder import build_vlm_rl_state
from src.rl.utils.io import ensure_dirs
from src.rl.utils.seeding import set_seed
from src.vlm.vlm_interface import QwenVLMClient, run_vlm_inference


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _parse_args() -> argparse.Namespace:
    config = RLConfig()
    vlm_model_dir = config.project_root / "models" / "vlm_rl"
    default_model = _first_existing([
        vlm_model_dir / "dqn_model.pth",
        vlm_model_dir / "pedestrian_collision_episode_020_dqn_model.pth",
        vlm_model_dir / "pedestrian_collision_episode_010_dqn_model.pth",
        vlm_model_dir / "car_collision_dqn_model.pth",
        vlm_model_dir / "night_dqn_model.pth",
        vlm_model_dir / "rain_dqn_model.pth",
    ])

    parser = argparse.ArgumentParser(
        description="Evaluate a trained VLM+RL DQN on the combined CARLA edge case."
    )
    parser.add_argument("--model", type=Path, default=default_model)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps-per-episode", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=config.seed)
    parser.add_argument("--scenario", default="combined_edge_case")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.project_root / "outputs" / "results" / "online_evaluation",
    )
    parser.add_argument(
        "--output-prefix",
        "--prefix",
        dest="output_prefix",
        default="vlm_rl_combined_edge",
    )
    parser.add_argument("--append", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = RLConfig()
    config.max_steps_per_episode = int(args.max_steps_per_episode)
    set_seed(int(args.seed))
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
    all_step_rows: list[dict] = []
    episode_rows: list[dict] = []

    print(f"VLM+RL EVAL | model={args.model}")
    print(
        f"VLM+RL EVAL | scenario={args.scenario} | episodes={args.episodes} | "
        f"max_steps={config.max_steps_per_episode} | seed={args.seed}"
    )

    for episode_id in range(args.episodes):
        env = RemoteCarlaEnv(config=config, scenario_name=args.scenario)
        aggregator = VLMWindowAggregator(window_size=config.vlm_window_size)
        episode_step_rows: list[dict] = []
        cumulative_distance_m = 0.0

        try:
            obs = env.reset()

            for step in range(config.max_steps_per_episode):
                vlm_latency_ms = NAN

                if step % config.vlm_interval_steps == 0:
                    started_at = time.perf_counter()
                    raw_vlm_output = run_vlm_inference(obs.frame_path, client=vlm_client)
                    vlm_latency_ms = (time.perf_counter() - started_at) * 1000.0
                    aggregator.add(raw_vlm_output)

                vlm_output = aggregator.aggregate()
                state = build_vlm_rl_state(obs, vlm_output)
                action_id = int(agent.select_action(state_vector=state, epsilon=0.0))
                result = env.step(action_id=action_id, vlm_output=vlm_output)

                row = build_step_row(
                    agent_type="vlm_rl",
                    episode_id=episode_id,
                    step=step,
                    scenario=args.scenario,
                    action_id=action_id,
                    reward=float(result.reward),
                    done=bool(result.done),
                    obs=obs,
                    next_obs=result.observation,
                    fixed_delta_seconds=config.fixed_delta_seconds,
                    max_steps=config.max_steps_per_episode,
                    cumulative_distance_m=cumulative_distance_m,
                    vlm_output=vlm_output,
                    vlm_latency_ms=vlm_latency_ms,
                )
                cumulative_distance_m = float(row["distance_traveled_m"])
                episode_step_rows.append(row)

                obs = result.observation
                if result.done:
                    break
        finally:
            env.close()

        episode_row = compute_episode_metrics(
            agent_type="vlm_rl",
            episode_id=episode_id,
            scenario=args.scenario,
            model_path=args.model,
            step_rows=episode_step_rows,
        )
        all_step_rows.extend(episode_step_rows)
        episode_rows.append(episode_row)

        print(
            f"VLM+RL EVAL EPISODE | episode={episode_id} | "
            f"reward={float(episode_row['total_reward']):.2f} | "
            f"steps={episode_row['episode_steps']} | "
            f"collisions={episode_row['collision_count']} | "
            f"lane_invasions={episode_row['lane_invasion_count']}"
        )

    step_path, episode_path, aggregate_path = write_evaluation_outputs(
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        step_rows=all_step_rows,
        episode_rows=episode_rows,
        append=bool(args.append),
    )
    comparison_path = write_comparison_if_available(args.output_dir)

    print(f"VLM+RL EVAL SAVED | steps={step_path}")
    print(f"VLM+RL EVAL SAVED | episodes={episode_path}")
    print(f"VLM+RL EVAL SAVED | aggregate={aggregate_path}")
    if comparison_path is not None:
        print(f"VLM+RL EVAL SAVED | comparison={comparison_path}")


if __name__ == "__main__":
    main()
