from __future__ import annotations

import argparse
from pathlib import Path

from src.rl.config import RLConfig
from src.rl.offline.agents.dqn import DQNAgent
from src.rl.online.evaluation_metrics import (
    build_step_row,
    compute_episode_metrics,
    write_comparison_if_available,
    write_evaluation_outputs,
)
from src.rl.online.remote_carla_env import RemoteCarlaEnv
from src.rl.state_encoder import build_rl_only_state
from src.rl.utils.io import ensure_dirs
from src.rl.utils.seeding import set_seed


def _parse_args() -> argparse.Namespace:
    config = RLConfig()
    default_model = (
        config.project_root
        / "models"
        / "online"
        / "pedestrian_collision_resume_episode_010_dqn_model.pth"
    )

    parser = argparse.ArgumentParser(
        description="Evaluate a trained RL-only DQN on the combined CARLA edge case."
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
        default="rl_only_combined_edge",
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
        state_dim=config.rl_only_state_dim,
        num_actions=config.action_dim,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        replay_capacity=config.replay_capacity,
    )
    agent.load(args.model)
    agent.policy_net.eval()
    agent.target_net.eval()

    all_step_rows: list[dict] = []
    episode_rows: list[dict] = []

    print(f"RL-ONLY EVAL | model={args.model}")
    print(
        f"RL-ONLY EVAL | scenario={args.scenario} | episodes={args.episodes} | "
        f"max_steps={config.max_steps_per_episode} | seed={args.seed}"
    )

    for episode_id in range(args.episodes):
        env = RemoteCarlaEnv(config=config, scenario_name=args.scenario)
        episode_step_rows: list[dict] = []
        cumulative_distance_m = 0.0

        try:
            obs = env.reset()

            for step in range(config.max_steps_per_episode):
                state = build_rl_only_state(obs)
                action_id = int(agent.select_action(state_vector=state, epsilon=0.0))
                result = env.step(action_id=action_id, vlm_output=None)

                row = build_step_row(
                    agent_type="rl_only",
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
                )
                cumulative_distance_m = float(row["distance_traveled_m"])
                episode_step_rows.append(row)

                obs = result.observation
                if result.done:
                    break
        finally:
            env.close()

        episode_row = compute_episode_metrics(
            agent_type="rl_only",
            episode_id=episode_id,
            scenario=args.scenario,
            model_path=args.model,
            step_rows=episode_step_rows,
        )
        all_step_rows.extend(episode_step_rows)
        episode_rows.append(episode_row)

        print(
            f"RL-ONLY EVAL EPISODE | episode={episode_id} | "
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

    print(f"RL-ONLY EVAL SAVED | steps={step_path}")
    print(f"RL-ONLY EVAL SAVED | episodes={episode_path}")
    print(f"RL-ONLY EVAL SAVED | aggregate={aggregate_path}")
    if comparison_path is not None:
        print(f"RL-ONLY EVAL SAVED | comparison={comparison_path}")


if __name__ == "__main__":
    main()
