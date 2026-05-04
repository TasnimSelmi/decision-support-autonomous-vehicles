from __future__ import annotations

from src.rl.agents.dqn import DQNAgent
from src.rl.config import RLConfig
from src.rl.online.carla_env import OnlineCarlaEnv
from src.rl.online.online_logger import OnlineLogger
from src.rl.online.training_runner import run_online_training_episode
from src.rl.utils.io import ensure_dirs


def main() -> None:
    config = RLConfig()
    scenarios = config.train_scenarios

    ensure_dirs([
        config.model_dir,
        config.log_dir,
        config.result_dir,
        config.online_output_dir,
    ])

    agent = DQNAgent(
        state_dim=config.rl_only_state_dim,
        action_dim=config.action_dim,
        config=config,
    )

    global_step = 0

    for scenario_name in scenarios:
        env = OnlineCarlaEnv(
            config=config,
            scenario_name=scenario_name,
        )

        logger = OnlineLogger(
            config.online_output_dir / scenario_name / "online_training_results.csv"
        )

        try:
            for episode_id in range(config.num_episodes):
                total_reward, global_step = run_online_training_episode(
                    env=env,
                    agent=agent,
                    logger=logger,
                    episode_id=episode_id,
                    global_step=global_step,
                    batch_size=config.batch_size,
                )

                print(
                    f"ONLINE TRAIN | scenario={scenario_name} | "
                    f"episode={episode_id} | reward={total_reward:.2f} | "
                    f"steps={global_step}"
                )

        finally:
            logger.save()
            env.close()

        save_path = config.project_root / "models" / "online" / f"{scenario_name}_dqn_model.pth"
        agent.save(save_path)


if __name__ == "__main__":
    main()
