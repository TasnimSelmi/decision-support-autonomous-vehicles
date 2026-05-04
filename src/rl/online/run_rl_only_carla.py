from src.rl.agents.dqn import DQNAgent
from src.rl.config import RLConfig
from src.rl.online.carla_env import OnlineCarlaEnv
from src.rl.online.episode_runner import run_online_episode
from src.rl.online.online_logger import OnlineLogger


def main():
    config = RLConfig()

    agent = DQNAgent(
        state_dim=config.rl_only_state_dim,
        action_dim=config.action_dim,
        config=config,
    )

    agent.load(config.rl_only_model_path)

    scenarios = config.train_scenarios + config.test_scenarios

    for scenario_name in scenarios:
        env = OnlineCarlaEnv(
            config=config,
            scenario_name=scenario_name,
        )

        logger = OnlineLogger(
            config.online_output_dir / scenario_name / "rl_only_results.csv"
        )

        try:
            for episode_id in range(config.online_episodes):
                total_reward = run_online_episode(
                    env=env,
                    agent=agent,
                    logger=logger,
                    episode_id=episode_id,
                    use_vlm=False,
                )

                print(
                    f"RL-only | scenario={scenario_name} | "
                    f"episode={episode_id} | reward={total_reward:.2f}"
                )

        finally:
            logger.save()
            env.close()


if __name__ == "__main__":
    main()