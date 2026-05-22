from __future__ import annotations

from src.rl.config import RLConfig
from src.rl.offline.agents.dqn import DQNAgent
from src.rl.online.online_logger import OnlineLogger
from src.rl.online.remote_carla_env import RemoteCarlaEnv
from src.rl.online.training_runner import run_online_training_episode
from src.rl.utils.io import ensure_dirs


def main() -> None:
    config = RLConfig()
    checkpoint_interval = 10
    online_model_dir = config.project_root / "models" / "online"

    def save_checkpoint(scenario_name: str, completed_episodes: int) -> None:
        checkpoint_path = (
            online_model_dir
            / f"{scenario_name}_episode_{completed_episodes:03d}_dqn_model.pth"
        )
        agent.save(checkpoint_path)
        print(f"ONLINE CHECKPOINT | saved={checkpoint_path}")

    ensure_dirs([
        config.model_dir,
        config.log_dir,
        config.result_dir,
        config.online_output_dir,
        online_model_dir,
    ])

    agent = DQNAgent(
        state_dim=config.rl_only_state_dim,
        num_actions=config.action_dim,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        replay_capacity=config.replay_capacity,
    )

    global_step = 0

    for scenario_name in config.train_scenarios:
        env = RemoteCarlaEnv(config=config, scenario_name=scenario_name)
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

                completed_episodes = episode_id + 1
                if completed_episodes % checkpoint_interval == 0:
                    save_checkpoint(scenario_name, completed_episodes)

                    if completed_episodes < config.num_episodes:
                        env.close()
                        env = RemoteCarlaEnv(config=config, scenario_name=scenario_name)
                        print(
                            "ONLINE WORKER RESTART | "
                            f"scenario={scenario_name} | after_episode={episode_id}"
                        )
        finally:
            logger.save()
            env.close()

        save_path = online_model_dir / f"{scenario_name}_dqn_model.pth"
        agent.save(save_path)


if __name__ == "__main__":
    main()
