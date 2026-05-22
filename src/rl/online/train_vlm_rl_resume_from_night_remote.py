from __future__ import annotations

from pathlib import Path

from src.rl.config import RLConfig
from src.rl.offline.agents.dqn import DQNAgent
from src.rl.online.online_logger import OnlineLogger
from src.rl.online.remote_carla_env import RemoteCarlaEnv
from src.rl.online.training_runner import run_online_training_episode
from src.rl.utils.io import ensure_dirs
from src.vlm.vlm_interface import QwenVLMClient, run_vlm_inference


class CachedVLMRunner:
    def __init__(self) -> None:
        self.client = QwenVLMClient()
        self.cache: dict[str, dict] = {}

    def __call__(self, image_path: str | Path) -> dict:
        key = str(image_path)
        if key not in self.cache:
            self.cache[key] = run_vlm_inference(image_path, client=self.client)
        return self.cache[key]


def main() -> None:
    config = RLConfig()
    checkpoint_interval = 10
    vlm_model_dir = config.project_root / "models" / "vlm_rl"
    resume_path = vlm_model_dir / "night_episode_010_dqn_model.pth"
    vlm_runner = CachedVLMRunner()

    ensure_dirs([
        config.model_dir,
        config.log_dir,
        config.result_dir,
        config.online_output_dir,
        vlm_model_dir,
    ])

    if not resume_path.exists() or resume_path.stat().st_size < 1000:
        raise FileNotFoundError(f"Resume checkpoint is missing or invalid: {resume_path}")

    agent = DQNAgent(
        state_dim=config.vlm_rl_state_dim,
        num_actions=config.action_dim,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        replay_capacity=config.replay_capacity,
    )
    agent.load(resume_path)
    agent.policy_net.train()
    agent.target_net.eval()

    # Continue with low exploration. The loaded checkpoint has already trained
    # through rain and 10 night episodes.
    global_step = max(config.epsilon_decay_steps, 29176)

    resume_plan = [
        ("night", 10),
        ("car_collision", 0),
        ("pedestrian_collision", 0),
    ]

    def save_checkpoint(scenario_name: str, completed_episodes: int) -> None:
        checkpoint_path = (
            vlm_model_dir
            / f"{scenario_name}_episode_{completed_episodes:03d}_dqn_model.pth"
        )
        agent.save(checkpoint_path)
        print(f"VLM+RL CHECKPOINT | saved={checkpoint_path}")

    print(f"VLM+RL RESUME | loaded={resume_path}")

    for scenario_name, start_episode in resume_plan:
        env = RemoteCarlaEnv(config=config, scenario_name=scenario_name)
        logger = OnlineLogger(
            config.online_output_dir
            / scenario_name
            / "vlm_rl_training_results_resume.csv"
        )

        try:
            for episode_id in range(start_episode, config.num_episodes):
                total_reward, global_step = run_online_training_episode(
                    env=env,
                    agent=agent,
                    logger=logger,
                    episode_id=episode_id,
                    global_step=global_step,
                    batch_size=config.batch_size,
                    use_vlm=True,
                    vlm_runner=vlm_runner,
                )

                print(
                    f"VLM+RL TRAIN RESUME | scenario={scenario_name} | "
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
                            "VLM+RL WORKER RESTART | "
                            f"scenario={scenario_name} | after_episode={episode_id}"
                        )
        finally:
            logger.save()
            env.close()

        save_path = vlm_model_dir / f"{scenario_name}_dqn_model.pth"
        agent.save(save_path)
        print(f"VLM+RL SCENARIO FINAL | saved={save_path}")

    final_path = vlm_model_dir / "combined_trained_dqn_model.pth"
    agent.save(final_path)
    print(f"VLM+RL FINAL | saved={final_path}")


if __name__ == "__main__":
    main()
