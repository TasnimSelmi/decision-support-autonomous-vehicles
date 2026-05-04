import numpy as np
import pandas as pd

from src.rl.agents.dqn import DQNAgent
from src.rl.config import RLConfig
from src.rl.envs.dummy_env import DummyDrivingEnv
from src.rl.utils.io import ensure_dirs
from src.rl.utils.seeding import set_seed


def linear_epsilon(step: int, config: RLConfig) -> float:
    fraction = min(step / config.epsilon_decay_steps, 1.0)
    return config.epsilon_start + fraction * (
        config.epsilon_end - config.epsilon_start
    )


def train() -> None:
    config = RLConfig()

    set_seed(config.seed)

    ensure_dirs(
        [
            config.model_dir,
            config.log_dir,
            config.result_dir,
        ]
    )

    env = DummyDrivingEnv(
        max_steps=config.max_steps_per_episode,
    )

    agent = DQNAgent(
        state_dim=config.state_dim,
        num_actions=config.num_actions,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        replay_capacity=config.replay_capacity,
    )

    logs: list[dict] = []
    global_step = 0

    for episode in range(1, config.num_episodes + 1):
        state = env.reset()
        state_vector = np.array(state.as_vector(), dtype=np.float32)

        total_reward = 0.0
        losses: list[float] = []

        for _ in range(config.max_steps_per_episode):
            epsilon = linear_epsilon(global_step, config)

            action = agent.select_action(
                state_vector=state_vector,
                epsilon=epsilon,
            )

            next_state, reward, done, _ = env.step(action)
            next_state_vector = np.array(next_state.as_vector(), dtype=np.float32)

            agent.replay_buffer.push(
                state=state_vector,
                action=int(action),
                reward=reward,
                next_state=next_state_vector,
                done=done,
            )

            loss = agent.optimize(config.batch_size)

            if loss is not None:
                losses.append(loss)

            if global_step % config.target_update_frequency == 0:
                agent.update_target_network()

            state_vector = next_state_vector
            total_reward += reward
            global_step += 1

            if done:
                break

        mean_loss = sum(losses) / len(losses) if losses else None

        logs.append(
            {
                "episode": episode,
                "total_reward": total_reward,
                "epsilon": linear_epsilon(global_step, config),
                "mean_loss": mean_loss,
                "global_step": global_step,
            }
        )

        if episode % 25 == 0:
            print(
                f"Episode {episode:04d} | "
                f"Reward {total_reward:8.2f} | "
                f"Epsilon {linear_epsilon(global_step, config):.3f}"
            )

    agent.save(config.dqn_model_path)

    log_df = pd.DataFrame(logs)
    log_df.to_csv(config.training_log_path, index=False)

    print("\nTraining finished.")
    print(f"Model saved to: {config.dqn_model_path}")
    print(f"Logs saved to: {config.training_log_path}")


if __name__ == "__main__":
    train()