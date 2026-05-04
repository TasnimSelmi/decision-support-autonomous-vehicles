from __future__ import annotations

import json

import numpy as np

from src.rl.action_space import ACTION_NAMES
from src.rl.online.vlm_window import VLMWindowAggregator
from src.rl.state_encoder import build_rl_only_state


def linear_epsilon(step: int, config) -> float:
    fraction = min(step / max(config.epsilon_decay_steps, 1), 1.0)
    return config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)


def run_online_training_episode(
    env,
    agent,
    logger,
    episode_id: int,
    global_step: int,
    batch_size: int,
    use_vlm: bool = False,
    vlm_runner=None,
):
    if use_vlm:
        raise NotImplementedError(
            "Online CARLA training is currently implemented for RL-only state."
        )

    obs = env.reset()
    total_reward = 0.0
    last_loss = None

    for step in range(env.config.max_steps_per_episode):
        state = build_rl_only_state(obs)
        epsilon = linear_epsilon(global_step, env.config)
        action_id = agent.select_action(state_vector=state, epsilon=epsilon)

        result = env.step(
            action_id=int(action_id),
            vlm_output=None,
        )

        next_obs = result.observation
        next_state = build_rl_only_state(next_obs)

        agent.replay_buffer.push(
            state=state,
            action=int(action_id),
            reward=float(result.reward),
            next_state=next_state,
            done=bool(result.done),
        )

        if len(agent.replay_buffer) >= batch_size:
            last_loss = agent.optimize(batch_size)

        if global_step % env.config.target_update_frequency == 0:
            agent.update_target_network()

        total_reward += float(result.reward)

        logger.log_step({
            "episode_id": episode_id,
            "scenario_name": env.scenario_name,
            "step": step,
            "frame_path": obs.frame_path,
            "speed": obs.speed,
            "steering": obs.steering,
            "throttle": obs.throttle,
            "brake": obs.brake,
            "collision": obs.collision,
            "lane_invasion": obs.lane_invasion,
            "action_id": int(action_id),
            "action_name": ACTION_NAMES.get(int(action_id), str(action_id)),
            "reward": float(result.reward),
            "total_reward": total_reward,
            "done": bool(result.done),
            "epsilon": epsilon,
            "loss": last_loss if last_loss is not None else "",
            "replay_size": len(agent.replay_buffer),
            "raw_vlm_output": "",
            "aggregated_vlm_output": "",
            "next_frame_path": next_obs.frame_path,
            "next_speed": next_obs.speed,
            "next_collision": next_obs.collision,
            "next_lane_invasion": next_obs.lane_invasion,
        })

        obs = next_obs
        global_step += 1

        if result.done:
            break

    return total_reward, global_step
