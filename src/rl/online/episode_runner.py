from src.rl.action_space import ACTION_NAMES
from src.rl.online.vlm_window import VLMWindowAggregator
from src.rl.state_encoder import build_rl_only_state, build_vlm_rl_state
import json


def run_online_episode(
    env,
    agent,
    logger,
    episode_id: int,
    use_vlm: bool = False,
    vlm_runner=None,
):
    obs = env.reset()
    total_reward = 0.0

    aggregator = VLMWindowAggregator(
        window_size=env.config.vlm_window_size
    )

    for step in range(env.config.max_steps_per_episode):
        raw_vlm_output = None
        aggregated_vlm_output = None
        vlm_fresh = 0

        if use_vlm:
            if vlm_runner is None:
                raise ValueError("vlm_runner is required when use_vlm=True")

            if step % env.config.vlm_interval_steps == 0:
                try:
                    raw_vlm_output = vlm_runner(obs.frame_path)
                    aggregator.add(raw_vlm_output)
                    vlm_fresh = 1
                except Exception as e:
                    print(f"[WARNING] VLM failed at step {step}: {e}")

            aggregated_vlm_output = aggregator.aggregate()
            state = build_vlm_rl_state(obs, aggregated_vlm_output)

        else:
            state = build_rl_only_state(obs)

        action_id = agent.act(state)

        result = env.step(
            action_id=action_id,
            vlm_output=aggregated_vlm_output,
        )

        total_reward += result.reward

        current_obs = result.observation

        row = {
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
            "reward": result.reward,
            "total_reward": total_reward,
            "total_episode_reward": total_reward,
            "done": result.done,
            "vlm_fresh": vlm_fresh,
            "raw_vlm_output": json.dumps(raw_vlm_output) if raw_vlm_output is not None else "",
            "aggregated_vlm_output": json.dumps(aggregated_vlm_output) if aggregated_vlm_output is not None else "",
        }

        row.update({
            "next_frame_path": current_obs.frame_path,
            "next_speed": current_obs.speed,
            "next_collision": current_obs.collision,
            "next_lane_invasion": current_obs.lane_invasion,
        })

        logger.log_step(row)

        obs = current_obs

        if result.done:
            break

    return total_reward