from __future__ import annotations

import time

from src.rl.online.evaluation_metrics import NAN, build_step_row
from src.rl.online.vlm_window import VLMWindowAggregator
from src.rl.state_encoder import build_rl_only_state, build_vlm_rl_state


def run_online_evaluation_episode(
    env,
    agent,
    episode_id: int,
    use_vlm: bool = False,
    vlm_runner=None,
):
    obs = env.reset()
    total_reward = 0.0
    cumulative_distance_m = 0.0
    step_rows: list[dict] = []

    aggregator = VLMWindowAggregator(window_size=env.config.vlm_window_size)

    for step in range(env.config.max_steps_per_episode):
        raw_vlm_output = None
        aggregated_vlm_output = None
        vlm_latency_ms = NAN

        if use_vlm:
            if vlm_runner is None:
                raise ValueError("vlm_runner is required when use_vlm=True")

            if step % env.config.vlm_interval_steps == 0:
                started_at = time.perf_counter()
                raw_vlm_output = vlm_runner(obs.frame_path)
                vlm_latency_ms = (time.perf_counter() - started_at) * 1000.0
                aggregator.add(raw_vlm_output)

            aggregated_vlm_output = aggregator.aggregate()
            state = build_vlm_rl_state(obs, aggregated_vlm_output)
        else:
            state = build_rl_only_state(obs)

        action_id = int(agent.select_action(state_vector=state, epsilon=0.0))

        result = env.step(
            action_id=action_id,
            vlm_output=aggregated_vlm_output,
        )

        next_obs = result.observation
        total_reward += float(result.reward)

        row = build_step_row(
            agent_type="vlm_rl" if use_vlm else "rl_only",
            episode_id=episode_id,
            step=step,
            scenario=env.scenario_name,
            action_id=action_id,
            reward=float(result.reward),
            done=bool(result.done),
            obs=obs,
            next_obs=next_obs,
            fixed_delta_seconds=env.config.fixed_delta_seconds,
            max_steps=env.config.max_steps_per_episode,
            cumulative_distance_m=cumulative_distance_m,
            vlm_output=aggregated_vlm_output,
            vlm_latency_ms=vlm_latency_ms,
        )

        cumulative_distance_m = float(row["distance_traveled_m"])
        step_rows.append(row)

        obs = next_obs

        if result.done:
            break

    return total_reward, step_rows