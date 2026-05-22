# Combined-Edge Evaluation Metrics

This evaluation compares the existing RL-only agent and VLM+RL agent on the same CARLA `combined_edge_case` scenario. The scenario is not changed by the evaluation pipeline.

## Logged Timestep Metrics

Each step CSV records a comparable schema for both agents:

- `episode_id`, `step`, `agent_type`, `scenario`
- `selected_action_id`, `selected_action_name`
- `reward`, `done`, `termination_reason`
- `speed_kmh`, `throttle`, `brake`, `steer`
- `collision_flag`, `collision_type`, `lane_invasion_flag`
- `distance_traveled_m`, `route_completion`
- `nearest_vehicle_distance_m`, `nearest_pedestrian_distance_m`
- `min_hazard_distance_m`, `ttc_seconds`
- `critical_moment`, `unsafe_action`

For VLM+RL, the raw step CSV also records:

- `vlm_hazard_level`
- `vlm_collision_risk`
- `vlm_pedestrian_presence`
- `vlm_lane_safety`
- `vlm_action_urgency`
- `vlm_recommended_action`
- `vlm_latency_ms`

Some fields are not currently exposed by the existing `CarlaObservation`/worker API. These are logged as `NaN` or empty values instead of being fabricated. Examples include TTC, route completion, nearest vehicle distance, nearest pedestrian distance, hazard distance, and collision type.

## Episode Metrics

At the end of each episode, the evaluator computes:

- `total_reward`: sum of rewards over the episode.
- `episode_steps`: number of executed evaluation steps.
- `success`: true when no collision occurred, no lane-invasion/timeout/stuck termination occurred, and route completion is sufficient if available.
- `collision_occurred`: whether any collision flag occurred.
- `collision_count`: number of collision-flagged timesteps.
- `collision_speed_kmh_mean` and `collision_speed_kmh_max`: ego speed at collision timesteps.
- `average_speed_kmh` and `max_speed_kmh`: driving efficiency and aggressiveness indicators.
- `total_distance_m`: integrated from ego speed and fixed CARLA timestep.
- `route_completion_final`: final route completion when available, otherwise `NaN`.
- `lane_invasion_count`: number of lane-invasion-flagged timesteps.
- `emergency_brake_count`: brake `>= 0.7` or selected discrete action `brake`.
- `hard_steer_count`: absolute steering `>= 0.7`.
- `min_ttc_seconds`, `mean_ttc_seconds`: risk timing metrics when TTC is available.
- `min_hazard_distance_m`, `mean_hazard_distance_m`: proximity risk metrics when hazard distance is available.
- `tcf_collisions_per_1000_steps`: collisions divided by steps, multiplied by 1000.
- `dcf_collisions_per_km`: collisions divided by driven distance in km.
- `unsafe_action_count`, `unsafe_action_rate`: continue/accelerate during a critical moment with low brake, when TTC or hazard distance is computable.
- `mean_vlm_latency_ms`: average VLM inference latency for VLM+RL when available.
- `mean_vlm_hazard_level`: encoded hazard severity, where low=0, medium=1, high=2.

Thresholds are intentionally transparent:

- TTC critical threshold: `3.0` seconds
- Hazard distance critical threshold: `5.0` meters
- Emergency brake threshold: `0.7`
- Hard steer threshold: `0.7`

## Aggregate Metrics

Each agent receives an aggregate CSV with:

- `episodes`
- `success_rate`
- `collision_rate`
- `mean_route_completion`, `std_route_completion`
- `mean_total_reward`, `std_total_reward`
- `mean_average_speed_kmh`, `std_average_speed_kmh`
- `mean_total_distance_m`, `std_total_distance_m`
- `mean_collision_speed_kmh`
- `mean_tcf_collisions_per_1000_steps`
- `mean_dcf_collisions_per_km`
- `mean_min_ttc_seconds`
- `mean_min_hazard_distance_m`
- `mean_unsafe_action_rate`
- `mean_vlm_latency_ms` when available

When both agent aggregate files exist, the pipeline also writes `combined_edge_comparison_aggregate.csv` with:

- `success_rate_delta = vlm_rl - rl_only`
- `route_completion_delta = vlm_rl - rl_only`
- `collision_rate_reduction = rl_only - vlm_rl`
- `dcf_reduction = rl_only - vlm_rl`
- `collision_speed_reduction = rl_only - vlm_rl`
- `min_ttc_improvement = vlm_rl - rl_only`
- `min_hazard_distance_improvement = vlm_rl - rl_only`

## Why These Metrics Matter

The evaluation follows common autonomous-driving RL practice:

- Driving efficiency: average speed, route completion, total distance.
- Safety: collision rate, collision speed, TCF, DCF.
- Risk behavior over time: TTC and minimum hazard distance.
- Reliability: success rate across multiple episodes.
- VLM contribution: semantic signals and VLM latency.

The goal is to test whether VLM+RL improves safety, robustness, and decision quality over RL-only in the same unseen combined edge-case scenario.

## Output Structure

New evaluation outputs are saved under `outputs/results/online_evaluation/`:

```text
raw/
  rl_only_combined_edge_steps.csv
  vlm_rl_combined_edge_steps.csv

episodes/
  rl_only_combined_edge_episodes.csv
  vlm_rl_combined_edge_episodes.csv

aggregate/
  rl_only_combined_edge_aggregate.csv
  vlm_rl_combined_edge_aggregate.csv
  combined_edge_comparison_aggregate.csv

figures/
  success_route_completion.png
  collision_safety_comparison.png
  ttc_over_time.png
  hazard_distance_over_time.png
  speed_over_time.png
  action_distribution.png
  vlm_semantic_signals.png
  linkedin_summary_figure.png
```

Existing old result CSVs are not deleted.

## Dry-Run Commands

Dry run RL-only:

```bash
python -m src.rl.online.evaluate_combined_edge_remote --episodes 2 --max-steps-per-episode 1000 --output-prefix rl_only_combined_edge
```

Dry run VLM+RL:

```bash
python -m src.rl.online.evaluate_vlm_rl_combined_edge_remote --episodes 2 --max-steps-per-episode 1000 --output-prefix vlm_rl_combined_edge
```

## Final 20-Episode Evaluation

Final RL-only evaluation:

```bash
python -m src.rl.online.evaluate_combined_edge_remote --episodes 20 --max-steps-per-episode 1000 --output-prefix rl_only_combined_edge
```

Final VLM+RL evaluation:

```bash
python -m src.rl.online.evaluate_vlm_rl_combined_edge_remote --episodes 20 --max-steps-per-episode 1000 --output-prefix vlm_rl_combined_edge
```

Generate plots:

```bash
python -m src.rl.online.plot_combined_edge_comparison
```

## Nohup Final Runs

RL-only final run:

```bash
mkdir -p outputs/logs
nohup python -u -m src.rl.online.evaluate_combined_edge_remote --episodes 20 --max-steps-per-episode 1000 --output-prefix rl_only_combined_edge > outputs/logs/evaluate_rl_only_combined_edge.log 2>&1 &
echo $!
```

VLM+RL final run:

```bash
mkdir -p outputs/logs
nohup python -u -m src.rl.online.evaluate_vlm_rl_combined_edge_remote --episodes 20 --max-steps-per-episode 1000 --output-prefix vlm_rl_combined_edge > outputs/logs/evaluate_vlm_rl_combined_edge.log 2>&1 &
echo $!
```

Plot generation after both runs:

```bash
python -m src.rl.online.plot_combined_edge_comparison
```
