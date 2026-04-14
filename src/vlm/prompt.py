VLM_PROMPT = """
You are analyzing a driving scene from the ego vehicle perspective in the CARLA simulator.

Return ONLY a valid JSON object with exactly these fields and no extra text:

{
  "hazard_level": "low|medium|high",
  "obstacle_presence": "yes|no",
  "collision_risk": "yes|no",
  "lane_safety": "safe|unsafe",
  "crossing_pedestrian_presence": "yes|no",
  "action_urgency": "continue|slow|brake"
}

Definitions:
- hazard_level:
  - low: no immediate danger is visible
  - medium: caution is needed, a developing or moderate risk is visible
  - high: immediate or severe danger is visible

- obstacle_presence:
  - yes: an obstacle, blocked path, stopped vehicle, or object appears to affect the ego vehicle's path
  - no: otherwise

- collision_risk:
  - yes: a near-term collision risk appears likely if the ego vehicle continues unchanged
  - no: otherwise

- lane_safety:
  - safe: the ego vehicle appears properly aligned and safely positioned in lane
  - unsafe: the ego vehicle appears drifting, off-lane, or in a dangerous lane position

- crossing_pedestrian_presence:
  - yes: a pedestrian is crossing or likely interfering with the ego vehicle's path
  - no: otherwise

- action_urgency:
  - continue: no urgent reaction is needed
  - slow: slowing down is advisable
  - brake: immediate braking is advisable

Rules:
- Output JSON only.
- Do not explain.
- Do not include markdown.
- Do not include any keys other than the six required keys.
"""