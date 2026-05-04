from collections import deque
from collections import Counter


SAFE_DEFAULT_VLM_OUTPUT = {
    "hazard_level": "medium",
    "obstacle_presence": "yes",
    "collision_risk": "yes",
    "lane_safety": "unsafe",
    "crossing_pedestrian_presence": "yes",
    "action_urgency": "slow",
}


class VLMWindowAggregator:
    def __init__(self, window_size: int = 5):
        self.buffer = deque(maxlen=window_size)

    def add(self, vlm_output: dict) -> None:
        if vlm_output:
            self.buffer.append(vlm_output)

    def aggregate(self) -> dict:
        if not self.buffer:
            return SAFE_DEFAULT_VLM_OUTPUT.copy()

        outputs = list(self.buffer)

        return {
            "hazard_level": self._max_hazard(outputs),
            "obstacle_presence": self._majority(outputs, "obstacle_presence", "no"),
            "collision_risk": self._any_yes(outputs, "collision_risk"),
            "lane_safety": self._any_unsafe(outputs),
            "crossing_pedestrian_presence": self._majority(
                outputs,
                "crossing_pedestrian_presence",
                "no",
            ),
            "action_urgency": self._max_urgency(outputs),
        }

    def _majority(self, outputs, key: str, default: str) -> str:
        values = [str(output.get(key, default)).lower() for output in outputs]
        counts = Counter(values)
        if not counts:
            return default
        return counts.most_common(1)[0][0]

    def _any_yes(self, outputs, key: str) -> str:
        return "yes" if any(o.get(key, "no") == "yes" for o in outputs) else "no"

    def _any_unsafe(self, outputs) -> str:
        return "unsafe" if any(
            o.get("lane_safety", "safe") == "unsafe" for o in outputs
        ) else "safe"

    def _max_hazard(self, outputs) -> str:
        order = {"low": 0, "medium": 1, "high": 2}
        reverse = {0: "low", 1: "medium", 2: "high"}

        max_value = max(
            order.get(o.get("hazard_level", "low"), 0)
            for o in outputs
        )
        return reverse[max_value]

    def _max_urgency(self, outputs) -> str:
        order = {
            "continue": 0,
            "slow": 1,
            "slow_down": 1,
            "brake": 2,
            "stop": 2,
        }
        reverse = {
            0: "continue",
            1: "slow",
            2: "brake",
        }

        max_value = max(
            order.get(o.get("action_urgency", "continue"), 0)
            for o in outputs
        )
        return reverse[max_value]