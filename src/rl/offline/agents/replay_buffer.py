from collections import deque
from dataclasses import dataclass
import random

import numpy as np


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.buffer, batch_size)

        return Transition(
            state=np.stack([item.state for item in batch]),
            action=np.array([item.action for item in batch], dtype=np.int64),
            reward=np.array([item.reward for item in batch], dtype=np.float32),
            next_state=np.stack([item.next_state for item in batch]),
            done=np.array([item.done for item in batch], dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)