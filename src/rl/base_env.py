from abc import ABC, abstractmethod

from src.rl.action_space import DrivingAction
from src.rl.schemas import DrivingState


class BaseDrivingEnv(ABC):
    @abstractmethod
    def reset(self) -> DrivingState:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: DrivingAction) -> tuple[DrivingState, float, bool, dict]:
        raise NotImplementedError
    

