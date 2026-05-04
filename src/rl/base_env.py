from abc import ABC, abstractmethod


class BaseRLEnv(ABC):

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action_id: int, vlm_output: dict | None = None):
        pass

    @abstractmethod
    def close(self):
        pass