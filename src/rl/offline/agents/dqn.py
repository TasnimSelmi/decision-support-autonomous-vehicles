from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from src.rl.action_space import DrivingAction
from src.rl.agents.replay_buffer import ReplayBuffer


class DQNNetwork(nn.Module):
    def __init__(self, state_dim: int, num_actions: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        learning_rate: float,
        gamma: float,
        replay_capacity: int,
        device: str | None = None,
    ):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.policy_net = DQNNetwork(state_dim, num_actions).to(self.device)
        self.target_net = DQNNetwork(state_dim, num_actions).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate,
        )

        self.loss_fn = nn.SmoothL1Loss()

        self.replay_buffer = ReplayBuffer(replay_capacity)

    def select_action(self, state_vector: np.ndarray, epsilon: float) -> DrivingAction:
        """
        Epsilon-greedy action selection.

        With probability epsilon:
            choose random action.
        Otherwise:
            choose action with highest Q-value.
        """

        if random.random() < epsilon:
            return DrivingAction(random.randrange(self.num_actions))

        with torch.no_grad():
            state_tensor = torch.tensor(
                state_vector,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            q_values = self.policy_net(state_tensor)
            action_id = int(torch.argmax(q_values, dim=1).item())

        return DrivingAction(action_id)

    def optimize(self, batch_size: int) -> float | None:
        if len(self.replay_buffer) < batch_size:
            return None

        batch = self.replay_buffer.sample(batch_size)

        states = torch.tensor(
            batch.state,
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.tensor(
            batch.action,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(1)

        rewards = torch.tensor(
            batch.reward,
            dtype=torch.float32,
            device=self.device,
        )

        next_states = torch.tensor(
            batch.next_state,
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.tensor(
            batch.done,
            dtype=torch.float32,
            device=self.device,
        )

        current_q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1).values
            target_q_values = rewards + self.gamma * next_q_values * (1.0 - dones)

        loss = self.loss_fn(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            max_norm=10.0,
        )

        self.optimizer.step()

        return float(loss.item())

    def update_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "state_dim": self.state_dim,
                "num_actions": self.num_actions,
            },
            path,
        )

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])

        self.policy_net.eval()
        self.target_net.eval()