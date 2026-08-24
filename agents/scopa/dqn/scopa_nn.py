from dataclasses import dataclass
from collections import deque
import random

from numpy.typing import NDArray
import numpy as np

import torch.nn as nn
import torch

from .play_candidates import ACTION_FEATURE_DIM


class ScopaNN(nn.Module):
    def __init__(self, observation_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim + ACTION_FEATURE_DIM, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, observations: torch.Tensor, action_features: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat((observations, action_features), dim=-1)
        return self.net(inputs).squeeze(-1)


@dataclass(frozen=True, slots=True)
class ScopaTransition:
    state: NDArray[np.float32]
    action_features: NDArray[np.float32]
    reward: float
    next_state: NDArray[np.float32]
    next_candidate_features: NDArray[np.float32]
    next_candidate_mask: NDArray[np.bool_]
    done: bool


class ScopaReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: deque[ScopaTransition] = deque(maxlen=capacity)

    def add(self, transition: ScopaTransition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[ScopaTransition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)
