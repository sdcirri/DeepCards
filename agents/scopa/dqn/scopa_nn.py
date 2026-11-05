import torch.nn as nn
import torch

from .play_candidates import ACTION_FEATURE_DIM


class ScopaNN(nn.Module):
    def __init__(self, observation_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim + ACTION_FEATURE_DIM, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, observations: torch.Tensor, action_features: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat((observations, action_features), dim=-1)
        return self.net(inputs).squeeze(-1)
