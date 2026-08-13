from collections import deque
from typing import Any
import random

import matplotlib.pyplot as plt

import numpy as np

import torch.optim as optim
import torch.nn as nn
import torch

from games.deck import DECK

from .agent import AgentId, Cards2PAgent



class CardNN(nn.Module):
    def __init__(self, obs_dim: int, actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: deque[tuple] = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, next_mask, done) -> None:
        self.buffer.append((state, action, reward, next_state, next_mask, done))

    def sample(self, batch_size: int) -> list[tuple]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


def choose_action(
    state: np.ndarray,
    mask: np.ndarray,
    net: CardNN,
    epsilon: float,
    device: torch.device,
) -> int:
    legal_actions = np.flatnonzero(mask)

    if legal_actions.size == 0:
        raise ValueError('No legal actions available')

    if random.random() < epsilon:
        return int(np.random.choice(legal_actions))

    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        q_values = net(state_t).cpu().numpy().flatten()

    q_values[mask == 0] = -np.inf
    return int(q_values.argmax())


def train_step(
    online_net: CardNN,
    target_net: CardNN,
    optimizer: optim.Optimizer,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    device: torch.device,
) -> float:
    states, actions, rewards, next_states, next_masks, dones = zip(
        *replay_buffer.sample(batch_size)
    )

    states = torch.as_tensor(np.stack(states), dtype=torch.float32, device=device)
    actions = torch.as_tensor(actions, dtype=torch.long, device=device)
    rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    next_states = torch.as_tensor(np.stack(next_states), dtype=torch.float32, device=device)
    next_masks = torch.as_tensor(np.stack(next_masks), dtype=torch.float32, device=device)
    dones = torch.as_tensor(dones, dtype=torch.float32, device=device)

    q_values = online_net(states)
    chosen_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = target_net(next_states)
        next_q_values[next_masks == 0] = -float('inf')
        best_next_q_values = next_q_values.max(dim=1).values
        best_next_q_values = torch.where(
            next_masks.sum(dim=1) > 0,
            best_next_q_values,
            torch.zeros_like(best_next_q_values),
        )
        targets = rewards + gamma * best_next_q_values * (1 - dones)

    loss = nn.functional.smooth_l1_loss(chosen_q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10.0)
    optimizer.step()

    return loss.item()


def plot_losses(losses: list[float]) -> None:
    if not losses:
        return

    window = min(100, max(1, len(losses) // 10))
    kernel = np.ones(window, dtype=np.float64) / window
    smooth = np.convolve(losses, kernel, mode='valid')
    plt.plot(losses, alpha=0.25, label='batch loss')
    plt.plot(
        range(window - 1, window - 1 + len(smooth)),
        smooth,
        label=f'moving avg (w={window})',
    )
    plt.xlabel('train step')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


def train(
    env: Any,
    whoami: AgentId,
    episodes: int,
    actions: int,
    obs_dim: int,
    device: torch.device,
    training_opponent: Cards2PAgent,
    verbose: bool = True,
) -> CardNN:
    online_net = CardNN(obs_dim, actions).to(device)
    target_net = CardNN(obs_dim, actions).to(device)
    target_net.load_state_dict(online_net.state_dict())

    optimizer = optim.Adam(online_net.parameters(), lr=1e-4)
    replay_buffer = ReplayBuffer(capacity=100_000)

    batch_size = 64
    gamma = 0.99
    epsilon, epsilon_min, epsilon_decay = 1.0, 0.05, 0.995
    target_update_frequency = 1000
    total_steps = 0
    losses: list[float] = []

    for episode in range(episodes):
        env.reset()

        while whoami in env.agents and not (env.terminations[whoami] or env.truncations[whoami]):
            if env.agent_selection != whoami:
                env.step(training_opponent.step(env))
                continue

            obs = env.observe(whoami)
            state, mask = obs['observation'], obs['action_mask']

            action = choose_action(state, mask, online_net, epsilon, device)
            env.step(action)
            # Include rewards from our step and intervening opponent steps
            # (trick points are often assigned when the opponent follows).
            reward = float(env.rewards[whoami])
            done = env.terminations[whoami]

            while (
                not done
                and whoami in env.agents
                and env.agent_selection != whoami
                and not (env.terminations[whoami] or env.truncations[whoami])
            ):
                env.step(training_opponent.step(env))
                reward += float(env.rewards[whoami])
                done = env.terminations[whoami]

            next_obs = env.observe(whoami)
            next_state = next_obs['observation']
            next_mask = next_obs['action_mask']

            replay_buffer.add(state, action, reward, next_state, next_mask, done)
            total_steps += 1

            if len(replay_buffer) >= batch_size:
                loss = train_step(
                    online_net,
                    target_net,
                    optimizer,
                    replay_buffer,
                    batch_size,
                    gamma,
                    device,
                )
                if verbose:
                    print(f'Episode: {episode}, Steps: {total_steps}, Loss: {loss}')
                losses.append(loss)

            if total_steps % target_update_frequency == 0:
                target_net.load_state_dict(online_net.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if verbose:
        print(f'Training done, avg loss: {sum(losses) / len(losses) if losses else float("nan")}')
        plot_losses(losses)

    return online_net


class DQNAgent(Cards2PAgent):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net: CardNN

    def __init__(self, whoami: AgentId, trained_net: CardNN) -> None:
        super().__init__(whoami, 'DQN Agent')
        self.net = trained_net

    def step(self, env: Any) -> int | None:
        if env.agent_selection != self.whoami:
            return None

        lead = None if env.lead_play is None else env.lead_play.card
        legal = env.hands[self.whoami].legal_plays(lead)
        while True:
            obs = env.observe(self.whoami)
            state, mask = obs['observation'], obs['action_mask']
            action = choose_action(state, mask, self.net, 0, self.device)
            if DECK[action] in legal:
                return action
