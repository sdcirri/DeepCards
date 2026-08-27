import matplotlib.pyplot as plt
from typing import Any, cast
import random
import time

import numpy as np

from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer
from tensordict import TensorDict
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


def choose_action(
    state: np.ndarray,
    mask: np.ndarray,
    net: nn.Module,
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
        q = net(state_t).squeeze(0)
        q[mask == 0] = -float('inf')
        return int(q.argmax().item())


def train_step(
    online_net: nn.Module,
    target_net: nn.Module,
    optimizer: optim.Optimizer,
    replay_buffer: TensorDictPrioritizedReplayBuffer,
    batch_size: int,
    gamma: float,
    device: torch.device,
) -> float:
    batch = replay_buffer.sample(batch_size)

    states = cast(torch.Tensor, batch['state']).to(device)
    actions = cast(torch.Tensor, batch['action']).to(device)
    rewards = cast(torch.Tensor, batch['reward']).to(device)
    next_states = cast(torch.Tensor, batch['next_state']).to(device)
    next_masks = cast(torch.Tensor, batch['next_mask']).to(device)
    dones = cast(torch.Tensor, batch['done']).to(device)
    weights = cast(torch.Tensor, batch['priority_weight']).to(device)

    q_values = online_net(states)
    chosen_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        online_next = online_net(next_states)
        online_next[next_masks == 0] = -float('inf')
        next_actions = online_next.argmax(dim=1)

        target_next = target_net(next_states)
        best_next_q_values = target_next.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        best_next_q_values = torch.where(
            next_masks.sum(dim=1) > 0,
            best_next_q_values,
            torch.zeros_like(best_next_q_values),
        )
        targets = rewards + gamma * best_next_q_values * ~dones

    td_error = targets - chosen_q_values
    element_loss = nn.functional.smooth_l1_loss(chosen_q_values, targets, reduction='none')
    loss = (weights * element_loss).mean()
    batch['td_error'] = td_error.detach().abs().cpu()
    replay_buffer.update_tensordict_priority(batch)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10.0)
    optimizer.step()

    return loss.item()


def plot_losses(losses: list[float]) -> None:
    if not losses:
        return

    window = min(100, max(1, len(losses) // 10))
    kernel = np.ones(window, dtype=np.float64) / np.float64(window)
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
        actions: int,
        obs_dim: int,
        device: torch.device,
        training_opponents: list[Cards2PAgent],
        opponents_distribution: list[float],
        total_episodes: int,
        verbose: bool = True,
) -> CardNN:
    if len(training_opponents) != len(opponents_distribution):
        raise RuntimeError('Length mismatch between training_opponents and opponents_distribution')

    # MLPs this small are only penalized by multithreading overhead
    torch.set_num_threads(1)

    online_net = CardNN(obs_dim, actions).to(device)
    target_net = CardNN(obs_dim, actions).to(device)
    target_net.load_state_dict(online_net.state_dict())

    optimizer = optim.AdamW(online_net.parameters(), lr=1e-4)
    replay_buffer = TensorDictPrioritizedReplayBuffer(
        storage=LazyTensorStorage(100_000),
        alpha=0.6, beta=0.4, eps=1e-6,
        priority_key='td_error'
    )

    batch_size = 128
    gamma = 0.99
    target_update_frequency = 1000
    total_steps = 0
    losses: list[float] = []
    epsilon_start, epsilon_min, epsilon_decay_steps = 1.0, 0.05, 200_000
    epsilon = epsilon_start
    train_start = time.time()

    for episode in range(total_episodes):
        env.reset()
        opponent = random.choices(training_opponents, weights=opponents_distribution, k=1)[0]

        while whoami in env.agents and not (env.terminations[whoami] or env.truncations[whoami]):
            if env.agent_selection != whoami:
                env.step(opponent.step(env))
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
                env.step(opponent.step(env))
                reward += float(env.rewards[whoami])
                done = env.terminations[whoami]

            next_obs = env.observe(whoami)
            next_state = next_obs['observation']
            next_mask = next_obs['action_mask']

            replay_buffer.add(TensorDict(
                {
                    'state': torch.tensor(state, dtype=torch.float32),
                    'action': torch.tensor(action, dtype=torch.long),
                    'reward': torch.tensor(reward, dtype=torch.float32),
                    'next_state': torch.tensor(next_state, dtype=torch.float32),
                    'next_mask': torch.tensor(next_mask, dtype=torch.bool),
                    'done': torch.tensor(done, dtype=torch.bool),
                    'td_error': torch.tensor(1.0)
                },
                batch_size=[]
            ))
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
                if verbose and total_steps % 1000 == 0:
                    print(f'Episode: {episode+1}, Steps: {total_steps}, Loss: {loss}, Time: {time.time() - train_start:.4f}s')
                    losses.append(loss)

            if total_steps % target_update_frequency == 0:
                target_net.load_state_dict(online_net.state_dict())

        epsilon = max(epsilon_min, epsilon_start - ((epsilon_start - epsilon_min) * total_steps / epsilon_decay_steps))

    if verbose:
        print(f'Training done, avg loss: {sum(losses) / len(losses) if losses else float("nan")}')
        plot_losses(losses)

    return online_net


class DQNAgent(Cards2PAgent):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net: nn.Module

    def __init__(self, whoami: AgentId, trained_net: nn.Module) -> None:
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
