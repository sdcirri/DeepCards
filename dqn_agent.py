from collections import deque
import random

import matplotlib.pyplot as plt

import numpy as np

import torch.optim as optim
import torch.nn as nn
import torch

from agent import Tressette2PAgent
from game import DECK


class CardNN(nn.Module):
    def __init__(self, actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(120, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, actions)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[tuple]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


def choose_action(state, mask, net: CardNN, epsilon: float, actions: int, device) -> int:
    legal_actions = np.flatnonzero(mask)  # indices where mask == 1

    if legal_actions.size == 0:
        raise ValueError('No legal actions available')

    # Exploration: random LEGAL action
    if random.random() < epsilon:
        return int(np.random.choice(legal_actions))

    # Exploitation: argmax Q among legal actions only
    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        q_values = net(state_t).cpu().numpy().flatten()

    q_values[mask == 0] = -np.inf  # hide illegal actions
    return int(q_values.argmax())


def train_step(
        online_net: CardNN,
        target_net: CardNN,
        optimizer,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        gamma: float,
        device
):
    states, actions, rewards, next_states, dones = zip(*replay_buffer.sample(batch_size))

    states = torch.tensor(states, dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.long, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)

    q_values = online_net(states)
    chosen_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = target_net(next_states)
        best_next_q_values = next_q_values.max(dim=1).values

        targets = rewards + gamma * best_next_q_values * (1 - dones)

    loss = nn.functional.smooth_l1_loss(chosen_q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10.0)
    optimizer.step()

    return loss.item()


def train(
        env,
        whoami,
        episodes: int,
        actions: int,
        device: torch.device,
        verbose: bool = True
) -> CardNN:
    online_net = CardNN(actions).to(device)
    target_net = CardNN(actions).to(device)

    target_net.load_state_dict(online_net.state_dict())

    optimizer = optim.Adam(online_net.parameters(), lr=1e-4)
    replay_buffer = ReplayBuffer(capacity=100_000)

    batch_size = 64
    gamma = 0.99

    epsilon, epsilon_min, epsilon_decay = 1.0, 0.05, 0.995
    target_update_frequency = 1000
    total_steps = 0
    losses = []

    for episode in range(episodes):
        env.reset()
        done = False
        ep_reward = 0

        while whoami in env.agents and not (env.terminations[whoami] or env.truncations[whoami]):
            if env.agent_selection != whoami:
                m = env.observe(env.agent_selection)['action_mask']
                legal = np.flatnonzero(m)
                env.step(int(np.random.choice(legal)) if legal.size else None)
                continue
            obs = env.observe(whoami)
            state, mask = obs['observation'], obs['action_mask']

            action = choose_action(state, mask, online_net, epsilon, actions, device)
            env.step(action)
            next_state = env.observe(whoami)['observation']
            reward = env.rewards[whoami]
            done = env.terminations[whoami]
            replay_buffer.add(state, action, reward, next_state, done)
            ep_reward += reward
            total_steps += 1

            if len(replay_buffer) >= batch_size:
                loss = train_step(online_net, target_net, optimizer, replay_buffer, batch_size, gamma, device)
                if verbose:
                    print(f'Episode: {episode}, Steps: {total_steps}, Loss: {loss}')
                losses.append(loss)

            if total_steps % target_update_frequency == 0:
                target_net.load_state_dict(online_net.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if verbose:
        print(f'Training done, avg loss: {sum(losses) / len(losses)}')
        plt.plot(losses)
        plt.show()

    return online_net


class DQNAgent(Tressette2PAgent):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net: CardNN

    def __init__(self, whoami, training_env) -> None:
        super().__init__(whoami)
        self.net = train(training_env, whoami, 200, 40, self.device)

    def step(self, env):
        if env.agent_selection != self.whoami:
            return None

        lead = None if env.lead_play is None else env.lead_play.card
        legal = env.hands[self.whoami].legal_plays(lead)
        while True:
            obs = env.observe(self.whoami)
            state, mask = obs['observation'], obs['action_mask']
            action = choose_action(state, mask, self.net, 0.05, 40, self.device)
            if DECK[action] in legal:
                print(f'{self.whoami} plays {DECK[action]}')
                return action
