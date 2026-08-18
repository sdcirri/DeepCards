"""
Scopa-specific Double DQN implementation
"""

from collections import deque
from typing import Any
import numpy as np
import random

from torch import optim
import torch.nn as nn
import torch

from games.deck import Card, CARD_INDEX, DECK
from games.scopa import ScopaHand

from ..agent import AgentId, Cards2PAgent
from ..dqn import CardNN, plot_losses


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: deque[tuple] = deque(maxlen=capacity)

    def add(self, state, play_idx, take_idx, reward, next_state, done) -> None:
        self.buffer.append((state, play_idx, take_idx, reward, next_state, done))

    def sample(self, batch_size: int) -> list[tuple]:
        return random.sample(self.buffer, batch_size)

    def clear(self) -> None:
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)


def _sorted_legal(hand: ScopaHand, table: list[Card]):
    legal = hand.scopa_legal_plays(table)
    return sorted(
        legal,
        key=lambda pt: (
            CARD_INDEX[pt[0]],
            tuple(sorted(CARD_INDEX[c] for c in pt[1])),
        ),
    )


def choose_action(
        state: np.ndarray,
        net: CardNN,
        epsilon: float,
        device: torch.device,
        hand: ScopaHand,
        table: list[Card],
) -> tuple[dict[str, int | np.ndarray], int, int]:
    if not (legal := _sorted_legal(hand, table)):
        raise RuntimeError('No legal move possible')

    play_mask = np.zeros(40, dtype=np.int8)
    by_play: dict[int, list[list[Card]]] = {}
    for played, taken in legal:
        i = CARD_INDEX[played]
        play_mask[i] = 1
        by_play.setdefault(i, []).append(taken)

    state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        q = net(state_t).cpu().numpy().flatten()
    q_play, q_take = q[:40], q[40:]

    if random.random() < epsilon:
        play_idx = int(np.random.choice(np.flatnonzero(play_mask)))
    else:
        qp = q_play.copy()
        qp[play_mask == 0] = -np.inf
        play_idx = int(qp.argmax())

    take_mask = np.zeros(40, dtype=np.int8)
    take_mask[:len(by_play[play_idx])] = 1

    if random.random() < epsilon:
        take_idx = random.randint(0, len(by_play[play_idx]) - 1)
    else:
        qt = q_take.copy()
        qt[take_mask == 0] = -np.inf
        take_idx = int(qt.argmax())

    taken = np.zeros(40, dtype=np.int8)
    for card in by_play[play_idx][take_idx]:
        taken[CARD_INDEX[card]] = 1

    return {'played': play_idx, 'taken': taken}, play_idx, take_idx


def train_step(
    online_net: CardNN,
    target_net: CardNN,
    optimizer: optim.Optimizer,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    device: torch.device,
) -> float:
    states, play_idxs, take_idxs, rewards, next_states, dones = zip(
        *replay_buffer.sample(batch_size)
    )

    states = torch.as_tensor(np.stack(states), dtype=torch.float32, device=device)
    play_idxs = torch.as_tensor(play_idxs, dtype=torch.long, device=device)
    take_idxs = torch.as_tensor(take_idxs, dtype=torch.long, device=device)
    rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    next_states = torch.as_tensor(np.stack(next_states), dtype=torch.float32, device=device)
    dones = torch.as_tensor(dones, dtype=torch.float32, device=device)

    q_values = online_net(states)
    chosen_q_values = (
        q_values.gather(1, play_idxs.unsqueeze(1)).squeeze(1)
        + q_values.gather(1, (take_idxs + 40).unsqueeze(1)).squeeze(1)
    )

    with torch.no_grad():
        online_next = online_net(next_states)
        target_next = target_net(next_states)

        best_next_q_values = torch.zeros(next_states.size(0), device=device)
        next_np = next_states.detach().cpu().numpy()

        for b in range(next_states.size(0)):
            obs = next_np[b]
            hand_cards = [DECK[i] for i in range(40) if obs[i] == 1]
            table_cards = [DECK[i] for i in range(40) if obs[80 + i] == 1]
            legal = _sorted_legal(ScopaHand(hand_cards), table_cards)
            if not legal:
                continue

            play_mask = np.zeros(40, dtype=np.int8)
            by_play: dict[int, list[list[Card]]] = {}
            for played, taken in legal:
                i = CARD_INDEX[played]
                play_mask[i] = 1
                by_play.setdefault(i, []).append(taken)

            q = online_next[b].detach().cpu().numpy()
            qp, qt = q[:40].copy(), q[40:].copy()

            qp[play_mask == 0] = -np.inf
            play_idx = int(qp.argmax())

            take_mask = np.zeros(40, dtype=np.int8)
            take_mask[: len(by_play[play_idx])] = 1
            qt[take_mask == 0] = -np.inf
            take_idx = int(qt.argmax())

            tq = target_next[b]
            best_next_q_values[b] = tq[play_idx] + tq[40 + take_idx]
        targets = rewards + gamma * best_next_q_values * (1 - dones)

    loss = nn.functional.smooth_l1_loss(chosen_q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10.0)
    optimizer.step()

    return loss.item()


def train(
        env: Any,
        whoami: AgentId,
        actions: int,
        obs_dim: int,
        device: torch.device,
        training_opponents: list[Cards2PAgent],
        episodes_per_opponent: list[int],
        verbose: bool = True,
) -> CardNN:
    if len(training_opponents) != len(episodes_per_opponent):
        raise RuntimeError('Length mismatch between training_opponents and episodes_per_opponent')

    online_net = CardNN(obs_dim, actions).to(device)
    target_net = CardNN(obs_dim, actions).to(device)
    target_net.load_state_dict(online_net.state_dict())

    optimizer = optim.AdamW(online_net.parameters(), lr=1e-4)
    replay_buffer = ReplayBuffer(capacity=100_000)

    batch_size = 64
    gamma = 0.99
    target_update_frequency = 1000
    total_steps = 0
    losses: list[float] = []

    for i, opponent in enumerate(training_opponents):
        epsilon, epsilon_min, epsilon_decay = 1.0, 0.05, 0.995
        replay_buffer.clear()

        for episode in range(episodes_per_opponent[i]):
            env.reset()

            while whoami in env.agents and not (env.terminations[whoami] or env.truncations[whoami]):
                if env.agent_selection != whoami:
                    env.step(opponent.step(env))
                    continue

                state = env.observe(whoami)
                action, play_idx, take_idx = choose_action(
                    state,
                    online_net,
                    epsilon,
                    device,
                    env.hands[whoami],
                    env.table
                )
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

                next_state = env.observe(whoami)

                replay_buffer.add(state, play_idx, take_idx, reward, next_state, done)
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
