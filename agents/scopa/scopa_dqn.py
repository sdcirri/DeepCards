import numpy as np
import random

from torch import optim, nn
import torch

from environments.scopa_env import Scopa2PEnv, AgentId
from games.deck import CARD_INDEX, Card, DECK

from ..dqn import (
    CardNN,
    train_step as play_train_step,
    ReplayBuffer,
    plot_losses,
    choose_action as choose_play
)

from ..agent import Cards2PAgent


OBS_DIM = Scopa2PEnv.OBSERVATION_PLANES * len(DECK) + Scopa2PEnv.EXTRA_OBSERVATIONS


class PlayCardNN(CardNN):
    def __init__(self, play_actions: int) -> None:
        super().__init__(OBS_DIM, play_actions)


class TakeCardNN(CardNN):
    def __init__(self, play_actions: int, take_actions: int) -> None:
        super().__init__(OBS_DIM + play_actions, take_actions)


def choose_take(
        take_state: np.ndarray,
        take_mask: np.ndarray,
        net: TakeCardNN,
        epsilon: float,
        device: torch.device,
        take_options: list[list[Card]]
) -> int:
    legal_actions = np.flatnonzero(take_mask)

    if legal_actions.size == 0:
        raise ValueError('No legal actions available')

    if random.random() < epsilon:
        return random.randint(0, len(take_options)-1)

    state_t = torch.tensor(take_state, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        q = net(state_t).squeeze(0)
        q[take_mask == 0] = -float('inf')

    return max(
        range(len(take_options)),
        key=lambda i: sum(q[CARD_INDEX[c]] for c in take_options[i])
    )


def take_train_step(
        online_net: CardNN,
        optimizer: optim.Optimizer,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        device: torch.device,
) -> float:
    states, card_masks, rewards, *_ = zip(*replay_buffer.sample(batch_size))

    states = torch.as_tensor(np.stack(states), dtype=torch.float32, device=device)
    card_masks = torch.as_tensor(np.stack(card_masks), dtype=torch.float32, device=device)
    rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)

    q = online_net(states)
    chosen_q = (q * card_masks).sum(dim=1)  # Σ q[card] for cards taken

    loss = nn.functional.smooth_l1_loss(chosen_q, rewards)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10.0)
    optimizer.step()
    return loss.item()


def train(
        env: Scopa2PEnv,
        whoami: AgentId,
        play_actions: int,
        take_actions: int,
        device: torch.device,
        training_opponents: list[Cards2PAgent],
        episodes_per_opponent: list[int],
        verbose: bool = True,
) -> tuple[PlayCardNN, TakeCardNN]:
    if len(training_opponents) != len(episodes_per_opponent):
        raise RuntimeError('Length mismatch between training_opponents and episodes_per_opponent')

    online_play_net = PlayCardNN(len(DECK)).to(device)
    target_play_net = PlayCardNN(len(DECK)).to(device)
    take_net = TakeCardNN(play_actions, take_actions).to(device)
    target_play_net.load_state_dict(online_play_net.state_dict())

    play_optimizer = optim.AdamW(online_play_net.parameters(), lr=1e-4)
    take_optimizer = optim.AdamW(take_net.parameters(), lr=1e-4)
    play_buffer, take_buffer = ReplayBuffer(capacity=100_000), ReplayBuffer(capacity=100_000)

    batch_size = 64
    gamma = 0.99
    target_update_frequency = 1000
    total_steps = 0
    play_losses, take_losses = [], []

    for i, opponent in enumerate(training_opponents):
        epsilon, epsilon_min, epsilon_decay = 1.0, 0.05, 0.995

        for episode in range(episodes_per_opponent[i]):
            env.reset()

            while whoami in env.agents and not (env.terminations[whoami] or env.truncations[whoami]):
                if env.agent_selection != whoami:
                    env.step(opponent.step(env))
                    continue

                obs = env.observe(whoami)
                state, play_mask, take_mask = obs['observation'], obs['play_mask'], obs['take_mask']

                play = choose_play(state, play_mask, online_play_net, epsilon, device)
                play_t = np.zeros(play_actions, dtype=np.int8)
                play_t[play] = 1

                legal = env.hands[whoami].scopa_legal_plays(env.table)
                take_opts = [
                    opt for opt in legal
                    if CARD_INDEX[opt[0]] == play
                ]
                take_state = np.concatenate((state, play_t))

                if len(take_opts) > 1:
                    take = choose_take(
                        take_state,
                        take_mask,
                        take_net,
                        epsilon,
                        device,
                        [opt[1] for opt in take_opts]
                    )
                else:
                    take = 0

                env.step((play, take))
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
                next_state, next_play_mask, next_take_mask = next_obs['observation'], next_obs['play_mask'], next_obs['take_mask']
                play_buffer.add(state, play, reward, next_state, next_play_mask, done)

                if len(take_opts) > 1:
                    take_next_state = np.concatenate((next_state, np.zeros(play_actions, dtype=np.int8)))
                    capture = take_opts[take][1]
                    mask = np.zeros(len(DECK), dtype=np.float32)
                    for c in capture:
                        mask[CARD_INDEX[c]] = 1.0
                    take_buffer.add(take_state, mask, reward, take_next_state, next_take_mask, True)
                total_steps += 1

                play_loss, take_loss = 0, 0
                if len(play_buffer) >= batch_size:
                    play_loss = play_train_step(
                        online_play_net,
                        target_play_net,
                        play_optimizer,
                        play_buffer,
                        batch_size,
                        gamma,
                        device,
                    )
                if len(take_buffer) >= batch_size:
                    take_loss = take_train_step(
                        take_net,
                        take_optimizer,
                        take_buffer,
                        batch_size,
                        device,
                    )
                if verbose and total_steps % 1000 == 0:
                    print(f'Episode: {episode}, Steps: {total_steps}, Losses: {play_loss=}, {take_loss=}')
                    play_losses.append(play_loss)
                    take_losses.append(take_loss)

                if total_steps % target_update_frequency == 0:
                    target_play_net.load_state_dict(online_play_net.state_dict())

            epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if verbose:
        print(f'Training done\n\tPlay avg loss: {sum(play_losses) / len(play_losses) if play_losses else float("nan")}')
        print(f'\tTake avg loss: {sum(take_losses) / len(take_losses) if take_losses else float("nan")}')
        plot_losses(play_losses)
        plot_losses(take_losses)

    return online_play_net, take_net
