from typing import Any, Callable

from torch import optim
import torch

from environments.scopa_env import AgentId
from games.deck import DECK, Card

from ..dqn import CardNN, ReplayBuffer, train_step, choose_action, plot_losses
from ..agent import Cards2PAgent


def train(
        env: Any,
        whoami: AgentId,
        actions: int,
        obs_dim: int,
        device: torch.device,
        training_opponents: list[Cards2PAgent],
        episodes_per_opponent: list[int],
        take_strategy: Callable[[list[list[Card]]], int],
        verbose: bool = True
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

        for episode in range(episodes_per_opponent[i]):
            env.reset()

            while whoami in env.agents and not (env.terminations[whoami] or env.truncations[whoami]):
                if env.agent_selection != whoami:
                    env.step(opponent.step(env))
                    continue

                obs = env.observe(whoami)
                state, mask = obs['observation'], obs['play_mask']

                play = choose_action(state, mask, online_net, epsilon, device)
                take_opts = [
                    opt[1]
                    for opt in env.hands[whoami].scopa_legal_plays(env.table)
                    if opt[0] == DECK[play]
                ]
                take = take_strategy(take_opts)

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
                next_state = next_obs['observation']
                next_mask = next_obs['play_mask']

                replay_buffer.add(state, play, reward, next_state, next_mask, done)
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
                        print(f'Episode: {episode}, Steps: {total_steps}, Loss: {loss}')
                        losses.append(loss)

                if total_steps % target_update_frequency == 0:
                    target_net.load_state_dict(online_net.state_dict())

            epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if verbose:
        print(f'Training done, avg loss: {sum(losses) / len(losses) if losses else float("nan")}')
        plot_losses(losses)

    return online_net
