from typing import Any
import random

from numpy.typing import NDArray
import numpy as np

from torch import optim, nn
import torch

from environments.scopa_env import AgentId
from agents.agent import Cards2PAgent

from .play_candidates import build_play_candidates, TakeStrategy, PlayCandidate, ACTION_FEATURE_DIM
from .scopa_nn import ScopaNN, ScopaReplayBuffer, ScopaTransition

from ...dqn import plot_losses


def choose_candidate(
         state: np.ndarray,
         candidates: list[PlayCandidate],
         online_net: ScopaNN,
         epsilon: float,
         device: torch.device,
 ) -> PlayCandidate:
     if not candidates:
         raise ValueError('No legal play candidates')
 
     if random.random() < epsilon:
         return random.choice(candidates)
 
     observations = torch.as_tensor(
         state,
         dtype=torch.float32,
         device=device,
     ).unsqueeze(0).expand(len(candidates), -1)

     features = torch.from_numpy(np.stack([
         candidate.features for candidate in candidates
     ])).to(device)
 
     with torch.no_grad():
         q_values = online_net(observations, features)
         selected_index = int(q_values.argmax().item())
 
     return candidates[selected_index]


def train_step(
        online_net: ScopaNN,
        target_net: ScopaNN,
        optimizer: optim.Optimizer,
        replay_buffer: ScopaReplayBuffer,
        batch_size: int,
        gamma: float,
        device: torch.device,
) -> float:
    transitions = replay_buffer.sample(batch_size)

    states = torch.as_tensor(
        np.stack([transition.state for transition in transitions]),
        dtype=torch.float32,
        device=device,
    )
    action_features = torch.as_tensor(
        np.stack([transition.action_features for transition in transitions]),
        dtype=torch.float32,
        device=device,
    )
    rewards = torch.as_tensor(
        [transition.reward for transition in transitions],
        dtype=torch.float32,
        device=device,
    )
    next_states = torch.as_tensor(
        np.stack([transition.next_state for transition in transitions]),
        dtype=torch.float32,
        device=device,
    )
    next_candidate_features = torch.as_tensor(
        np.stack([
            transition.next_candidate_features
            for transition in transitions
        ]),
        dtype=torch.float32,
        device=device,
    )
    next_candidate_mask = torch.as_tensor(
        np.stack([
            transition.next_candidate_mask
            for transition in transitions
        ]),
        dtype=torch.bool,
        device=device,
    )
    dones = torch.as_tensor(
        [transition.done for transition in transitions],
        dtype=torch.float32,
        device=device,
    )

    chosen_q = online_net(states, action_features)

    with torch.no_grad():
        batch_size_, candidate_count, feature_dim = next_candidate_features.shape

        expanded_next_states = (
            next_states
                .unsqueeze(1)
                .expand(-1, candidate_count, -1)
                .reshape(batch_size_ * candidate_count, -1)
        )
        flattened_features = next_candidate_features.reshape(
            batch_size_ * candidate_count,
            feature_dim,
        )

        online_next_q = online_net(
            expanded_next_states,
            flattened_features,
        ).reshape(batch_size_, candidate_count)

        online_next_q = online_next_q.masked_fill(
            ~next_candidate_mask,
            -torch.inf,
        )
        next_actions = online_next_q.argmax(dim=1)

        target_next_q = target_net(
            expanded_next_states,
            flattened_features,
        ).reshape(batch_size_, candidate_count)

        best_next_q = target_next_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)

        best_next_q = torch.where(
            next_candidate_mask.any(dim=1),
            best_next_q,
            torch.zeros_like(best_next_q),
        )

        targets = rewards + gamma * best_next_q * (1 - dones)

    loss = nn.functional.smooth_l1_loss(chosen_q, targets)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10.0)
    optimizer.step()

    return loss.item()


def pad_candidates(candidates: list[PlayCandidate]) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    features = np.zeros(
        (3, ACTION_FEATURE_DIM),
        dtype=np.float32,
    )
    mask = np.zeros(3, dtype=np.bool_)

    for index, candidate in enumerate(candidates):
        features[index] = candidate.features
        mask[index] = True

    return features, mask


def train(
        env: Any,
        whoami: AgentId,
        obs_dim: int,
        device: torch.device,
        training_opponents: list[Cards2PAgent],
        episodes_per_opponent: list[int],
        take_strategy: TakeStrategy,
        verbose: bool = True
) -> ScopaNN:
    if len(training_opponents) != len(episodes_per_opponent):
        raise RuntimeError('Length mismatch between training_opponents and episodes_per_opponent')

    online_net = ScopaNN(obs_dim).to(device)
    target_net = ScopaNN(obs_dim).to(device)
    target_net.load_state_dict(online_net.state_dict())

    optimizer = optim.AdamW(online_net.parameters(), lr=1e-4)
    replay_buffer = ScopaReplayBuffer(capacity=100_000)

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

                candidates = build_play_candidates(env, whoami, take_strategy)
                chosen = choose_candidate(state, candidates, online_net, epsilon, device)

                env.step((chosen.play, chosen.take))
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
                next_candidates = [] if done else build_play_candidates(env, whoami, take_strategy)
                next_features, next_candidate_mask = pad_candidates(next_candidates)

                replay_buffer.add(ScopaTransition(
                    state=state, action_features=chosen.features, reward=reward,
                    next_state=next_state, next_candidate_features=next_features,
                    next_candidate_mask=next_candidate_mask, done=done
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
                        print(f'Episode: {episode}, Steps: {total_steps}, Loss: {loss}')
                        losses.append(loss)

                if total_steps % target_update_frequency == 0:
                    target_net.load_state_dict(online_net.state_dict())

            epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if verbose:
        print(f'Training done, avg loss: {sum(losses) / len(losses) if losses else float("nan")}')
        plot_losses(losses)

    return online_net
