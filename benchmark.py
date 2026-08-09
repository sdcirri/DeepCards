from typing import Callable
from pathlib import Path

import torch

from environment import env as env_factory, AgentId

from points_aware_greedy_agent import PointsAwareGreedyAgent
from dqn_agent import DQNAgent, train, CardNN
from random_agent import RandomAgent
from greedy_agent import GreedyAgent

from agent import challenge, Tressette2PAgent


AgentFactory = Callable[[AgentId], Tressette2PAgent]
MODEL_PATH = Path('dqn_net.pt')

print('Initializing agents ...')


if MODEL_PATH.exists():
    print(f'Loading model from {MODEL_PATH}')
    dqn_net = CardNN(40).to(DQNAgent.device)
    dqn_net.load_state_dict(torch.load(MODEL_PATH, map_location=DQNAgent.device, weights_only=True))
    dqn_net.eval()
else:
    print('No saved model found, training ...')
    dqn_net = train(env_factory(render_mode=None), 'p1', 5000, 40, DQNAgent.device, False)
    torch.save(dqn_net.state_dict(), MODEL_PATH)
    print(f'Saved model to {MODEL_PATH}')


agent_factories: list[tuple[str, AgentFactory]] = [
    ('Random', lambda whoami: RandomAgent(whoami)),
    ('Greedy', lambda whoami: GreedyAgent(whoami)),
    ('Points-Aware Greedy', lambda whoami: PointsAwareGreedyAgent(whoami)),
    ('DQN', lambda whoami: DQNAgent(whoami, dqn_net))
]

EPISODES = 10_000

for name1, make1 in agent_factories:
    for name2, make2 in agent_factories:
        agent1, agent2 = make1('p1'), make2('p2')
        print(f'Starting match: {agent1.name} [p1] vs {agent2.name} [p2], {EPISODES} episodes')
        score1, score2 = challenge(agent1, agent2, EPISODES)

        print(f'Agent 1: {agent1.name}: avg score: {score1 / EPISODES:.4f}')
        print(f'Agent 2: {agent2.name}: avg score: {score2 / EPISODES:.4f}')
