from typing import Callable
from pathlib import Path
from sys import argv

import torch

from agents.briscola.points_aware_greedy_agent import PointsAwareGreedyAgent
from agents.briscola.dqn_agent import DQNAgent, train, CardNN
from agents.briscola.random_agent import RandomAgent
from agents.briscola.greedy_agent import GreedyAgent

from environments.briscola_env import env as env_factory, AgentId
from agents.briscola.agent import challenge
from agents.agent import Cards2PAgent


AgentFactory = Callable[[AgentId], Cards2PAgent]
MODEL_PATH = Path('briscola_dqn_net.pt')

print('Initializing agents ...')


if MODEL_PATH.exists():
    print(f'Loading model from {MODEL_PATH}')
    dqn_net = CardNN(40).to(DQNAgent.device)
    dqn_net.load_state_dict(torch.load(MODEL_PATH, map_location=DQNAgent.device, weights_only=True))
    dqn_net.eval()
else:
    print('No saved model found, training ...')
    dqn_net = train(
        env_factory(render_mode=None),
        'p1',
        40,
        DQNAgent.device,
        [RandomAgent('p2'), GreedyAgent('p2'), PointsAwareGreedyAgent('p2')],
        [2000, 2000, 15000],
        False
    )
    torch.save(dqn_net.state_dict(), MODEL_PATH)
    print(f'Saved model to {MODEL_PATH}')


agent_factories: list[tuple[str, AgentFactory]] = [
    ('Random', lambda whoami: RandomAgent(whoami)),
    ('Greedy', lambda whoami: GreedyAgent(whoami)),
    ('Points-Aware Greedy', lambda whoami: PointsAwareGreedyAgent(whoami)),
    ('DQN', lambda whoami: DQNAgent(whoami, dqn_net))
]

EPISODES = 10_000
if len(argv) > 1:
    EPISODES = int(argv[1])

render_mode = None
if len(argv) > 2:
    render_mode = argv[2]

for name1, make1 in agent_factories:
    for name2, make2 in agent_factories:
        agent1, agent2 = make1('p1'), make2('p2')
        print(f'Starting match: {agent1.name} [p1] vs {agent2.name} [p2], {EPISODES} episodes')
        score1, score2 = challenge(agent1, agent2, EPISODES, render_mode)

        print(f'Agent 1: {agent1.name}: avg score: {score1 / EPISODES:.4f}')
        print(f'Agent 2: {agent2.name}: avg score: {score2 / EPISODES:.4f}')
