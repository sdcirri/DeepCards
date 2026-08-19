from typing import Callable
from pathlib import Path
from sys import argv

import torch

from agents.scopa.value_aware_greedy_agent import ValueAwareGreedyAgent
from agents.scopa.greedy_take_dqn_agent import GreedyTakeDQNAgent
from agents.scopa.rl_take_dqn_agent import RLTakeDQNAgent
from agents.scopa.dqn_agent import DQNAgent, PlayCardNN
from agents.scopa.scopa_dqn import TakeCardNN, train
from agents.scopa.greedy_agent import GreedyAgent
from agents.scopa.random_agent import RandomAgent

from environments.scopa_env import env as env_factory, AgentId
from agents.scopa.agent import challenge
from agents.agent import Cards2PAgent


AgentFactory = Callable[[AgentId], Cards2PAgent]


PLAY_MODEL_PATH, TAKE_MODEL_PATH = Path('scopa_play_dqn_net.pt'), Path('scopa_take_dqn_net.pt')

print('Initializing agents ...')

if PLAY_MODEL_PATH.exists() and TAKE_MODEL_PATH.exists():
    print(f'Loading models from {PLAY_MODEL_PATH} and {TAKE_MODEL_PATH}')
    play_dqn_net, take_dqn_net = PlayCardNN(40).to(DQNAgent.device), TakeCardNN(40, 40).to(DQNAgent.device)
    play_dqn_net.load_state_dict(torch.load(PLAY_MODEL_PATH, map_location=DQNAgent.device, weights_only=True))
    take_dqn_net.load_state_dict(torch.load(TAKE_MODEL_PATH, map_location=DQNAgent.device, weights_only=True))
    play_dqn_net.eval()
else:
    print('No saved models found, training ...')
    play_dqn_net, take_dqn_net = train(
        env_factory(render_mode=None),
        'p1',
        40,
        40,
        DQNAgent.device,
        [RandomAgent('p2'), GreedyAgent('p2'), ValueAwareGreedyAgent('p2')],
        [1000, 5000, 25000],
        True
    )
    torch.save(play_dqn_net.state_dict(), PLAY_MODEL_PATH)
    torch.save(take_dqn_net.state_dict(), TAKE_MODEL_PATH)
    print(f'Saved models to {PLAY_MODEL_PATH} and {TAKE_MODEL_PATH}')


agent_factories: list[tuple[str, AgentFactory]] = [
    ('Random', lambda whoami: RandomAgent(whoami)),
    ('Greedy', lambda whoami: GreedyAgent(whoami)),
    ('Value-Aware Greedy', lambda whoami: ValueAwareGreedyAgent(whoami)),
    ('DQN + Greedy Take', lambda whoami: GreedyTakeDQNAgent(whoami, play_dqn_net)),
    ('DQN + RL Take', lambda whoami: RLTakeDQNAgent(whoami, play_dqn_net, take_dqn_net)),
]

EPISODES = 10_000
if len(argv) > 1:
    EPISODES = int(argv[1])

render_mode = None
if len(argv) > 2:
    render_mode = argv[2]

for i, (name1, make1) in enumerate(agent_factories):
    for name2, make2 in agent_factories[i:]:
        agent1, agent2 = make1('p1'), make2('p2')
        print(f'Starting match: {agent1.name} [p1] vs {agent2.name} [p2], {EPISODES} episodes')
        score1, score2 = challenge(agent1, agent2, EPISODES, render_mode)

        print(f'Agent 1: {agent1.name}: avg score: {score1 / EPISODES:.4f}')
        print(f'Agent 2: {agent2.name}: avg score: {score2 / EPISODES:.4f}')
