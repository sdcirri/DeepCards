from typing import Callable

from random_agent import RandomAgent
from greedy_agent import GreedyAgent
from dqn_agent import DQNAgent, train
from agent import challenge, Tressette2PAgent

from environment import env as env_factory, AgentId

AgentFactory = Callable[[AgentId], Tressette2PAgent]

print('Initializing agents ...')

dqn_net = train(
    env_factory(render_mode=None),
    'p1',
    5000,
    40,
    DQNAgent.device,
    True
)

agent_factories: list[tuple[str, AgentFactory]] = [
    ('Random', lambda whoami: RandomAgent(whoami)),
    ('Greedy', lambda whoami: GreedyAgent(whoami)),
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
