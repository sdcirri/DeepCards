from concurrent.futures import ProcessPoolExecutor
from typing import Any, TypeAlias, Literal
from collections.abc import Callable
from gymnasium.utils import seeding
from abc import abstractmethod, ABC
import os


AgentId: TypeAlias = Literal['p1', 'p2']


class Cards2PAgent(ABC):
    whoami: AgentId
    name: str

    def __init__(self, whoami: AgentId, name: str) -> None:
        self.whoami = whoami
        self.name = name
        self.np_random, self.np_random_seed = seeding.np_random(None)

    @abstractmethod
    def step(self, env: Any) -> Any:
        pass


def challenge_step(env: Any, agent1: Cards2PAgent, agent2: Cards2PAgent) -> None:
    env.reset()
    while not env.terminations['p1']:
        if env.agent_selection == 'p1':
            env.step(agent1.step(env))
        else:
            env.step(agent2.step(env))


def run_challenge(
        agent1: Cards2PAgent,
        agent2: Cards2PAgent,
        episodes: int,
        one_episode: Callable[[Cards2PAgent, Cards2PAgent], tuple[int, int]],
) -> tuple[int, int]:
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = executor.map(one_episode, [agent1] * episodes, [agent2] * episodes)

    points_1, points_2 = 0, 0
    for result in results:
        points_1 += result[0]
        points_2 += result[1]

    return points_1, points_2
