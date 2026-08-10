from concurrent.futures import ProcessPoolExecutor
from gymnasium.utils import seeding
from abc import abstractmethod, ABC
import os

from environments.tressette_env import Tressette2PEnv, AgentId, Action, env as env_factory


class Tressette2PAgent(ABC):
    whoami: AgentId
    name: str

    def __init__(self, whoami: AgentId, name: str) -> None:
        self.whoami = whoami
        self.name = name
        self.np_random, self.np_random_seed = seeding.np_random(None)

    @abstractmethod
    def step(self, env: Tressette2PEnv) -> Action | None:
        pass


def challenge_step(env: Tressette2PEnv, agent1: Tressette2PAgent, agent2: Tressette2PAgent) -> None:
    env.reset()
    while not env.terminations['p1']:
        if env.agent_selection == 'p1':
            env.step(agent1.step(env))
        else:
            env.step(agent2.step(env))


def _one_episode(a1: Tressette2PAgent, a2: Tressette2PAgent) -> tuple[int, int]:
    env = env_factory(render_mode=None)
    challenge_step(env, a1, a2)
    raw = env.unwrapped
    # Points are stored in thirds, score is truncated at the end of the game
    return raw.scores['p1'] // 3, raw.scores['p2'] // 3


def challenge(agent1: Tressette2PAgent, agent2: Tressette2PAgent, episodes: int) -> tuple[int, int]:
    with (ProcessPoolExecutor(max_workers=os.cpu_count()) as executor):
        results = executor.map(_one_episode, [agent1] * episodes, [agent2] * episodes)

    points_1, points_2 = 0, 0
    for result in results:
        points_1 += result[0]
        points_2 += result[1]

    return points_1, points_2
