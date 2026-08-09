from gymnasium.utils import seeding
from abc import abstractmethod, ABC

from environment import Tressette2PEnv, AgentId, Action, env as env_factory


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


def challenge(agent1: Tressette2PAgent, agent2: Tressette2PAgent, episodes: int) -> tuple[int, int]:
    points_1, points_2 = 0, 0

    for _ in range(episodes):
        env = env_factory(render_mode=None)
        challenge_step(env, agent1, agent2)

        # Points are stored in thirds, score is truncated at the end of the game
        points_1 += env.scores['p1'] // 3
        points_2 += env.scores['p2'] // 3

    return points_1, points_2
