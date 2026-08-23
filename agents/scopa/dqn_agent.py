from abc import abstractmethod, ABC

from environments.scopa_env import Scopa2PEnv, Action
from games.deck import CARD_INDEX, Card, DECK

from ..dqn import choose_action, CardNN as BaseCardNN, DQNAgent as BaseDQNAgent

from ..agent import AgentId


OBS_DIM = Scopa2PEnv.OBSERVATION_PLANES * len(DECK) + Scopa2PEnv.EXTRA_OBSERVATIONS


class CardNN(BaseCardNN):
    def __init__(self, actions: int) -> None:
        super().__init__(OBS_DIM, actions)


class DQNAgent(BaseDQNAgent, ABC):
    """
    Base DQN agent for Scopa
    """

    def __init__(self, name: str, whoami: AgentId, net: CardNN) -> None:
        super().__init__(whoami, net)
        self.name = name

    @staticmethod
    @abstractmethod
    def take_strategy(play: int, env: Scopa2PEnv, legal: list[tuple[Card, list[Card]]]) -> int:
        ...

    def step(self, env: Scopa2PEnv) -> Action | None:
        if env.agent_selection != self.whoami:
            return None

        obs = env.observe(self.whoami)

        play = choose_action(
                obs['observation'],
                obs['play_mask'],
                self.net,
                0,
                self.device
        )

        legal = env.hands[self.whoami].scopa_legal_plays(env.table)
        take_opts = [
            opt for opt in legal
            if CARD_INDEX[opt[0]] == play
        ]
        if len(take_opts) <= 1:
            return play, 0

        return play, self.take_strategy(play, env, legal)
