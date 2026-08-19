from abc import abstractmethod, ABC

from environments.scopa_env import Scopa2PEnv, Action
from games.deck import CARD_INDEX, Card

from .scopa_dqn import PlayCardNN, choose_play
from ..dqn import DQNAgent as BaseDQNAgent

from ..agent import AgentId


class DQNAgent(BaseDQNAgent, ABC):
    """
    Base DQN agent for Scopa
    """

    def __init__(self, name: str, whoami: AgentId, net: PlayCardNN) -> None:
        super().__init__(whoami, net)
        self.name = name

    @abstractmethod
    def take_strategy(self, play: int, env: Scopa2PEnv, legal: list[tuple[Card, list[Card]]]) -> int:
        ...

    def step(self, env: Scopa2PEnv) -> Action | None:
        if env.agent_selection != self.whoami:
            return None

        obs = env.observe(self.whoami)

        play = choose_play(
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
