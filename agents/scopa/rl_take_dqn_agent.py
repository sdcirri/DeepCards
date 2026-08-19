import numpy as np
import torch

from environments.scopa_env import Scopa2PEnv
from games.deck import DECK, Card

from .scopa_dqn import PlayCardNN, TakeCardNN, train as dqn_train, choose_take
from .dqn_agent import DQNAgent

from ..agent import AgentId, Cards2PAgent


def train(
        env: Scopa2PEnv,
        whoami: AgentId,
        play_actions: int,
        take_actions: int,
        device: torch.device,
        training_opponents: list[Cards2PAgent],
        episodes_per_opponent: list[int],
        verbose: bool = True,
) -> tuple[PlayCardNN, TakeCardNN]:
    return dqn_train(
        env,
        whoami,
        play_actions,
        take_actions,
        device,
        training_opponents,
        episodes_per_opponent,
        verbose,
    )


class RLTakeDQNAgent(DQNAgent):
    """
    Use an auxiliary net to predict the best
    cards to take
    """

    def __init__(self, whoami: AgentId, play_net: PlayCardNN, take_net: TakeCardNN) -> None:
        super().__init__('DQN Agent with RL Take', whoami, play_net)
        self.take_net = take_net

    @staticmethod
    def train(
            whoami: AgentId,
            training_env: Scopa2PEnv,
            training_opponents: list[Cards2PAgent],
            episodes_per_opponent: list[int],
            verbose_training: bool = False,
    ) -> 'RLTakeDQNAgent':
        play_net, take_net = train(
                training_env,
                whoami,
                len(DECK),
                len(DECK),
                DQNAgent.device,
                training_opponents,
                episodes_per_opponent,
                verbose_training,
        )
        return RLTakeDQNAgent(whoami, play_net, take_net)

    def take_strategy(self, play: int, env: Scopa2PEnv, legal: list[tuple[Card, list[Card]]]) -> int:
        obs = env.observe(self.whoami)
        play_t = np.zeros(len(DECK), dtype=np.int8)
        play_t[play] = 1
        options = [opt[1] for opt in legal if opt[0] == DECK[play]]
        return choose_take(
            np.concatenate((obs['observation'], play_t)),
            obs['take_mask'],
            self.take_net,
            0,
            self.device,
            options
        )
