import torch

from environments.briscola_env import AgentId, Briscola2PEnv
from games.deck import DECK

from ..dqn import choose_action as choose_action
from ..dqn import DQNAgent as BaseDQNAgent
from ..dqn import CardNN as BaseCardNN
from ..dqn import train as dqn_train
from ..agent import Cards2PAgent


OBS_DIM = Briscola2PEnv.OBSERVATION_PLANES * len(DECK) + Briscola2PEnv.EXTRA_OBSERVATIONS


class CardNN(BaseCardNN):
    def __init__(self, actions: int) -> None:
        super().__init__(OBS_DIM, actions)


def train(
        env: Briscola2PEnv,
        whoami: AgentId,
        actions: int,
        device: torch.device,
        training_opponents: list[Cards2PAgent],
        episodes_per_opponent: list[int],
        verbose: bool = True,
) -> CardNN:
    return dqn_train(
        env,
        whoami,
        actions,
        OBS_DIM,
        device,
        training_opponents,
        episodes_per_opponent,
        verbose,
    )


class DQNAgent(BaseDQNAgent):
    @staticmethod
    def train(
            whoami: AgentId,
            training_env: Briscola2PEnv,
            training_opponents: list[Cards2PAgent],
            episodes_per_opponent: list[int],
            verbose_training: bool = False,
    ) -> 'DQNAgent':
        net = train(
                training_env,
                whoami,
                40,
                DQNAgent.device,
                training_opponents,
                episodes_per_opponent,
                verbose_training,
        )
        return DQNAgent(whoami, net)
