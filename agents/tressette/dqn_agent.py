from environments.tressette_env import AgentId, Tressette2PEnv

from ..dqn import choose_action as choose_action
from ..dqn import DQNAgent as BaseDQNAgent
from ..dqn import CardNN as BaseCardNN
from ..dqn import train as dqn_train
from ..agent import Cards2PAgent

from .random_agent import RandomAgent


OBS_DIM = 120


class CardNN(BaseCardNN):
    def __init__(self, actions: int) -> None:
        super().__init__(OBS_DIM, actions)


def train(
    env: Tressette2PEnv,
    whoami: AgentId,
    episodes: int,
    actions: int,
    device,
    training_opponent: Cards2PAgent | None = None,
    verbose: bool = True,
) -> CardNN:
    if training_opponent is None:
        training_opponent = RandomAgent('p2' if whoami == 'p1' else 'p1')
    return dqn_train(  # type: ignore[return-value]
        env,
        whoami,
        episodes,
        actions,
        OBS_DIM,
        device,
        training_opponent,
        verbose,
    )


class DQNAgent(BaseDQNAgent):
    @staticmethod
    def train(
        whoami: AgentId,
        training_env: Tressette2PEnv,
        training_opponent: Cards2PAgent | None = None,
        verbose_training: bool = False,
    ) -> 'DQNAgent':
        net = train(
            training_env,
            whoami,
            5000,
            40,
            DQNAgent.device,
            training_opponent,
            verbose_training,
        )
        return DQNAgent(whoami, net)
