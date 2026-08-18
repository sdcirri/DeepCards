import torch

from environments.scopa_env import Scopa2PEnv

from ..dqn import choose_action as choose_action
from ..dqn import DQNAgent as BaseDQNAgent
from ..dqn import CardNN as BaseCardNN
from ..dqn import train as dqn_train

from ..agent import Cards2PAgent, AgentId


OBS_DIM = 200


class CardNN(BaseCardNN):
    def __init__(self, actions: int) -> None:
        super().__init__(OBS_DIM, actions)


def train(
        env: Scopa2PEnv,
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
            training_env: Scopa2PEnv,
            training_opponents: list[Cards2PAgent],
            episodes_per_opponent: list[int],
            verbose_training: bool = False,
    ) -> 'DQNAgent':
        net = train(
                training_env,
                whoami,
                80,         # card to play + cards to take
                DQNAgent.device,
                training_opponents,
                episodes_per_opponent,
                verbose_training,
        )
        return DQNAgent(whoami, net)

    def step(self, env: Scopa2PEnv) -> int | None:
        if env.agent_selection != self.whoami:
            return None

        obs = env.observe(self.whoami)

        return choose_action(
                obs['observation'],
                obs['action_mask'],
                self.net,
                0,
                self.device
        )
