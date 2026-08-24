from environments.scopa_env import Scopa2PEnv, Action
from games.deck import DECK

from .dqn.play_candidates import build_play_candidates
from .dqn.training import choose_candidate
from .dqn.scopa_nn import ScopaNN

from ..dqn import DQNAgent as BaseDQNAgent
from ..agent import AgentId


OBS_DIM = Scopa2PEnv.OBSERVATION_PLANES * len(DECK) + Scopa2PEnv.EXTRA_OBSERVATIONS


class DQNAgent(BaseDQNAgent):
    """
    Base DQN agent for Scopa
    """

    def __init__(self, whoami: AgentId, net: ScopaNN) -> None:
        super().__init__(whoami, net)

    def step(self, env: Scopa2PEnv) -> Action | None:
        if env.agent_selection != self.whoami:
            return None

        obs = env.observe(self.whoami)
        candidates = build_play_candidates(
            env,
            self.whoami,
            lambda opt: max(range(len(opt)), key=lambda i: len(opt[i]))
        )
        candidate = choose_candidate(
                obs['observation'],
                candidates,
                self.net,
                0,
                self.device
        )
        return candidate.play, candidate.take
