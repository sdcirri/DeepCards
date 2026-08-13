import numpy as np

from environments.tressette_env import Tressette2PEnv, AgentId, Action
from games.deck import DECK

from agents.tressette.agent import Tressette2PAgent


class RandomAgent(Tressette2PAgent):
    """
    Play a random card as long as it's playable
    """
    def __init__(self, whoami: AgentId) -> None:
        super().__init__(whoami, 'Random Agent')

    def step(self, env: Tressette2PEnv) -> Action | None:
        if env.agent_selection != self.whoami:
            return None

        lead = None if env.lead_play is None else env.lead_play.card
        legal = env.hands[self.whoami].legal_plays(lead)
        while True:
            action = self.np_random.integers(low=0, high=40, dtype=np.int32)
            if DECK[action] in legal:
                return action
