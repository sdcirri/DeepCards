from environments.briscola_env import Briscola2PEnv, AgentId, Action
from games.deck import CARD_INDEX

from ..agent import Cards2PAgent


class RandomAgent(Cards2PAgent):
    """
    Play a random card as long as it's playable
    """
    def __init__(self, whoami: AgentId) -> None:
        super().__init__(whoami, 'Random Agent')

    def step(self, env: Briscola2PEnv) -> Action | None:
        if env.agent_selection != self.whoami:
            return None

        lead = None if env.lead_play is None else env.lead_play.card
        legal = env.hands[self.whoami].legal_plays(lead)
        action = self.np_random.choice(legal)
        return CARD_INDEX[action]
