from environments.scopa_env import Scopa2PEnv, AgentId, Action
from games.deck import CARD_INDEX

from ..agent import Cards2PAgent


class RandomAgent(Cards2PAgent):
    """
    Play a random card
    """
    def __init__(self, whoami: AgentId) -> None:
        super().__init__(whoami, 'Random Agent')

    def step(self, env: Scopa2PEnv) -> Action | None:
        if env.agent_selection != self.whoami:
            return None

        legal = env.hands[self.whoami].scopa_legal_plays(env.table)
        play = self.np_random.choice(env.hands[self.whoami].cards)
        options = [opt[1] for opt in legal if opt[0] == play]
        take = self.np_random.integers(0, len(options))

        return CARD_INDEX[play], take
