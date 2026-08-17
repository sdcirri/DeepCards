import numpy as np

from environments.scopa_env import Scopa2PEnv, AgentId, Action
from games.deck import CARD_INDEX, DECK

from agents.agent import Cards2PAgent


class RandomAgent(Cards2PAgent):
    """
    Play a random card as long as it's playable
    """
    def __init__(self, whoami: AgentId) -> None:
        super().__init__(whoami, 'Random Agent')

    def step(self, env: Scopa2PEnv) -> Action | None:
        if env.agent_selection != self.whoami:
            return None

        legal = env.hands[self.whoami].scopa_legal_plays(env.table)
        played, taken_cards = legal[int(self.np_random.integers(len(legal)))]

        take_mask = np.zeros(len(DECK), dtype=np.int8)
        for card in taken_cards:
            take_mask[CARD_INDEX[card]] = 1

        return {'played': CARD_INDEX[played], 'taken': take_mask}
