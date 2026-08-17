import numpy as np

from environments.scopa_env import Scopa2PEnv, AgentId, Action
from games.deck import CARD_INDEX, DECK

from agents.agent import Cards2PAgent


class GreedyAgent(Cards2PAgent):
    """
    Try to take as many cards as possible
    """
    def __init__(self, whoami: AgentId) -> None:
        super().__init__(whoami, 'Greedy Agent')

    def step(self, env: Scopa2PEnv) -> Action | None:
        if env.agent_selection != self.whoami:
            return None

        played, taken_cards = sorted(
            env.hands[self.whoami].scopa_legal_plays(env.table),
            key=lambda l: len(l[1]),
            reverse=True
        )[0]

        take_mask = np.zeros(len(DECK), dtype=np.int8)
        for card in taken_cards:
            take_mask[CARD_INDEX[card]] = 1

        return {'played': CARD_INDEX[played], 'taken': take_mask}
