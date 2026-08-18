from environments.scopa_env import Scopa2PEnv, AgentId, Action
from games.deck import CARD_INDEX

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

        legal = env.hands[self.whoami].scopa_legal_plays(env.table)
        played, taken_cards = max(legal, key=lambda pt: len(pt[1]))

        options = [take for play, take in legal if play == played]
        take_idx = options.index(taken_cards)

        return {'played': CARD_INDEX[played], 'taken': take_idx}
