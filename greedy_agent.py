from environment import Tressette2PEnv, AgentId, Action
from game import Card, CARD_INDEX

from agent import Tressette2PAgent


class GreedyAgent(Tressette2PAgent):
    """
    Always respond with the strongest card, but
    always "vola" with the weakest
    """
    def __init__(self, whoami: AgentId) -> None:
        super().__init__(whoami, 'Greedy Agent')

    def step(self, env: Tressette2PEnv) -> Action | None:
        def greedy_strategy(card: Card, lead: Card | None) -> int:
            if lead is None:
                return card.power
            return card.power if card.suit == lead.suit else -card.power

        if env.agent_selection != self.whoami:
            return None

        lead = None if env.lead_play is None else env.lead_play.card
        legal = sorted(
                env.hands[self.whoami].legal_plays(lead),
                key=lambda c: greedy_strategy(c, lead),
                reverse=True
        )
        return CARD_INDEX[legal[0]]

