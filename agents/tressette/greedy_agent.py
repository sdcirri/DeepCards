from environments.tressette_env import Tressette2PEnv, AgentId, Action

from agents.tressette.agent import Tressette2PAgent

from games.deck import CARD_INDEX, Card
from games.tressette import CARD_POWER


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
                return CARD_POWER[card.number]
            return CARD_POWER[card.number] if card.suit == lead.suit else -CARD_POWER[card.number]

        if env.agent_selection != self.whoami:
            return None

        lead = None if env.lead_play is None else env.lead_play.card
        legal = sorted(
                env.hands[self.whoami].legal_plays(lead),
                key=lambda c: greedy_strategy(c, lead),
                reverse=True
        )
        return CARD_INDEX[legal[0]]
