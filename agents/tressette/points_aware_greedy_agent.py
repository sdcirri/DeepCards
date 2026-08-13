from environments.tressette_env import Tressette2PEnv, AgentId, Action

from agents.tressette.agent import Tressette2PAgent

from games.tressette import CARD_POWER, card_point_thirds
from games.deck import CARD_INDEX


class PointsAwareGreedyAgent(Tressette2PAgent):
    """
    Point-aware heuristic:
    - Leading: strongest 0-point card, else lowest-value lead
    - Following: cheapest winner, else shed lowest-value card
    """
    def __init__(self, whoami: AgentId) -> None:
        super().__init__(whoami, 'Points-Aware Greedy Agent')

    def step(self, env: Tressette2PEnv) -> Action | None:
        if env.agent_selection != self.whoami:
            return None

        lead = None if env.lead_play is None else env.lead_play.card
        legal = env.hands[self.whoami].legal_plays(lead)

        if lead is None:
            # Leading: win cheaply with 0-point cards, else lead lowest value
            zero_point = [c for c in legal if card_point_thirds(c) == 0]
            if zero_point:
                card = max(zero_point, key=lambda c: CARD_POWER[c.number])
            else:
                card = min(
                    legal,
                    key=lambda c: (card_point_thirds(c), CARD_POWER[c.number])
                )
        else:
            winners = [
                c for c in legal
                if c.suit == lead.suit and CARD_POWER[c.number] > CARD_POWER[lead.number]
            ]
            if winners:
                # Cheapest card that still wins the trick
                card = min(
                    winners,
                    key=lambda c: (CARD_POWER[c.number], card_point_thirds(c))
                )
            else:
                # Losing: shed lowest-value card (opponent already winning)
                card = min(
                    legal, key=lambda c: (card_point_thirds(c), CARD_POWER[c.number])
                )

        return CARD_INDEX[card]
