from environment import Tressette2PEnv, AgentId, Action
from game import CARD_INDEX

from agent import Tressette2PAgent


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
            zero_point = [c for c in legal if c.point_thirds == 0]
            if zero_point:
                card = max(zero_point, key=lambda c: c.power)
            else:
                card = min(legal, key=lambda c: (c.point_thirds, c.power))
        else:
            winners = [c for c in legal if c.suit == lead.suit and c.power > lead.power]
            if winners:
                # Cheapest card that still wins the trick
                card = min(winners, key=lambda c: (c.power, c.point_thirds))
            else:
                # Losing: shed lowest-value card (opponent already winning)
                card = min(legal, key=lambda c: (c.point_thirds, c.power))

        return CARD_INDEX[card]
