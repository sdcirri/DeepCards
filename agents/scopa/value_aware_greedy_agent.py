from environments.scopa_env import Scopa2PEnv, AgentId, Action
from games.scopa import ScopaHand
from games.deck import Card, Suit

from ..agent import Cards2PAgent


class ValueAwareGreedyAgent(Cards2PAgent):
    """
    Try to take as many cards as possible, but be
    aware of their value
    """
    def __init__(self, whoami: AgentId) -> None:
        super().__init__(whoami, 'Value-Aware Greedy Agent')

    @staticmethod
    def card_value(card: Card) -> float:
        # Carte and primiera
        value = 1 / 40 + ScopaHand.PRIMIERA_BY_NUM[card.number-1] / 139
        # Settebello
        if card == Card(Suit.DENARI, 7):
            value += 1
        # Denari (also applies to Settebello)
        if card.suit == Suit.DENARI:
            value += 1 / 10
        return value

    @staticmethod
    def play_value(play: tuple[Card, list[Card]], table: list[Card]) -> float:
        if len(play[1]) == 0:
            # Leaving a card on the table is potentially lost value
            return -ValueAwareGreedyAgent.card_value(play[0])

        value = ValueAwareGreedyAgent.card_value(play[0]) + \
            sum(ValueAwareGreedyAgent.card_value(c) for c in play[1])

        # Scopa reward
        if len(play[1]) == len(table):
            value += 1

        return value

    def step(self, env: Scopa2PEnv) -> Action | None:
        if env.agent_selection != self.whoami:
            return None

        legal = env.hands[self.whoami].scopa_legal_plays(env.table)
        played = max(legal, key=lambda pt: self.play_value(pt, env.table))

        return legal.index(played)
