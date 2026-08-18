from dataclasses import dataclass

from .deck import Suit, Card, Hand


CARD_POWER_ORDER = (1, 3, 10, 9, 8, 7, 6, 5, 4, 2)
CARD_POWER = {
    number: len(CARD_POWER_ORDER) - position
    for position, number in enumerate(CARD_POWER_ORDER)
}


@dataclass(slots=True)
class BriscolaHand(Hand):

    def legal_plays(self, trick: Card | None) -> list[Card]:
        # There are no illegal plays in Briscola
        return self.cards.copy()


def card_points(card: Card) -> int:
    CARD_POINTS = (11, 0, 10, 0, 0, 0, 0, 2, 3, 4)
    return CARD_POINTS[card.number-1]


def first_card_wins(card1: Card, card2: Card, briscola: Suit) -> bool:
    """
    Returns True if the first played card (card1)
    wins, False otherwise
    """
    if card1.suit != card2.suit:
        if card2.suit != briscola:
            return True
        if card1.suit != briscola:
            return False

    return CARD_POWER[card1.number] > CARD_POWER[card2.number]
