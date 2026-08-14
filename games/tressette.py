from dataclasses import dataclass, field

from .deck import Suit, Card, Hand


CARD_POWER_ORDER = (3, 2, 1, 10, 9, 8, 7, 6, 5, 4)
CARD_POWER = {
    number: len(CARD_POWER_ORDER) - position
    for position, number in enumerate(CARD_POWER_ORDER)
}


@dataclass(slots=True)
class TressetteHand(Hand):
    known_opponent_hand: list[Card] = field(init=False, default_factory=list)

    def get_accusi_points(self) -> int:
        napoli = {s: 0 for s in Suit}
        buongiochi = [0] * 3

        for card in self.cards:
            if card.number in (1, 2, 3):
                buongiochi[card.number-1] += 1
                napoli[card.suit] += 1

        return sum(3 if n == 3 else 0 for n in napoli.values()) + sum(b if b >= 3 else 0 for b in buongiochi)

    def legal_plays(self, trick: Card | None) -> list[Card]:
        cards = self.cards.copy()
        if trick is None:
            return cards
        same_suit = [c for c in cards if c.suit == trick.suit]
        if not same_suit:
            return cards
        return same_suit

    def see_opponent_draw(self, drawn_card: Card) -> None:
        """
        Remember what the opponent drawn
        """
        self.see_card(drawn_card)
        self.known_opponent_hand.append(drawn_card)

    def see_card(self, card: Card) -> None:
        """
        Also need to update self.known_opponent_hand
        when a card is played
        """
        Hand.see_card(self, card)
        if card in self.known_opponent_hand:
            self.known_opponent_hand.remove(card)


def card_point_thirds(card: Card) -> int:
    if card.number == 1:
        return 3
    elif card.number in (2, 3, 8, 9, 10):
        return 1
    return 0


def first_card_wins(card1: Card, card2: Card) -> bool:
    """
    Returns True if the first played card (card1)
    wins, False otherwise
    """
    if card1.suit != card2.suit:
        return True

    return CARD_POWER[card1.number] > CARD_POWER[card2.number]
