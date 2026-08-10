from dataclasses import dataclass, field
from enum import Enum


class Suit(Enum):
    DENARI = 'Denari'
    BASTONI = 'Bastoni'
    COPPE = 'Coppe'
    SPADE = 'Spade'


CARD_NUMBERS = tuple(range(1, 11))
CARD_POWER_ORDER = (3, 2, 1, 10, 9, 8, 7, 6, 5, 4)
CARD_POWER = {
    number: len(CARD_POWER_ORDER) - position
    for position, number in enumerate(CARD_POWER_ORDER)
}


@dataclass(frozen=True, slots=True)
class Card:
    suit: Suit
    number: int

    @property
    def point_thirds(self) -> int:
        if self.number == 1:
            return 3
        elif self.number in (2, 3, 8, 9, 10):
            return 1
        return 0

    @property
    def power(self) -> int:
        return CARD_POWER[self.number]

    def __str__(self) -> str:
        return f'Card [{self.number} of {self.suit.value}]'


@dataclass(slots=True)
class Hand:
    cards: list[Card]
    seen: set[Card] = field(init=False)

    def __post_init__(self) -> None:
        self.cards = list(self.cards)
        if len(self.cards) != len(set(self.cards)):
            raise ValueError('A hand cannot contain duplicate cards')
        self.seen = set(self.cards)

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

    def see_card(self, card: Card) -> None:
        """
        For when adversary either plays a trick
        or takes a card
        """
        self.seen.add(card)

    def play_card(self, card: Card) -> None:
        self.cards.remove(card)

    def take_card(self, card: Card) -> None:
        self.cards.append(card)
        self.seen.add(card)

    def __len__(self) -> int:
        return len(self.cards)


DECK = [
    Card(s, n)
    for s in Suit
    for n in range(1, 11)
]

CARD_INDEX: dict[Card, int] = {
    card: index
    for index, card in enumerate(DECK)
}


def first_card_wins(card1: Card, card2: Card) -> bool:
    """
    Returns True if the first played card (card1)
    wins, False otherwise
    """
    if card1.suit != card2.suit:
        return True

    return card1.power > card2.power
