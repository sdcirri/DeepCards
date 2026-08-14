from dataclasses import dataclass, field
from enum import Enum


class Suit(Enum):
    DENARI = 'Denari'
    BASTONI = 'Bastoni'
    COPPE = 'Coppe'
    SPADE = 'Spade'


CARD_NUMBERS = tuple(range(1, 11))


@dataclass(frozen=True, slots=True)
class Card:
    suit: Suit
    number: int

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

    def legal_plays(self, trick: Card | None) -> list[Card]:
        ...

    def see_card(self, card: Card) -> None:
        """
        For when opponent either plays a trick
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
