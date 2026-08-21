from dataclasses import dataclass, field
from functools import cache

from .deck import Suit, Card, Hand, CARD_INDEX


PRIMIERA_BY_NUM = (16, 12, 13, 14, 15, 18, 21, 10, 10, 10)


def find_takes(card: Card, table: tuple[Card]) -> list[list[Card]]:
    """
    Finds all possible combinations of cards
    that can be taken playing `card`, assuming
    no card on the table has the same number
    as `card`
    """

    # dp[s] = all possible combinations of cards
    # that sum up to s

    dp = [[] for _ in range(card.number+1)]
    dp[0] = [[]]

    for i, x in enumerate(table):
        if x.number >= card.number:
            continue
        for s in range(card.number, x.number-1, -1):
            for subset in dp[s - x.number]:
                dp[s].append(subset + [x])

    return dp[card.number]


def legal_plays(hand: tuple[Card], table: tuple[Card]) -> list[tuple[Card, list[Card]]]:
    """
    List legal plays, returns a list of
    tuples representing legal plays like this:
    (playable_card, [list of cards on the table taken by that card])
    """
    ret = []
    for card in hand:
        same_num = [c for c in table if c.number == card.number]
        if len(same_num) > 0:
            # If same number is already on the table you are
            #   forced to take
            ret += [(card, [c]) for c in same_num]
        else:
            takes = find_takes(card, table)
            if len(takes) == 0:
                # If played, card will just stack up on the table
                ret += [(card, [])]
            else:
                ret += [(card, take) for take in takes]
    return ret


@cache
def sorted_legal_plays(hand: tuple[Card], table: tuple[Card]) -> list[tuple[Card, list[Card]]]:
    """
    List legal plays, returns a list of
    tuples representing legal plays like this:
    (playable_card, [list of cards on the table taken by that card]),
    sorted by card indexes
    """
    return sorted(
        legal_plays(hand, table),
        key=lambda pt: (
            CARD_INDEX[pt[0]],
            tuple(sorted(CARD_INDEX[c] for c in pt[1])),
        ),
    )


def card_value(card: Card) -> float:
    """
    Estimation of card advantage when taken
    :param card: card to evaluate
    :return: the card value
    """
    # Carte and primiera
    value = 1 / 40 + PRIMIERA_BY_NUM[card.number-1] / 84
    # Settebello
    if card == Card(Suit.DENARI, 7):
        value += 1
    # Denari (also applies to Settebello)
    if card.suit == Suit.DENARI:
        value += 1 / 10
    return value


def play_value(play: tuple[Card, list[Card]], table: list[Card]) -> float:
    if len(play[1]) == 0:
        # Leaving a card on the table is potentially lost value
        return -card_value(play[0])

    value = card_value(play[0]) + sum(card_value(c) for c in play[1])

    # Scopa reward
    if len(play[1]) == len(table):
        value += 1

    return value


@dataclass
class ScopaScope:
    scope: int = 0
    primiera: dict[Suit, int] = field(default_factory=lambda: {s: 0 for s in Suit})
    settebello: int = 0
    carte: int = 0
    denari: int = 0


@dataclass(slots=True)
class ScopaHand(Hand):
    taken_cards: list[Card] = field(default_factory=list)
    opponent_taken_cards: list[Card] = field(default_factory=list)
    score: ScopaScope = field(default_factory=ScopaScope)

    def scopa_legal_plays(self, table: list[Card]) -> list[tuple[Card, list[Card]]]:
        """
        List legal plays, returns a list of
        tuples representing legal plays like this:
        (playable_card, [list of cards on the table taken by that card]),
        sorted by card indexes
        """
        return sorted_legal_plays(tuple(self.cards), tuple(table))

    def see_opponent_taken(self, cards: list[Card]) -> None:
        self.opponent_taken_cards.extend(cards)

    def scopa(self) -> None:
        self.score.scope += 1

    def update_score(self, taken: list[Card]) -> None:
        for card in taken:
            self.taken_cards.append(card)
            self.score.carte += 1

            if card == Card(Suit.DENARI, 7):
                self.score.settebello = 1

            if card.suit == Suit.DENARI:
                self.score.denari += 1

            if (p := PRIMIERA_BY_NUM[card.number-1]) > self.score.primiera[card.suit]:
                self.score.primiera[card.suit] = p

    def get_score(self, opponent_primiera: int) -> int:
        return self.score.scope + self.score.settebello \
            + (1 if sum(self.score.primiera.values()) > opponent_primiera else 0) \
            + (1 if self.score.carte > 20 else 0) \
            + (1 if self.score.denari > 5 else 0)
