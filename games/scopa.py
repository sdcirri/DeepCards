from dataclasses import dataclass, field

from .deck import Suit, Card, Hand


def find_takes(card: Card, table: list[Card]) -> list[list[Card]]:
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


@dataclass
class ScopaScope:
    scope: int = 0
    primiera: int = 0
    settebello: int = 0
    carte: int = 0
    denari: int = 0


@dataclass(slots=True)
class ScopaHand(Hand):
    PRIMIERA_BY_NUM = [16, 12, 13, 14, 15, 18, 21, 10, 10, 10]

    taken_cards: list[Card] = field(default_factory=list)
    opponent_taken_cards: list[Card] = field(default_factory=list)
    score: ScopaScope = field(default_factory=ScopaScope)

    def scopa_legal_plays(self, table: list[Card]) -> list[tuple[Card, list[Card]]]:
        """
        List legal plays, returns a list of
        tuples representing legal plays like this:
        (playable_card, [list of cards on the table taken by that card])
        """
        ret = []
        for card in self.cards:
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

            self.score.primiera += self.PRIMIERA_BY_NUM[card.number-1]

            if card.suit == Suit.DENARI:
                self.score.denari += 1
