from dataclasses import dataclass
from typing import Callable

import numpy as np

from environments.cards_env import AgentId, encode_cards
from environments.scopa_env import Scopa2PEnv

from games.deck import CARD_INDEX, DECK, Card


ACTION_FEATURE_DIM = len(DECK) * 3 + 1

TakeStrategy = Callable[[list[list[Card]]], int]


@dataclass(frozen=True, slots=True)
class PlayCandidate:
    play: int
    take: int
    features: np.ndarray


def build_play_candidates(
    env: Scopa2PEnv,
    agent: AgentId,
    take_strategy: TakeStrategy,
) -> list[PlayCandidate]:
    legal = env.hands[agent].scopa_legal_plays(env.table)

    options_by_card: dict[Card, list[list[Card]]] = {}
    for played, take in legal:
        options_by_card.setdefault(played, []).append(take)

    opponent = 'p2' if agent == 'p1' else 'p1'
    game_finishes = not env.pile and len(env.hands[agent].cards) == 1 and len(env.hands[opponent].cards) == 0

    n = len(DECK)
    table_base = encode_cards(env.table)
    candidates: list[PlayCandidate] = []

    for card in sorted(options_by_card, key=CARD_INDEX.__getitem__):
        options = options_by_card[card]
        take = take_strategy(options)
        captured = options[take]
        play_idx = CARD_INDEX[card]

        row = np.zeros(ACTION_FEATURE_DIM, dtype=np.float32)
        row[play_idx] = 1.0
        for c in captured:
            row[n + CARD_INDEX[c]] = 1.0

        row[2 * n : 3 * n] = table_base
        if captured:
            for c in captured:
                row[2 * n + CARD_INDEX[c]] = 0.0
        else:
            row[2 * n + play_idx] = 1.0

        clears_table = bool(captured) and len(captured) == len(env.table)
        row[3 * n] = float(clears_table and not game_finishes)

        candidates.append(PlayCandidate(play_idx, take, row))

    return candidates
