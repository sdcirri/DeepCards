from dataclasses import dataclass
from typing import Callable

import numpy as np

from environments.scopa_env import Scopa2PEnv
from environments.cards_env import AgentId

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
    opponent = 'p2' if agent == 'p1' else 'p1'
    candidates: list[PlayCandidate] = []

    for card in sorted(env.hands[agent].cards, key=CARD_INDEX.__getitem__):
        options = [take for played, take in legal if played == card]
        take = take_strategy(options)
        captured = options[take]

        played_encoding = np.zeros(len(DECK), dtype=np.float32)
        played_encoding[CARD_INDEX[card]] = 1.0

        captured_encoding = np.zeros(len(DECK), dtype=np.float32)
        for captured_card in captured:
            captured_encoding[CARD_INDEX[captured_card]] = 1.0

        resulting_table = set(env.table)
        if captured:
            resulting_table.difference_update(captured)
        else:
            resulting_table.add(card)
        table_encoding = np.fromiter(
            (table_card in resulting_table for table_card in DECK),
            dtype=np.float32,
            count=len(DECK),
        )

        game_finishes = (
            not env.pile
            and len(env.hands[agent].cards) == 1
            and len(env.hands[opponent].cards) == 0
        )
        makes_scopa = float(bool(captured) and not resulting_table and not game_finishes)

        features = np.concatenate((
            played_encoding,
            captured_encoding,
            table_encoding,
            np.asarray([makes_scopa], dtype=np.float32),
        ))
        candidates.append(PlayCandidate(CARD_INDEX[card], take, features))

    return candidates
