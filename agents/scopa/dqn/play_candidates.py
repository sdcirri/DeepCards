from dataclasses import dataclass

import numpy as np

from environments.cards_env import AgentId, encode_cards
from environments.scopa_env import Scopa2PEnv

from games.deck import CARD_INDEX, DECK, Card


ACTION_FEATURE_DIM = len(DECK) * 3 + 1


@dataclass(frozen=True, slots=True)
class PlayCandidate:
    play: int
    take: int
    features: np.ndarray


def build_play_candidates(env: Scopa2PEnv, agent: AgentId) -> list[PlayCandidate]:
    legal = env.hands[agent].scopa_legal_plays(env.table)
    opponent = 'p2' if agent == 'p1' else 'p1'

    game_finishes = not env.pile and len(env.hands[agent].cards) == 1 and len(env.hands[opponent].cards) == 0

    candidates: list[PlayCandidate] = []

    for card in sorted(env.hands[agent].cards, key=CARD_INDEX.__getitem__):
        options = [take for played, take in legal if played == card]
        played_encoding = np.zeros(len(DECK), dtype=np.float32)
        played_encoding[CARD_INDEX[card]] = 1.0

        for take, captured in enumerate(options):
            captured_encoding = encode_cards(captured)

            resulting_table = set(env.table)
            if captured:
                resulting_table.difference_update(captured)
            else:
                resulting_table.add(card)

            table_encoding = encode_cards(resulting_table)
            makes_scopa = float(bool(captured) and not resulting_table and not game_finishes)
            features = np.concatenate((
                played_encoding,
                captured_encoding,
                table_encoding,
                np.asarray([makes_scopa], dtype=np.float32),
            ))
            candidates.append(PlayCandidate(
                play=CARD_INDEX[card],
                take=take,
                features=features,
            ))

    return candidates
