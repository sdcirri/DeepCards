from dataclasses import dataclass
from functools import lru_cache
from math import comb
import numpy as np

from environments.cards_env import AgentId, encode_cards
from environments.scopa_env import Scopa2PEnv

from games.scopa import card_value, play_value, legal_plays
from games.deck import CARD_INDEX, DECK


VALUES = np.asarray([card_value(c) for c in DECK], dtype=np.float32)
ACTION_FEATURE_DIM = len(DECK) * 3 + 11


@dataclass(frozen=True, slots=True)
class PlayCandidate:
    play: int
    take: int
    features: np.ndarray


@lru_cache(maxsize=100_000)
def expected_reply(table_ids: list[int], unknown_ids: list[int], hand_size: int) -> float:
    if hand_size == 0 or not unknown_ids:
        return 0.0

    table = tuple(DECK[i] for i in table_ids)
    values = sorted(
        max(play_value(play, table) for play in legal_plays((DECK[i],), table))
        for i in unknown_ids
    )
    h = min(hand_size, len(values))
    denominator = comb(len(values), h)

    return sum(
        value * comb(j, h - 1) / denominator
        for j, value in enumerate(values)
        if j >= h - 1
    )


def enrich_candidates(env: Scopa2PEnv, agent: AgentId, candidates: list[PlayCandidate]) -> list[PlayCandidate]:
    hand = env.hands[agent]
    unknown = tuple(
        i for i, card in enumerate(DECK)
        if card not in hand.seen
    )
    opponent_size = (
        40 - len(env.pile) - len(hand.cards) - len(env.table)
        - len(hand.taken_cards) - len(hand.opponent_taken_cards)
    )
    if opponent_size == 0 and len(hand.cards) == 1 and env.pile:
        opponent_size = 3

    enriched = []
    for candidate in candidates:
        f = candidate.features
        played_value = float(f[:40] @ VALUES)
        captured_value = float(f[40:80] @ VALUES)
        captured_count = float(f[40:80].sum())
        table_ids = tuple(np.flatnonzero(f[80:120]))
        immediate_value = (
            played_value + captured_value
            if captured_count else -played_value
        ) + f[120]

        extra = np.asarray([
            played_value,
            captured_value,
            immediate_value,
            expected_reply(table_ids, unknown, opponent_size),
            captured_count / 10,
            len(table_ids) / 10,
            sum(DECK[i].number for i in table_ids) / 40,
            f[40:50].sum() / 10,    # Taken Denari
            f[46],                  # Settebello
            float(opponent_size == 0),
        ], dtype=np.float32)

        enriched.append(type(candidate)(
            candidate.play,
            candidate.take,
            np.concatenate((f, extra)),
        ))

    return enriched


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

    return enrich_candidates(env, agent, candidates)
