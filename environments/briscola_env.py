from collections import deque
from typing import Any

from pettingzoo import AECEnv
import numpy as np

from games.briscola import BriscolaHand, first_card_wins, card_points
from games.deck import DECK, Card, CARD_INDEX

from environments.cards_env import (
    Action as Action,
    AgentId,
    BinaryArray,
    Cards2PEnv,
    wrap_cards_env,
)


class Briscola2PEnv(Cards2PEnv):
    metadata = {
        'name': 'briscola2p_v0',
        'render_modes': ['human', 'ansi'],
        'is_parallelizable': False,
    }
    OBSERVATION_PLANES = 4

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__(render_mode=render_mode)
        self.briscola: Card | None = None

    def _deal(self, shuffled_cards: list[Card]) -> None:
        self.hands = {
            'p1': BriscolaHand(shuffled_cards[:3]),
            'p2': BriscolaHand(shuffled_cards[3:6]),
        }
        self.pile = deque(shuffled_cards[6:])
        self.briscola = shuffled_cards[-1]
        self.hands['p1'].see_card(self.briscola)
        self.hands['p2'].see_card(self.briscola)

    def _extra_observation_planes(self, agent: AgentId) -> list[BinaryArray]:
        del agent
        briscola_encoding = np.zeros(len(DECK), dtype=np.int8)
        if self.briscola is not None:
            briscola_encoding[CARD_INDEX[self.briscola]] = 1
        return [briscola_encoding]

    def _lead_wins(self, leader_card: Card, follower_card: Card) -> bool:
        if self.briscola is None:
            raise RuntimeError('Cannot resolve a trick without a briscola')
        return first_card_wins(leader_card, follower_card, self.briscola.suit)

    def _trick_points(self, leader_card: Card, follower_card: Card) -> int:
        return card_points(follower_card) + card_points(leader_card)

    def _give_drawn_cards(self, winner: AgentId, loser: AgentId) -> None:
        self.hands[winner].take_card(self.pile.popleft())
        self.hands[loser].take_card(self.pile.popleft())

    def _score_info(self, agent: AgentId, opponent: AgentId) -> dict[str, Any]:
        return {
            'score': self.scores[agent],
            'opponent_score': self.scores[opponent],
        }

    def _render_text(self) -> str:
        lead_text = 'none' if self.lead_play is None else f'{self.lead_play.agent}: {self.lead_play.card}'
        return (
            f'briscola={self.briscola} | '
            f'turn={self.agent_selection} | '
            f'lead={lead_text} | '
            f'pile={len(self.pile)} | '
            f'score={self.scores}'
        )


def env(render_mode: str | None = None) -> AECEnv:
    return wrap_cards_env(raw_env(render_mode=render_mode))


def raw_env(render_mode: str | None = None) -> Briscola2PEnv:
    return Briscola2PEnv(render_mode=render_mode)
