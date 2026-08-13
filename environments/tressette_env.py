from collections import deque
from typing import Any

from pettingzoo import AECEnv

from games.tressette import TressetteHand, first_card_wins, card_point_thirds
from games.deck import Card

from environments.cards_env import (
    Action as Action,
    AgentId,
    Cards2PEnv,
    wrap_cards_env,
)


class Tressette2PEnv(Cards2PEnv):
    metadata = {
        'name': 'tressette2p_v0',
        'render_modes': ['human', 'ansi'],
        'is_parallelizable': False,
    }
    OBSERVATION_PLANES = 3

    def _deal(self, shuffled_cards: list[Card]) -> None:
        self.hands = {
            'p1': TressetteHand(shuffled_cards[:10]),
            'p2': TressetteHand(shuffled_cards[10:20]),
        }
        self.pile = deque(shuffled_cards[20:])

    def _lead_wins(self, leader_card: Card, follower_card: Card) -> bool:
        return first_card_wins(leader_card, follower_card)

    def _trick_points(self, leader_card: Card, follower_card: Card) -> int:
        return card_point_thirds(follower_card) + card_point_thirds(leader_card)

    def _give_drawn_cards(self, winner: AgentId, loser: AgentId) -> None:
        winner_card = self.pile.popleft()
        loser_card = self.pile.popleft()

        self.hands[winner].take_card(winner_card)
        self.hands[loser].see_card(winner_card)

        self.hands[loser].take_card(loser_card)
        self.hands[winner].see_card(loser_card)

    def _on_game_finished(self, winner: AgentId) -> None:
        # Extra point (three thirds) for last win
        self.rewards[winner] += 3

    def _score_info(self, agent: AgentId, opponent: AgentId) -> dict[str, Any]:
        return {
            'score_thirds': self.scores[agent],
            'opponent_score_thirds': self.scores[opponent],
        }

    def _render_text(self) -> str:
        lead_text = 'none' if self.lead_play is None else f'{self.lead_play.agent}: {self.lead_play.card}'
        return (
            f'turn={self.agent_selection} | '
            f'lead={lead_text} | '
            f'pile={len(self.pile)} | '
            f'score_thirds={self.scores}'
        )


def env(render_mode: str | None = None) -> AECEnv:
    return wrap_cards_env(raw_env(render_mode=render_mode))


def raw_env(render_mode: str | None = None) -> Tressette2PEnv:
    return Tressette2PEnv(render_mode=render_mode)
