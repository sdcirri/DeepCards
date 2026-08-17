"""Shared viz constants and live session state."""

from typing import Any, Literal
from dataclasses import dataclass
from pathlib import Path

from games.deck import Card
from environments.cards_env import AgentId

ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / 'assets' / 'CartePiacentineITA'
CARD_BACK = ASSETS_DIR / '_Dorso.png'
TABLE_COLOR = (27, 107, 58)
BORDER_COLOR = (12, 60, 32)
WHITE = (255, 255, 255)
STEP_DELAY_MS = 450
ANIM_MS = 280
ANIM_HOLD_MS = 180
OPPONENT_CARD_ALPHA = 140
MAX_PARALLEL_EPISODES = 100

GameKind = Literal['briscola', 'tressette', 'scopa']

_LIVE: dict[str, Any] = {
    'screen': None,
    'kind': None,
    'clock': None,
    'font_cache': {},
    'card_cache': {},
    'n_slots': 1,
    'cols': 1,
    'rows': 1,
    'pane_size': (900, 700),
    'scale': 1.0,
    'panes': {},
    'prev': {},
    'title': '',
    'parallel_envs': [],
}


@dataclass(frozen=True, slots=True)
class _Flight:
    card: Card
    agent: AgentId
    start: tuple[int, int]
    end: tuple[int, int]
    face_up: bool = True


def card_image_path(card: Card) -> Path:
    return ASSETS_DIR / f'{card.suit.value}{card.number:02d}.png'


def _slot_prev(slot: int) -> dict[str, Any] | None:
    return _LIVE['prev'].get(slot)


def _set_slot_prev(slot: int, snapshot: dict[str, Any] | None) -> None:
    if snapshot is None:
        _LIVE['prev'].pop(slot, None)
    else:
        _LIVE['prev'][slot] = snapshot
