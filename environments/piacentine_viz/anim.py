"""Animation helpers and shared flight playback."""

from typing import Callable

import pygame

from games.deck import Card
from environments.cards_env import AgentId

from .state import ANIM_HOLD_MS, ANIM_MS, CARD_BACK, _Flight, _LIVE, card_image_path
from .draw import _blit_card
from .window import _present, _pump_events, _wait_step

def _ease_out_cubic(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3


def _lerp(a: tuple[int, int], b: tuple[int, int], t: float) -> tuple[int, int]:
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
    )
def _new_cards(before: list[Card], after: list[Card]) -> list[Card]:
    remaining = list(before)
    gained: list[Card] = []
    for card in after:
        if card in remaining:
            remaining.remove(card)
        else:
            gained.append(card)
    return gained
def _won_pile_pos(
    surface: pygame.Surface,
    winner: AgentId,
    *,
    viewpoint: AgentId,
    index: int = 0,
) -> tuple[int, int]:
    width = surface.get_width()
    height = surface.get_height()
    x = int(width * 0.88) + index * max(6, int(14 * _LIVE['scale']))
    if winner == viewpoint:
        y = int(height * 0.72)
    else:
        y = int(height * 0.28)
    return x, y
def _animate_flights(
    surface: pygame.Surface,
    flights: list[_Flight],
    *,
    draw_frame: Callable[[], None],
    card_size: tuple[int, int],
    hold_ms: int | None = None,
) -> None:
    if not flights:
        return
    if hold_ms is None:
        if ANIM_HOLD_MS <= 0:
            hold_ms = 0
        elif _LIVE['n_slots'] <= 1:
            hold_ms = ANIM_HOLD_MS
        else:
            hold_ms = max(40, ANIM_HOLD_MS // 3)

    if ANIM_MS <= 0:
        draw_frame()
        for flight in flights:
            path = card_image_path(flight.card) if flight.face_up else CARD_BACK
            _blit_card(surface, path, flight.end, card_size)
        _present()
        _wait_step(hold_ms)
        return

    clock = _LIVE['clock']
    anim_ms = ANIM_MS if _LIVE['n_slots'] <= 1 else max(90, ANIM_MS // 2)
    start_ticks = pygame.time.get_ticks()

    while True:
        if not _pump_events():
            return

        elapsed = pygame.time.get_ticks() - start_ticks
        t = min(1.0, elapsed / anim_ms)
        eased = _ease_out_cubic(t)

        draw_frame()
        for flight in flights:
            path = card_image_path(flight.card) if flight.face_up else CARD_BACK
            pos = _lerp(flight.start, flight.end, eased)
            _blit_card(surface, path, pos, card_size)
        _present()

        if clock is not None:
            clock.tick(60)
        if t >= 1.0:
            break

    draw_frame()
    for flight in flights:
        path = card_image_path(flight.card) if flight.face_up else CARD_BACK
        _blit_card(surface, path, flight.end, card_size)
    _present()
    _wait_step(hold_ms)
def _missing_cards(before: list[Card], after: list[Card]) -> list[Card]:
    remaining = list(after)
    missing: list[Card] = []
    for card in before:
        if card in remaining:
            remaining.remove(card)
        else:
            missing.append(card)
    return missing
