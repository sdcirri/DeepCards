"""Pygame window, grid layout, and parallel session lifecycle."""

from typing import Any
import math

import pygame

from .state import (
    BORDER_COLOR,
    MAX_PARALLEL_EPISODES,
    STEP_DELAY_MS,
    TABLE_COLOR,
    GameKind,
    _LIVE,
)

def _ensure_pygame() -> None:
    if not pygame.get_init():
        pygame.init()
    if not pygame.font.get_init():
        pygame.font.init()


def _base_pane_size(kind: GameKind) -> tuple[int, int]:
    if kind == 'tressette':
        return (1100, 700)
    if kind == 'scopa':
        return (1000, 700)
    return (900, 700)


def _grid_dims(n: int) -> tuple[int, int]:
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = max(1, math.ceil(n / cols))
    return cols, rows

def _display_budget() -> tuple[int, int]:
    try:
        info = pygame.display.Info()
        current_w = info.current_w or 1280
        current_h = info.current_h or 720
    except pygame.error:
        current_w, current_h = 1280, 720
    # Leave room for desktop panels / window chrome.
    max_w = max(640, int(current_w * 0.92))
    max_h = max(480, int(current_h * 0.88))
    return max_w, max_h


def _pump_events() -> bool:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            close_live_window()
            return False
        if event.type == pygame.VIDEORESIZE and _LIVE['screen'] is not None:
            _relayout_window(event.w, event.h)
            from .parallel import _redraw_parallel_envs_static
            _redraw_parallel_envs_static()
    return _LIVE['screen'] is not None


def _step_delay_ms() -> int:
    if STEP_DELAY_MS <= 0:
        return 0
    n = _LIVE['n_slots']
    if n <= 1:
        return STEP_DELAY_MS
    # Keep the grid watchable as N grows.
    return max(40, STEP_DELAY_MS // max(1, n // 2))


def _wait_step(delay_ms: int | None = None) -> None:
    if delay_ms is None:
        delay_ms = _step_delay_ms()
    if delay_ms <= 0 or _LIVE['screen'] is None:
        return

    deadline = pygame.time.get_ticks() + delay_ms
    clock = _LIVE['clock']
    while pygame.time.get_ticks() < deadline:
        if not _pump_events():
            return
        if clock is not None:
            clock.tick(30)
        else:
            pygame.time.delay(10)

def close_live_window() -> None:
    if _LIVE['screen'] is not None:
        pygame.display.quit()
    _LIVE['screen'] = None
    _LIVE['kind'] = None
    _LIVE['clock'] = None
    _LIVE['font_cache'] = {}
    _LIVE['card_cache'] = {}
    _LIVE['n_slots'] = 1
    _LIVE['cols'] = 1
    _LIVE['rows'] = 1
    _LIVE['pane_size'] = (900, 700)
    _LIVE['scale'] = 1.0
    _LIVE['panes'] = {}
    _LIVE['prev'] = {}
    _LIVE['title'] = ''
    _LIVE['parallel_envs'] = []


def _compute_layout(
    kind: GameKind,
    n_slots: int,
    window_w: int,
    window_h: int,
) -> tuple[int, int, int, int, float]:
    cols, rows = _grid_dims(n_slots)
    base_w, base_h = _base_pane_size(kind)
    pane_w = max(120, window_w // cols)
    pane_h = max(100, window_h // rows)
    scale = min(pane_w / base_w, pane_h / base_h)
    return cols, rows, pane_w, pane_h, scale


def _rebuild_panes(pane_w: int, pane_h: int, n_slots: int) -> dict[int, pygame.Surface]:
    panes = {
        slot: pygame.Surface((pane_w, pane_h)).convert()
        for slot in range(n_slots)
    }
    for pane in panes.values():
        pane.fill(TABLE_COLOR)
    return panes


def _relayout_window(window_w: int, window_h: int) -> None:
    kind = _LIVE['kind']
    n_slots = _LIVE['n_slots']
    if kind is None or _LIVE['screen'] is None:
        return

    window_w = max(320, window_w)
    window_h = max(240, window_h)
    cols, rows, pane_w, pane_h, scale = _compute_layout(kind, n_slots, window_w, window_h)
    screen = pygame.display.set_mode((window_w, window_h), pygame.RESIZABLE)
    _LIVE.update(
        {
            'screen': screen,
            'cols': cols,
            'rows': rows,
            'pane_size': (pane_w, pane_h),
            'scale': scale,
            'panes': _rebuild_panes(pane_w, pane_h, n_slots),
            'card_cache': {},
            'font_cache': {},
        }
    )
def begin_parallel_session(
    kind: GameKind,
    n_slots: int,
    *,
    title: str = '',
) -> None:
    """Open (or recreate) a resizable window split into n_slots episode panes."""
    if n_slots < 1:
        raise ValueError('n_slots must be >= 1')
    if n_slots > MAX_PARALLEL_EPISODES:
        raise ValueError(f'n_slots must be <= {MAX_PARALLEL_EPISODES}')

    close_live_window()
    _ensure_pygame()

    cols, rows = _grid_dims(n_slots)
    base_w, base_h = _base_pane_size(kind)
    max_w, max_h = _display_budget()
    scale = min(max_w / (cols * base_w), max_h / (rows * base_h), 1.0)
    pane_w = max(120, int(base_w * scale))
    pane_h = max(100, int(base_h * scale))
    window_w = pane_w * cols
    window_h = pane_h * rows

    screen = pygame.display.set_mode((window_w, window_h), pygame.RESIZABLE)
    caption = title or f'{kind.title()} — Carte Piacentine'
    if n_slots > 1:
        caption = f'{caption}  [{n_slots} episodes]'
    pygame.display.set_caption(caption)

    _LIVE.update(
        {
            'screen': screen,
            'kind': kind,
            'clock': pygame.time.Clock(),
            'font_cache': {},
            'card_cache': {},
            'n_slots': n_slots,
            'cols': cols,
            'rows': rows,
            'pane_size': (pane_w, pane_h),
            'scale': scale,
            'panes': _rebuild_panes(pane_w, pane_h, n_slots),
            'prev': {},
            'title': caption,
            'parallel_envs': [],
        }
    )
    _present()


def end_parallel_session() -> None:
    close_live_window()


def set_parallel_envs(envs: list[Any]) -> None:
    """Raw envs used to redraw panes after a window resize."""
    _LIVE['parallel_envs'] = list(envs)


def _pane_origin(slot: int) -> tuple[int, int]:
    cols = _LIVE['cols']
    pane_w, pane_h = _LIVE['pane_size']
    return (slot % cols) * pane_w, (slot // cols) * pane_h


def _present() -> None:
    screen = _LIVE['screen']
    if screen is None:
        return
    screen.fill(BORDER_COLOR)
    for slot, pane in _LIVE['panes'].items():
        origin = _pane_origin(slot)
        screen.blit(pane, origin)
        rect = pygame.Rect(origin, _LIVE['pane_size'])
        pygame.draw.rect(screen, BORDER_COLOR, rect, width=max(1, int(2 * _LIVE['scale'])))
    pygame.display.flip()


def _get_pane(kind: GameKind, slot: int) -> pygame.Surface:
    if (
        _LIVE['screen'] is None
        or _LIVE['kind'] != kind
        or not pygame.display.get_init()
        or slot not in _LIVE['panes']
    ):
        # Single-game fallback session.
        begin_parallel_session(kind, max(slot + 1, 1))
    return _LIVE['panes'][slot]
