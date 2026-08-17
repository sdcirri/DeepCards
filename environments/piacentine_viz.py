"""
Pygame table renderer using Carte Piacentine assets.

Supports a single table or a grid of up to MAX_PARALLEL_EPISODES panes
for watching many episodes of one challenge at once.

Assets from: https://deerlike.itch.io/piacentine-cards
"""

from typing import Any, Callable, Literal
from dataclasses import dataclass
from pathlib import Path
import math

import pygame

from games.deck import Card

from environments.briscola_env import Briscola2PEnv
from environments.cards_env import AgentId, LeadPlay
from environments.scopa_env import Scopa2PEnv
from environments.tressette_env import Tressette2PEnv

ASSETS_DIR = Path(__file__).resolve().parent.parent / 'assets' / 'CartePiacentineITA'
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
    'panes': {},  # slot -> Surface
    'prev': {},  # slot -> snapshot
    'title': '',
    'parallel_envs': [],  # raw envs for resize redraw
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


def _card_size(hand_count: int, scale: float = 1.0) -> tuple[int, int]:
    if hand_count <= 3:
        base = (90, 160)
    elif hand_count <= 6:
        base = (72, 128)
    else:
        base = (58, 103)
    return max(12, int(base[0] * scale)), max(20, int(base[1] * scale))


def _load_card_surface(path: Path, size: tuple[int, int]) -> pygame.Surface:
    cache = _LIVE['card_cache']
    key = (str(path), size)
    surface = cache.get(key)
    if surface is None:
        image = pygame.image.load(str(path)).convert_alpha()
        surface = pygame.transform.smoothscale(image, size)
        cache[key] = surface
    return surface


def _fan_xs(count: int, center: int, gap: int) -> list[int]:
    if count <= 0:
        return []
    if count == 1:
        return [center]
    start = center - gap * (count - 1) // 2
    return [start + gap * index for index in range(count)]


def _blit_card(
    surface: pygame.Surface,
    path: Path,
    center: tuple[int, int],
    size: tuple[int, int],
    *,
    alpha: int = 255,
) -> None:
    card = _load_card_surface(path, size)
    if alpha < 255:
        card = card.copy()
        card.fill((255, 255, 255, alpha), special_flags=pygame.BLEND_RGBA_MULT)
    rect = card.get_rect(center=center)
    surface.blit(card, rect)


def _font(size: int = 22) -> pygame.font.Font:
    size = max(9, size)
    cache = _LIVE['font_cache']
    font = cache.get(size)
    if font is None:
        font = pygame.font.SysFont('dejavusans', size, bold=True)
        cache[size] = font
    return font


def _label_size(base: int) -> int:
    return max(9, int(base * _LIVE['scale']))


def _draw_label(
    surface: pygame.Surface,
    text: str,
    center: tuple[int, int],
    *,
    size: int = 22,
) -> None:
    rendered = _font(size).render(text, True, WHITE)
    rect = rendered.get_rect(center=center)
    surface.blit(rendered, rect)


def _draw_label_left(
    surface: pygame.Surface,
    text: str,
    topleft: tuple[int, int],
    *,
    size: int = 22,
) -> None:
    rendered = _font(size).render(text, True, WHITE)
    surface.blit(rendered, topleft)


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


def _hand_gap(card_width: int, count: int) -> int:
    scale = _LIVE['scale']
    if count <= 3:
        return card_width + max(4, int(18 * scale))
    if count <= 6:
        return card_width + max(3, int(10 * scale))
    return max(card_width - max(4, int(8 * scale)), max(16, int(42 * scale)))


def _hand_y(surface: pygame.Surface, *, bottom: bool) -> int:
    height = surface.get_height()
    return int(height * 0.84) if bottom else int(height * 0.14)


def _trick_pos(
    surface: pygame.Surface,
    agent: AgentId,
    *,
    viewpoint: AgentId,
) -> tuple[int, int]:
    width = surface.get_width()
    height = surface.get_height()
    center_x = int(width * 0.62)
    offset = max(12, int(40 * _LIVE['scale']))
    x = center_x - offset if agent == viewpoint else center_x + offset
    return x, int(height * 0.48)


def _deck_pos(surface: pygame.Surface, *, kind: GameKind) -> tuple[int, int]:
    width = surface.get_width()
    height = surface.get_height()
    if kind == 'briscola':
        return int(width * 0.33), int(height * 0.48)
    if kind == 'scopa':
        return int(width * 0.18), int(height * 0.48)
    return int(width * 0.31), int(height * 0.48)


def _hand_slot_pos(
    surface: pygame.Surface,
    *,
    hand_count: int,
    index: int,
    is_viewer: bool,
) -> tuple[int, int]:
    scale = _LIVE['scale']
    count = max(hand_count, 1)
    card_w, _ = _card_size(count, scale)
    gap = _hand_gap(card_w, count)
    xs = _fan_xs(hand_count, center=surface.get_width() // 2, gap=gap)
    if not xs:
        return surface.get_width() // 2, _hand_y(surface, bottom=is_viewer)
    return xs[min(max(index, 0), len(xs) - 1)], _hand_y(surface, bottom=is_viewer)


def _hand_card_pos(
    surface: pygame.Surface,
    *,
    hand: list[Card],
    card: Card,
    is_viewer: bool,
) -> tuple[int, int]:
    try:
        index = hand.index(card)
    except ValueError:
        index = len(hand)
    return _hand_slot_pos(
        surface,
        hand_count=max(len(hand), index + 1),
        index=index,
        is_viewer=is_viewer,
    )


def _draw_hands(
    surface: pygame.Surface,
    *,
    viewer_hand: list[Card],
    opponent_hand: list[Card],
    card_size: tuple[int, int],
    gap: int,
) -> None:
    width = surface.get_width()

    for x, card in zip(
        _fan_xs(len(opponent_hand), center=width // 2, gap=gap),
        opponent_hand,
        strict=True,
    ):
        _blit_card(
            surface,
            card_image_path(card),
            (x, _hand_y(surface, bottom=False)),
            card_size,
            alpha=OPPONENT_CARD_ALPHA,
        )

    for x, card in zip(
        _fan_xs(len(viewer_hand), center=width // 2, gap=gap),
        viewer_hand,
        strict=True,
    ):
        _blit_card(
            surface,
            card_image_path(card),
            (x, _hand_y(surface, bottom=True)),
            card_size,
        )


def _draw_trick_cards(
    surface: pygame.Surface,
    cards: list[tuple[AgentId, Card]],
    *,
    viewpoint: AgentId,
    card_size: tuple[int, int],
) -> None:
    if not cards:
        return

    width = surface.get_width()
    height = surface.get_height()
    for agent, card in cards:
        _blit_card(
            surface,
            card_image_path(card),
            _trick_pos(surface, agent, viewpoint=viewpoint),
            card_size,
        )
    _draw_label(
        surface,
        'trick',
        (int(width * 0.62), int(height * 0.63)),
        size=_label_size(18),
    )


def _agent_label(env: Any, agent: AgentId) -> str:
    names = getattr(env, 'display_names', None)
    if isinstance(names, dict) and names.get(agent):
        return str(names[agent])
    return agent


def _draw_scores(
    surface: pygame.Surface,
    *,
    env: Any,
    viewpoint: AgentId,
    opponent: AgentId,
    scores: dict[AgentId, int],
    turn: AgentId,
    score_label: str,
    slot: int,
) -> None:
    height = surface.get_height()
    pad = max(4, int(16 * _LIVE['scale']))
    top_name = _agent_label(env, opponent)
    bottom_name = _agent_label(env, viewpoint)
    n_slots = _LIVE['n_slots']

    if n_slots > 1:
        _draw_label_left(surface, f'#{slot + 1}', (pad, pad), size=_label_size(16))
        name_y = pad + _label_size(18)
    else:
        name_y = pad

    _draw_label_left(surface, top_name, (pad, name_y), size=_label_size(22))
    _draw_label_left(
        surface,
        f'{scores[opponent]} {score_label}',
        (pad, name_y + _label_size(24)),
        size=_label_size(18),
    )

    _draw_label_left(
        surface,
        bottom_name,
        (pad, height - pad - _label_size(44)),
        size=_label_size(22),
    )
    _draw_label_left(
        surface,
        f'{scores[viewpoint]} {score_label}  ·  turn={turn}',
        (pad, height - pad - _label_size(20)),
        size=_label_size(18),
    )


def _ease_out_cubic(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3


def _lerp(a: tuple[int, int], b: tuple[int, int], t: float) -> tuple[int, int]:
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
    )


def _slot_prev(slot: int) -> dict[str, Any] | None:
    return _LIVE['prev'].get(slot)


def _set_slot_prev(slot: int, snapshot: dict[str, Any] | None) -> None:
    if snapshot is None:
        _LIVE['prev'].pop(slot, None)
    else:
        _LIVE['prev'][slot] = snapshot


def _detect_play_flight(
    surface: pygame.Surface,
    *,
    slot: int,
    viewpoint: AgentId,
    hands: dict[AgentId, list[Card]],
    lead_play: LeadPlay | None,
    pile_size: int,
) -> _Flight | None:
    prev = _slot_prev(slot)
    if prev is not None and prev.get('kind') != _LIVE['kind']:
        prev = None
        _set_slot_prev(slot, None)

    if prev is not None and pile_size > prev.get('pile_size', pile_size):
        prev = None
        _set_slot_prev(slot, None)

    prev_lead: LeadPlay | None = None if prev is None else prev['lead']
    prev_hands: dict[AgentId, list[Card]] | None = None if prev is None else prev['hands']

    if lead_play is not None and (prev_lead is None or prev_lead.card != lead_play.card):
        agent = lead_play.agent
        card = lead_play.card
        if prev_hands is not None and card in prev_hands[agent]:
            source_hand = prev_hands[agent]
        else:
            source_hand = [*hands[agent], card]
        start = _hand_card_pos(
            surface,
            hand=source_hand,
            card=card,
            is_viewer=(agent == viewpoint),
        )
        end = _trick_pos(surface, agent, viewpoint=viewpoint)
        return _Flight(card=card, agent=agent, start=start, end=end)

    if prev_lead is not None and lead_play is None and prev_hands is not None:
        follower: AgentId = 'p2' if prev_lead.agent == 'p1' else 'p1'
        missing = [card for card in prev_hands[follower] if card not in hands[follower]]
        if not missing:
            return None
        card = missing[0]
        start = _hand_card_pos(
            surface,
            hand=prev_hands[follower],
            card=card,
            is_viewer=(follower == viewpoint),
        )
        end = _trick_pos(surface, follower, viewpoint=viewpoint)
        return _Flight(card=card, agent=follower, start=start, end=end)

    return None


def _new_cards(before: list[Card], after: list[Card]) -> list[Card]:
    remaining = list(before)
    gained: list[Card] = []
    for card in after:
        if card in remaining:
            remaining.remove(card)
        else:
            gained.append(card)
    return gained


def _detect_draw_flights(
    surface: pygame.Surface,
    *,
    slot: int,
    kind: GameKind,
    viewpoint: AgentId,
    hands: dict[AgentId, list[Card]],
    lead_play: LeadPlay | None,
    pile_size: int,
    winner: AgentId,
) -> list[_Flight]:
    prev = _slot_prev(slot)
    if prev is None or prev.get('kind') != kind:
        return []
    if lead_play is not None or prev.get('lead') is None:
        return []
    if pile_size >= prev.get('pile_size', pile_size):
        return []

    prev_hands: dict[AgentId, list[Card]] = prev['hands']
    loser: AgentId = 'p2' if winner == 'p1' else 'p1'
    deck = _deck_pos(surface, kind=kind)
    flights: list[_Flight] = []

    for agent in (winner, loser):
        gained = _new_cards(prev_hands[agent], hands[agent])
        if not gained:
            continue
        card = gained[0]
        end = _hand_card_pos(
            surface,
            hand=hands[agent],
            card=card,
            is_viewer=(agent == viewpoint),
        )
        face_up = kind == 'tressette' or agent == viewpoint
        flights.append(_Flight(card=card, agent=agent, start=deck, end=end, face_up=face_up))

    return flights


def _pre_draw_hands(
    *,
    hands: dict[AgentId, list[Card]],
    draw_flights: list[_Flight],
) -> dict[AgentId, list[Card]]:
    drawn = {flight.card for flight in draw_flights}
    return {
        agent: [card for card in cards if card not in drawn]
        for agent, cards in hands.items()
    }


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


def _snapshot(
    *,
    slot: int,
    viewpoint: AgentId,
    hands: dict[AgentId, list[Card]],
    lead_play: LeadPlay | None,
    pile_size: int,
) -> None:
    _set_slot_prev(
        slot,
        {
            'kind': _LIVE['kind'],
            'viewpoint': viewpoint,
            'hands': {
                'p1': list(hands['p1']),
                'p2': list(hands['p2']),
            },
            'lead': lead_play,
            'pile_size': pile_size,
        },
    )


def _render_table(
    env: Briscola2PEnv | Tressette2PEnv,
    *,
    kind: GameKind,
    viewpoint: AgentId,
    score_label: str,
    draw_center: Callable[[pygame.Surface, tuple[int, int]], None],
) -> pygame.Surface:
    slot = int(getattr(env, 'render_slot', 0))
    surface = _get_pane(kind, slot)
    scale = _LIVE['scale']
    opponent: AgentId = 'p2' if viewpoint == 'p1' else 'p1'
    hands = {
        'p1': list(env.hands['p1'].cards),
        'p2': list(env.hands['p2'].cards),
    }
    pile_size = len(env.pile)

    play_flight = _detect_play_flight(
        surface,
        slot=slot,
        viewpoint=viewpoint,
        hands=hands,
        lead_play=env.lead_play,
        pile_size=pile_size,
    )
    draw_flights = _detect_draw_flights(
        surface,
        slot=slot,
        kind=kind,
        viewpoint=viewpoint,
        hands=hands,
        lead_play=env.lead_play,
        pile_size=pile_size,
        winner=env.agent_selection,
    )

    shown_hands = _pre_draw_hands(hands=hands, draw_flights=draw_flights) if draw_flights else {
        'p1': list(hands['p1']),
        'p2': list(hands['p2']),
    }
    trick_cards: list[tuple[AgentId, Card]] = []
    prev = _slot_prev(slot)
    if env.lead_play is not None:
        trick_cards = [(env.lead_play.agent, env.lead_play.card)]
    elif play_flight is not None and prev is not None and prev['lead'] is not None:
        prev_lead = prev['lead']
        trick_cards = [(prev_lead.agent, prev_lead.card)]

    def paint(active_trick: list[tuple[AgentId, Card]] | None = None) -> None:
        viewer_hand = shown_hands[viewpoint]
        opponent_hand = shown_hands[opponent]
        hand_count = max(len(viewer_hand), len(opponent_hand), 1)
        card_size = _card_size(hand_count, scale)
        gap = _hand_gap(card_size[0], hand_count)

        surface.fill(TABLE_COLOR)
        _draw_hands(
            surface,
            viewer_hand=viewer_hand,
            opponent_hand=opponent_hand,
            card_size=card_size,
            gap=gap,
        )
        draw_center(surface, card_size)
        _draw_trick_cards(
            surface,
            trick_cards if active_trick is None else active_trick,
            viewpoint=viewpoint,
            card_size=card_size,
        )
        _draw_scores(
            surface,
            env=env,
            viewpoint=viewpoint,
            opponent=opponent,
            scores=env.scores,
            turn=env.agent_selection,
            score_label=score_label,
            slot=slot,
        )

    def card_size_for(hand_count: int) -> tuple[int, int]:
        return _card_size(max(hand_count, 1), scale)

    if play_flight is not None:
        settled: list[tuple[AgentId, Card]] = []
        if env.lead_play is None and prev is not None and prev['lead'] is not None:
            prev_lead = prev['lead']
            settled = [(prev_lead.agent, prev_lead.card)]

        def play_frame() -> None:
            paint(settled)

        size = card_size_for(max(len(shown_hands[viewpoint]), len(shown_hands[opponent]), 1))
        _animate_flights(
            surface,
            [play_flight],
            draw_frame=play_frame,
            card_size=size,
        )
        trick_cards = [*settled, (play_flight.agent, play_flight.card)]

        if env.lead_play is None and len(trick_cards) == 2:
            winner: AgentId = env.agent_selection
            claim_flights = [
                _Flight(
                    card=card,
                    agent=agent,
                    start=_trick_pos(surface, agent, viewpoint=viewpoint),
                    end=_won_pile_pos(surface, winner, viewpoint=viewpoint, index=index),
                    face_up=True,
                )
                for index, (agent, card) in enumerate(trick_cards)
            ]

            def claim_frame() -> None:
                paint([])

            _animate_flights(
                surface,
                claim_flights,
                draw_frame=claim_frame,
                card_size=size,
            )
            trick_cards = []

    for draw_flight in draw_flights:
        hand_count = max(len(shown_hands[viewpoint]), len(shown_hands[opponent]), 1)

        def draw_frame() -> None:
            paint(trick_cards)

        _animate_flights(
            surface,
            [draw_flight],
            draw_frame=draw_frame,
            card_size=card_size_for(hand_count),
        )
        shown_hands[draw_flight.agent].append(draw_flight.card)

    shown_hands['p1'] = list(hands['p1'])
    shown_hands['p2'] = list(hands['p2'])
    final_trick: list[tuple[AgentId, Card]] = []
    if env.lead_play is not None:
        final_trick = [(env.lead_play.agent, env.lead_play.card)]
    paint(final_trick)
    _present()
    _wait_step()
    _snapshot(
        slot=slot,
        viewpoint=viewpoint,
        hands=hands,
        lead_play=env.lead_play,
        pile_size=pile_size,
    )
    return surface


def _briscola_draw_center(env: Briscola2PEnv) -> Callable[[pygame.Surface, tuple[int, int]], None]:
    def draw_center(surface: pygame.Surface, card_size: tuple[int, int]) -> None:
        width = surface.get_width()
        height = surface.get_height()
        if env.briscola is not None:
            _blit_card(
                surface,
                card_image_path(env.briscola),
                (int(width * 0.24), int(height * 0.48)),
                card_size,
            )
        if env.pile:
            _blit_card(surface, CARD_BACK, (int(width * 0.32), int(height * 0.47)), card_size)
            _blit_card(surface, CARD_BACK, (int(width * 0.33), int(height * 0.48)), card_size)

        if env.briscola is not None or env.pile:
            label = 'briscola'
            if env.pile:
                label = f'briscola / deck ({len(env.pile)})'
            _draw_label(
                surface,
                label,
                (int(width * 0.29), int(height * 0.63)),
                size=_label_size(18),
            )

    return draw_center


def _tressette_draw_center(env: Tressette2PEnv) -> Callable[[pygame.Surface, tuple[int, int]], None]:
    def draw_center(surface: pygame.Surface, card_size: tuple[int, int]) -> None:
        width = surface.get_width()
        height = surface.get_height()
        if env.pile:
            _blit_card(surface, CARD_BACK, (int(width * 0.30), int(height * 0.47)), card_size)
            _blit_card(surface, CARD_BACK, (int(width * 0.31), int(height * 0.48)), card_size)
            _draw_label(
                surface,
                f'deck ({len(env.pile)})',
                (int(width * 0.31), int(height * 0.63)),
                size=_label_size(18),
            )

    return draw_center


def render_briscola_table(
    env: Briscola2PEnv,
    *,
    viewpoint: AgentId = 'p1',
    show: bool = True,
) -> pygame.Surface | None:
    if not show:
        return None
    return _render_table(
        env,
        kind='briscola',
        viewpoint=viewpoint,
        score_label='pts',
        draw_center=_briscola_draw_center(env),
    )


def render_tressette_table(
    env: Tressette2PEnv,
    *,
    viewpoint: AgentId = 'p1',
    show: bool = True,
) -> pygame.Surface | None:
    if not show:
        return None
    return _render_table(
        env,
        kind='tressette',
        viewpoint=viewpoint,
        score_label='thirds',
        draw_center=_tressette_draw_center(env),
    )


def _scopa_table_card_pos(
    surface: pygame.Surface,
    *,
    index: int,
    count: int,
) -> tuple[int, int]:
    width = surface.get_width()
    height = surface.get_height()
    scale = _LIVE['scale']
    card_w = _card_size(max(count, 3), scale)[0]
    gap = max(card_w - max(4, int(10 * scale)), max(20, int(48 * scale)))
    center_x = int(width * 0.58)
    xs = _fan_xs(count, center=center_x, gap=gap)
    return xs[index], int(height * 0.48)


def _draw_scopa_table_cards(
    surface: pygame.Surface,
    table: list[Card],
    *,
    card_size: tuple[int, int],
) -> None:
    if not table:
        return
    width = surface.get_width()
    height = surface.get_height()
    for index, card in enumerate(table):
        _blit_card(
            surface,
            card_image_path(card),
            _scopa_table_card_pos(surface, index=index, count=len(table)),
            card_size,
        )
    _draw_label(
        surface,
        'table',
        (int(width * 0.58), int(height * 0.63)),
        size=_label_size(18),
    )


def _scopa_live_scores(env: Scopa2PEnv) -> dict[AgentId, int]:
    if env.terminations.get('p1') or env.terminations.get('p2'):
        return {'p1': env.scores['p1'], 'p2': env.scores['p2']}
    return {
        'p1': env.hands['p1'].score.scope + env.hands['p1'].score.settebello,
        'p2': env.hands['p2'].score.scope + env.hands['p2'].score.settebello,
    }


def _scopa_snapshot(
    *,
    slot: int,
    viewpoint: AgentId,
    hands: dict[AgentId, list[Card]],
    table: list[Card],
    pile_size: int,
) -> None:
    _set_slot_prev(
        slot,
        {
            'kind': 'scopa',
            'viewpoint': viewpoint,
            'hands': {
                'p1': list(hands['p1']),
                'p2': list(hands['p2']),
            },
            'table': list(table),
            'pile_size': pile_size,
            'lead': None,
        },
    )


def _paint_scopa_state(
    env: Scopa2PEnv,
    *,
    viewpoint: AgentId,
    shown_hands: dict[AgentId, list[Card]] | None = None,
    shown_table: list[Card] | None = None,
) -> pygame.Surface:
    slot = int(getattr(env, 'render_slot', 0))
    surface = _get_pane('scopa', slot)
    scale = _LIVE['scale']
    opponent: AgentId = 'p2' if viewpoint == 'p1' else 'p1'
    hands = shown_hands or {
        'p1': list(env.hands['p1'].cards),
        'p2': list(env.hands['p2'].cards),
    }
    table = list(env.table if shown_table is None else shown_table)
    viewer_hand = hands[viewpoint]
    opponent_hand = hands[opponent]
    # Size by hand, not by table (table can grow large).
    hand_count = max(len(viewer_hand), len(opponent_hand), 1)
    card_size = _card_size(hand_count, scale)
    gap = _hand_gap(card_size[0], hand_count)
    height = surface.get_height()

    surface.fill(TABLE_COLOR)
    _draw_hands(
        surface,
        viewer_hand=viewer_hand,
        opponent_hand=opponent_hand,
        card_size=card_size,
        gap=gap,
    )
    if env.pile:
        deck = _deck_pos(surface, kind='scopa')
        _blit_card(surface, CARD_BACK, deck, card_size)
        _blit_card(
            surface,
            CARD_BACK,
            (deck[0] + max(2, int(4 * scale)), deck[1] + max(2, int(4 * scale))),
            card_size,
        )
        _draw_label(
            surface,
            f'deck ({len(env.pile)})',
            (deck[0], int(height * 0.63)),
            size=_label_size(18),
        )
    _draw_scopa_table_cards(surface, table, card_size=card_size)
    _draw_scores(
        surface,
        env=env,
        viewpoint=viewpoint,
        opponent=opponent,
        scores=_scopa_live_scores(env),
        turn=env.agent_selection,
        score_label='pts',
        slot=slot,
    )
    return surface


def _missing_cards(before: list[Card], after: list[Card]) -> list[Card]:
    remaining = list(after)
    missing: list[Card] = []
    for card in before:
        if card in remaining:
            remaining.remove(card)
        else:
            missing.append(card)
    return missing


@dataclass(slots=True)
class _ScopaAnim:
    hands: dict[AgentId, list[Card]]
    table: list[Card]
    pile_size: int
    anim_hands: dict[AgentId, list[Card]]
    anim_table: list[Card]
    play_flights: list[_Flight]
    claim_flights: list[_Flight]
    deal_flights: list[_Flight]


def _detect_scopa_anim(
    env: Scopa2PEnv,
    surface: pygame.Surface,
    *,
    viewpoint: AgentId,
) -> _ScopaAnim:
    slot = int(getattr(env, 'render_slot', 0))
    hands = {
        'p1': list(env.hands['p1'].cards),
        'p2': list(env.hands['p2'].cards),
    }
    table = list(env.table)
    pile_size = len(env.pile)
    prev = _slot_prev(slot)
    if prev is not None and prev.get('kind') != 'scopa':
        prev = None
        _set_slot_prev(slot, None)

    play_flights: list[_Flight] = []
    claim_flights: list[_Flight] = []
    deal_flights: list[_Flight] = []
    anim_hands = {
        'p1': list(hands['p1']),
        'p2': list(hands['p2']),
    }
    anim_table = list(table)

    if prev is not None:
        prev_hands: dict[AgentId, list[Card]] = prev['hands']
        prev_table: list[Card] = list(prev.get('table', []))
        prev_pile = int(prev.get('pile_size', pile_size))

        actor: AgentId | None = None
        played: Card | None = None
        for agent in ('p1', 'p2'):
            gained = set(_new_cards(prev_hands[agent], hands[agent]))
            post_play_hand = [c for c in hands[agent] if c not in gained]
            lost = _missing_cards(prev_hands[agent], post_play_hand)
            if len(lost) == 1:
                actor = agent
                played = lost[0]
                break

        if actor is not None and played is not None:
            start = _hand_card_pos(
                surface,
                hand=prev_hands[actor],
                card=played,
                is_viewer=(actor == viewpoint),
            )
            anim_hands = {
                'p1': list(prev_hands['p1']),
                'p2': list(prev_hands['p2']),
            }
            anim_hands[actor] = [c for c in prev_hands[actor] if c != played]
            anim_table = list(prev_table)

            if played in table and played not in prev_table:
                play_flights.append(
                    _Flight(
                        card=played,
                        agent=actor,
                        start=start,
                        end=_scopa_table_card_pos(
                            surface,
                            index=table.index(played),
                            count=max(len(table), 1),
                        ),
                    )
                )
            else:
                play_flights.append(
                    _Flight(
                        card=played,
                        agent=actor,
                        start=start,
                        end=_won_pile_pos(surface, actor, viewpoint=viewpoint, index=0),
                    )
                )
                for index, taken_card in enumerate(_missing_cards(prev_table, table)):
                    claim_flights.append(
                        _Flight(
                            card=taken_card,
                            agent=actor,
                            start=_scopa_table_card_pos(
                                surface,
                                index=prev_table.index(taken_card),
                                count=max(len(prev_table), 1),
                            ),
                            end=_won_pile_pos(
                                surface,
                                actor,
                                viewpoint=viewpoint,
                                index=index + 1,
                            ),
                        )
                    )

        if prev_pile - pile_size >= 6:
            for agent in ('p1', 'p2'):
                for card in _new_cards(prev_hands[agent], hands[agent]):
                    deal_flights.append(
                        _Flight(
                            card=card,
                            agent=agent,
                            start=_deck_pos(surface, kind='scopa'),
                            end=_hand_card_pos(
                                surface,
                                hand=hands[agent],
                                card=card,
                                is_viewer=(agent == viewpoint),
                            ),
                            face_up=(agent == viewpoint),
                        )
                    )

    return _ScopaAnim(
        hands=hands,
        table=table,
        pile_size=pile_size,
        anim_hands=anim_hands,
        anim_table=anim_table,
        play_flights=play_flights,
        claim_flights=claim_flights,
        deal_flights=deal_flights,
    )


def render_scopa_table(
    env: Scopa2PEnv,
    *,
    viewpoint: AgentId = 'p1',
    show: bool = True,
) -> pygame.Surface | None:
    """
    Scopa table: 3-card hands, face-up table, stock.
    Cards are NOT redrawn after every play — only when both hands are empty
    and three more are dealt to each player from the stock.
    """
    if not show:
        return None

    slot = int(getattr(env, 'render_slot', 0))
    surface = _get_pane('scopa', slot)
    scale = _LIVE['scale']
    anim = _detect_scopa_anim(env, surface, viewpoint=viewpoint)
    hands = anim.hands
    table = anim.table
    pile_size = anim.pile_size
    play_flights = anim.play_flights
    claim_flights = anim.claim_flights
    deal_flights = anim.deal_flights
    anim_hands = anim.anim_hands
    anim_table = anim.anim_table

    hand_count = max(len(hands['p1']), len(hands['p2']), 1)
    card_size = _card_size(hand_count, scale)

    if play_flights or claim_flights:
        def play_frame() -> None:
            _paint_scopa_state(
                env,
                viewpoint=viewpoint,
                shown_hands=anim_hands,
                shown_table=anim_table,
            )

        _animate_flights(
            surface,
            [*play_flights, *claim_flights],
            draw_frame=play_frame,
            card_size=card_size,
        )

    if deal_flights:
        pre_deal_hands = {
            agent: [c for c in hands[agent] if c not in {f.card for f in deal_flights if f.agent == agent}]
            for agent in ('p1', 'p2')
        }

        def deal_frame() -> None:
            _paint_scopa_state(
                env,
                viewpoint=viewpoint,
                shown_hands=pre_deal_hands,
                shown_table=table,
            )

        _animate_flights(
            surface,
            deal_flights,
            draw_frame=deal_frame,
            card_size=card_size,
        )

    surface = _paint_scopa_state(env, viewpoint=viewpoint)
    _scopa_snapshot(
        slot=slot,
        viewpoint=viewpoint,
        hands=hands,
        table=table,
        pile_size=pile_size,
    )
    _present()
    _wait_step()
    return surface


@dataclass(slots=True)
class _ScopaPaneFrame:
    env: Scopa2PEnv
    slot: int
    surface: pygame.Surface
    viewpoint: AgentId
    hands: dict[AgentId, list[Card]]
    table: list[Card]
    pile_size: int
    shown_hands: dict[AgentId, list[Card]]
    shown_table: list[Card]
    play_flights: list[_Flight]
    claim_flights: list[_Flight]
    deal_flights: list[_Flight]


def _prepare_scopa_pane_frame(env: Scopa2PEnv) -> _ScopaPaneFrame:
    viewpoint: AgentId = 'p1'
    slot = int(getattr(env, 'render_slot', 0))
    surface = _get_pane('scopa', slot)
    anim = _detect_scopa_anim(env, surface, viewpoint=viewpoint)
    if anim.deal_flights:
        shown_hands = {
            agent: [
                c for c in anim.hands[agent]
                if c not in {f.card for f in anim.deal_flights if f.agent == agent}
            ]
            for agent in ('p1', 'p2')
        }
    else:
        shown_hands = {
            'p1': list(anim.anim_hands['p1']),
            'p2': list(anim.anim_hands['p2']),
        }
    return _ScopaPaneFrame(
        env=env,
        slot=slot,
        surface=surface,
        viewpoint=viewpoint,
        hands=anim.hands,
        table=anim.table,
        pile_size=anim.pile_size,
        shown_hands=shown_hands,
        shown_table=list(anim.anim_table),
        play_flights=anim.play_flights,
        claim_flights=anim.claim_flights,
        deal_flights=anim.deal_flights,
    )


def _paint_scopa_pane_frame(
    frame: _ScopaPaneFrame,
    *,
    flying: list[_Flight] | None = None,
    t: float = 1.0,
) -> None:
    _paint_scopa_state(
        frame.env,
        viewpoint=frame.viewpoint,
        shown_hands=frame.shown_hands,
        shown_table=frame.shown_table,
    )
    if flying:
        scale = _LIVE['scale']
        hand_count = max(
            len(frame.shown_hands['p1']),
            len(frame.shown_hands['p2']),
            1,
        )
        card_size = _card_size(hand_count, scale)
        eased = _ease_out_cubic(t)
        for flight in flying:
            path = card_image_path(flight.card) if flight.face_up else CARD_BACK
            pos = _lerp(flight.start, flight.end, eased)
            _blit_card(frame.surface, path, pos, card_size)


def _animate_scopa_parallel_phase(
    frames: list[_ScopaPaneFrame],
    flights_of: Callable[[_ScopaPaneFrame], list[_Flight]],
) -> None:
    active = [(frame, flights_of(frame)) for frame in frames]
    if not any(flights for _, flights in active):
        return

    if ANIM_MS <= 0:
        for frame, flights in active:
            _paint_scopa_pane_frame(frame, flying=flights, t=1.0)
        _present()
        return

    clock = _LIVE['clock']
    anim_ms = max(90, ANIM_MS // 2) if _LIVE['n_slots'] > 1 else ANIM_MS
    start_ticks = pygame.time.get_ticks()
    while True:
        if not _pump_events():
            return
        elapsed = pygame.time.get_ticks() - start_ticks
        t = min(1.0, elapsed / anim_ms)
        for frame, flights in active:
            _paint_scopa_pane_frame(frame, flying=flights, t=t)
        _present()
        if clock is not None:
            clock.tick(60)
        if t >= 1.0:
            break


def _render_scopa_parallel_round(envs: list[Any]) -> None:
    frames = [_prepare_scopa_pane_frame(env) for env in envs]

    # 1) Play + capture (synced across panes).
    _animate_scopa_parallel_phase(
        frames,
        lambda frame: [*frame.play_flights, *frame.claim_flights],
    )
    for frame in frames:
        if frame.play_flights or frame.claim_flights:
            frame.shown_table = list(frame.table)
            # Keep post-play / pre-deal hands already set in prepare.
            if not frame.deal_flights:
                frame.shown_hands = {
                    'p1': list(frame.hands['p1']),
                    'p2': list(frame.hands['p2']),
                }

    # 2) Redeal from stock when both hands emptied.
    _animate_scopa_parallel_phase(frames, lambda frame: frame.deal_flights)

    for frame in frames:
        frame.shown_hands = {
            'p1': list(frame.hands['p1']),
            'p2': list(frame.hands['p2']),
        }
        frame.shown_table = list(frame.table)
        _paint_scopa_pane_frame(frame)
        _scopa_snapshot(
            slot=frame.slot,
            viewpoint=frame.viewpoint,
            hands=frame.hands,
            table=frame.table,
            pile_size=frame.pile_size,
        )

    _present()
    _wait_step()


@dataclass(slots=True)
class _PaneFrame:
    env: Any
    slot: int
    surface: pygame.Surface
    viewpoint: AgentId
    opponent: AgentId
    score_label: str
    draw_center: Callable[[pygame.Surface, tuple[int, int]], None]
    hands: dict[AgentId, list[Card]]
    shown_hands: dict[AgentId, list[Card]]
    trick_cards: list[tuple[AgentId, Card]]
    play_flight: _Flight | None
    claim_flights: list[_Flight]
    draw_flights: list[_Flight]


def _score_label_for(kind: GameKind) -> str:
    if kind == 'briscola':
        return 'pts'
    if kind == 'scopa':
        return 'pts'
    return 'thirds'


def _prepare_pane_frame(env: Any, kind: GameKind) -> _PaneFrame:
    slot = int(getattr(env, 'render_slot', 0))
    surface = _get_pane(kind, slot)
    viewpoint: AgentId = 'p1'
    opponent: AgentId = 'p2'
    hands = {
        'p1': list(env.hands['p1'].cards),
        'p2': list(env.hands['p2'].cards),
    }
    pile_size = len(env.pile)
    play_flight = _detect_play_flight(
        surface,
        slot=slot,
        viewpoint=viewpoint,
        hands=hands,
        lead_play=env.lead_play,
        pile_size=pile_size,
    )
    draw_flights = _detect_draw_flights(
        surface,
        slot=slot,
        kind=kind,
        viewpoint=viewpoint,
        hands=hands,
        lead_play=env.lead_play,
        pile_size=pile_size,
        winner=env.agent_selection,
    )
    shown_hands = _pre_draw_hands(hands=hands, draw_flights=draw_flights) if draw_flights else {
        'p1': list(hands['p1']),
        'p2': list(hands['p2']),
    }
    prev = _slot_prev(slot)
    trick_cards: list[tuple[AgentId, Card]] = []
    if env.lead_play is not None:
        trick_cards = [(env.lead_play.agent, env.lead_play.card)]
    elif play_flight is not None and prev is not None and prev['lead'] is not None:
        prev_lead = prev['lead']
        trick_cards = [(prev_lead.agent, prev_lead.card)]

    claim_flights: list[_Flight] = []
    if play_flight is not None and env.lead_play is None and prev is not None and prev['lead'] is not None:
        settled = [(prev['lead'].agent, prev['lead'].card), (play_flight.agent, play_flight.card)]
        winner: AgentId = env.agent_selection
        claim_flights = [
            _Flight(
                card=card,
                agent=agent,
                start=_trick_pos(surface, agent, viewpoint=viewpoint),
                end=_won_pile_pos(surface, winner, viewpoint=viewpoint, index=index),
                face_up=True,
            )
            for index, (agent, card) in enumerate(settled)
        ]

    if kind == 'briscola':
        draw_center = _briscola_draw_center(env)
    else:
        draw_center = _tressette_draw_center(env)

    return _PaneFrame(
        env=env,
        slot=slot,
        surface=surface,
        viewpoint=viewpoint,
        opponent=opponent,
        score_label=_score_label_for(kind),
        draw_center=draw_center,
        hands=hands,
        shown_hands=shown_hands,
        trick_cards=trick_cards,
        play_flight=play_flight,
        claim_flights=claim_flights,
        draw_flights=draw_flights,
    )


def _paint_pane_frame(
    frame: _PaneFrame,
    *,
    trick_cards: list[tuple[AgentId, Card]] | None = None,
    flying: list[_Flight] | None = None,
    t: float = 1.0,
) -> None:
    scale = _LIVE['scale']
    viewer_hand = frame.shown_hands[frame.viewpoint]
    opponent_hand = frame.shown_hands[frame.opponent]
    hand_count = max(len(viewer_hand), len(opponent_hand), 1)
    card_size = _card_size(hand_count, scale)
    gap = _hand_gap(card_size[0], hand_count)
    trick = frame.trick_cards if trick_cards is None else trick_cards

    frame.surface.fill(TABLE_COLOR)
    _draw_hands(
        frame.surface,
        viewer_hand=viewer_hand,
        opponent_hand=opponent_hand,
        card_size=card_size,
        gap=gap,
    )
    frame.draw_center(frame.surface, card_size)
    _draw_trick_cards(
        frame.surface,
        trick,
        viewpoint=frame.viewpoint,
        card_size=card_size,
    )
    _draw_scores(
        frame.surface,
        env=frame.env,
        viewpoint=frame.viewpoint,
        opponent=frame.opponent,
        scores=frame.env.scores,
        turn=frame.env.agent_selection,
        score_label=frame.score_label,
        slot=frame.slot,
    )
    if flying:
        eased = _ease_out_cubic(t)
        for flight in flying:
            path = card_image_path(flight.card) if flight.face_up else CARD_BACK
            pos = _lerp(flight.start, flight.end, eased)
            _blit_card(frame.surface, path, pos, card_size)


def _animate_parallel_phase(
    frames: list[_PaneFrame],
    flights_of: Callable[[_PaneFrame], list[_Flight]],
    trick_of: Callable[[_PaneFrame], list[tuple[AgentId, Card]]],
) -> None:
    active = [(frame, flights_of(frame)) for frame in frames]
    if not any(flights for _, flights in active):
        return

    if ANIM_MS <= 0:
        for frame, flights in active:
            _paint_pane_frame(frame, trick_cards=trick_of(frame), flying=flights, t=1.0)
        _present()
        return

    clock = _LIVE['clock']
    anim_ms = max(90, ANIM_MS // 2) if _LIVE['n_slots'] > 1 else ANIM_MS
    start_ticks = pygame.time.get_ticks()
    while True:
        if not _pump_events():
            return
        elapsed = pygame.time.get_ticks() - start_ticks
        t = min(1.0, elapsed / anim_ms)
        for frame, flights in active:
            _paint_pane_frame(frame, trick_cards=trick_of(frame), flying=flights, t=t)
        _present()
        if clock is not None:
            clock.tick(60)
        if t >= 1.0:
            break


def _paint_env_static(env: Any, kind: GameKind) -> None:
    if kind == 'scopa':
        viewpoint: AgentId = 'p1'
        _paint_scopa_state(env, viewpoint=viewpoint)
        _scopa_snapshot(
            slot=int(getattr(env, 'render_slot', 0)),
            viewpoint=viewpoint,
            hands={
                'p1': list(env.hands['p1'].cards),
                'p2': list(env.hands['p2'].cards),
            },
            table=list(env.table),
            pile_size=len(env.pile),
        )
        return

    frame = _prepare_pane_frame(env, kind)
    # Static view uses final hands / trick (no in-flight cards).
    frame.shown_hands = {
        'p1': list(frame.hands['p1']),
        'p2': list(frame.hands['p2']),
    }
    final_trick: list[tuple[AgentId, Card]] = []
    if env.lead_play is not None:
        final_trick = [(env.lead_play.agent, env.lead_play.card)]
    frame.trick_cards = final_trick
    _paint_pane_frame(frame)
    _snapshot(
        slot=frame.slot,
        viewpoint=frame.viewpoint,
        hands=frame.hands,
        lead_play=env.lead_play,
        pile_size=len(env.pile),
    )


def _redraw_parallel_envs_static() -> None:
    kind = _LIVE['kind']
    if kind is None:
        return
    for env in _LIVE.get('parallel_envs', []):
        if getattr(env, 'hands', None) is None:
            continue
        _paint_env_static(env, kind)
    _present()


def render_parallel_round(envs: list[Any], kind: GameKind) -> None:
    """Step-synced render: all panes animate their latest action together."""
    if _LIVE['screen'] is None:
        return

    if kind == 'scopa':
        _render_scopa_parallel_round(envs)
        return

    frames = [_prepare_pane_frame(env, kind) for env in envs]

    # 1) Cards played to the trick.
    def play_flights(frame: _PaneFrame) -> list[_Flight]:
        return [frame.play_flight] if frame.play_flight is not None else []

    def play_trick(frame: _PaneFrame) -> list[tuple[AgentId, Card]]:
        if frame.play_flight is None:
            return frame.trick_cards
        # Keep only the prior lead while the new card flies in.
        return [card for card in frame.trick_cards if card[1] != frame.play_flight.card]

    _animate_parallel_phase(frames, play_flights, play_trick)

    for frame in frames:
        if frame.play_flight is not None:
            settled = play_trick(frame)
            frame.trick_cards = [*settled, (frame.play_flight.agent, frame.play_flight.card)]

    # 2) Claim both cards toward the winner.
    _animate_parallel_phase(
        frames,
        lambda frame: frame.claim_flights,
        lambda frame: frame.trick_cards if not frame.claim_flights else [],
    )
    for frame in frames:
        if frame.claim_flights:
            frame.trick_cards = []

    # 3) Draws — winner then loser, still synced across panes.
    max_draws = max((len(frame.draw_flights) for frame in frames), default=0)
    for draw_index in range(max_draws):
        def draw_flights(frame: _PaneFrame, index: int = draw_index) -> list[_Flight]:
            if index < len(frame.draw_flights):
                return [frame.draw_flights[index]]
            return []

        _animate_parallel_phase(
            frames,
            draw_flights,
            lambda frame: frame.trick_cards,
        )
        for frame in frames:
            if draw_index < len(frame.draw_flights):
                flight = frame.draw_flights[draw_index]
                frame.shown_hands[flight.agent].append(flight.card)

    # Final settled frames + snapshots.
    for frame in frames:
        frame.shown_hands = {
            'p1': list(frame.hands['p1']),
            'p2': list(frame.hands['p2']),
        }
        final_trick: list[tuple[AgentId, Card]] = []
        if frame.env.lead_play is not None:
            final_trick = [(frame.env.lead_play.agent, frame.env.lead_play.card)]
        frame.trick_cards = final_trick
        _paint_pane_frame(frame)
        _snapshot(
            slot=frame.slot,
            viewpoint=frame.viewpoint,
            hands=frame.hands,
            lead_play=frame.env.lead_play,
            pile_size=len(frame.env.pile),
        )

    _present()
    _wait_step()


def paint_parallel_static(envs: list[Any], kind: GameKind) -> None:
    """Paint current env states with no animation (deals / resize)."""
    for env in envs:
        _paint_env_static(env, kind)
    _present()

