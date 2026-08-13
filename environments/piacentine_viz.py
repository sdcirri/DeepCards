"""
Pygame table renderer using Carte Piacentine assets.

Human rendering keeps a single window and redraws it each step,
with a short flight animation when a card is played to the table.

Assets from: https://deerlike.itch.io/piacentine-cards
"""

from typing import Any, Callable, Literal
from dataclasses import dataclass
from pathlib import Path

import pygame

from games.deck import Card

from environments.briscola_env import Briscola2PEnv
from environments.cards_env import AgentId, LeadPlay
from environments.tressette_env import Tressette2PEnv

ASSETS_DIR = Path(__file__).resolve().parent.parent / 'assets' / 'CartePiacentineITA'
CARD_BACK = ASSETS_DIR / '_Dorso.png'
TABLE_COLOR = (27, 107, 58)
WHITE = (255, 255, 255)
STEP_DELAY_MS = 450
ANIM_MS = 280
ANIM_HOLD_MS = 180
OPPONENT_CARD_ALPHA = 140

GameKind = Literal['briscola', 'tressette']

_LIVE: dict[str, Any] = {
    'screen': None,
    'kind': None,
    'clock': None,
    'font': None,
    'card_cache': {},
    'prev': None,
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


def _window_size(kind: GameKind) -> tuple[int, int]:
    return (1100, 700) if kind == 'tressette' else (900, 700)


def _card_size(hand_count: int) -> tuple[int, int]:
    if hand_count <= 3:
        return 90, 160
    if hand_count <= 6:
        return 72, 128
    return 58, 103


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
    screen: pygame.Surface,
    path: Path,
    center: tuple[int, int],
    size: tuple[int, int],
    *,
    alpha: int = 255,
) -> None:
    surface = _load_card_surface(path, size)
    if alpha < 255:
        surface = surface.copy()
        surface.fill((255, 255, 255, alpha), special_flags=pygame.BLEND_RGBA_MULT)
    rect = surface.get_rect(center=center)
    screen.blit(surface, rect)


def _font(size: int = 22) -> pygame.font.Font:
    font = _LIVE['font']
    if font is None or font.get_height() != size + 4:
        font = pygame.font.SysFont('dejavusans', size, bold=True)
        _LIVE['font'] = font
    return font


def _draw_label(screen: pygame.Surface, text: str, center: tuple[int, int], *, size: int = 22) -> None:
    surface = _font(size).render(text, True, WHITE)
    rect = surface.get_rect(center=center)
    screen.blit(surface, rect)


def _draw_label_left(
    screen: pygame.Surface,
    text: str,
    topleft: tuple[int, int],
    *,
    size: int = 22,
) -> None:
    surface = _font(size).render(text, True, WHITE)
    screen.blit(surface, topleft)


def _pump_events() -> bool:
    """Process window events. Returns False if the window was closed."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            close_live_window()
            return False
    return _LIVE['screen'] is not None


def _wait_step(delay_ms: int = STEP_DELAY_MS) -> None:
    """Pause between plays while keeping the window responsive."""
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
    screen = _LIVE['screen']
    if screen is not None:
        pygame.display.quit()
    _LIVE['screen'] = None
    _LIVE['kind'] = None
    _LIVE['clock'] = None
    _LIVE['font'] = None
    _LIVE['card_cache'] = {}
    _LIVE['prev'] = None


def _get_screen(kind: GameKind) -> pygame.Surface:
    _ensure_pygame()
    screen = _LIVE['screen']
    stale = screen is None or _LIVE['kind'] != kind or not pygame.display.get_init()
    if stale:
        close_live_window()
        _ensure_pygame()
        width, height = _window_size(kind)
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(f'{kind.title()} — Carte Piacentine')
        _LIVE['screen'] = screen
        _LIVE['kind'] = kind
        _LIVE['clock'] = pygame.time.Clock()
        _LIVE['font'] = pygame.font.SysFont('dejavusans', 22, bold=True)
        _LIVE['card_cache'] = {}
        _LIVE['prev'] = None
    return screen


def _hand_gap(card_width: int, count: int) -> int:
    if count <= 3:
        return card_width + 18
    if count <= 6:
        return card_width + 10
    return max(card_width - 8, 42)


def _hand_y(screen: pygame.Surface, *, bottom: bool) -> int:
    height = screen.get_height()
    return int(height * 0.84) if bottom else int(height * 0.14)


def _trick_pos(
    screen: pygame.Surface,
    agent: AgentId,
    *,
    viewpoint: AgentId,
) -> tuple[int, int]:
    width = screen.get_width()
    height = screen.get_height()
    center_x = int(width * 0.62)
    x = center_x - 40 if agent == viewpoint else center_x + 40
    return x, int(height * 0.48)


def _deck_pos(screen: pygame.Surface, *, kind: GameKind) -> tuple[int, int]:
    width = screen.get_width()
    height = screen.get_height()
    if kind == 'briscola':
        return int(width * 0.33), int(height * 0.48)
    return int(width * 0.31), int(height * 0.48)


def _hand_slot_pos(
    screen: pygame.Surface,
    *,
    hand_count: int,
    index: int,
    is_viewer: bool,
) -> tuple[int, int]:
    count = max(hand_count, 1)
    card_w, _ = _card_size(count)
    gap = _hand_gap(card_w, count)
    xs = _fan_xs(hand_count, center=screen.get_width() // 2, gap=gap)
    if not xs:
        return screen.get_width() // 2, _hand_y(screen, bottom=is_viewer)
    return xs[min(max(index, 0), len(xs) - 1)], _hand_y(screen, bottom=is_viewer)


def _hand_card_pos(
    screen: pygame.Surface,
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
        screen,
        hand_count=max(len(hand), index + 1),
        index=index,
        is_viewer=is_viewer,
    )


def _draw_hands(
    screen: pygame.Surface,
    *,
    viewer_hand: list[Card],
    opponent_hand: list[Card],
    card_size: tuple[int, int],
    gap: int,
) -> None:
    width = screen.get_width()

    for x, card in zip(
        _fan_xs(len(opponent_hand), center=width // 2, gap=gap),
        opponent_hand,
        strict=True,
    ):
        _blit_card(
            screen,
            card_image_path(card),
            (x, _hand_y(screen, bottom=False)),
            card_size,
            alpha=OPPONENT_CARD_ALPHA,
        )

    for x, card in zip(
        _fan_xs(len(viewer_hand), center=width // 2, gap=gap),
        viewer_hand,
        strict=True,
    ):
        _blit_card(screen, card_image_path(card), (x, _hand_y(screen, bottom=True)), card_size)


def _draw_trick_cards(
    screen: pygame.Surface,
    cards: list[tuple[AgentId, Card]],
    *,
    viewpoint: AgentId,
    card_size: tuple[int, int],
) -> None:
    if not cards:
        return

    width = screen.get_width()
    height = screen.get_height()
    for agent, card in cards:
        _blit_card(screen, card_image_path(card), _trick_pos(screen, agent, viewpoint=viewpoint), card_size)
    _draw_label(screen, 'trick', (int(width * 0.62), int(height * 0.63)), size=18)


def _agent_label(env: Any, agent: AgentId) -> str:
    names = getattr(env, 'display_names', None)
    if isinstance(names, dict) and names.get(agent):
        return str(names[agent])
    return agent


def _draw_scores(
    screen: pygame.Surface,
    *,
    env: Any,
    viewpoint: AgentId,
    opponent: AgentId,
    scores: dict[AgentId, int],
    turn: AgentId,
    score_label: str,
) -> None:
    height = screen.get_height()
    top_name = _agent_label(env, opponent)
    bottom_name = _agent_label(env, viewpoint)

    _draw_label_left(screen, top_name, (16, 12), size=22)
    _draw_label_left(
        screen,
        f'{scores[opponent]} {score_label}',
        (16, 40),
        size=18,
    )

    _draw_label_left(screen, bottom_name, (16, height - 52), size=22)
    _draw_label_left(
        screen,
        f'{scores[viewpoint]} {score_label}  ·  turn={turn}',
        (16, height - 28),
        size=18,
    )


def _ease_out_cubic(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3


def _lerp(a: tuple[int, int], b: tuple[int, int], t: float) -> tuple[int, int]:
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
    )


def _detect_play_flight(
    screen: pygame.Surface,
    *,
    viewpoint: AgentId,
    hands: dict[AgentId, list[Card]],
    lead_play: LeadPlay | None,
    pile_size: int,
) -> _Flight | None:
    prev = _LIVE['prev']
    if prev is not None and prev.get('kind') != _LIVE['kind']:
        prev = None
        _LIVE['prev'] = None

    # New deal / reset: pile grows again — don't animate across episodes.
    if prev is not None and pile_size > prev.get('pile_size', pile_size):
        prev = None
        _LIVE['prev'] = None

    prev_lead: LeadPlay | None = None if prev is None else prev['lead']
    prev_hands: dict[AgentId, list[Card]] | None = None if prev is None else prev['hands']

    # Lead play: a new face-up card on the table.
    if lead_play is not None and (prev_lead is None or prev_lead.card != lead_play.card):
        agent = lead_play.agent
        card = lead_play.card
        if prev_hands is not None and card in prev_hands[agent]:
            source_hand = prev_hands[agent]
        else:
            # First painted frame of an episode: reconstruct the pre-play hand.
            source_hand = [*hands[agent], card]
        start = _hand_card_pos(
            screen,
            hand=source_hand,
            card=card,
            is_viewer=(agent == viewpoint),
        )
        end = _trick_pos(screen, agent, viewpoint=viewpoint)
        return _Flight(card=card, agent=agent, start=start, end=end)

    # Follow play: trick just resolved, so lead cleared. Infer the follower card.
    if prev_lead is not None and lead_play is None and prev_hands is not None:
        follower: AgentId = 'p2' if prev_lead.agent == 'p1' else 'p1'
        missing = [card for card in prev_hands[follower] if card not in hands[follower]]
        if not missing:
            return None
        card = missing[0]

        start = _hand_card_pos(
            screen,
            hand=prev_hands[follower],
            card=card,
            is_viewer=(follower == viewpoint),
        )
        end = _trick_pos(screen, follower, viewpoint=viewpoint)
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
    screen: pygame.Surface,
    *,
    kind: GameKind,
    viewpoint: AgentId,
    hands: dict[AgentId, list[Card]],
    lead_play: LeadPlay | None,
    pile_size: int,
    winner: AgentId,
) -> list[_Flight]:
    """Deck→hand draw flights after a trick.

    Tressette: face-up (drawn cards are public).
    Briscola: face-down (drawn cards stay hidden).
    """
    prev = _LIVE['prev']
    if prev is None or prev.get('kind') != kind:
        return []
    if lead_play is not None or prev.get('lead') is None:
        return []
    if pile_size >= prev.get('pile_size', pile_size):
        return []

    prev_hands: dict[AgentId, list[Card]] = prev['hands']
    loser: AgentId = 'p2' if winner == 'p1' else 'p1'
    deck = _deck_pos(screen, kind=kind)
    flights: list[_Flight] = []

    for agent in (winner, loser):
        gained = _new_cards(prev_hands[agent], hands[agent])
        if not gained:
            continue
        card = gained[0]
        end = _hand_card_pos(
            screen,
            hand=hands[agent],
            card=card,
            is_viewer=(agent == viewpoint),
        )
        # Tressette draws are public; Briscola only reveals your own card.
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
    screen: pygame.Surface,
    winner: AgentId,
    *,
    viewpoint: AgentId,
    index: int = 0,
) -> tuple[int, int]:
    """Landing spot for claimed trick cards on the winner's side."""
    width = screen.get_width()
    height = screen.get_height()
    x = int(width * 0.88) + index * 14
    if winner == viewpoint:
        y = int(height * 0.72)
    else:
        y = int(height * 0.28)
    return x, y


def _animate_flight(
    screen: pygame.Surface,
    flight: _Flight,
    *,
    draw_frame: Callable[[], None],
    card_size: tuple[int, int],
    hold_ms: int = ANIM_HOLD_MS,
) -> None:
    _animate_flights(
        screen,
        [flight],
        draw_frame=draw_frame,
        card_size=card_size,
        hold_ms=hold_ms,
    )


def _animate_flights(
    screen: pygame.Surface,
    flights: list[_Flight],
    *,
    draw_frame: Callable[[], None],
    card_size: tuple[int, int],
    hold_ms: int = ANIM_HOLD_MS,
) -> None:
    if not flights:
        return

    clock = _LIVE['clock']
    start_ticks = pygame.time.get_ticks()

    while True:
        if not _pump_events():
            return

        elapsed = pygame.time.get_ticks() - start_ticks
        t = min(1.0, elapsed / ANIM_MS)
        eased = _ease_out_cubic(t)

        draw_frame()
        for flight in flights:
            path = card_image_path(flight.card) if flight.face_up else CARD_BACK
            pos = _lerp(flight.start, flight.end, eased)
            _blit_card(screen, path, pos, card_size)
        pygame.display.flip()

        if clock is not None:
            clock.tick(60)
        if t >= 1.0:
            break

    draw_frame()
    for flight in flights:
        path = card_image_path(flight.card) if flight.face_up else CARD_BACK
        _blit_card(screen, path, flight.end, card_size)
    pygame.display.flip()
    _wait_step(hold_ms)


def _snapshot(
    *,
    viewpoint: AgentId,
    hands: dict[AgentId, list[Card]],
    lead_play: LeadPlay | None,
    pile_size: int,
) -> None:
    _LIVE['prev'] = {
        'kind': _LIVE['kind'],
        'viewpoint': viewpoint,
        'hands': {
            'p1': list(hands['p1']),
            'p2': list(hands['p2']),
        },
        'lead': lead_play,
        'pile_size': pile_size,
    }


def _render_table(
    env: Briscola2PEnv | Tressette2PEnv,
    *,
    kind: GameKind,
    viewpoint: AgentId,
    score_label: str,
    draw_center: Callable[[pygame.Surface, tuple[int, int]], None],
) -> pygame.Surface:
    screen = _get_screen(kind)
    opponent: AgentId = 'p2' if viewpoint == 'p1' else 'p1'
    hands = {
        'p1': list(env.hands['p1'].cards),
        'p2': list(env.hands['p2'].cards),
    }
    pile_size = len(env.pile)

    play_flight = _detect_play_flight(
        screen,
        viewpoint=viewpoint,
        hands=hands,
        lead_play=env.lead_play,
        pile_size=pile_size,
    )
    draw_flights = _detect_draw_flights(
        screen,
        kind=kind,
        viewpoint=viewpoint,
        hands=hands,
        lead_play=env.lead_play,
        pile_size=pile_size,
        winner=env.agent_selection,
    )

    # While plays/draws animate, keep hands at the pre-draw composition.
    shown_hands = _pre_draw_hands(hands=hands, draw_flights=draw_flights) if draw_flights else {
        'p1': list(hands['p1']),
        'p2': list(hands['p2']),
    }
    trick_cards: list[tuple[AgentId, Card]] = []
    if env.lead_play is not None:
        trick_cards = [(env.lead_play.agent, env.lead_play.card)]
    elif play_flight is not None and _LIVE['prev'] is not None and _LIVE['prev']['lead'] is not None:
        # Follow play in progress: keep the lead visible until draws finish.
        prev_lead = _LIVE['prev']['lead']
        trick_cards = [(prev_lead.agent, prev_lead.card)]

    def paint(active_trick: list[tuple[AgentId, Card]] | None = None) -> None:
        viewer_hand = shown_hands[viewpoint]
        opponent_hand = shown_hands[opponent]
        hand_count = max(len(viewer_hand), len(opponent_hand), 1)
        card_size = _card_size(hand_count)
        gap = _hand_gap(card_size[0], hand_count)

        screen.fill(TABLE_COLOR)
        _draw_hands(
            screen,
            viewer_hand=viewer_hand,
            opponent_hand=opponent_hand,
            card_size=card_size,
            gap=gap,
        )
        draw_center(screen, card_size)
        _draw_trick_cards(
            screen,
            trick_cards if active_trick is None else active_trick,
            viewpoint=viewpoint,
            card_size=card_size,
        )
        _draw_scores(
            screen,
            env=env,
            viewpoint=viewpoint,
            opponent=opponent,
            scores=env.scores,
            turn=env.agent_selection,
            score_label=score_label,
        )

    def card_size_for(hand_count: int) -> tuple[int, int]:
        return _card_size(max(hand_count, 1))

    if play_flight is not None:
        settled: list[tuple[AgentId, Card]] = []
        if env.lead_play is None and _LIVE['prev'] is not None and _LIVE['prev']['lead'] is not None:
            prev_lead = _LIVE['prev']['lead']
            settled = [(prev_lead.agent, prev_lead.card)]

        def play_frame() -> None:
            paint(settled)

        size = card_size_for(
            max(len(shown_hands[viewpoint]), len(shown_hands[opponent]), 1)
        )
        _animate_flight(
            screen,
            play_flight,
            draw_frame=play_frame,
            card_size=size,
        )
        trick_cards = [*settled, (play_flight.agent, play_flight.card)]

        # After the follower lands, slide both trick cards to the winner's side.
        if env.lead_play is None and len(trick_cards) == 2:
            winner: AgentId = env.agent_selection
            claim_flights = [
                _Flight(
                    card=card,
                    agent=agent,
                    start=_trick_pos(screen, agent, viewpoint=viewpoint),
                    end=_won_pile_pos(screen, winner, viewpoint=viewpoint, index=index),
                    face_up=True,
                )
                for index, (agent, card) in enumerate(trick_cards)
            ]

            def claim_frame() -> None:
                paint([])

            _animate_flights(
                screen,
                claim_flights,
                draw_frame=claim_frame,
                card_size=size,
                hold_ms=max(ANIM_HOLD_MS // 2, 80),
            )
            trick_cards = []

    for draw_flight in draw_flights:
        hand_count = max(len(shown_hands[viewpoint]), len(shown_hands[opponent]), 1)

        def draw_frame() -> None:
            paint(trick_cards)

        _animate_flight(
            screen,
            draw_flight,
            draw_frame=draw_frame,
            card_size=card_size_for(hand_count),
            hold_ms=max(ANIM_HOLD_MS // 2, 80),
        )
        shown_hands[draw_flight.agent].append(draw_flight.card)

    # Final settled frame for the current env state.
    shown_hands['p1'] = list(hands['p1'])
    shown_hands['p2'] = list(hands['p2'])
    final_trick: list[tuple[AgentId, Card]] = []
    if env.lead_play is not None:
        final_trick = [(env.lead_play.agent, env.lead_play.card)]
    paint(final_trick)
    pygame.display.flip()
    _wait_step()
    _snapshot(
        viewpoint=viewpoint,
        hands=hands,
        lead_play=env.lead_play,
        pile_size=pile_size,
    )
    return screen


def render_briscola_table(
    env: Briscola2PEnv,
    *,
    viewpoint: AgentId = 'p1',
    show: bool = True,
) -> pygame.Surface | None:
    """Draw a top-down Briscola table from one player's viewpoint."""
    if not show:
        return None

    def draw_center(screen: pygame.Surface, card_size: tuple[int, int]) -> None:
        width = screen.get_width()
        height = screen.get_height()
        if env.briscola is not None:
            _blit_card(
                screen,
                card_image_path(env.briscola),
                (int(width * 0.24), int(height * 0.48)),
                card_size,
            )
        if env.pile:
            _blit_card(screen, CARD_BACK, (int(width * 0.32), int(height * 0.47)), card_size)
            _blit_card(screen, CARD_BACK, (int(width * 0.33), int(height * 0.48)), card_size)

        if env.briscola is not None or env.pile:
            label = 'briscola'
            if env.pile:
                label = f'briscola / deck ({len(env.pile)})'
            _draw_label(screen, label, (int(width * 0.29), int(height * 0.63)), size=18)

    return _render_table(
        env,
        kind='briscola',
        viewpoint=viewpoint,
        score_label='pts',
        draw_center=draw_center,
    )


def render_tressette_table(
    env: Tressette2PEnv,
    *,
    viewpoint: AgentId = 'p1',
    show: bool = True,
) -> pygame.Surface | None:
    """Draw a top-down Tressette table from one player's viewpoint."""
    if not show:
        return None

    def draw_center(screen: pygame.Surface, card_size: tuple[int, int]) -> None:
        width = screen.get_width()
        height = screen.get_height()
        if env.pile:
            _blit_card(screen, CARD_BACK, (int(width * 0.30), int(height * 0.47)), card_size)
            _blit_card(screen, CARD_BACK, (int(width * 0.31), int(height * 0.48)), card_size)
            _draw_label(
                screen,
                f'deck ({len(env.pile)})',
                (int(width * 0.31), int(height * 0.63)),
                size=18,
            )

    return _render_table(
        env,
        kind='tressette',
        viewpoint=viewpoint,
        score_label='thirds',
        draw_center=draw_center,
    )


def _lead_first_card(env: Briscola2PEnv | Tressette2PEnv, agent: AgentId = 'p1') -> None:
    from environments.cards_env import LeadPlay

    if env.agent_selection != agent or not env.hands[agent].cards:
        return

    led = env.hands[agent].cards[0]
    env.hands[agent].play_card(led)
    env.lead_play = LeadPlay(agent=agent, card=led)
    env.agent_selection = 'p2' if agent == 'p1' else 'p1'


def sketch_briscola(*, seed: int = 0, save_path: Path | None = None) -> None:
    from environments.briscola_env import raw_env

    env = raw_env(render_mode=None)
    env.reset(seed=seed)
    _lead_first_card(env)
    screen = render_briscola_table(env, viewpoint='p1', show=True)
    if save_path is not None and screen is not None:
        pygame.image.save(screen, str(save_path))
        print(f'Saved sketch to {save_path}')
    close_live_window()


def sketch_tressette(*, seed: int = 0, save_path: Path | None = None) -> None:
    from environments.tressette_env import raw_env

    env = raw_env(render_mode=None)
    env.reset(seed=seed)
    _lead_first_card(env)
    screen = render_tressette_table(env, viewpoint='p1', show=True)
    if save_path is not None and screen is not None:
        pygame.image.save(screen, str(save_path))
        print(f'Saved sketch to {save_path}')
    close_live_window()
