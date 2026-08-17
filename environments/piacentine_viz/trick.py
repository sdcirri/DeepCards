"""Briscola / Tressette single-pane render and parallel trick frames."""

from typing import Any, Callable
from dataclasses import dataclass

import pygame

from games.deck import Card
from environments.briscola_env import Briscola2PEnv
from environments.cards_env import AgentId, LeadPlay
from environments.tressette_env import Tressette2PEnv

from .state import (
    ANIM_MS,
    CARD_BACK,
    TABLE_COLOR,
    GameKind,
    _Flight,
    _LIVE,
    _slot_prev,
    _set_slot_prev,
    card_image_path,
)
from .draw import (
    _blit_card,
    _card_size,
    _deck_pos,
    _draw_hands,
    _draw_label,
    _draw_scores,
    _draw_trick_cards,
    _hand_card_pos,
    _hand_gap,
    _label_size,
    _trick_pos,
)
from .anim import _animate_flights, _ease_out_cubic, _lerp, _new_cards, _won_pile_pos
from .window import _get_pane, _present, _pump_events, _wait_step

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
