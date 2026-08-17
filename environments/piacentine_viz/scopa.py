"""Scopa single-pane render and synced parallel panes."""

from typing import Any, Callable
from dataclasses import dataclass

import pygame

from games.deck import Card
from environments.cards_env import AgentId
from environments.scopa_env import Scopa2PEnv

from .state import (
    ANIM_MS,
    CARD_BACK,
    TABLE_COLOR,
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
    _fan_xs,
    _hand_card_pos,
    _hand_gap,
    _label_size,
)
from .anim import (
    _animate_flights,
    _ease_out_cubic,
    _lerp,
    _missing_cards,
    _new_cards,
    _won_pile_pos,
)
from .window import _get_pane, _present, _pump_events, _wait_step

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

