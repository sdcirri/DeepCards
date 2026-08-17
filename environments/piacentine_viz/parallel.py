"""Step-synced multi-pane orchestration for human challenges."""

from typing import Any

from games.deck import Card
from environments.cards_env import AgentId

from .state import GameKind, _Flight, _LIVE
from .window import _present, _wait_step
from .trick import (
    _PaneFrame,
    _animate_parallel_phase,
    _paint_pane_frame,
    _prepare_pane_frame,
    _snapshot,
)
from .scopa import (
    _paint_scopa_state,
    _render_scopa_parallel_round,
    _scopa_snapshot,
)

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

