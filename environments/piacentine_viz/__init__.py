"""
Pygame table renderer using Carte Piacentine assets.

Supports a single table or a grid of up to MAX_PARALLEL_EPISODES panes
for watching many episodes of one challenge at once.

Assets from: https://deerlike.itch.io/piacentine-cards
"""

from .state import MAX_PARALLEL_EPISODES, GameKind, card_image_path
from .window import (
    begin_parallel_session,
    close_live_window,
    end_parallel_session,
    set_parallel_envs,
)
from .trick import render_briscola_table, render_tressette_table
from .scopa import render_scopa_table
from .parallel import paint_parallel_static, render_parallel_round

__all__ = [
    'MAX_PARALLEL_EPISODES',
    'GameKind',
    'begin_parallel_session',
    'card_image_path',
    'close_live_window',
    'end_parallel_session',
    'paint_parallel_static',
    'render_briscola_table',
    'render_parallel_round',
    'render_scopa_table',
    'render_tressette_table',
    'set_parallel_envs',
]
