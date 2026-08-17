"""Card blitting, hand fans, trick positions, and score HUD."""

from typing import Any
from pathlib import Path

import pygame

from games.deck import Card
from environments.cards_env import AgentId

from .state import (
    CARD_BACK,
    OPPONENT_CARD_ALPHA,
    WHITE,
    GameKind,
    _LIVE,
    card_image_path,
)

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
