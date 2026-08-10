from __future__ import annotations

import numpy as np
from pettingzoo.test import api_test, seed_test

from environments.tressette_env import env
from games.tressette import DECK, Card, Suit, first_card_wins


def test_deck() -> None:
    assert len(DECK) == 40
    assert len(set(DECK)) == 40


def test_card_power() -> None:
    three = Card(Suit.DENARI, 3)
    two = Card(Suit.DENARI, 2)
    ace = Card(Suit.DENARI, 1)
    king = Card(Suit.DENARI, 10)
    off_suit_three = Card(Suit.COPPE, 3)

    assert first_card_wins(three, two)
    assert first_card_wins(two, ace)
    assert first_card_wins(ace, king)
    assert first_card_wins(king, off_suit_three)


def test_complete_random_game() -> None:
    game = env()
    game.reset(seed=42)

    for agent in game.agent_iter():
        observation, _, terminated, truncated, _ = game.last()

        if terminated or truncated:
            action = None
        else:
            legal_actions = np.flatnonzero(
                observation['action_mask']
            )
            assert legal_actions.size > 0
            action = int(legal_actions[0])

        game.step(action)

    assert not game.agents
    assert sum(game.unwrapped.scores.values()) == 32


def test_pettingzoo_api() -> None:
    api_test(
        env(),
        num_cycles=500,
        verbose_progress=False,
    )


def test_reproducibility() -> None:
    seed_test(env, num_cycles=50)
