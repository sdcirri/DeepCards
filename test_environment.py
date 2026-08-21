from __future__ import annotations

from collections import deque
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pettingzoo.test import api_test, seed_test

from environments.briscola_env import Briscola2PEnv, env as briscola_env, raw_env as briscola_raw
from environments.cards_env import (
    AgentId,
    Cards2PEnv,
    LeadPlay,
    card_from_action,
    wrap_cards_env,
)
from environments.scopa_env import Scopa2PEnv, env as scopa_env, raw_env as scopa_raw
from environments.tressette_env import Tressette2PEnv, env as tressette_env, raw_env as tressette_raw
from games.briscola import BriscolaHand, card_points, first_card_wins as briscola_first_wins
from games.deck import CARD_INDEX, CARD_NUMBERS, DECK, Card, Hand, Suit
from games.scopa import (
    ScopaHand,
    ScopaScore,
    card_value,
    find_takes,
    legal_plays as scopa_legal_plays,
    play_value,
    sorted_legal_plays,
)
from games.tressette import (
    TressetteHand,
    card_point_thirds,
    first_card_wins as tressette_first_wins,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _scopa_first_legal_action(raw: Scopa2PEnv, agent: AgentId) -> tuple[int, int]:
    legal = raw.hands[agent].scopa_legal_plays(raw.table)
    assert legal
    play_card, capture = legal[0]
    options = [opt[1] for opt in legal if opt[0] == play_card]
    return CARD_INDEX[play_card], options.index(capture)


def _play_random_episode(factory, seed: int = 0) -> Any:
    game = factory()
    game.reset(seed=seed)
    for agent in game.agent_iter():
        observation, _, terminated, truncated, _ = game.last()
        if terminated or truncated:
            action = None
        elif 'action_mask' in observation:
            legal = np.flatnonzero(observation['action_mask'])
            assert legal.size > 0
            action = int(legal[0])
        else:
            action = _scopa_first_legal_action(game.unwrapped, agent)
        game.step(action)
    assert not game.agents
    return game


def _c(suit: Suit, number: int) -> Card:
    return Card(suit, number)


# ---------------------------------------------------------------------------
# games.deck
# ---------------------------------------------------------------------------


def test_deck_unique() -> None:
    assert len(DECK) == 40
    assert len(set(DECK)) == 40
    assert len(CARD_NUMBERS) == 10
    assert len(CARD_INDEX) == 40
    assert str(DECK[0]) == 'Card [1 of Denari]'


def test_hand_basics() -> None:
    a, b = _c(Suit.DENARI, 1), _c(Suit.COPPE, 2)
    hand = Hand([a, b])
    assert len(hand) == 2
    assert a in hand.seen
    hand.see_card(_c(Suit.SPADE, 3))
    assert _c(Suit.SPADE, 3) in hand.seen
    hand.play_card(a)
    assert a not in hand.cards
    hand.take_card(a)
    assert a in hand.cards and a in hand.seen
    # base legal_plays body is Ellipsis (no-op → returns None)
    assert hand.legal_plays(None) is None


def test_hand_rejects_duplicates() -> None:
    card = _c(Suit.DENARI, 1)
    with pytest.raises(ValueError, match='duplicate'):
        Hand([card, card])


# ---------------------------------------------------------------------------
# games.tressette
# ---------------------------------------------------------------------------


def test_tressette_card_points_and_power() -> None:
    assert card_point_thirds(_c(Suit.DENARI, 1)) == 3
    assert card_point_thirds(_c(Suit.DENARI, 2)) == 1
    assert card_point_thirds(_c(Suit.DENARI, 3)) == 1
    assert card_point_thirds(_c(Suit.DENARI, 8)) == 1
    assert card_point_thirds(_c(Suit.DENARI, 4)) == 0
    three, two, ace, king, off = (
        _c(Suit.DENARI, 3),
        _c(Suit.DENARI, 2),
        _c(Suit.DENARI, 1),
        _c(Suit.DENARI, 10),
        _c(Suit.COPPE, 3),
    )
    assert tressette_first_wins(three, two)
    assert tressette_first_wins(two, ace)
    assert tressette_first_wins(ace, king)
    assert tressette_first_wins(king, off)
    assert not tressette_first_wins(two, three)


def test_tressette_hand_legal_and_known() -> None:
    lead = _c(Suit.DENARI, 5)
    hand = TressetteHand([
        _c(Suit.DENARI, 1),
        _c(Suit.COPPE, 2),
        _c(Suit.BASTONI, 3),
    ])
    assert set(hand.legal_plays(None)) == set(hand.cards)
    assert hand.legal_plays(lead) == [_c(Suit.DENARI, 1)]
    void = TressetteHand([_c(Suit.COPPE, 2)])
    assert void.legal_plays(lead) == [_c(Suit.COPPE, 2)]

    drawn = _c(Suit.SPADE, 7)
    hand.see_opponent_draw(drawn)
    assert drawn in hand.known_opponent_hand
    hand.see_card(drawn)
    assert drawn not in hand.known_opponent_hand
    hand.see_card(_c(Suit.SPADE, 8))  # never known — no error


def test_tressette_accusi_points() -> None:
    # Napoli in denari + buongioco of three aces
    hand = TressetteHand([
        _c(Suit.DENARI, 1),
        _c(Suit.DENARI, 2),
        _c(Suit.DENARI, 3),
        _c(Suit.COPPE, 1),
        _c(Suit.BASTONI, 1),
        _c(Suit.SPADE, 4),
    ])
    # napoli denari = 3, buongioco aces = 3 → 6
    assert hand.get_accusi_points() == 6
    emptyish = TressetteHand([_c(Suit.DENARI, 4)])
    assert emptyish.get_accusi_points() == 0


# ---------------------------------------------------------------------------
# games.briscola
# ---------------------------------------------------------------------------


def test_briscola_points_and_trumps() -> None:
    assert card_points(_c(Suit.DENARI, 1)) == 11
    assert card_points(_c(Suit.DENARI, 3)) == 10
    assert card_points(_c(Suit.DENARI, 10)) == 4
    assert card_points(_c(Suit.DENARI, 2)) == 0
    trump = Suit.COPPE
    ace_d = _c(Suit.DENARI, 1)
    three_c = _c(Suit.COPPE, 3)
    two_c = _c(Suit.COPPE, 2)
    # offsuit loses to trump
    assert not briscola_first_wins(ace_d, three_c, trump)
    assert briscola_first_wins(three_c, ace_d, trump)
    # same suit by power (ace > 3 in briscola)
    assert briscola_first_wins(_c(Suit.COPPE, 1), three_c, trump)
    assert not briscola_first_wins(two_c, three_c, trump)
    # both non-trump different suits: lead wins
    assert briscola_first_wins(ace_d, _c(Suit.BASTONI, 3), trump)
    hand = BriscolaHand([ace_d, three_c])
    assert hand.legal_plays(_c(Suit.SPADE, 5)) == [ace_d, three_c]


# ---------------------------------------------------------------------------
# games.scopa
# ---------------------------------------------------------------------------


def test_scopa_find_takes_and_legal() -> None:
    table = (_c(Suit.DENARI, 2), _c(Suit.COPPE, 3), _c(Suit.BASTONI, 5))
    takes = find_takes(_c(Suit.SPADE, 5), table)
    assert any(sorted(t, key=id) for t in takes)
    # exact 2+3
    assert [_c(Suit.DENARI, 2), _c(Suit.COPPE, 3)] in takes or \
        [_c(Suit.COPPE, 3), _c(Suit.DENARI, 2)] in takes

    hand = (_c(Suit.SPADE, 5), _c(Suit.DENARI, 7))
    # same-number force take
    table2 = (_c(Suit.COPPE, 5), _c(Suit.BASTONI, 1))
    plays = scopa_legal_plays(hand, table2)
    assert all(p[0].number != 5 or len(p[1]) == 1 for p in plays)
    # no take → empty capture list
    lonely = scopa_legal_plays((_c(Suit.SPADE, 1),), (_c(Suit.DENARI, 10),))
    assert ( _c(Suit.SPADE, 1), [] ) in lonely

    sorted_plays = sorted_legal_plays(hand, table2)
    assert sorted_plays == sorted(
        plays,
        key=lambda pt: (CARD_INDEX[pt[0]], tuple(sorted(CARD_INDEX[c] for c in pt[1]))),
    )


def test_scopa_hand_scoring() -> None:
    hand = ScopaHand([_c(Suit.DENARI, 1)])
    hand.scopa()
    assert hand.score.scope == 1
    hand.see_opponent_taken([_c(Suit.COPPE, 4)])
    assert _c(Suit.COPPE, 4) in hand.opponent_taken_cards
    hand.update_score([_c(Suit.DENARI, 7), _c(Suit.BASTONI, 1)])
    assert hand.score.settebello == 1
    assert hand.score.denari == 1
    assert hand.score.carte == 2
    assert sum(hand.score.primiera.values()) > 0
    assert isinstance(hand.score, ScopaScore)
    assert len(hand.scopa_legal_plays([])) == 1
    assert hand.get_score(opponent_primiera=0) >= 2


def test_scopa_card_and_play_value() -> None:
    settebello = _c(Suit.DENARI, 7)
    assert card_value(settebello) > card_value(_c(Suit.SPADE, 2))
    table = [_c(Suit.COPPE, 3), _c(Suit.BASTONI, 4)]
    assert play_value((_c(Suit.SPADE, 1), []), table) < 0
    capture = (_c(Suit.SPADE, 7), table)
    assert play_value(capture, table) > play_value((_c(Suit.SPADE, 7), [table[0]]), table)


def test_scopa_get_score_branches() -> None:
    hand = ScopaHand([])
    hand.score.scope = 1
    hand.score.settebello = 1
    hand.score.carte = 30
    hand.score.denari = 8
    hand.score.primiera = {suit: 21 for suit in Suit}
    assert hand.get_score(opponent_primiera=0) == 5

    hand.score.scope = hand.score.settebello = 0
    hand.score.carte = 5
    hand.score.denari = 1
    hand.score.primiera = {suit: 0 for suit in Suit}
    assert hand.get_score(opponent_primiera=80) == 0

    hand.score.carte = 20
    hand.score.denari = 5
    hand.score.primiera = {suit: 10 for suit in Suit}
    assert hand.get_score(opponent_primiera=40) == 0


# ---------------------------------------------------------------------------
# environments.cards_env (via concrete env + direct helpers)
# ---------------------------------------------------------------------------


def test_card_from_action() -> None:
    assert card_from_action(0) == DECK[0]
    with pytest.raises(ValueError):
        card_from_action(-1)
    with pytest.raises(ValueError):
        card_from_action(40)


def test_cards_env_init_validation() -> None:
    class BadPlanes(Tressette2PEnv):
        OBSERVATION_PLANES = 2

    class BadPoints(Tressette2PEnv):
        MAX_HAND_POINTS = 0

    with pytest.raises(ValueError, match='render mode'):
        Tressette2PEnv(render_mode='opengl')
    with pytest.raises(ValueError, match='OBSERVATION_PLANES'):
        BadPlanes()
    with pytest.raises(ValueError, match='MAX_HAND_POINTS'):
        BadPoints()


def test_cards_env_spaces_and_ansi() -> None:
    raw = tressette_raw(render_mode='ansi')
    raw.reset(seed=1)
    assert raw.observation_space('p1') is not None
    assert raw.action_space('p1') is not None
    text = raw.render()
    assert text is not None and 'turn=' in text
    raw.close()


def test_cards_env_illegal_and_none_action() -> None:
    raw = tressette_raw()
    raw.reset(seed=2)
    agent = raw.agent_selection
    with pytest.raises(ValueError, match='live agent'):
        raw.step(None)
    # pick an illegal card index if possible
    mask = raw.observe(agent)['action_mask']
    illegal = int(np.flatnonzero(mask == 0)[0])
    with pytest.raises(ValueError, match='Illegal'):
        raw.step(illegal)


def test_cards_env_action_mask_edges() -> None:
    raw = tressette_raw()
    raw.reset(seed=3)
    # not selected agent → empty mask
    other = 'p2' if raw.agent_selection == 'p1' else 'p1'
    assert raw.observe(other)['action_mask'].sum() == 0
    # agent not in agents
    raw.agents = [raw.agent_selection]
    assert raw._action_mask(other).sum() == 0
    # terminated
    raw.agents = ['p1', 'p2']
    raw.terminations[raw.agent_selection] = True
    assert raw._action_mask(raw.agent_selection).sum() == 0


def test_cards_env_resolve_trick_guards() -> None:
    raw = tressette_raw()
    raw.reset(seed=4)
    with pytest.raises(RuntimeError, match='without a lead'):
        raw._resolve_trick('p1', DECK[0])
    raw.lead_play = LeadPlay(agent='p1', card=DECK[0])
    with pytest.raises(RuntimeError, match='same agent'):
        raw._resolve_trick('p1', DECK[1])


def test_cards_env_draw_guards() -> None:
    raw = tressette_raw()
    raw.reset(seed=5)
    raw.pile = deque()
    raw._draw_cards('p1', 'p2')  # no-op
    raw.pile = deque([DECK[0]])
    with pytest.raises(RuntimeError, match='invalid number'):
        raw._draw_cards('p1', 'p2')


def test_cards_env_default_extra_planes_and_finish_hook() -> None:
    # Exercise Cards2PEnv defaults via a minimal concrete subclass.
    class Minimal(Cards2PEnv):
        OBSERVATION_PLANES = 3
        EXTRA_OBSERVATIONS = 0
        MAX_HAND_POINTS = 6

        def _deal(self, shuffled_cards: list[Card]) -> None:
            self.hands = {
                'p1': TressetteHand(shuffled_cards[:10]),
                'p2': TressetteHand(shuffled_cards[10:20]),
            }
            self.pile = deque(shuffled_cards[20:])

        def _lead_wins(self, leader_card: Card, follower_card: Card) -> bool:
            return True

        def _trick_points(self, leader_card: Card, follower_card: Card) -> int:
            return 0

        def _give_drawn_cards(self, winner: AgentId, loser: AgentId) -> None:
            return None

        def _score_info(self, agent: AgentId, opponent: AgentId) -> dict[str, Any]:
            return {}

        def _render_text(self) -> str:
            return 'minimal'

    env = Minimal(render_mode=None)
    env.reset(seed=0)
    assert env._extra_observation_planes('p1') == []
    env._on_game_finished('p1')
    assert env.render() is None
    ansi = Minimal(render_mode='ansi')
    ansi.reset(seed=0)
    assert ansi.render() == 'minimal'
    human = Minimal(render_mode='human')
    human.reset(seed=0)
    with patch('builtins.print') as mocked:
        human.render()
        mocked.assert_called()
        mask = human.observe(human.agent_selection)['action_mask']
        human.step(int(np.flatnonzero(mask)[0]))
        assert mocked.call_count >= 2
    # abstract base implementations still raise if called unbound
    with pytest.raises(NotImplementedError):
        Cards2PEnv._deal(env, [])
    with pytest.raises(NotImplementedError):
        Cards2PEnv._lead_wins(env, DECK[0], DECK[1])
    with pytest.raises(NotImplementedError):
        Cards2PEnv._trick_points(env, DECK[0], DECK[1])
    with pytest.raises(NotImplementedError):
        Cards2PEnv._give_drawn_cards(env, 'p1', 'p2')
    with pytest.raises(NotImplementedError):
        Cards2PEnv._score_info(env, 'p1', 'p2')
    with pytest.raises(NotImplementedError):
        Cards2PEnv._render_text(env)
    wrapped = wrap_cards_env(Minimal())
    assert wrapped is not None


# ---------------------------------------------------------------------------
# Tressette env
# ---------------------------------------------------------------------------


def test_tressette_complete_random_game() -> None:
    game = _play_random_episode(tressette_env, seed=42)
    # 32 card thirds + 3 ultima
    assert sum(game.unwrapped.scores.values()) == 35


def test_tressette_api_and_seed() -> None:
    api_test(tressette_env(), num_cycles=200, verbose_progress=False)
    seed_test(tressette_env, num_cycles=30)


def test_tressette_known_hand_plane_and_draw() -> None:
    raw = tressette_raw()
    raw.reset(seed=7)
    obs0 = raw.observe(raw.agent_selection)['observation']
    assert obs0.shape == (40 * Tressette2PEnv.OBSERVATION_PLANES + Tressette2PEnv.EXTRA_OBSERVATIONS,)
    # play until a draw happens (pile shrinks)
    pile0 = len(raw.pile)
    for _ in range(4):
        if raw.terminations['p1']:
            break
        agent = raw.agent_selection
        mask = raw.observe(agent)['action_mask']
        raw.step(int(np.flatnonzero(mask)[0]))
    assert len(raw.pile) <= pile0


def test_tressette_human_render_mocked() -> None:
    raw = tressette_raw(render_mode='human')
    raw.reset(seed=0)
    with patch('environments.piacentine_viz.render_tressette_table') as r, \
            patch('environments.piacentine_viz.close_live_window') as c:
        assert raw.render() is None
        r.assert_called_once()
        raw.close()
        c.assert_called_once()


# ---------------------------------------------------------------------------
# Briscola env
# ---------------------------------------------------------------------------


def test_briscola_complete_random_game() -> None:
    game = _play_random_episode(briscola_env, seed=42)
    total = sum(game.unwrapped.scores.values())
    assert total == sum(card_points(c) for c in DECK)


def test_briscola_api_and_seed() -> None:
    api_test(briscola_env(), num_cycles=200, verbose_progress=False)
    seed_test(briscola_env, num_cycles=30)


def test_briscola_planes_and_trump_error() -> None:
    raw = briscola_raw()
    raw.reset(seed=1)
    assert raw.briscola is not None
    assert raw.observe('p1')['observation'].shape == (
        40 * Briscola2PEnv.OBSERVATION_PLANES + Briscola2PEnv.EXTRA_OBSERVATIONS,
    )
    raw.briscola = None
    with pytest.raises(RuntimeError, match='without a briscola'):
        raw._lead_wins(DECK[0], DECK[1])


def test_briscola_ansi_and_human() -> None:
    ansi = briscola_raw(render_mode='ansi')
    ansi.reset(seed=0)
    assert 'briscola=' in (ansi.render() or '')
    ansi.close()
    human = briscola_raw(render_mode='human')
    human.reset(seed=0)
    with patch('environments.piacentine_viz.render_briscola_table') as r, \
            patch('environments.piacentine_viz.close_live_window') as c:
        human.render()
        human.close()
        r.assert_called_once()
        c.assert_called_once()


# ---------------------------------------------------------------------------
# Scopa env
# ---------------------------------------------------------------------------


def test_scopa_complete_random_game() -> None:
    game = _play_random_episode(scopa_env, seed=42)
    raw = game.unwrapped
    assert not game.agents
    assert isinstance(raw.scores['p1'], int)
    assert isinstance(raw.scores['p2'], int)


def test_scopa_seed_reproducible_with_legal_actions() -> None:
    """
    PettingZoo api_test/seed_test sample the full Tuple action space, which
    includes illegal take indices. Seed determinism is checked with legal plays.
    """
    def run(seed: int) -> list[tuple[int, int]]:
        game = scopa_env()
        game.reset(seed=seed)
        actions: list[tuple[int, int]] = []
        for agent in game.agent_iter():
            _, _, terminated, truncated, _ = game.last()
            if terminated or truncated:
                game.step(None)
                continue
            action = _scopa_first_legal_action(game.unwrapped, agent)
            actions.append(action)
            game.step(action)
        return actions

    assert run(11) == run(11)


def test_scopa_init_validation_and_spaces() -> None:
    with pytest.raises(ValueError, match='render mode'):
        Scopa2PEnv(render_mode='bad')

    class BadPlanes(Scopa2PEnv):
        OBSERVATION_PLANES = 1

    with pytest.raises(ValueError, match='OBSERVATION_PLANES'):
        BadPlanes()

    raw = scopa_raw(render_mode='ansi')
    raw.reset(seed=0)
    obs = raw.observe('p1')
    expected = (
        len(DECK) * Scopa2PEnv.OBSERVATION_PLANES + Scopa2PEnv.EXTRA_OBSERVATIONS
    )
    assert raw.observation_space('p1') is not None
    assert raw.action_space('p1').spaces[0].n == len(DECK)
    assert raw.action_space('p1').spaces[1].n == 10
    assert obs['observation'].shape == (expected,)
    assert 'play_mask' in obs and 'take_mask' in obs
    assert 'table=' in (raw.render() or '')
    raw.close()


def test_scopa_step_errors_and_dead() -> None:
    raw = scopa_raw()
    raw.reset(seed=1)
    with pytest.raises(ValueError, match='live agent'):
        raw.step(None)
    raw.terminations = {'p1': True, 'p2': True}
    raw.agent_selection = 'p1'
    # dead step should not raise
    raw.step(None)


def test_scopa_final_scores_via_get_score() -> None:
    raw = scopa_raw()
    raw.reset(seed=2)
    p1, p2 = raw.hands['p1'].score, raw.hands['p2'].score
    p1.scope, p1.settebello = 1, 1
    p1.carte, p1.denari = 30, 8
    p1.primiera = {suit: 21 for suit in Suit}
    p2.scope = p2.settebello = 0
    p2.carte, p2.denari = 10, 2
    p2.primiera = {suit: 0 for suit in Suit}
    p1_primiera = sum(p1.primiera.values())
    p2_primiera = sum(p2.primiera.values())
    raw.scores['p1'] = raw.hands['p1'].get_score(p2_primiera)
    raw.scores['p2'] = raw.hands['p2'].get_score(p1_primiera)
    assert raw.scores['p1'] == 5
    assert raw.scores['p2'] == 0


def test_scopa_redeal_exhausted() -> None:
    raw = scopa_raw()
    raw.reset(seed=3)
    raw.pile.clear()
    raw.hands['p1'].cards.clear()
    raw.hands['p2'].cards.clear()
    raw.pile.append(DECK[0])  # only one card — not enough for 3+3
    with pytest.raises(RuntimeError, match='Stock exhausted'):
        raw._redeal_hands()


def test_scopa_human_render_mocked() -> None:
    raw = scopa_raw(render_mode='human')
    raw.reset(seed=0)
    with patch('environments.piacentine_viz.render_scopa_table') as r, \
            patch('environments.piacentine_viz.close_live_window') as c:
        raw.render()
        action = _scopa_first_legal_action(raw, raw.agent_selection)
        raw.step(action)
        raw.close()
        assert r.call_count >= 1
        c.assert_called_once()


def test_scopa_capture_and_scopa_point() -> None:
    """Force a table-clearing capture mid-game to hit the scopa() branch."""
    raw = scopa_raw()
    raw.reset(seed=10)
    agent = raw.agent_selection
    # Rebuild a controllable position: one card on table matching hand
    table_card = _c(Suit.DENARI, 4)
    play_card = _c(Suit.COPPE, 4)
    raw.table = [table_card]
    raw.hands[agent].cards = [play_card]
    raw.hands[agent].seen.add(play_card)
    raw.hands[agent].seen.add(table_card)
    # Ensure game not finished
    assert raw.pile
    legal = raw.hands[agent].scopa_legal_plays(raw.table)
    assert any(c == play_card and t == [table_card] for c, t in legal)
    before = raw.hands[agent].score.scope
    raw.step((CARD_INDEX[play_card], 0))
    assert raw.hands[agent].score.scope == before + 1
    assert table_card not in raw.table


def test_scopa_dump_on_table_and_endgame_reward() -> None:
    raw = scopa_raw()
    raw.reset(seed=5)
    agent = raw.agent_selection
    play_card = _c(Suit.SPADE, 1)
    raw.table = [_c(Suit.DENARI, 10)]
    raw.hands[agent].cards = [play_card]
    raw.hands[agent].seen.update(raw.table)
    raw.hands[agent].seen.add(play_card)
    raw.step((CARD_INDEX[play_card], 0))
    assert play_card in raw.table

    # Finish the hand to exercise terminal scoring / rewards.
    raw.pile.clear()
    raw.hands['p1'].cards.clear()
    raw.hands['p2'].cards.clear()
    raw.table = [_c(Suit.COPPE, 2)]
    raw.last_winner = 'p1'
    # One more forced capture that ends the game
    raw.hands['p1'].cards = [_c(Suit.BASTONI, 2)]
    raw.agent_selection = 'p1'
    raw.step((CARD_INDEX[_c(Suit.BASTONI, 2)], 0))
    assert raw.terminations['p1'] and raw.terminations['p2']
    assert isinstance(raw.scores['p1'], int)