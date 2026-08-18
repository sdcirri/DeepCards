from typing import Any, TypeAlias
from collections import deque
import numpy as np
import random

from gymnasium.utils import seeding
from gymnasium import spaces

from pettingzoo.utils import wrappers
from pettingzoo import AECEnv

from games.scopa import ScopaHand
from games.deck import DECK, Card

from .cards_env import AgentId, Action, Observation, BinaryArray


class Scopa2PEnv(AECEnv[AgentId, Observation, Action]):
    """
    Scopa 2-player environment. Since the Scopa rules are
    so different it cannot subclass `Cards2PEnv`
    """

    metadata = {
        'name': 'scopa2p_v0',
        'render_modes': ['human', 'ansi'],
        'is_parallelizable': False,
    }

    ILLEGAL_PLAY_PENALTY = -1000.0
    OBSERVATION_PLANES = 5

    table: list[Card]
    last_winner: AgentId

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()

        if render_mode not in (None, 'human', 'ansi'):
            raise ValueError(f'Unsupported render mode: {render_mode}')

        if not isinstance(self.OBSERVATION_PLANES, int) or self.OBSERVATION_PLANES < 3:
            raise ValueError('OBSERVATION_PLANES must be an int >= 3')

        self.render_mode = render_mode
        self.possible_agents: list[AgentId] = ['p1', 'p2']
        self.agent_name_mapping = {
            agent: index
            for index, agent in enumerate(self.possible_agents)
        }
        self.first_at_hand = random.choice(('p1', 'p2'))
        self.last_winner = self.first_at_hand

        # Index of the "legal" array from scopa_legal_plays, which is sorted
        #   and only depends on the state
        self._action_spaces = {
            agent: spaces.Discrete(80)
            for agent in self.possible_agents
        }

        self._observation_spaces = {
            agent: spaces.Dict(
                {
                    'observation': spaces.MultiBinary(len(DECK) * self.OBSERVATION_PLANES),
                    'action_mask': spaces.MultiBinary(len(DECK) * 2),
                }
            )
            for agent in self.possible_agents
        }

        self.np_random, self.np_random_seed = seeding.np_random(None)

        self.agents: list[AgentId] = []
        self.agent_selection: AgentId = 'p1'
        self._skip_agent_selection = None
        self.rewards: dict[AgentId, float] = {}
        self._cumulative_rewards: dict[AgentId, float] = {}
        self.terminations: dict[AgentId, bool] = {}
        self.truncations: dict[AgentId, bool] = {}
        self.infos: dict[AgentId, dict[str, Any]] = {}

        self.hands: dict[AgentId, ScopaHand] = {}
        self.pile: deque[Card] = deque()
        self.table: list[Card] = []
        self.scores = {
            'p1': 0,
            'p2': 0,
        }

    def _deal(self, shuffled_cards: list[Card]) -> None:
        self.hands = {
            'p1': ScopaHand(shuffled_cards[:3]),
            'p2': ScopaHand(shuffled_cards[3:6]),
        }
        self.table += shuffled_cards[6:10]
        self.pile = deque(shuffled_cards[10:])

        for card in self.table:
            self.hands['p1'].see_card(card)
            self.hands['p2'].see_card(card)

    def observation_space(self, agent: AgentId) -> spaces.Space[Observation]:
        return self._observation_spaces[agent]

    def action_space(self, agent: AgentId) -> spaces.Space[Action]:
        return self._action_spaces[agent]

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        del options

        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)
            for offset, agent in enumerate(self.possible_agents):
                self._action_spaces[agent].seed(seed + offset)

        self._skip_agent_selection = None
        self.agents = self.possible_agents.copy()
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        indices = self.np_random.permutation(len(DECK))
        shuffled_cards = [DECK[int(index)] for index in indices]

        self.scores = {
            'p1': 0,
            'p2': 0,
        }
        self.table = []
        self._deal(shuffled_cards)

        starting_index = int(self.np_random.integers(0, 2))
        self.agent_selection = self.possible_agents[starting_index]
        self._update_infos()

    def observe(self, agent: AgentId) -> Observation:
        hand = self.hands[agent]
        hand_encoding = np.fromiter(
            (card in hand.cards for card in DECK),
            dtype=np.int8,
            count=len(DECK),
        )
        seen_encoding = np.fromiter(
            (card in hand.seen for card in DECK),
            dtype=np.int8,
            count=len(DECK),
        )
        table_encoding = np.fromiter(
            (card in self.table for card in DECK),
            dtype=np.int8,
            count=len(DECK),
        )
        taken_encoding = np.fromiter(
            (card in hand.taken_cards for card in DECK),
            dtype=np.int8,
            count=len(DECK),
        )
        opponent_taken_encoding = np.fromiter(
            (card in hand.opponent_taken_cards for card in DECK),
            dtype=np.int8,
            count=len(DECK),
        )

        observation = np.concatenate((
            hand_encoding,
            seen_encoding,
            table_encoding,
            taken_encoding,
            opponent_taken_encoding
        ))

        legal = self.hands[agent].scopa_legal_plays(self.table)
        action_mask = np.zeros(len(DECK) * 2, dtype=np.int8)
        action_mask[:len(legal)] = 1
        return {
            'observation': observation,
            'action_mask': action_mask
        }

    def step(self, action: Action | None) -> None:
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        if action is None:
            raise ValueError('A live agent must provide an action')

        self._cumulative_rewards[agent] = 0.0
        self._clear_rewards()

        legal = self.hands[agent].scopa_legal_plays(self.table)
        played_card, taken_cards = legal[action]

        opponent = 'p2' if agent == 'p1' else 'p1'
        self.hands[agent].play_card(played_card)
        self.hands[opponent].see_card(played_card)

        score, opponent_score = self.hands[agent].score, self.hands[opponent].score
        old_scope, old_settebello = score.scope, score.settebello
        old_carte, old_denari, old_primiera = score.carte, score.denari, score.primiera
        old_opponent_scope, old_opponent_settebello = opponent_score.scope, opponent_score.settebello
        old_opponent_carte, old_opponent_denari = opponent_score.carte, opponent_score.denari
        old_opponent_primiera = opponent_score.primiera

        if len(taken_cards) == 0:
            self.table.append(played_card)
        else:
            self.hands[opponent].see_opponent_taken([played_card] + taken_cards)
            for card in taken_cards:
                self.table.remove(card)
            if len(self.table) == 0 and not self._game_finished():
                # Last one does not count as scopa!
                self.hands[agent].scopa()
            self.hands[agent].update_score([played_card, *taken_cards])
            self.last_winner = agent

        if self._game_finished():
            self.hands[self.last_winner].update_score(self.table)
            self.table.clear()
            self.terminations = {agent: True for agent in self.agents}
            self._compute_final_scores()
            self._deads_step_first()
        else:
            self.agent_selection = opponent
            # After both players have spent their 3 cards,
            #   deal 3 more each from the stock
            if self._hands_empty() and self.pile:
                self._redeal_hands()

        scopa_reward = score.scope - old_scope
        settebello_reward = score.settebello - old_settebello
        carte_reward = (score.carte - old_carte) / 40
        denari_reward = (score.denari - old_denari) / 10
        primiera_reward = (score.primiera - old_primiera) / 139
        opponent_scopa_reward = opponent_score.scope - old_opponent_scope
        opponent_settebello_reward = opponent_score.settebello - old_opponent_settebello
        opponent_carte_reward = (opponent_score.carte - old_opponent_carte) / 40
        opponent_denari_reward = (opponent_score.denari - old_opponent_denari) / 10
        opponent_primiera_reward = (opponent_score.primiera - old_opponent_primiera) / 139

        delta = (
                + scopa_reward
                + settebello_reward
                + carte_reward
                + denari_reward
                + primiera_reward
                - opponent_scopa_reward
                - opponent_settebello_reward
                - opponent_carte_reward
                - opponent_denari_reward
                - opponent_primiera_reward
        )
        self.rewards[agent], self.rewards[opponent] = delta, -delta

        self._update_infos()
        self._accumulate_rewards()

        if self.render_mode == 'human':
            self.render()

    def _compute_final_scores(self) -> None:
        p1_score, p2_score = self.hands['p1'].score, self.hands['p2'].score
        self.scores['p1'] = p1_score.scope + p1_score.settebello
        self.scores['p2'] = p2_score.scope + p2_score.settebello

        if p1_score.carte > p2_score.carte:
            self.scores['p1'] += 1
        elif p2_score.carte > p1_score.carte:
            self.scores['p2'] += 1

        if p1_score.denari > p2_score.denari:
            self.scores['p1'] += 1
        elif p2_score.denari > p1_score.denari:
            self.scores['p2'] += 1

        if p1_score.primiera > p2_score.primiera:
            self.scores['p1'] += 1
        elif p2_score.primiera > p1_score.primiera:
            self.scores['p2'] += 1

    def _redeal_hands(self) -> None:
        """
        Deal three cards to each player from the stock; table is unchanged.
        """
        for agent in (self.first_at_hand, 'p1' if self.first_at_hand == 'p2' else 'p2'):
            for _ in range(3):
                if not self.pile:
                    raise RuntimeError('Stock exhausted mid-deal')
                card = self.pile.popleft()
                self.hands[agent].take_card(card)

    def _hands_empty(self) -> bool:
        return not self.hands['p1'].cards and not self.hands['p2'].cards

    def _game_finished(self) -> bool:
        return not self.pile and self._hands_empty()

    def _update_infos(self) -> None:
        for agent in self.agents:
            opponent = 'p2' if agent == 'p1' else 'p1'
            self.infos[agent] = {
                **self._score_info(agent, opponent),
                'pile_size': len(self.pile),
                'hand_size': len(self.hands[agent]),
                'opponent_hand_size': len(self.hands[opponent]),
            }

    def render(self) -> str | None:
        output = self._render_text()

        if self.render_mode == 'ansi':
            return output

        if self.render_mode == 'human':
            from environments.piacentine_viz import render_scopa_table

            render_scopa_table(self, viewpoint='p1')

        return None

    def close(self) -> None:
        if self.render_mode == 'human':
            from environments.piacentine_viz import close_live_window

            close_live_window()
        return super().close()

    def _score_info(self, agent: AgentId, opponent: AgentId) -> dict[str, Any]:
        return {
            'score': self.scores[agent],
            'opponent_score': self.scores[opponent],
        }

    def _render_text(self) -> str:
        return (
            f'table={self.table} | '
            f'turn={self.agent_selection} | '
            f'pile={len(self.pile)} | '
            f'score={self.scores}'
        )


def env(render_mode: str | None = None) -> AECEnv:
    environment = raw_env(render_mode)
    environment = wrappers.AssertOutOfBoundsWrapper(environment)
    environment = wrappers.OrderEnforcingWrapper(environment)
    return environment


def raw_env(render_mode: str | None = None) -> Scopa2PEnv:
    return Scopa2PEnv(render_mode=render_mode)
