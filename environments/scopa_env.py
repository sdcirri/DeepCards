from typing import Any, TypeAlias
from collections import deque

import numpy as np

from gymnasium.utils import seeding
from gymnasium import spaces

from pettingzoo.utils import wrappers
from pettingzoo import AECEnv

from games.scopa import ScopaHand
from games.deck import DECK, Card

from .cards_env import AgentId, Observation


Action: TypeAlias = tuple[int, int]


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
    EXTRA_OBSERVATIONS = 9

    table: list[Card]
    last_winner: AgentId
    first_at_hand: AgentId

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()

        if render_mode not in (None, 'human', 'ansi'):
            raise ValueError(f'Unsupported render mode: {render_mode}')

        if not isinstance(self.OBSERVATION_PLANES, int) or self.OBSERVATION_PLANES < 3:
            raise ValueError('OBSERVATION_PLANES must be an int >= 3')

        self.np_random, self.np_random_seed = seeding.np_random(None)

        self.render_mode = render_mode
        self.possible_agents: list[AgentId] = ['p1', 'p2']
        self.agent_name_mapping = {
            agent: index
            for index, agent in enumerate(self.possible_agents)
        }
        self.first_at_hand = self.np_random.choice(self.possible_agents)
        self.last_winner = self.first_at_hand

        # Index of the "legal" array from scopa_legal_plays, which is sorted
        #   and only depends on the state
        self._action_spaces = {
            agent: spaces.Tuple((
                    spaces.Discrete(len(DECK)),
                    spaces.Discrete(10)     # Arbitrary, in practice never exceeds 3 or 4
                ))
            for agent in self.possible_agents
        }

        self._observation_spaces = {
            agent: spaces.Dict(
                {
                    'observation': spaces.Box(
                        low=0.0, high=1.0,
                        shape=(len(DECK) * self.OBSERVATION_PLANES + self.EXTRA_OBSERVATIONS,),
                        dtype=np.float32
                    ),
                    'play_mask': spaces.MultiBinary(len(DECK)),
                    'take_mask': spaces.MultiBinary(len(DECK))
                }
            )
            for agent in self.possible_agents
        }

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
        # Who receives stock cards first on redeals — must be seeded.
        self.first_at_hand = self.possible_agents[int(self.np_random.integers(0, 2))]
        self.last_winner = self.first_at_hand
        self._update_infos()

    def observe(self, agent: AgentId) -> Observation:
        hand = self.hands[agent]
        hand_encoding = np.fromiter(
            (card in hand.cards for card in DECK),
            dtype=np.float32,
            count=len(DECK),
        )
        seen_encoding = np.fromiter(
            (card in hand.seen for card in DECK),
            dtype=np.float32,
            count=len(DECK),
        )
        table_encoding = np.fromiter(
            (card in self.table for card in DECK),
            dtype=np.float32,
            count=len(DECK),
        )
        taken_encoding = np.fromiter(
            (card in hand.taken_cards for card in DECK),
            dtype=np.float32,
            count=len(DECK),
        )
        opponent_taken_encoding = np.fromiter(
            (card in hand.opponent_taken_cards for card in DECK),
            dtype=np.float32,
            count=len(DECK),
        )

        opp = 'p1' if agent == 'p2' else 'p2'
        my_score, opp_score = self.hands[agent].score, self.hands[opp].score
        extra_obs = np.fromiter((
            len(self.pile) / 30,        # 6 cards have been dealt + 4 put on the table
            my_score.carte / 40, opp_score.carte / 40,
            my_score.denari / 10, opp_score.denari / 10,
            sum(my_score.primiera.values()) / 84, sum(opp_score.primiera.values()) / 84,
            my_score.scope, opp_score.scope
        ), dtype=np.float32)

        observation = np.concatenate((
            hand_encoding,
            seen_encoding,
            table_encoding,
            taken_encoding,
            opponent_taken_encoding,
            extra_obs
        ))

        return {
            'observation': observation,
            'play_mask': hand_encoding,
            'take_mask': table_encoding
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

        played_card = DECK[action[0]]
        options = [opt[1] for opt in self.hands[agent].scopa_legal_plays(self.table) if opt[0] == played_card]
        taken_cards = options[action[1]]

        opponent = 'p2' if agent == 'p1' else 'p1'
        self.hands[agent].play_card(played_card)
        self.hands[opponent].see_card(played_card)

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

            p1_primiera, p2_primiera = sum(self.hands['p1'].score.primiera.values()), sum(self.hands['p2'].score.primiera.values())

            self.scores['p1'] = self.hands['p1'].get_score(p2_primiera)
            self.scores['p2'] = self.hands['p2'].get_score(p1_primiera)
            self.terminations = {'p1': True, 'p2': True}

            delta = self.hands['p1'].get_score(p2_primiera) - self.hands['p2'].get_score(p1_primiera)
            self.rewards['p1'], self.rewards['p2'] = delta, -delta

            self._deads_step_first()
        else:
            self.rewards = {'p1': 0.0, 'p2': 0.0}
            self.agent_selection = opponent
            # After both players have spent their 3 cards,
            #   deal 3 more each from the stock
            if self._hands_empty() and self.pile:
                self._redeal_hands()

        self._update_infos()
        self._accumulate_rewards()

        if self.render_mode == 'human':
            self.render()

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
