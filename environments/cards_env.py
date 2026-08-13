from typing import Any, TypeAlias, Literal
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import deque

from numpy.typing import NDArray
import numpy as np

from gymnasium.utils import seeding
from gymnasium import spaces

from pettingzoo.utils import wrappers
from pettingzoo import AECEnv

from games.deck import DECK, Card, CARD_INDEX, Hand


AgentId: TypeAlias = Literal['p1', 'p2']
Action: TypeAlias = int | np.integer[Any]
BinaryArray: TypeAlias = NDArray[np.int8]
Observation: TypeAlias = dict[str, BinaryArray]


@dataclass(frozen=True, slots=True)
class LeadPlay:
    agent: AgentId
    card: Card


def card_from_action(action: int) -> Card:
    if not 0 <= action < len(DECK):
        raise ValueError(f'Invalid action: {action}')
    return DECK[action]


class Cards2PEnv(AECEnv[AgentId, Observation, Action], ABC):
    """
    Shared 2-player trick-taking AEC shell.
    """

    metadata = {
        'render_modes': ['human', 'ansi'],
        'is_parallelizable': False,
    }

    ILLEGAL_PLAY_PENALTY = -1000.0
    OBSERVATION_PLANES: int
    # Max points (or thirds) in a single trick; used to scale RL rewards into ~[-1, 1].
    MAX_HAND_POINTS: int

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()

        if render_mode not in (None, 'human', 'ansi'):
            raise ValueError(f'Unsupported render mode: {render_mode}')

        if not isinstance(self.OBSERVATION_PLANES, int) or self.OBSERVATION_PLANES < 3:
            raise ValueError('OBSERVATION_PLANES must be an int >= 3')

        if not isinstance(self.MAX_HAND_POINTS, int) or self.MAX_HAND_POINTS <= 0:
            raise ValueError('MAX_HAND_POINTS must be a positive int')

        self.render_mode = render_mode
        self.possible_agents: list[AgentId] = ['p1', 'p2']
        self.agent_name_mapping = {
            agent: index
            for index, agent in enumerate(self.possible_agents)
        }

        self._action_spaces = {
            agent: spaces.Discrete(len(DECK), dtype=np.int32)
            for agent in self.possible_agents
        }

        self._observation_spaces = {
            agent: spaces.Dict(
                {
                    'observation': spaces.MultiBinary(len(DECK) * self.OBSERVATION_PLANES),
                    'action_mask': spaces.MultiBinary(len(DECK)),
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

        self.hands: dict[AgentId, Hand] = {}
        self.pile: deque[Card] = deque()
        self.lead_play: LeadPlay | None = None
        self.scores: dict[AgentId, int] = {
            'p1': 0,
            'p2': 0,
        }

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

        self.lead_play = None
        self.scores = {
            'p1': 0,
            'p2': 0,
        }
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
        trick_encoding = np.zeros(len(DECK), dtype=np.int8)

        if self.lead_play is not None:
            trick_encoding[CARD_INDEX[self.lead_play.card]] = 1

        planes = [hand_encoding, seen_encoding, trick_encoding]
        planes.extend(self._extra_observation_planes(agent))
        observation = np.concatenate(planes)

        return {
            'observation': observation,
            'action_mask': self._action_mask(agent),
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

        action_index = int(action)
        played_card = card_from_action(action_index)
        legal_cards = self._legal_cards(agent)

        if played_card not in legal_cards:
            raise ValueError(f'Illegal action for {agent}: {played_card}')

        opponent = self._opponent(agent)
        self.hands[agent].play_card(played_card)
        self.hands[opponent].see_card(played_card)

        if self.lead_play is None:
            self.lead_play = LeadPlay(agent=agent, card=played_card)
            self.agent_selection = opponent
        else:
            self._resolve_trick(follower=agent, follower_card=played_card)

        self._update_infos()
        self._accumulate_rewards()

        if self.render_mode == 'human':
            self.render()

    def render(self) -> str | None:
        output = self._render_text()

        if self.render_mode == 'ansi':
            return output

        if self.render_mode == 'human':
            print(output)

        return None

    def close(self) -> None:
        return None

    def _resolve_trick(self, follower: AgentId, follower_card: Card) -> None:
        if self.lead_play is None:
            raise RuntimeError('Cannot resolve a trick without a lead card')

        leader = self.lead_play.agent
        leader_card = self.lead_play.card

        if leader == follower:
            raise RuntimeError('The same agent cannot play both cards')

        if self._lead_wins(leader_card, follower_card):
            winner = leader
            loser = follower
        else:
            winner = follower
            loser = leader

        trick_points = self._trick_points(leader_card, follower_card)

        # scores: real points/thirds for UI and challenge feedback
        # rewards: scaled for RL (points / max per hand)
        self.scores[winner] += trick_points
        scaled = float(trick_points) / self.MAX_HAND_POINTS
        self.rewards[winner] = scaled
        self.rewards[loser] = -scaled
        self.lead_play = None

        self._draw_cards(winner=winner, loser=loser)
        self.agent_selection = winner

        if self._game_finished():
            self._on_game_finished(winner)
            self.terminations = {agent: True for agent in self.agents}
            self._deads_step_first()

    def _draw_cards(self, winner: AgentId, loser: AgentId) -> None:
        if not self.pile:
            return

        if len(self.pile) < 2:
            raise RuntimeError('The pile contains an invalid number of cards')

        self._give_drawn_cards(winner=winner, loser=loser)

    def _legal_cards(self, agent: AgentId) -> list[Card]:
        lead_card = None if self.lead_play is None else self.lead_play.card
        return self.hands[agent].legal_plays(lead_card)

    def _action_mask(self, agent: AgentId) -> BinaryArray:
        mask = np.zeros(len(DECK), dtype=np.int8)

        if agent not in self.agents:
            return mask

        if agent != self.agent_selection:
            return mask

        if self.terminations[agent] or self.truncations[agent]:
            return mask

        for card in self._legal_cards(agent):
            mask[CARD_INDEX[card]] = 1

        return mask

    def _game_finished(self) -> bool:
        return (
            not self.pile
            and not self.hands['p1'].cards
            and not self.hands['p2'].cards
        )

    def _update_infos(self) -> None:
        for agent in self.agents:
            opponent = self._opponent(agent)
            self.infos[agent] = {
                **self._score_info(agent, opponent),
                'pile_size': len(self.pile),
                'hand_size': len(self.hands[agent]),
                'opponent_hand_size': len(self.hands[opponent]),
            }

    def _extra_observation_planes(self, agent: AgentId) -> list[BinaryArray]:
        del agent
        return []

    def _on_game_finished(self, winner: AgentId) -> None:
        del winner

    @abstractmethod
    def _deal(self, shuffled_cards: list[Card]) -> None:
        raise NotImplementedError

    @abstractmethod
    def _lead_wins(self, leader_card: Card, follower_card: Card) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _trick_points(self, leader_card: Card, follower_card: Card) -> int:
        raise NotImplementedError

    @abstractmethod
    def _give_drawn_cards(self, winner: AgentId, loser: AgentId) -> None:
        raise NotImplementedError

    @abstractmethod
    def _score_info(self, agent: AgentId, opponent: AgentId) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def _render_text(self) -> str:
        raise NotImplementedError

    @staticmethod
    def _opponent(agent: AgentId) -> AgentId:
        return 'p2' if agent == 'p1' else 'p1'


def wrap_cards_env(environment: Cards2PEnv) -> AECEnv:
    environment = wrappers.TerminateIllegalWrapper(
        environment,
        illegal_reward=environment.ILLEGAL_PLAY_PENALTY,
    )
    environment = wrappers.AssertOutOfBoundsWrapper(environment)
    environment = wrappers.OrderEnforcingWrapper(environment)
    return environment
