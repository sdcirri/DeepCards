from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Literal, TypeAlias

from gymnasium.utils import seeding


AgentId: TypeAlias = Literal['p1', 'p2']


class Cards2PAgent(ABC):
    whoami: AgentId
    name: str

    def __init__(self, whoami: AgentId, name: str) -> None:
        self.whoami = whoami
        self.name = name
        self.np_random, self.np_random_seed = seeding.np_random(None)

    @abstractmethod
    def step(self, env: Any) -> Any:
        pass


def challenge_step(env: Any, agent1: Cards2PAgent, agent2: Cards2PAgent) -> None:
    env.reset()
    while not env.terminations['p1']:
        if env.agent_selection == 'p1':
            env.step(agent1.step(env))
        else:
            env.step(agent2.step(env))


def run_challenge(
        agent1: Cards2PAgent,
        agent2: Cards2PAgent,
        episodes: int,
        one_episode: Callable[[Cards2PAgent, Cards2PAgent, str | None], tuple[int, int]],
        render_mode: str | None
) -> tuple[int, int]:
    # Render disables process parallelism (pygame is process-local).
    pool_size = 1 if render_mode is not None else os.cpu_count()
    with ProcessPoolExecutor(max_workers=pool_size) as executor:
        results = executor.map(
            one_episode,
            [agent1] * episodes,
            [agent2] * episodes,
            [render_mode] * episodes
        )

    points_1, points_2 = 0, 0
    for result in results:
        points_1 += result[0]
        points_2 += result[1]

    return points_1, points_2


def run_human_challenge(
    agent1: Cards2PAgent,
    agent2: Cards2PAgent,
    episodes: int,
    *,
    env_factory: Callable[[str | None], Any],
    kind: Literal['briscola', 'tressette', 'scopa'],
    score_of: Callable[[Any], tuple[int, int]],
) -> tuple[int, int]:
    """
    Run episodes in-process, showing them in one split pygame window when possible.
    """
    from environments.piacentine_viz import (
        MAX_PARALLEL_EPISODES,
        begin_parallel_session,
        end_parallel_session,
        paint_parallel_static,
        render_parallel_round,
        set_parallel_envs,
    )

    if episodes < 1:
        return 0, 0

    parallel = episodes <= MAX_PARALLEL_EPISODES
    n_slots = episodes if parallel else 1
    title = f'{agent1.name} vs {agent2.name}'
    begin_parallel_session(kind, n_slots, title=title)

    points_1 = 0
    points_2 = 0

    try:
        if parallel:
            # Logic envs do not auto-render; the viz advances every pane together.
            wrapped_envs = []
            raw_envs = []
            for slot in range(episodes):
                env = env_factory(None)
                raw = env.unwrapped
                raw.display_names = {'p1': agent1.name, 'p2': agent2.name}
                raw.render_slot = slot
                env.reset(seed=slot)
                wrapped_envs.append(env)
                raw_envs.append(raw)

            set_parallel_envs(raw_envs)
            paint_parallel_static(raw_envs, kind)

            alive = list(range(episodes))
            while alive:
                stepped_raw: list[Any] = []
                next_alive: list[int] = []
                for slot in alive:
                    env = wrapped_envs[slot]
                    if env.terminations['p1']:
                        continue
                    if env.agent_selection == 'p1':
                        env.step(agent1.step(env))
                    else:
                        env.step(agent2.step(env))
                    stepped_raw.append(raw_envs[slot])
                    if not env.terminations['p1']:
                        next_alive.append(slot)

                if stepped_raw:
                    render_parallel_round(stepped_raw, kind)
                alive = next_alive

            for env in wrapped_envs:
                raw = env.unwrapped
                s1, s2 = score_of(raw)
                points_1 += s1
                points_2 += s2
                env.close()
        else:
            # Too many episodes for a grid: play them one-by-one in a single pane.
            for episode in range(episodes):
                env = env_factory('human')
                raw = env.unwrapped
                raw.display_names = {'p1': agent1.name, 'p2': agent2.name}
                raw.render_slot = 0
                challenge_step(env, agent1, agent2)
                s1, s2 = score_of(raw)
                points_1 += s1
                points_2 += s2
                env.close()
    finally:
        end_parallel_session()

    return points_1, points_2
