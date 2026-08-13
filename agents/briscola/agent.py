from agents.agent import Cards2PAgent, challenge_step, run_challenge
from environments.briscola_env import env as env_factory


def _one_episode(a1: Cards2PAgent, a2: Cards2PAgent, render_mode: str) -> tuple[int, int]:
    env = env_factory(render_mode=render_mode)
    raw = env.unwrapped
    raw.display_names = {'p1': a1.name, 'p2': a2.name}
    challenge_step(env, a1, a2)
    return raw.scores['p1'], raw.scores['p2']


def challenge(agent1: Cards2PAgent, agent2: Cards2PAgent, episodes: int, render_mode: str | None) -> tuple[int, int]:
    return run_challenge(agent1, agent2, episodes, _one_episode, render_mode)
