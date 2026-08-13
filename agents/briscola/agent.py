from agents.agent import Cards2PAgent, challenge_step, run_challenge
from environments.briscola_env import env as env_factory


def _one_episode(a1: Cards2PAgent, a2: Cards2PAgent) -> tuple[int, int]:
    env = env_factory(render_mode=None)
    challenge_step(env, a1, a2)
    raw = env.unwrapped
    return raw.scores['p1'], raw.scores['p2']


def challenge(agent1: Cards2PAgent, agent2: Cards2PAgent, episodes: int) -> tuple[int, int]:
    return run_challenge(agent1, agent2, episodes, _one_episode)
