from environments.scopa_env import Scopa2PEnv, AgentId, Action

from ..agent import Cards2PAgent


class RandomAgent(Cards2PAgent):
    """
    Play a random card
    """
    def __init__(self, whoami: AgentId) -> None:
        super().__init__(whoami, 'Random Agent')

    def step(self, env: Scopa2PEnv) -> Action | None:
        if env.agent_selection != self.whoami:
            return None

        legal = env.hands[self.whoami].scopa_legal_plays(env.table)
        return self.np_random.integers(0, len(legal))
