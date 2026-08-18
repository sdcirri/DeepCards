from environments.scopa_env import Scopa2PEnv, AgentId, Action

from ..agent import Cards2PAgent


class GreedyAgent(Cards2PAgent):
    """
    Try to take as many cards as possible
    """
    def __init__(self, whoami: AgentId) -> None:
        super().__init__(whoami, 'Greedy Agent')

    def step(self, env: Scopa2PEnv) -> Action | None:
        if env.agent_selection != self.whoami:
            return None

        legal = env.hands[self.whoami].scopa_legal_plays(env.table)
        played = max(legal, key=lambda pt: len(pt[1]))

        return legal.index(played)
