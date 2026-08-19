from environments.scopa_env import Scopa2PEnv
from games.scopa import card_value
from games.deck import Card, DECK
from .dqn_agent import DQNAgent
from .scopa_dqn import PlayCardNN
from ..agent import AgentId


class GreedyTakeDQNAgent(DQNAgent):
    """
    Use the standard NN for play choice, then
    use greedy strategy to choose which card to take
    """

    def __init__(self, whoami: AgentId, net: PlayCardNN) -> None:
        super().__init__('DQN Agent with Greedy Take', whoami, net)

    def take_strategy(self, play: int, env: Scopa2PEnv, legal: list[tuple[Card, list[Card]]]) -> int:
        options = [opt[1] for opt in legal if opt[0] == DECK[play]]
        return max(
            range(len(options)),
            key=lambda i: sum(card_value(c) for c in options[i])
        )
