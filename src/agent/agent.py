import random
import math
import torch

from src.action.actions import ALL_ACTIONS
from src.common.configs import RunConfig
from src.nn.dqn import DQN
from src.state.general import State
from src.map.bounds import Bounds


class Agent:
    def __init__(self, initial_state: State, policy_net: DQN, nn_config: RunConfig):
        self.state: State = initial_state
        self.policy_net: DQN = policy_net
        self.config: RunConfig = nn_config
        self.all_actions: dict[str, int] = {
            ALL_ACTIONS[i]: i for i in range(len(ALL_ACTIONS))
        }
        self.bounds: Bounds = Bounds(
            x=(self.state.x, self.state.x + 1), y=(self.state.y, self.state.y + 1)
        )

    # def select_action(self):
    #     sample = random.random()
    #     eps_th = self.config.EPS_END + (
    #         self.config.EPS_START - self.config.EPS_END
    #     ) * math.exp(-1.0 * self.state.ticks / self.config.EPS_DECAY)
    #     self.state.ticks += 1
    #     if sample > eps_th:
    #         return self.policy_net(self.state).max(1).indices.view(1, 1)
    #     else:
    #         return torch.tensor([[env.action_space.sample()]], device=, dtype=torch.long)
