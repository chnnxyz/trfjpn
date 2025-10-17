import random
import math
import torch

from action.actions import ALL_ACTIONS
from common.configs import RunConfig
from nn.dqn import DQN
from state.general import State



class Agent:
    def __init__(self, initial_state: State, policy_net: DQN, nn_config: RunConfig):
        self.state: State = initial_state
        self.policy_net: DQN = policy_net
        self.config: RunConfig = nn_config
        self.all_actions: dict[str, int] = {
            ALL_ACTIONS[i]: i for i in range(len(ALL_ACTIONS))
        }


    def select_action(self):
        sample = random.random()
        eps_th = self.config.EPS_END + (
            self.config.EPS_START - self.config.EPS_END
        ) * math.exp(-1.0 * self.state.ticks / self.config.EPS_DECAY)
        self.state.ticks += 1
        if sample > eps_th:
            return self.policy_net(self.state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=, dtype=torch.long)
