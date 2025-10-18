import random
import math
import torch

from src.action.actions import ALL_ACTIONS, LEVER_ACTIONS, EATER_ACTIONS
from src.common.configs import RunConfig
from src.geometry.collision import (
    check_collision_with_lever,
    check_collision_with_mag,
    solve_mapbox_colissions,
)
from src.nn.dqn import DQN
from src.state.general import State
from src.map.bounds import Bounds
from src.map.base_map import MapModel


class Agent:
    def __init__(
        self,
        initial_state: State,
        policy_net: DQN,
        nn_config: RunConfig,
        map: MapModel,
        device: torch.device = torch.device("cpu"),
    ):
        self.state: State = initial_state
        self.policy_net: DQN = policy_net
        self.config: RunConfig = nn_config
        self.map: MapModel = map
        self.device: torch.device = device
        self.all_actions: dict[str, int] = {
            ALL_ACTIONS[i]: i for i in range(len(ALL_ACTIONS))
        }
        self.bounds: Bounds = Bounds(
            x=(self.state.x, self.state.x + 1), y=(self.state.y, self.state.y + 1)
        )

    def _get_observables_from_state(self):
        """returns observable variables from state to use in NN"""
        return torch.Tensor([self.state.x, self.state.y])

    def _get_action_filter(self) -> list[int]:
        """Runs geometry heuristics to determine the list of allowed actions"""
        disallowed_actions: list[str] = []
        allow_mag_actions = True
        if self.map.mag is not None:
            allow_mag_actions = check_collision_with_mag(self, self.map.mag)
        allow_lever_actions = any(
            check_collision_with_lever(self, x) for x in self.map.levers
        )

        if not allow_mag_actions:
            disallowed_actions += EATER_ACTIONS
        if not allow_lever_actions:
            disallowed_actions += LEVER_ACTIONS

        disallowed_indices = [
            v for k, v in self.all_actions.items() if k in disallowed_actions
        ]
        return disallowed_indices

    def select_action(self, disallowed: list[int]) -> torch.Tensor:
        sample = random.random()
        allowed: list[int] = [
            a for a in self.all_actions.values() if a not in disallowed
        ]

        eps_th = self.config.EPS_END + (
            (self.config.EPS_START - self.config.EPS_END)
            * math.exp(-1.0 * self.state.ticks / self.config.EPS_DECAY)
        )

        if sample > eps_th:
            with torch.no_grad():
                state: torch.Tensor = self._get_observables_from_state()
                q_vals = self.policy_net(state).to(self.device)
                q_vals = q_vals.to(self.device)
                if len(disallowed) > 0:
                    min_val = torch.finfo(q_vals.dtype).min
                    for a in disallowed:
                        if 0 <= a < q_vals.shape[-1]:
                            q_vals[0, a] = min_val
                action_idx: torch.Tensor = q_vals.max(1).indices.view(1, 1)
                return action_idx.to(self.device, dtype=torch.long)

        chosen: int = random.choice(allowed)
        return torch.tensor([[chosen]], device=self.device, dtype=torch.long)
