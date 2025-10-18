import random
import math
import torch

from copy import deepcopy

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
from src.nn.memory import ReplayMemory, Transition


class Agent:
    def __init__(
        self,
        initial_state: State,
        policy_net: DQN,
        target_net: DQN,
        nn_config: RunConfig,
        map: MapModel,
        optimizer,
        device: torch.device = torch.device("cpu"),
        memory: ReplayMemory = ReplayMemory(10000),
    ):
        self.state: State = initial_state
        self.policy_net: DQN = policy_net
        self.target_net: DQN = target_net
        self.optimizer = optimizer
        self.config: RunConfig = nn_config
        self.map: MapModel = map
        self.map_backup: MapModel = deepcopy(map)
        self.device: torch.device = device
        self.all_actions: dict[str, int] = {
            ALL_ACTIONS[i]: i for i in range(len(ALL_ACTIONS))
        }
        self.bounds: Bounds = Bounds(
            x=(self.state.x, self.state.x + 1), y=(self.state.y, self.state.y + 1)
        )
        self.memory: ReplayMemory = memory

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

    def _optimize_model(self):
        if len(self.memory) < self.config.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.config.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.config.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )
        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self.config.GAMMA
        ) + reward_batch

        # Compute Huber loss
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
