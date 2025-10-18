import random

import torch
from torch.optim.adamw import AdamW

from src.common.configs import load_config
from src.nn.config import LayerConfig
from src.action.actions import ALL_ACTIONS
from src.nn.dqn import DQN
from src.nn.memory import ReplayMemory
from src.nn.setup import setup_torch_device
from src.agent.agent import Agent
from src.state.general import State
from src.map.base_map import MapModel

if __name__ == "__main__":
    cfg = load_config("./run_setup.yaml")
    device = setup_torch_device()
    actions = ALL_ACTIONS
    n_actions = len(ALL_ACTIONS)
    n_observations = 3
    layer_config = [
        LayerConfig(n_observations, 100),
        LayerConfig(100, 100),
        LayerConfig(100, n_actions),
    ]
    policy_net = DQN(layer_config).to(device)
    target_net = DQN(layer_config).to(device)
    print("device:", device)
    print("policy_net class:", policy_net.__class__)
    print("number of parameters:", sum(1 for _ in policy_net.parameters()))
    print("named parameters:", list(policy_net.named_parameters()))
    map = MapModel(w=5.26, h=4.21, ey=(4.21 - 0.842) / 2, l1y=0.421, l2y=None)
    target_net.load_state_dict(policy_net.state_dict())
    optim = AdamW(policy_net.parameters(), lr=cfg.LR, amsgrad=True)
    memory = ReplayMemory(10000)
    agent = Agent(
        initial_state=State(x=random.uniform(0, 5.26), y=random.uniform(0, 4.21)),
        policy_net=policy_net,
        nn_config=cfg,
        map=map,
        device=device,
        memory=memory,
        optimizer=optim,
        target_net=target_net,
    )
    torch.save(agent.policy_net, "dqn.pt")
