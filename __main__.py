from torch.optim.adamw import AdamW
from common.configs import load_config
from nn.config import LayerConfig
from action.actions import ALL_ACTIONS
from nn.dqn import DQN
from nn.memory import ReplayMemory
from nn.setup import setup_torch_device

if __name__ == "__main__":
    cfg = load_config("./run_setup.yaml")
    device = setup_torch_device()
    actions = ALL_ACTIONS
    n_actions = len(ALL_ACTIONS)
    n_observations = 2
    layer_config = [
        LayerConfig(n_observations, 100),
        LayerConfig(100, 100),
        LayerConfig(100, n_actions),
    ]
    policy_net = DQN(layer_config).to(device)
    target_net = DQN(layer_config).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optim = AdamW(policy_net.parameters(), lr=cfg.LR, amsgrad=True)
    memory = ReplayMemory(10000000)
