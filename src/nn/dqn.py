from collections import deque, namedtuple
from typing import List, Callable

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.nn.config import LayerConfig


class _ActivationModule(nn.Module):
    """Wrap a functional activation so it can be used inside nn.Sequential."""

    def __init__(self, fn: Callable[..., torch.Tensor]):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


class DQN(nn.Module):
    """Deep Q Network builder — builds and registers modules at construction time."""

    def __init__(self, layers: List[LayerConfig]):
        super().__init__()
        modules = []
        for i, lc in enumerate(layers):
            # instantiate the base layer (e.g. nn.Linear)
            modules.append(lc.base_layer_type(lc.n_in, lc.n_out))

            # append activation for all but the last layer
            if i < len(layers) - 1 and lc.activation is not None:
                modules.append(_ActivationModule(lc.activation))

        # register as a single module so parameters are visible to optimizers
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
