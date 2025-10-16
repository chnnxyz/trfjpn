from collections import deque, namedtuple
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from nn.config import LayerConfig


class DQN(nn.Module):
    """Deep Q Network builder"""

    def __init__(self, layers: List[LayerConfig]):
        super(DQN, self).__init__()
        self.layers = layers

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            compiled_layer = layer.base_layer_type(layer.n_in, layer.n_out)
            if i == len(self.layers) - 1:
                return compiled_layer(x)

            x = layer.activation(compiled_layer(x))
