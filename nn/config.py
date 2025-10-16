from dataclasses import dataclass

from torch import nn


@dataclass
class LayerConfig:
    n_in: int
    n_out: int
    base_layer_type: type[nn.Module] = nn.Linear
    activation: type[nn.Module] = nn.ReLU
