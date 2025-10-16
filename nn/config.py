from dataclasses import dataclass
from typing import Callable
from torch import nn, Tensor
import torch.nn.functional as F


@dataclass
class LayerConfig:
    n_in: int
    n_out: int
    base_layer_type: type[nn.Module] = nn.Linear
    activation: Callable[..., Tensor] = F.relu
