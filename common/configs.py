from dataclasses import dataclass
from typing import Any, Dict
import yaml


@dataclass
class RunConfig:
    BATCH_SIZE: int = 128
    GAMMA: float = 0.99
    EPS_START: float = 0.9
    EPS_END: float = 0.01
    EPS_DECAY: int = 2500
    TAU: float = 0.005
    LR: float = 0.0003


def load_config(path="config.yaml") -> RunConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)  # type: ignore[assignment]
    return RunConfig(**data)
