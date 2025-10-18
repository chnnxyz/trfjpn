import random

from dataclasses import dataclass
from functools import partial
from typing import Callable

from src.loss.hunger import linear_hunger


@dataclass
class State:
    """State of an agent including position, time, and reinforcement"""

    x: float  # observable by agent
    y: float  # observable by agent
    ticks: int = 0
    trial_ticks: int = 0
    available_reinforcers: int = 0
    reinforcers: int = 0
    initial_hunger: float = 0
    hunger: float = initial_hunger
    trial_completed: bool = False
    completed_at: int = 0
    iti: int = random.randint(10, 30)
    hunger_function: Callable[..., float] | partial[float] = partial(
        linear_hunger, a=0.2, b=20, h_0=initial_hunger
    )
