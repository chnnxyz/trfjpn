from dataclasses import dataclass
from functools import partial
from typing import Callable, Self

from loss.hunger import linear_hunger


@dataclass
class State:
    """State of an agent including position, time, and reinforcement"""

    x: int
    y: int
    ticks: int = 0
    since_reinforcement: int = 0
    available_reinforcers: int = 0
    reinforcers: int = 0
    initial_hunger: float = 0
    hunger: float = initial_hunger
    hunger_function: Callable[..., float] | partial[float] = partial(
        linear_hunger, a=1, b=1, h_0=initial_hunger
    )

    def update_pos(self, x: int, y: int):
        self.x = x
        self.y = y
        self.ticks += 1

    def reinforce(self, n_reinf: int = 1):
        self.reinforcers += n_reinf
        self.since_reinforcement = 0

    def update_hunger(self):
        self.hunger = self.hunger_function(
            t=self.ticks,
            r_n=self.reinforcers,
        )
