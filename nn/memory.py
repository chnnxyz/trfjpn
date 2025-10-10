import random

from collections import namedtuple, deque
from typing import Any


# TODO: Pyright types
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    """Memory object for DQN calculation"""

    def __init__(self, capacity: int):
        self.memory: deque[Transition] = deque([], maxlen=capacity)

    def __len__(self) -> int:
        return len(self.memory)

    def push(self, *args: Any):
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)
