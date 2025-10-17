from dataclasses import dataclass, field

from map.bounds import Bounds


@dataclass
class Mag:
    y0: float
    x1: float
    h: float
    bounds: Bounds = field(init=False)

    def __post_init__(self):
        self.bounds = Bounds(x=(self.x1, self.x1), y=(self.y0, self.y0 + self.h))
