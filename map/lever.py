from dataclasses import dataclass, field

from map.bounds import Bounds


@dataclass
class Lever:
    y0: float
    h: float
    d: float
    x1: float

    bounds: Bounds = field(init=False)

    def __post_init__(self):
        self.bounds = Bounds(
            x=(self.x1 - self.d, self.x1), y=(self.y0, self.y0 + self.h)
        )
