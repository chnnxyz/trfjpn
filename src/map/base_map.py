from src.map.lever import Lever
from src.map.mag import Mag


class MapModel:
    def __init__(
        self,
        w: float,
        h: float,
        l1y: float,
        l2y: float | None,
        ey: float | None,
        lw: float = 0.842,
        ld: float = 0.333,
        ew: float = 0.842,
    ):
        self.w: float = w
        self.h: float = h
        self.mag: Mag | None = None if ey is not None else Mag(ey, w, ew)
        self.levers: list[Lever] = [Lever(l1y, lw, ld, w)]
        if l2y is not None:
            self.levers.append(Lever(l2y, lw, ld, w))
