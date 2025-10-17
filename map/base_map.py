class BaseMap:
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
        self.l1y: float = l1y
        self.l2y: float = l2y if l2y is not None else -999999.0
        self.lw: float = lw
        self.ey: float = ey if ey is not None else -999999.0

    def init_state(self):
        pass

    def map_states_action(self):
        pass
