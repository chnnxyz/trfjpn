from typing import Optional
class BaseMap:
    def __init__(self,
                 w: int,
                 h: int,
                 l1y: Optional[int],
                 l2y: int,
                 ey: int
    ):
        self.w = w
        self.h = h
        self.l1y = l1y
        self.l2y = l2y
        self.ey = ey

    def init_state(self):
        pass


    def map_states_action(self):
        pass
