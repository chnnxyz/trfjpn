from typing import Dict, Tuple, Optional, List
from action.actions import DIRECTIONAL_ACTIONS, INACTION, EATER_ACTIONS, LEVER_ACTIONS

def build_actions_by_position(
    w: int,
    h:int,
    l1y: Optional[int],
    l2y: Optional[int],
    ey: Optional[int]
) -> Dict[Tuple[int,int], List[str]]:
    # This is ugly O(n^2) but we can fix that later.
    #
    # Assign special values so they never match if any skinnerbox element
    # is missing

    if l1y is None:
        l1y = -99999
    if l2y is None:
        l2y = -99999
    if ey is None:
        ey = -99999

    action_map = {}
    for i in range(w + 1):
        for j in range(h + 1):
            actions = DIRECTIONAL_ACTIONS + INACTION
            if i == 0:
                actions.remove("LEFT")
            if j == 0:
                actions.remove("UP")
            if i == w or ((j == l1y or j == l2y or j == ey) and i == w - 1):
                actions.remove("RIGHT")
            if j == h or (i == w and (j == l2y - 1 or j == l1y - 1 or j == ey - 1)):
                actions.remove("DOWN")
