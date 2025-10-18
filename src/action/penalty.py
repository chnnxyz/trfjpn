from src.action.actions import (
    LEVER_ACTIONS,
    EATER_ACTIONS,
    INACTION,
    ALL_ACTIONS,
)


def _calculate_penalty(action: str) -> float:
    if action in INACTION:
        return 0
    if action in EATER_ACTIONS:
        return 5
    if action in LEVER_ACTIONS:
        return 7

    return 8


ACTION_PENALTIES = {a: _calculate_penalty(a) for a in ALL_ACTIONS}
