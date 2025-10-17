from src.agent.agent import Agent
from src.map.lever import Lever
from src.map.base_map import MapModel
from src.map.mag import Mag
from src.map.bounds import Bounds


# Colissions against map
def _solve_left_collision(agent: Agent, x0: float):
    if agent.bounds.x[0] <= x0:
        agent.state.x = 0
        agent.bounds.x = (0, 1)


def _solve_right_collision(agent: Agent, x1: float):
    if agent.bounds.x[1] >= x1:
        agent.state.x = x1 - 1.0
        agent.bounds.x = (x1 - 1.0, x1)


def _solve_top_collision(agent: Agent, y0: float):
    if agent.bounds.y[0] <= y0:
        agent.state.y = y0
        agent.bounds.y = (y0, y0 + 1.0)


def _solve_bottom_collision(agent: Agent, y1: float):
    if agent.bounds.y[1] >= y1:
        agent.state.y = y1 - 1.0
        agent.bounds.y = (y1 - 1.0, y1)


def solve_mapbox_colissions(
    agent: Agent, x0: float, x1: float, y0: float, y1: float
) -> None:
    _solve_left_collision(agent, x0)
    _solve_right_collision(agent, x1)
    _solve_top_collision(agent, y0)
    _solve_bottom_collision(agent, y1)


# Collisions against box elements
def check_collision_with_mag(agent: Agent, mag: Mag) -> bool:
    """Checks if an agent is adjacent to the mag, and returns true if it is.
    This is used for action filtering.
    """
    if (agent.bounds.x[1] >= mag.x1) and (
        mag.bounds.y[0] <= agent.bounds.y[1] <= mag.bounds.y[1]
        or mag.bounds.y[0] <= agent.bounds.y[0] <= mag.bounds.y[1]
    ):
        # Solve collision and push agent back
        _solve_right_collision(agent, mag.x1)

        # allow for head entry action
        return True
    return False


def check_collision_with_lever(agent: Agent, lever: Lever) -> bool:
    """Checks if the agent is adjacent to a lever.
    Returns True if adjacent (touching or overlapping), otherwise False.
    """

    agent_bounds: Bounds = agent.bounds
    lever_bounds: Bounds = lever.bounds

    ax_min, ax_max = agent_bounds.x
    ay_min, ay_max = agent_bounds.y
    lx_min, lx_max = lever_bounds.x
    ly_min, ly_max = lever_bounds.y

    # Check for overlap or adjacency on both axes
    x_adjacent = ax_max >= lx_min and ax_min <= lx_max
    y_adjacent = ay_max >= ly_min and ay_min <= ly_max

    if x_adjacent:
        _solve_left_collision(agent, lx_min)
    if y_adjacent:
        _solve_top_collision(agent, ly_max)
        _solve_bottom_collision(agent, ly_min)

    return x_adjacent or y_adjacent
