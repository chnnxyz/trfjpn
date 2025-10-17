from src.agent.agent import Agent
from src.map.lever import Lever
from src.map.base_map import MapModel
from src.map.mag import Mag


def solve_left_collision(agent: Agent, x0: float):
    if agent.bounds.x[0] <= x0:
        agent.state.x = 0
        agent.bounds.x = (0, 1)


def solve_right_collision(agent: Agent, x1: float):
    if agent.bounds.x[1] >= x1:
        agent.state.x = x1 - 1.0
        agent.bounds.x = (x1 - 1.0, x1)


def solve_top_collision(agent: Agent, y0: float):
    if agent.bounds.y[0] <= y0:
        agent.state.y = y0
        agent.bounds.y = (y0, y0 + 1.0)


def solve_bottom_collision(agent: Agent, y1: float):
    if agent.bounds.y[1] >= y1:
        agent.state.y = y1 - 1.0
        agent.bounds.y = (y1 - 1.0, y1)


def check_collision_with_mag(agent: Agent, mag: Mag) -> bool:
    if (agent.bounds.x[1] >= mag.x1) and (
        mag.bounds.y[0] <= agent.bounds.y[1] <= mag.bounds.y[1]
        or mag.bounds.y[0] <= agent.bounds.y[0] <= mag.bounds.y[1]
    ):
        # Solve collision and push agent back

        # allow for head entry action
        return True
    return False
