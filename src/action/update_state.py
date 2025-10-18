"""Module to update state objects based o the selected action"""

from src.action.actions import (
    DIRECTIONAL_ACTIONS,
    LEVER_ACTIONS,
    ALL_ACTIONS,
    EATER_ACTIONS,
)
from src.action.penalty import ACTION_PENALTIES
from src.agent.agent import Agent


def update_state(action: str, agent: Agent, trial_type: str):
    if action in DIRECTIONAL_ACTIONS:
        if action == "LEFT":
            agent.state.x -= 0.175
            agent.bounds.x = (agent.state.x, agent.state.x + 1)
        if action == "RIGHT":
            agent.state.x += 0.175
            agent.bounds.x = (agent.state.x, agent.state.x + 1)
        if action == "UP":
            agent.state.y -= 0.175
            agent.bounds.y = (agent.state.y, agent.state.y + 1)
        if action == "DOWN":
            agent.state.y += 0.175
            agent.bounds.y = (agent.state.y, agent.state.y + 1)
    if action in LEVER_ACTIONS:
        # if fixed interval, check for reinforcer interval
        if trial_type == "FI":
            if agent.state.trial_ticks * 5 >= 30:
                # Increase available reinforcers
                agent.state.available_reinforcers += 1
                # Remove lever
                agent.map.levers = []
    if action in EATER_ACTIONS:
        agent.state.reinforcers += agent.state.available_reinforcers
    # finally update the state to use the action penalty
    agent.state.hunger = agent.state.hunger_function(
        t=agent.state.ticks,
        r_n=agent.state.reinforcers,
        penalty=0.2 * ACTION_PENALTIES[action],
    )
    agent.state.ticks += 1
