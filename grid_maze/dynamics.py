"""Pure, framework-free grid dynamics.

This module is the SINGLE source of truth for "how the maze works"
(movement, walls, rewards, termination). Both the Gymnasium environment and
the tabular Policy Iteration model import from here, so the planner can never
disagree with the environment it is planning for.

Coordinate convention
---------------------
Positions are ``(x, y)`` with ``(0, 0)`` in the TOP-LEFT corner, x growing to
the right and y growing downward (like screen pixels). Therefore the action
"up" means ``y - 1``.
"""
from __future__ import annotations

from typing import Iterable, Sequence

# --- Actions: {right, up, left, down} -------
RIGHT, UP, LEFT, DOWN = 0, 1, 2, 3
ACTION_NAMES = ("right", "up", "left", "down")
ACTION_ARROWS = ("→", "↑", "←", "↓")
ACTION_DELTAS = ((1, 0), (0, -1), (-1, 0), (0, 1))  # (dx, dy)
N_ACTIONS = 4

# --- Reward function ------------------------
GOAL_REWARD = 1.0
BAD_REWARD = -1.0
STEP_REWARD = 0.0

Cell = tuple[int, int]


def cell_to_xy(cell: int, size: int) -> Cell:
    """Flat index -> (x, y).  Row-major: index = y * size + x."""
    return cell % size, cell // size


def xy_to_cell(x: int, y: int, size: int) -> int:
    """(x, y) -> flat index."""
    return y * size + x


def move(x: int, y: int, action: int, size: int) -> Cell:
    """Deterministic move. Bumping into the outer wall leaves the agent in place."""
    dx, dy = ACTION_DELTAS[action]
    nx, ny = x + dx, y + dy
    if 0 <= nx < size and 0 <= ny < size:
        return nx, ny
    return x, y


def reward_and_done(pos: Cell, goals: Iterable[Cell], bads: Iterable[Cell]) -> tuple[float, bool]:
    """Reward for ENTERING ``pos`` and whether that ends the episode."""
    if pos in set(goals):
        return GOAL_REWARD, True
    if pos in set(bads):
        return BAD_REWARD, True
    return STEP_REWARD, False


def parse_observation(obs: Sequence[int], n_goals: int, n_bad: int) -> tuple[Cell, list[Cell], list[Cell]]:
    """Split the flat observation back into (agent, goals, bads).

    Layout of the observation vector (8 ints for 1 goal + 2 bad cells):
        [agent_x, agent_y, goal_x, goal_y, bad1_x, bad1_y, bad2_x, bad2_y]
    """
    pts = [(int(obs[i]), int(obs[i + 1])) for i in range(0, len(obs), 2)]
    agent = pts[0]
    goals = pts[1 : 1 + n_goals]
    bads = pts[1 + n_goals : 1 + n_goals + n_bad]
    return agent, goals, bads
