"""Small text helpers for printing policies / values / traced paths."""
from __future__ import annotations

import numpy as np

from .dynamics import ACTION_ARROWS, Cell, cell_to_xy, xy_to_cell
from .tabular_model import TabularModel


def policy_to_text(policy: np.ndarray, size: int, goals, bads, agent: Cell | None = None) -> str:
    """ASCII grid: arrows for the policy, G / X for special cells, S marks the start."""
    rows = []
    for y in range(size):
        row = []
        for x in range(size):
            c = (x, y)
            if c in goals:
                row.append(" G ")
            elif c in bads:
                row.append(" X ")
            else:
                arrow = ACTION_ARROWS[int(policy[xy_to_cell(x, y, size)])]
                row.append(f"[{arrow}]" if c == agent else f" {arrow} ")
        rows.append("".join(row))
    return "\n".join(rows)


def values_to_text(V: np.ndarray, size: int) -> str:
    return "\n".join(
        " ".join(f"{V[xy_to_cell(x, y, size)]:5.2f}" for x in range(size)) for y in range(size)
    )


def trace_policy(model: TabularModel, policy: np.ndarray, start: Cell, max_steps: int = 200) -> list[Cell]:
    """Follow the policy from ``start`` until a terminal cell (or max_steps)."""
    s = xy_to_cell(*start, model.size)
    path = [start]
    for _ in range(max_steps):
        if model.terminal[s]:
            break
        s = int(model.next_state[s, policy[s]])
        path.append(cell_to_xy(s, model.size))
    return path
