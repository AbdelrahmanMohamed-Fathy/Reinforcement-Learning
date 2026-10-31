"""Turn ONE maze layout into the tabular MDP that Policy Iteration needs.

Key idea (state-space reduction)
--------------------------------
The observation has 8 integers, but goal/bad cells never move during an
episode. Once a layout is fixed, the only thing that changes is the agent's
cell, so the MDP has just ``size * size`` states (25 for the 5x5 maze) instead
of ~345,000 joint observations. We build one tiny model per layout.

State id = flat cell index ``y * size + x``.
Terminal cells (goals + bad cells) are absorbing with reward 0 afterwards, so
their value is 0 and the reward for reaching them is paid on the transition INTO
them (matching the environment).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .dynamics import N_ACTIONS, Cell, cell_to_xy, move, reward_and_done, xy_to_cell


@dataclass(frozen=True)
class TabularModel:
    size: int
    next_state: np.ndarray  # (S, A) int  -> deterministic successor of (s, a)
    reward: np.ndarray      # (S, A) float -> reward for taking a in s
    terminal: np.ndarray    # (S,)  bool

    @property
    def n_states(self) -> int:
        return self.next_state.shape[0]

    @property
    def n_actions(self) -> int:
        return self.next_state.shape[1]


def build_model(size: int, goals: Sequence[Cell], bads: Sequence[Cell]) -> TabularModel:
    n_states = size * size
    next_state = np.zeros((n_states, N_ACTIONS), dtype=np.int64)
    reward = np.zeros((n_states, N_ACTIONS), dtype=np.float64)
    terminal = np.zeros(n_states, dtype=bool)

    for s in range(n_states):
        x, y = cell_to_xy(s, size)
        if (x, y) in goals or (x, y) in bads:
            terminal[s] = True
            next_state[s, :] = s  # absorbing, reward stays 0
            continue
        for a in range(N_ACTIONS):
            nx, ny = move(x, y, a, size)  # same dynamics as the env
            r, _ = reward_and_done((nx, ny), goals, bads)
            next_state[s, a] = xy_to_cell(nx, ny, size)
            reward[s, a] = r
    return TabularModel(size, next_state, reward, terminal)
