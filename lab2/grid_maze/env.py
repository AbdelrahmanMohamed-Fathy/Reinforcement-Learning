"""Custom Gymnasium environment: the Grid Maze.

Objective: walk from a random start cell "S" to a random goal cell "G" while
avoiding randomly placed bad cells "X". A new random layout is drawn on every
``reset()``.

Observation : MultiDiscrete of 8 ints (default) =
              [agent_x, agent_y, goal_x, goal_y, bad1_x, bad1_y, bad2_x, bad2_y]
Actions     : Discrete(4) = {0: right, 1: up, 2: left, 3: down}, deterministic
Reward      : +1 reaching a goal, -1 reaching a bad cell, 0 otherwise
Termination : reaching a goal or a bad cell
Truncation  : handled by the TimeLimit wrapper applied in ``gym.register``
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .dynamics import (
    N_ACTIONS,
    Cell,
    cell_to_xy,
    move,
    reward_and_done,
)


class GridMazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 2}

    def __init__(
        self,
        size: int = 5,
        n_goals: int = 1,
        n_bad: int = 2,
        render_mode: str | None = None,
    ):
        super().__init__()
        if 1 + n_goals + n_bad > size * size:
            raise ValueError("Grid too small for the requested number of special cells.")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']}")

        self.size = size
        self.n_goals = n_goals
        self.n_bad = n_bad
        self.render_mode = render_mode

        n_coords = 2 * (1 + n_goals + n_bad)  # = 8 for the default maze
        self.observation_space = spaces.MultiDiscrete(np.full(n_coords, size, dtype=np.int64))
        self.action_space = spaces.Discrete(N_ACTIONS)

        self._agent: Cell | None = None
        self._start: Cell | None = None
        self._goals: list[Cell] = []
        self._bads: list[Cell] = []
        self._renderer = None

    # ------------------------------------------------------------------ helpers
    def _get_obs(self) -> np.ndarray:
        pts = [self._agent, *self._goals, *self._bads]
        return np.array([c for p in pts for c in p], dtype=np.int64)

    def _get_info(self) -> dict[str, Any]:
        ax, ay = self._agent
        dist = min(abs(ax - gx) + abs(ay - gy) for gx, gy in self._goals)
        return {"distance_to_goal": dist}

    def _validate_layout(self, layout: dict) -> tuple[Cell, list[Cell], list[Cell]]:
        agent = tuple(layout["agent"])
        goals = [tuple(g) for g in layout["goals"]]
        bads = [tuple(b) for b in layout["bads"]]
        cells = [agent, *goals, *bads]
        if len(goals) != self.n_goals or len(bads) != self.n_bad:
            raise ValueError("Layout has the wrong number of goal/bad cells.")
        if len(set(cells)) != len(cells):
            raise ValueError("Layout cells must all be distinct.")
        if any(not (0 <= c < self.size) for cell in cells for c in cell):
            raise ValueError("Layout cell outside the grid.")
        return agent, goals, bads

    # --------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)  # seeds self.np_random

        if options and "layout" in options:  # fixed layout (tests / demos)
            agent, goals, bads = self._validate_layout(options["layout"])
        else:  # random layout, all cells distinct
            n = 1 + self.n_goals + self.n_bad
            flat = self.np_random.choice(self.size * self.size, size=n, replace=False)
            pts = [cell_to_xy(int(c), self.size) for c in flat]
            agent, goals, bads = pts[0], pts[1 : 1 + self.n_goals], pts[1 + self.n_goals :]

        self._agent, self._start = agent, agent
        self._goals, self._bads = goals, bads

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), self._get_info()

    def step(self, action):
        if self._agent is None:
            raise RuntimeError("Call reset() before step().")
        self._agent = move(*self._agent, int(action), self.size)
        reward, terminated = reward_and_done(self._agent, self._goals, self._bads)

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, False, self._get_info()

    def render(self):
        if self.render_mode is None:
            gym.logger.warn("render() called without a render_mode; nothing to draw.")
            return None
        if self._agent is None:
            raise RuntimeError("Call reset() before render().")
        if self._renderer is None:
            from .renderer import GridMazeRenderer  # lazy: pygame only when drawing

            self._renderer = GridMazeRenderer(
                self.size, self.render_mode, fps=self.metadata["render_fps"]
            )
        return self._renderer.draw(self._agent, self._start, self._goals, self._bads)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
