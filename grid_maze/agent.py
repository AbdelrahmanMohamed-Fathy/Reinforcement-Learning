"""Applies Policy Iteration to the Grid Maze environment.

The environment gives a NEW random layout at every reset, so the agent
  1. reads the layout out of the 8-integer observation,
  2. builds the 25-state tabular model for that layout,
  3. runs Policy Iteration on it (once per distinct layout; results are cached),
  4. then simply looks up ``policy[agent_cell]`` to act.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .dynamics import parse_observation, xy_to_cell
from .policy_iteration import PolicyIterationResult, policy_iteration
from .tabular_model import TabularModel, build_model


class PolicyIterationAgent:
    def __init__(self, size: int = 5, n_goals: int = 1, n_bad: int = 2, gamma: float = 0.99, **pi_kwargs):
        self.size, self.n_goals, self.n_bad, self.gamma = size, n_goals, n_bad, gamma
        self.pi_kwargs = pi_kwargs
        self._cache: dict[tuple, tuple[TabularModel, PolicyIterationResult]] = {}

    @classmethod
    def for_env(cls, env, **kwargs) -> "PolicyIterationAgent":
        u = env.unwrapped
        return cls(u.size, u.n_goals, u.n_bad, **kwargs)

    def solve(self, obs) -> tuple[TabularModel, PolicyIterationResult]:
        """Plan for the layout contained in ``obs`` (cached by layout, NOT by agent position)."""
        agent, goals, bads = parse_observation(obs, self.n_goals, self.n_bad)
        key = (frozenset(goals), frozenset(bads))
        if key not in self._cache:
            model = build_model(self.size, goals, bads)
            self._cache[key] = (model, policy_iteration(model, self.gamma, **self.pi_kwargs))
        return self._cache[key]

    def act(self, obs) -> int:
        agent, _, _ = parse_observation(obs, self.n_goals, self.n_bad)
        _, result = self.solve(obs)
        return int(result.policy[xy_to_cell(*agent, self.size)])


def run_episode(env, agent: PolicyIterationAgent, seed: int | None = None, options: dict | None = None) -> dict[str, Any]:
    """Play one episode with the agent; returns a summary dict."""
    obs, _ = env.reset(seed=seed, options=options)
    _, result = agent.solve(obs)
    total, steps, terminated, truncated = 0.0, 0, False, False
    while not (terminated or truncated):
        obs, r, terminated, truncated, _ = env.step(agent.act(obs))
        total += r
        steps += 1
    return {
        "return": total,
        "steps": steps,
        "success": terminated and total > 0,
        "pi_iterations": result.iterations,
        "truncated": truncated,
    }
