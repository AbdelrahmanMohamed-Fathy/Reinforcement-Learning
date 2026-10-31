"""Policy EVALUATION: compute V^pi for a fixed deterministic policy.

Bellman expectation equation (deterministic dynamics, so no sum over s'):

    V(s) = R(s, pi(s)) + gamma * V( next(s, pi(s)) )          for non-terminal s
    V(s) = 0                                                   for terminal s

Two solvers are provided:
  * iterative : repeated sweeps until the largest change < theta (textbook)
  * exact     : solve (I - gamma * P_pi) V = r_pi with a linear solver
                (fast for small S such as 25; O(S^3) so unusable for huge grids)
"""
from __future__ import annotations

import numpy as np

from .tabular_model import TabularModel


def _policy_transitions(model: TabularModel, policy: np.ndarray):
    idx = np.arange(model.n_states)
    return model.next_state[idx, policy], model.reward[idx, policy]


def evaluate_policy_iterative(
    model: TabularModel,
    policy: np.ndarray,
    gamma: float,
    V_init: np.ndarray | None = None,
    theta: float = 1e-10,
    max_sweeps: int = 100_000,
) -> tuple[np.ndarray, int]:
    """Returns (V, number_of_sweeps). ``V_init`` enables warm-starting."""
    ns, r = _policy_transitions(model, policy)
    V = np.zeros(model.n_states) if V_init is None else V_init.copy()
    V[model.terminal] = 0.0

    for sweep in range(1, max_sweeps + 1):
        V_new = np.where(model.terminal, 0.0, r + gamma * V[ns])  # vectorised sweep
        delta = np.max(np.abs(V_new - V))
        V = V_new
        if delta < theta:
            return V, sweep
    raise RuntimeError("Policy evaluation did not converge; check gamma < 1.")


def evaluate_policy_exact(model: TabularModel, policy: np.ndarray, gamma: float) -> tuple[np.ndarray, int]:
    """Direct linear solve. Returns (V, 1) so it is call-compatible with the iterative one."""
    ns, r = _policy_transitions(model, policy)
    S = model.n_states
    A = np.eye(S)
    live = np.flatnonzero(~model.terminal)
    A[live, ns[live]] -= gamma           # row s: V(s) - gamma * V(s') = r
    b = np.where(model.terminal, 0.0, r)
    return np.linalg.solve(A, b), 1
