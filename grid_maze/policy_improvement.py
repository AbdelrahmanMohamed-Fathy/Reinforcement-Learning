"""Policy IMPROVEMENT: act greedily with respect to the current V^pi.

    Q(s, a)  = R(s, a) + gamma * V(next(s, a))
    pi'(s)   = argmax_a Q(s, a)

Anti-cycling rule: a state's action is only switched when the new action is
STRICTLY better (by more than ``tol``) than the action it already uses.
Without this, ties between equally good actions (very common here, e.g. two
shortest paths) could flip back and forth forever and the policy would never
be declared "stable".
"""
from __future__ import annotations

import numpy as np

from .tabular_model import TabularModel


def q_values(model: TabularModel, V: np.ndarray, gamma: float) -> np.ndarray:
    return model.reward + gamma * V[model.next_state]  # (S, A)


def improve_policy(
    model: TabularModel, V: np.ndarray, policy: np.ndarray, gamma: float, tol: float = 1e-9
) -> tuple[np.ndarray, int]:
    """Returns (new_policy, number_of_states_whose_action_changed)."""
    Q = q_values(model, V, gamma)
    idx = np.arange(model.n_states)
    best = Q.argmax(axis=1)
    better = Q[idx, best] > Q[idx, policy] + tol
    changed = better & ~model.terminal
    return np.where(changed, best, policy), int(changed.sum())
