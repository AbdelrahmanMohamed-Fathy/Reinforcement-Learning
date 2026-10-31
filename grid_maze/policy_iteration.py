"""Policy Iteration = alternate evaluation and improvement until the policy is stable.

    pi_0 -> V^{pi_0} -> pi_1 -> V^{pi_1} -> ... -> pi_* (no state changes)

Note on ``warm_start``: reusing the previous V looks like a free speed-up but
measured SLOWER here (see analysis.py q3). Policies that loop forever (e.g. bump
a wall) make the stale values decay only as gamma^k, so evaluation needs
thousands of sweeps to reach theta. Cold start (V=0) is therefore the default.

One "iteration" below = one evaluation step + one improvement step. The LAST
iteration is the one in which improvement changes nothing (that is how we know
we have converged), so ``iterations`` includes that confirming pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .policy_evaluation import evaluate_policy_exact, evaluate_policy_iterative
from .policy_improvement import improve_policy
from .tabular_model import TabularModel


@dataclass
class PolicyIterationResult:
    policy: np.ndarray                 # (S,) greedy action per state
    V: np.ndarray                      # (S,) value of the final policy
    iterations: int                    # evaluate+improve rounds (incl. the confirming one)
    converged: bool
    history: list[dict] = field(default_factory=list)  # per-iteration diagnostics

    @property
    def total_eval_sweeps(self) -> int:
        return sum(h["eval_sweeps"] for h in self.history)


def policy_iteration(
    model: TabularModel,
    gamma: float = 0.99,
    theta: float = 1e-10,
    evaluation: str = "iterative",     # "iterative" | "exact"
    init: str = "zeros",               # "zeros" (always 'right') | "random"
    seed: int | None = None,
    warm_start: bool = False,          # reuse previous V as the next evaluation's start (see note)
    max_iterations: int = 10_000,
) -> PolicyIterationResult:
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must be in (0, 1): it makes evaluation well-defined and prefers short paths.")

    if init == "random":
        policy = np.random.default_rng(seed).integers(0, model.n_actions, model.n_states)
    elif init == "zeros":
        policy = np.zeros(model.n_states, dtype=np.int64)
    else:
        raise ValueError("init must be 'zeros' or 'random'")

    V = None
    history: list[dict] = []
    for it in range(1, max_iterations + 1):
        # 1) policy evaluation
        if evaluation == "exact":
            V, sweeps = evaluate_policy_exact(model, policy, gamma)
        else:
            V, sweeps = evaluate_policy_iterative(
                model, policy, gamma, V_init=V if warm_start else None, theta=theta
            )
        # 2) policy improvement
        policy, n_changed = improve_policy(model, V, policy, gamma)
        history.append({"iteration": it, "eval_sweeps": sweeps, "policy_changes": n_changed})

        if n_changed == 0:  # stable policy => optimal
            return PolicyIterationResult(policy, V, it, True, history)

    return PolicyIterationResult(policy, V, max_iterations, False, history)
