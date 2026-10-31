import numpy as np
import pytest

from grid_maze.policy_improvement import improve_policy, q_values
from grid_maze.policy_iteration import policy_iteration
from grid_maze.tabular_model import build_model
from grid_maze.dynamics import cell_to_xy, xy_to_cell
from grid_maze.policy_evaluation import evaluate_policy_exact, evaluate_policy_iterative

GAMMA = 0.99


def random_layout(rng, size=5, n_goals=1, n_bad=2):
    cells = [cell_to_xy(int(c), size) for c in rng.choice(size * size, 1 + n_goals + n_bad, replace=False)]
    return cells[1 : 1 + n_goals], cells[1 + n_goals :]


def value_iteration(model, gamma, iters=2000):
    V = np.zeros(model.n_states)
    for _ in range(iters):
        V = np.where(model.terminal, 0.0, q_values(model, V, gamma).max(axis=1))
    return V


def test_model_shapes_and_terminals():
    m = build_model(5, [(4, 0)], [(0, 0), (2, 2)])
    assert m.n_states == 25 and m.n_actions == 4
    assert m.terminal.sum() == 3
    s_next_to_goal = xy_to_cell(3, 0, 5)
    assert m.reward[s_next_to_goal, 0] == 1.0  # 'right' into the goal


def test_known_tiny_case():
    # Goal directly right of (3,0): optimal V = 1 there, gamma one step further away, ...
    m = build_model(5, [(4, 0)], [(0, 4), (4, 4)])
    r = policy_iteration(m, GAMMA)
    assert r.converged
    assert r.policy[xy_to_cell(3, 0, 5)] == 0
    assert r.V[xy_to_cell(3, 0, 5)] == pytest.approx(1.0)
    assert r.V[xy_to_cell(2, 0, 5)] == pytest.approx(GAMMA)
    assert r.V[xy_to_cell(1, 0, 5)] == pytest.approx(GAMMA**2)


@pytest.mark.parametrize("seed", range(25))
def test_matches_value_iteration_and_is_stable(seed):
    rng = np.random.default_rng(seed)
    goals, bads = random_layout(rng)
    m = build_model(5, goals, bads)
    res = policy_iteration(m, GAMMA)
    assert res.converged
    np.testing.assert_allclose(res.V, value_iteration(m, GAMMA), atol=1e-7)
    _, changed = improve_policy(m, res.V, res.policy, GAMMA)
    assert changed == 0  # no state can still be improved


@pytest.mark.parametrize("seed", range(10))
def test_exact_and_iterative_agree(seed):
    rng = np.random.default_rng(100 + seed)
    m = build_model(5, *random_layout(rng))
    pol = rng.integers(0, 4, 25)
    v1, _ = evaluate_policy_iterative(m, pol, GAMMA)
    v2, _ = evaluate_policy_exact(m, pol, GAMMA)
    np.testing.assert_allclose(v1, v2, atol=1e-7)
    a = policy_iteration(m, GAMMA, evaluation="iterative", init="random", seed=seed)
    b = policy_iteration(m, GAMMA, evaluation="exact", init="random", seed=seed)
    np.testing.assert_allclose(a.V, b.V, atol=1e-7)


def test_multiple_goals_prefers_nearest():
    m = build_model(5, [(0, 0), (4, 4)], [(2, 0), (0, 4)])
    r = policy_iteration(m, GAMMA)
    # (1,1) is 2 steps from (0,0) and 6 from (4,4) -> value is gamma^1
    assert r.V[xy_to_cell(1, 1, 5)] == pytest.approx(GAMMA)
