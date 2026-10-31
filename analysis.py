"""Experiments that back the numbers/answers in REPORT.md.

    python analysis.py            # everything
    python analysis.py q1 q3      # only some sections  (q1 q2 q3 q4 q5 q7)
"""
from __future__ import annotations

import sys
import time
from math import comb

import numpy as np

from grid_maze.dynamics import cell_to_xy, xy_to_cell
from grid_maze.policy_iteration import policy_iteration
from grid_maze.tabular_model import build_model
from grid_maze.utils import policy_to_text, trace_policy, values_to_text

GAMMA = 0.99


def random_layout(rng, size=5, n_goals=1, n_bad=2):
    cells = [cell_to_xy(int(c), size) for c in rng.choice(size * size, 1 + n_goals + n_bad, replace=False)]
    return cells[0], cells[1 : 1 + n_goals], cells[1 + n_goals :]


def header(t):
    print(f"\n{'=' * 8} {t} {'=' * 8}")


# --------------------------------------------------------------------- Q1
def q1():
    header("Q1  State-space size of the 5x5 maze")
    n = 25
    raw = 5**8
    layouts = n * (n - 1) * (n - 2)              # ordered (goal, bad1, bad2), all distinct
    joint = layouts * n                          # + agent anywhere (incl. on G/X after the last move)
    non_terminal = layouts * (n - 3)
    print(f"raw MultiDiscrete([5]*8) tuples          : {raw:,}")
    print(f"valid layouts (G, X1, X2 distinct)       : {layouts:,}   ({comb(n,1)*comb(n-1,2):,} if the two X are unordered)")
    print(f"joint states (layout x agent cell)       : {joint:,}")
    print(f"  of which non-terminal (agent not on G/X): {non_terminal:,}")
    print(f"MDP states once the layout is FIXED       : {n}  ({n-3} non-terminal + 3 terminal)")


# --------------------------------------------------------------------- Q2/Q3
def q3(n_layouts=1000):
    header(f"Q3  Iterations to a stable policy ({n_layouts} random 5x5 layouts)")
    rng = np.random.default_rng(0)
    layouts = [random_layout(rng) for _ in range(n_layouts)]
    configs = {
        "iterative eval, init=always-right (default)":   dict(evaluation="iterative", init="zeros"),
        "iterative eval, init=always-right, warm start": dict(evaluation="iterative", init="zeros", warm_start=True),
        "iterative eval, init=random policy":            dict(evaluation="iterative", init="random"),
        "exact (linear-solve) eval, init=always-right":  dict(evaluation="exact", init="zeros"),
    }
    print(f"{'configuration':<48}{'iters mean':>11}{'min':>5}{'max':>5}{'eval sweeps':>13}{'time ms/maze':>14}")
    for name, kw in configs.items():
        its, sweeps = [], []
        t0 = time.perf_counter()
        for i, (agent, goals, bads) in enumerate(layouts):
            r = policy_iteration(build_model(5, goals, bads), GAMMA, seed=i, **kw)
            its.append(r.iterations)
            sweeps.append(r.total_eval_sweeps)
        dt = (time.perf_counter() - t0) / n_layouts * 1000
        print(f"{name:<48}{np.mean(its):>11.2f}{min(its):>5}{max(its):>5}{np.mean(sweeps):>13.1f}{dt:>14.2f}")

    agent, goals, bads = layouts[0]
    r = policy_iteration(build_model(5, goals, bads), GAMMA)
    print(f"\nExample (first layout): goal={goals} bad={bads}")
    for h in r.history:
        print("  ", h)
    print("  (last row has policy_changes = 0 -> that is the confirming pass; 'iterations' counts it)")
    print(policy_to_text(r.policy, 5, goals, bads, agent))


# --------------------------------------------------------------------- Q4
def q4():
    header("Q4  Multiple goal cells")
    goals, bads = [(0, 0), (4, 4)], [(2, 2), (4, 0)]
    m = build_model(5, goals, bads)
    r = policy_iteration(m, GAMMA)
    print(f"(a) two goals {goals}, bad {bads} -> converged in {r.iterations} iterations")
    print(policy_to_text(r.policy, 5, goals, bads))
    print("values:\n" + values_to_text(r.V, 5))
    ends = {}
    for s in range(25):
        if not m.terminal[s]:
            end = trace_policy(m, r.policy, cell_to_xy(s, 5))[-1]
            ends.setdefault(end, []).append(cell_to_xy(s, 5))
    for end, cells in ends.items():
        print(f"  {len(cells):2d} start cells end at {end}")
    tie = [c for c in [cell_to_xy(s, 5) for s in range(25)]
           if c not in goals + bads and abs(c[0]-0)+abs(c[1]-0) == abs(c[0]-4)+abs(c[1]-4)]
    print(f"  cells equidistant from both goals (ties, broken by lowest action index): {tie}")

    # (b) goals with different rewards
    m2 = build_model(5, goals, bads)
    g0 = xy_to_cell(0, 0, 5)
    m2.reward[(m2.next_state == g0) & (~m2.terminal)[:, None]] = 0.5   # goal (0,0) now pays only +0.5
    r2 = policy_iteration(m2, GAMMA)
    print("\n(b) same maze, but goal (0,0) pays +0.5 and goal (4,4) pays +1")
    print(policy_to_text(r2.policy, 5, goals, bads))
    print("  -> the agent now walks PAST the near, small goal to reach the far, big one when 0.99^(d-1) > 0.5")


# --------------------------------------------------------------------- Q5
def q5(size=100):
    header(f"Q5  {size}x{size} maze")
    n = size * size
    print(f"MDP states for ONE fixed layout          : {n:,}")
    print(f"joint (layout x agent) states             : {n * (n-1) * (n-2) * n:.3e}  <- impossible to tabulate")
    rng = np.random.default_rng(1)
    agent, goals, bads = random_layout(rng, size)
    t0 = time.perf_counter()
    m = build_model(size, goals, bads)
    t1 = time.perf_counter()
    r = policy_iteration(m, GAMMA)
    t2 = time.perf_counter()
    path = trace_policy(m, r.policy, agent, 1000)
    print(f"layout: start={agent} goal={goals} bad={bads}")
    print(f"build model {t1-t0:.2f}s | PI {t2-t1:.2f}s | iterations={r.iterations} | total eval sweeps={r.total_eval_sweeps:,}")
    print(f"agent reaches {path[-1]} in {len(path)-1} steps (Manhattan distance {abs(agent[0]-goals[0][0]) + abs(agent[1]-goals[0][1])})")
    print(f"value at the start cell: {r.V[xy_to_cell(*agent, size)]:.4f}  (= gamma^(steps-1) = {GAMMA**(len(path)-2):.4f})")
    print("note: with gamma=0.99, values 100+ steps away are ~0.37 -- fine for 100x100, "
          "but a 1000x1000 maze would need gamma closer to 1 or the signal underflows relative to theta.")


# --------------------------------------------------------------------- Q7
def q7():
    header("Q7  Moving bad cells (ghosts)")
    n = 25
    print(f"static maze, fixed layout                : {n} states")
    print(f"+ 2 ghosts (positions in the state)      : {n**3:,} states per fixed goal ({n**3 * 4:,} state-action pairs)")
    print(f"+ ghost heading/phase (4 directions each): {n**3 * 16:,}")
    print(f"+ random goal as well                    : {n**4:,}")
    print(f"same idea on 100x100 with 2 ghosts       : {(100*100)**3:.2e}")


SECTIONS = {"q1": q1, "q2": q3, "q3": q3, "q4": q4, "q5": q5, "q7": q7}

if __name__ == "__main__":
    wanted = sys.argv[1:] or ["q1", "q3", "q4", "q5", "q7"]
    done = set()
    for w in wanted:
        fn = SECTIONS[w]
        if fn not in done:
            fn()
            done.add(fn)
