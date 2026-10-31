# Assignment 2 Report — Policy Iteration on a Custom Grid Maze

**Repository:** `<your GitHub URL here>`  |  **Videos:** `videos/` in the repo

## 1. Outcome summary

* **Environment.** `GridMaze-v0` (Gymnasium): 5×5 grid, random S/G/2×X each reset, 8-integer observation,
  4 deterministic actions, rewards +1 / −1 / 0, PyGame rendering (`human` and `rgb_array`).
* **Method.** For every new layout the agent builds a 25-state tabular MDP from the observation and runs
  Policy Iteration (γ = 0.99, iterative evaluation with θ = 1e-10, greedy improvement with strict-improvement
  tie handling). The resulting policy is cached per layout.
* **Result.** On 1000 random mazes (seeds 0–999) the agent reached the goal in **100 % of episodes**
  (mean return +1.00, mean 3.44 steps) and never touched a bad cell.
  Policy Iteration needed **6.08 iterations on average (min 3, max 9)**.
* **Verification.** The PI solution matches an independent Value Iteration solution on 25 random layouts, the iterative
  and exact evaluators agree, and no state can be improved after convergence (53 unit tests).
* **Videos.** 5 single-goal videos (`policy_iteration-episode-*.mp4`) and 2 two-goal videos (`multi_goal-episode-*.mp4`),
  recorded with `RecordVideo` (trivial one-or-two-step mazes are skipped so the videos show actual navigation).

## 2. Answers

### Q1. State-space size of the 5×5 maze
It depends on what you call "the state":

| View | Size |
|---|---|
| Raw observation space `MultiDiscrete([5]*8)` | 5⁸ = **390,625** tuples |
| Valid layouts (G, X₁, X₂ on distinct cells) | 25·24·23 = **13,800** |
| Joint states = layout × agent cell | 13,800 × 25 = **345,000** (303,600 non-terminal) |
| **MDP state once the layout is fixed** | **25** cells (22 non-terminal + 3 terminal) |

Goal/bad cells never move inside an episode, so the part of the observation that actually changes is the agent's cell.
That is why Policy Iteration only needs 25 states per maze.

### Q2. How to optimize Policy Iteration for this problem
Measured over 1000 random layouts (`python analysis.py q3`):

| Configuration | PI iterations (mean) | Evaluation sweeps | Time per maze |
|---|---|---|---|
| Iterative eval, init = "always right" (baseline) | 6.05 | 33.7 | 0.45 ms |
| + warm-start V between iterations | 6.05 | 139.9 | 1.15 ms (**slower**) |
| Random initial policy | 4.79 | 27.1 | 0.43 ms |
| Exact evaluation (linear solve) | 6.05 | 6.0 | 0.49 ms |

1. **Reduce the state space** (biggest win): solve the 25-state per-layout MDP instead of 345,000 joint states.
2. **Cache** the policy per layout (keyed by the goal/bad cells, not the agent position) — the agent plans once per maze.
3. **Vectorize** evaluation and improvement with NumPy over a precomputed `next_state`/`reward` table; terminal states are absorbing with V = 0.
4. **Evaluate exactly** with a linear solve when S is tiny (6 solves instead of ~34 sweeps); use iterative sweeps when S is large.
5. **Better initial policy** (random or greedy-toward-goal) reduces improvement rounds (6.05 → 4.79).
6. **γ < 1** (0.99) makes evaluation well-defined for looping policies and prefers shorter paths.
7. **Strict-improvement tie rule** prevents endless flip-flopping between equally good actions.
8. *Negative result:* warm-starting V did **not** help — policies that loop (e.g. bumping a wall) make the stale values decay only as γᵏ, so evaluation needs far more sweeps.

### Q3. Iterations to converge on a stable policy (5×5)
One iteration = evaluation + improvement; the last iteration is the one where improvement changes nothing.
* Over 1000 random mazes: **mean ≈ 6 iterations (range 3–10)** (6.05 in `analysis.py`, 6.08 on seeds 0–999 in `run_policy_iteration.py`).
* Example (seed 0 layout, goal (2,2)): **6 iterations** — policy changes per iteration: 7, 7, 3, 2, 1, **0**.
  So 5 improving rounds + 1 confirming round.

The count grows with the longest shortest-path distance, because each improvement round extends the "correct" region by roughly
one step from the goal.

### Q4. Multiple goal cells
Policy Iteration handles them with no change: all goals are just terminal states with a positive reward. With equal rewards and γ < 1,
V(s) = γ^(d−1) where d is the distance to the **nearest** goal, so the greedy policy heads for the closest goal.

Example (`python analysis.py q4`): goals at (0,0) and (4,4), bad cells at (2,2) and (4,0). Converges in 4 iterations; 9 start
cells end at (0,0) and 12 end at (4,4). Cells equidistant from both goals, e.g. (3,1), (1,3), (0,4), are ties — several optimal policies exist,
and our tie rule picks the lowest-index action. The **value function is unique even though the optimal policy is not.**
With unequal rewards (goal (0,0) pays +0.5, goal (4,4) pays +1) the policy flips: the agent walks *past* the near small goal to reach the
far large one, since 0.99^(d−1) of +1 still beats +0.5.

### Q5. Can Policy Iteration work on a 100×100 maze?
**Yes for a fixed layout, no for the joint state space.**
* Per layout there are 100² = 10,000 states. PI solved a random 100×100 maze in **134 iterations / 0.57 s** (10,083 evaluation sweeps;
  the path found had 57 steps = the Manhattan distance). Each sweep costs O(S·A) = 40k operations, trivial for NumPy.
* The joint (layout × agent) space is about **10¹⁶** states, which cannot be tabulated — we only avoid it because we plan after seeing the layout.
* Caveats: iterations scale with path length (a 100×100 maze needs ~134 vs ~6); exact dense solves need O(S²) memory
  (10⁴×10⁴ doubles ≈ 800 MB) and O(S³) time, so use iterative evaluation or sparse solvers; and with γ = 0.99 a +1 reward 400 steps away
  is worth γ⁴⁰⁰ ≈ 0.018, so far-away goals get tiny values (γ must be raised for much larger mazes).

### Q6. Can Policy Iteration work on a continuous-space maze?
**Not in its exact tabular form.** PI must evaluate and improve the value of *every* state, which needs a finite, enumerable state set.
A continuous maze has infinitely many states, so the value table cannot exist, the sweep over states cannot be performed, and the improvement
`argmax_a Q(s,a)` becomes an optimization at every point (if actions are continuous too). Workarounds change the method:
* **Discretize** the space into a grid: PI then works on the discretized MDP, but the answer is only approximate (boundaries, obstacle
  thin-ness, resolution) and states grow as (1/ε)ᵈ — the curse of dimensionality.
* **Function approximation** (approximate PI/fitted value iteration, DQN, actor-critic) represents V or π with a parametric model that generalizes
  across states; this needs sampled data and loses the exact convergence guarantee.
* PI also needs the transition model `P(s'|s,a)` to be known, which is rarely true in continuous settings.

### Q7. Can Policy Iteration work with moving bad cells (Pac-Man ghosts)?
**Only if the ghosts become part of the state and their dynamics are known.**
* Our per-layout 25-state model assumes the bad cells are fixed, so transitions depend only on the agent's cell. With moving ghosts, the same
  agent move is safe at one time step and fatal at another, so a table `π(agent cell)` cannot work — the process is not Markov in the agent cell alone.
* If the state is augmented to `(agent, ghost₁, ghost₂)` (plus heading/phase if they patrol), the problem is again a finite MDP and
  PI applies, provided we know how the ghosts move (deterministic patrol, or a known probability distribution, in which case the evaluation
  takes an expectation over ghost moves). Collisions such as swapping cells also have to be modeled.
* The price is state-space explosion: 25 → 25³ = **15,625** states (per goal) with 2 ghosts, 250,000 with heading information, ~10¹² on a 100×100 maze.
* If ghost behavior is unknown or reacts to the agent (chasing), classical DP cannot be applied directly: we need to learn the model or use
  model-free RL, or treat it as a game (minimax).

## 3. Limitations of classical DP seen in this assignment
Policy Iteration is exact and fast here, but it needs (1) a known model, (2) a small enumerable state space, and (3) a Markov state. Larger grids (Q5) only
work after cutting the state space down, continuous spaces (Q6) break the enumeration, and moving ghosts (Q7) blow up the state space or the known-model assumption.
