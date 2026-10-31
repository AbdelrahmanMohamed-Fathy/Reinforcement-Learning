# Grid Maze — Custom Gymnasium Environment + Policy Iteration

CMPS458 Reinforcement Learning, Assignment 2 (Fall 2026).

A 5×5 grid maze with a random start `S`, a random goal `G` and two random bad cells `X`.
The environment is a custom **Gymnasium** env rendered with **PyGame**; the agent is solved with
tabular **Policy Iteration** (dynamic programming) and recorded with **RecordVideo**.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python run_policy_iteration.py --episodes 1000        # stats over 1000 random mazes
python run_policy_iteration.py --episodes 3 --verbose # print policy arrows + values
python run_policy_iteration.py --episodes 5 --render  # watch it in a PyGame window
python record_video.py --episodes 5                   # -> videos/*.mp4
python record_video.py --episodes 2 --n-goals 2 --prefix multi_goal
python analysis.py                                    # numbers used in REPORT.md
pytest -q                                             # 53 tests
```

## Layout
```
grid_maze/
  dynamics.py           movement / walls / rewards (single source of truth)
  env.py                GridMazeEnv (Gymnasium API)
  renderer.py           PyGame drawing (human + rgb_array)
  tabular_model.py      layout -> 25-state tabular MDP
  policy_evaluation.py  iterative + exact V^pi
  policy_improvement.py greedy improvement
  policy_iteration.py   the evaluate/improve loop
  agent.py              applies PI to the env (+ run_episode)
  utils.py              ASCII policy/value printing, path tracing
run_policy_iteration.py   record_video.py   analysis.py
tests/                    REPORT.md
```

## Environment spec
| | |
|---|---|
| Observation | `MultiDiscrete([5]*8)` = `[agent_x, agent_y, goal_x, goal_y, bad1_x, bad1_y, bad2_x, bad2_y]` |
| Actions | `Discrete(4)`: 0 right, 1 up, 2 left, 3 down (deterministic; walls = stay in place) |
| Reward | +1 goal, −1 bad cell, 0 otherwise |
| Episode end | terminated on goal/bad cell; truncated after 50 steps |
| Coordinates | `(0,0)` is the top-left cell; "up" means `y − 1` |
