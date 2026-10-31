import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import grid_maze  # noqa: F401  (registers GridMaze-v0)
from grid_maze import GridMazeEnv
from grid_maze.dynamics import DOWN, RIGHT, UP

LAYOUT = {"agent": (0, 0), "goals": [(2, 0)], "bads": [(0, 1), (4, 4)]}


def make(**kw):
    env = GridMazeEnv(**kw)
    env.reset(options={"layout": LAYOUT})
    return env


def test_gymnasium_api_compliance():
    check_env(GridMazeEnv(), skip_render_check=True)


def test_spaces_match_assignment():
    env = GridMazeEnv()
    assert env.observation_space.shape == (8,)   # 8 discrete integers
    assert env.action_space.n == 4               # {right, up, left, down}
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)


def test_random_layout_cells_are_distinct_and_seeded():
    env = GridMazeEnv()
    a, _ = env.reset(seed=7)
    b, _ = env.reset(seed=7)
    assert np.array_equal(a, b)
    pts = {tuple(a[i : i + 2]) for i in range(0, 8, 2)}
    assert len(pts) == 4


def test_goal_gives_plus_one_and_terminates():
    env = make()
    env.step(RIGHT)
    _, r, term, trunc, _ = env.step(RIGHT)
    assert (r, term, trunc) == (1.0, True, False)


def test_bad_cell_gives_minus_one_and_terminates():
    env = make()
    _, r, term, _, _ = env.step(DOWN)
    assert (r, term) == (-1.0, True)


def test_normal_move_gives_zero_and_wall_bump_stays():
    env = make()
    obs, r, term, _, _ = env.step(UP)  # wall
    assert (r, term) == (0.0, False)
    assert tuple(obs[:2]) == (0, 0)


def test_time_limit_truncates():
    env = gym.make("GridMaze-v0")
    env.reset(options={"layout": LAYOUT})
    trunc = False
    for _ in range(50):
        _, _, term, trunc, _ = env.step(UP)  # bump the wall forever
    assert trunc and not term


def test_invalid_layout_rejected():
    env = GridMazeEnv()
    with pytest.raises(ValueError):
        env.reset(options={"layout": {"agent": (0, 0), "goals": [(0, 0)], "bads": [(1, 1), (2, 2)]}})
