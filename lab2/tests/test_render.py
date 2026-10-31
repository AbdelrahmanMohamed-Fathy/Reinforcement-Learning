import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless-safe

import numpy as np

from grid_maze import GridMazeEnv


def test_rgb_array_frame_shape_and_content():
    env = GridMazeEnv(render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame.shape == (480, 480, 3) and frame.dtype == np.uint8
    assert frame.std() > 0  # something was actually drawn
    env.close()


def test_frame_changes_after_step():
    env = GridMazeEnv(render_mode="rgb_array")
    env.reset(seed=1)
    f0 = env.render().copy()
    env.step(0)
    assert not np.array_equal(f0, env.render())
    env.close()
