"""Grid Maze: a custom Gymnasium environment plus tabular Policy Iteration."""
import gymnasium as gym

from .env import GridMazeEnv

ENV_ID = "GridMaze-v0"

# max_episode_steps adds the TimeLimit wrapper (=> truncation). Without it an
# agent that wanders safely would never end its episode.
if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point="grid_maze.env:GridMazeEnv", max_episode_steps=50)

__all__ = ["GridMazeEnv", "ENV_ID"]
