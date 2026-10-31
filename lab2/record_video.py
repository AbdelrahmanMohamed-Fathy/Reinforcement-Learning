"""Record the trained (planned) agent with Gymnasium's RecordVideo wrapper.

    python record_video.py                       # 5 videos -> videos/
    python record_video.py --episodes 3 --min-steps 6
    python record_video.py --n-goals 2 --prefix multi_goal
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # no window needed to record

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

import grid_maze  # noqa: F401
from grid_maze.agent import PolicyIterationAgent, run_episode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--folder", default="videos")
    p.add_argument("--prefix", default="policy_iteration")
    p.add_argument("--n-goals", type=int, default=1)
    p.add_argument("--n-bad", type=int, default=2)
    p.add_argument("--min-steps", type=int, default=5,
                   help="skip trivial mazes solved in fewer steps (boring videos)")
    a = p.parse_args()

    kw = dict(n_goals=a.n_goals, n_bad=a.n_bad)
    probe = gym.make("GridMaze-v0", **kw)               # cheap env, no rendering
    agent = PolicyIterationAgent.for_env(probe)

    # Pick seeds whose optimal solution is interesting and successful.
    seeds, s = [], a.seed
    while len(seeds) < a.episodes:
        r = run_episode(probe, agent, seed=s)
        if r["success"] and r["steps"] >= a.min_steps:
            seeds.append(s)
        s += 1
    probe.close()

    env = gym.make("GridMaze-v0", render_mode="rgb_array", **kw)
    env = RecordVideo(env, video_folder=a.folder, name_prefix=a.prefix,
                      episode_trigger=lambda ep: True)   # record every episode
    for seed in seeds:
        r = run_episode(env, agent, seed=seed)
        print(f"seed {seed}: {r['steps']} steps, return {r['return']:+.0f}")
    env.close()  # flushes the last video
    print(f"Videos saved in ./{a.folder}/")


if __name__ == "__main__":
    main()
