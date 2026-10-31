"""Run Policy Iteration on the Grid Maze and report the results.

    python run_policy_iteration.py                  # 100 random mazes, print stats
    python run_policy_iteration.py --render         # watch the agent in a PyGame window
    python run_policy_iteration.py --episodes 5 --verbose
"""
from __future__ import annotations

import argparse

import numpy as np
import gymnasium as gym

import grid_maze  # noqa: F401  (registers GridMaze-v0)
from grid_maze.agent import PolicyIterationAgent, run_episode
from grid_maze.dynamics import parse_observation
from grid_maze.utils import policy_to_text, values_to_text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--n-goals", type=int, default=1)
    p.add_argument("--n-bad", type=int, default=2)
    p.add_argument("--evaluation", choices=["iterative", "exact"], default="iterative")
    p.add_argument("--render", action="store_true", help="open a PyGame window")
    p.add_argument("--verbose", action="store_true", help="print policy + values per episode")
    a = p.parse_args()

    env = gym.make("GridMaze-v0", n_goals=a.n_goals, n_bad=a.n_bad,
                   render_mode="human" if a.render else None)
    agent = PolicyIterationAgent.for_env(env, gamma=a.gamma, evaluation=a.evaluation)

    results = []
    for ep in range(a.episodes):
        seed = a.seed + ep
        if a.verbose:
            obs, _ = env.reset(seed=seed)
            model, res = agent.solve(obs)
            start, goals, bads = parse_observation(obs, a.n_goals, a.n_bad)
            print(f"\n--- episode {ep} (seed {seed}) | PI converged in {res.iterations} iterations ---")
            print(policy_to_text(res.policy, model.size, goals, bads, agent=start))
            print("values:\n" + values_to_text(res.V, model.size))
        results.append(run_episode(env, agent, seed=seed))
        if a.verbose:
            print(results[-1])
    env.close()

    succ = np.mean([r["success"] for r in results])
    its = [r["pi_iterations"] for r in results]
    print(f"\nEpisodes: {len(results)} | success rate: {succ:.1%} | "
          f"mean return: {np.mean([r['return'] for r in results]):.3f} | "
          f"mean steps: {np.mean([r['steps'] for r in results]):.2f}")
    print(f"PI iterations to a stable policy: mean {np.mean(its):.2f}, min {min(its)}, max {max(its)}")
    fails = [i + a.seed for i, r in enumerate(results) if not r["success"]]
    if fails:
        print(f"Non-successful seeds (goal unreachable without touching a bad cell): {fails}")


if __name__ == "__main__":
    main()
