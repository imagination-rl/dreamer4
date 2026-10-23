"""Real-state policy statistics of each loop, from the replay buffer of each run.

The replay stores, for each real step, the native Beta action (in [0, 1]) and its log-probability under the
policy that collected it. The mean of -log pi(a|s) over sampled actions is an unbiased estimate of the policy
entropy on the real states (Monte Carlo). The script adds 6 * log 2, as the training log does, so the value is
in the same units as cleanrl_updated_ppo/entropy (the entropy on the dream states of the last PPO minibatch).

Loop L collects episodes 64 L .. 64 L + 63 with the policy after the PPO update of loop L - 1.

Runs: the a2_grouped runs of ../../10-4-26-stage1-combinations/data/loop_metrics.csv that still have a replay.
Writes ../data/replay_loops.csv (one row for each run and loop).
"""

import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
RUNS = REPO / "runs" / "gym_imagination"
STAGE1 = REPO / "report" / "10-4-26-stage1-combinations" / "data" / "loop_metrics.csv"
EPISODES_PER_LOOP = 64
STEPS = 1000
LOG_RANGE = 6 * np.log(2.)
ROOT_ANGLE, X_VELOCITY = 1, 8


def run_names():
    names = set()
    with open(STAGE1) as file:
        for row in csv.DictReader(file):
            names.add(row["run"])
    return sorted(n for n in names if n.startswith("a2_grouped_"))


def wrap(x):
    return np.abs(np.angle(np.exp(1j * x)))


def run_stats(name):
    replay = Path(os.path.realpath(RUNS / name)) / "replay_buffer"
    if not (replay / "log_probs_continuous.data.npy").exists():
        return []
    load = lambda field: np.load(replay / f"{field}.npy", mmap_mode = "r")
    log_probs, actions, states = load("log_probs_continuous.data"), load("actions_continuous.data"), load("critic_state.data")
    returns = np.load(replay / "episode_return.data.meta.npy")
    num_episodes = int(np.load(replay / "num_episodes.state.npy"))
    rows = []
    for loop in range(num_episodes // EPISODES_PER_LOOP):
        episodes = slice(loop * EPISODES_PER_LOOP, (loop + 1) * EPISODES_PER_LOOP)
        lp = np.asarray(log_probs[episodes, :STEPS]).sum(-1)  # (64, 1000)
        state = np.asarray(states[episodes, :STEPS])
        act = np.asarray(actions[episodes, :STEPS])
        flipped = wrap(state[..., ROOT_ANGLE]) > np.pi / 2
        neg = -lp + LOG_RANGE
        ep_ret = returns[episodes]
        flipped_ep = flipped.mean(1)
        rows.append({
            "run": name, "loop": loop,
            "real_entropy": float(neg.mean()),
            "real_entropy_upright": float(neg[~flipped].mean()) if (~flipped).any() else float("nan"),
            "real_entropy_flipped": float(neg[flipped].mean()) if flipped.any() else float("nan"),
            "real_entropy_first100": float(neg[:, :100].mean()),
            "flipped_frac": float(flipped.mean()),
            "flipped_episode_frac": float((flipped_ep > 0.5).mean()),
            "x_velocity": float(state[..., X_VELOCITY].mean()),
            "saturation": float((np.abs(act - 0.5) > 0.45).mean()),
            "return_mean": float(ep_ret.mean()), "return_std": float(ep_ret.std()), "return_min": float(ep_ret.min()),
            "return_upright_eps": float(ep_ret[flipped_ep < 0.1].mean()) if (flipped_ep < 0.1).any() else float("nan"),
            "return_flipped_eps": float(ep_ret[flipped_ep > 0.5].mean()) if (flipped_ep > 0.5).any() else float("nan"),
        })
    return rows


def main():
    names = run_names()
    with ProcessPoolExecutor(max_workers = 12) as pool:
        results = list(pool.map(run_stats, names))
    rows = [row for result in results for row in result]
    print(f"{sum(1 for r in results if r)} runs with a replay, {len(rows)} loops")
    with open(ROOT / "data" / "replay_loops.csv", "w", newline = "") as file:
        writer = csv.DictWriter(file, fieldnames = list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
