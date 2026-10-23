"""Per-loop return and flipped fraction of finished runs, read from their replays (np.load, mmap; no file is created).

Output: report/10-4-26-pretrain-data/data/run_curves.csv
Usage:  python report/10-4-26-pretrain-data/scripts/run_curves.py run_name [run_name ...]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "data" / "run_curves.csv"


def curve(run):
    folder = REPO / "runs" / "gym_imagination" / run / "replay_buffer"
    rewards = np.load(folder / "rewards.data.npy", mmap_mode = "r")
    state = np.load(folder / "critic_state.data.npy", mmap_mode = "r")
    num = int(np.load(folder / "num_episodes.state.npy"))
    rows = []
    for loop in range(num // 64):
        episodes = slice(loop * 64, loop * 64 + 64)
        returns = np.asarray(rewards[episodes, :1000], dtype = np.float64).sum(1)
        angle = np.asarray(state[episodes, :1001, 1], dtype = np.float64)
        flipped = np.abs(np.angle(np.exp(1j * angle))) > np.pi / 2
        rows.append(dict(run = run, loop = loop, ret = returns.mean(), ret_std = returns.std(),
                         flipped_frac = flipped.mean(), flipped_episodes = flipped[:, -1].mean()))
    return rows


rows = [row for run in sys.argv[1:] for row in curve(run)]
table = pd.DataFrame(rows)
if OUT.exists():
    old = pd.read_csv(OUT)
    table = pd.concat([old[~old.run.isin(table.run.unique())], table])
table.to_csv(OUT, index = False)
