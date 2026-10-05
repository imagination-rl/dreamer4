"""Episode returns of each past-run replay, in the return buckets of the past_runs mode.

The saved CSV was made when PAST_RUN_REPLAYS listed 26 runs (14 scratch + 12 pretrained); it now lists the 12 pretrained runs.

Output: report/9-30-26-past-runs/data/replay_return_buckets.csv
Usage:  python report/9-30-26-past-runs/scripts/replay_return_buckets.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from memmap_replay_buffer.replay_buffer import ReplayBuffer

from train.ablations import PAST_RUN_REPLAYS

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data" / "replay_return_buckets.csv"
EDGES = [-np.inf, 0., 500., 1500., np.inf]
T = 1000


def main():
    rows = []
    for folder in PAST_RUN_REPLAYS:
        replay = ReplayBuffer.from_folder(folder, read_only = True)
        meta = replay.meta_data
        full = np.flatnonzero((np.asarray(meta["episode_lens"]) == T + 1) & np.asarray(meta["is_truncated"], dtype = bool)
                              & ~np.asarray(meta["terminals"], dtype = bool) & ~np.asarray(meta["is_from_world_model"], dtype = bool))
        returns = np.asarray(replay.data["rewards"][:len(meta["episode_lens"]), :T]).sum(-1)[full]
        counts = np.histogram(returns, EDGES)[0]
        rows.append(dict(run = folder.split("/")[2], pretrained = "offline_pretrained" in folder, full = len(full),
                         below_0 = counts[0], r0_500 = counts[1], r500_1500 = counts[2], above_1500 = counts[3],
                         median = float(np.median(returns)), p90 = float(np.percentile(returns, 90))))
    table = pd.DataFrame(rows)
    OUT.parent.mkdir(parents = True, exist_ok = True)
    table.to_csv(OUT, index = False)
    pd.set_option("display.width", 250)
    print(table.round(0).to_string(index = False))


if __name__ == "__main__":
    main()
