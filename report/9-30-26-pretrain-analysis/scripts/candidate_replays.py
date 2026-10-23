"""Episodes that new past-run replays can give: returns, speed, action size, and upright-to-flipped transitions.

The candidates are the local online-only A2 runs of runs/local_9-28-26 and the scaled40 RL runs (entries 47-50).
The local repro runs stay out: their replays hold the online states of the probes.

Output: report/9-30-26-pretrain-analysis/data/candidate_replays.csv
Usage:  python -m report.9-30-26-pretrain-analysis.scripts.candidate_replays
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from memmap_replay_buffer.replay_buffer import ReplayBuffer

from train import offline_pretraining as off
from train.ablations import PAST_RUN_REPLAYS

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
T = 1000
EDGES = [-np.inf, 0., 500., 1500., 2500., np.inf]
CANDIDATES = [
    *PAST_RUN_REPLAYS,
    *[f"runs/local_9-28-26/{name}/replay_buffer" for name in (
        "attnpool_on_s42", "attnpool_on_s4", "attnpool_on_s1", "attnres_full_s42", "attnres_full_s4",
        "attnres_block2_s42", "attnres_block2_s4", "attnpool_off_s42", "attnpool_off_s4", "attnpool_off_s1",
        "attnpool_off_depth5_s42", "attnpool_off_depth5_s4",
    )],
    *[f"runs/gym_imagination/offline_pretrained_a2_scaled40_{name}/replay_buffer" for name in (
        "rehearsal_s42", "rehearsal_s4", "no_rehearsal_s42", "no_rehearsal_s4",
    )],
]


def main():
    rows = []
    for folder in CANDIDATES:
        if not (REPO / folder).is_dir():
            continue
        replay = ReplayBuffer.from_folder(str(REPO / folder), read_only = True)
        meta = replay.meta_data
        n = len(meta["episode_lens"])
        full = np.flatnonzero((np.asarray(meta["episode_lens"]) == T + 1) & np.asarray(meta["is_truncated"], dtype = bool)
                              & ~np.asarray(meta["terminals"], dtype = bool) & ~np.asarray(meta["is_from_world_model"], dtype = bool))
        returns = np.asarray(replay.data["rewards"][:n, :T]).sum(-1)[full]
        obs_key = "critic_state"
        obs = np.asarray(replay.data[obs_key][:n, :T])[full]
        velocity = obs[..., off.HALF_CHEETAH_X_VELOCITY_INDEX]
        angle = np.abs(off._wrap_angle(obs[..., off.HALF_CHEETAH_ROOT_ANGLE_INDEX]))
        flipped = angle > np.pi / 2
        counts = np.histogram(returns, EDGES)[0]
        good = returns >= 1500
        rows.append(dict(
            run = folder.split("/")[-2], full = len(full), below_0 = counts[0], r0_500 = counts[1], r500_1500 = counts[2],
            r1500_2500 = counts[3], above_2500 = counts[4], median = float(np.median(returns)),
            vx_good = float(velocity[good].mean()) if good.any() else np.nan,
            flips = int((~flipped[:, :-1] & flipped[:, 1:]).sum()), flipped_frac = float(flipped.mean()),
        ))
        if "actions_continuous" in replay.data:
            actions = np.asarray(replay.data["actions_continuous"][:n, :T])[full]
            # A replay with native Beta actions in [0, 1] gives the env action 2a - 1.
            env_size = np.abs(2 * actions - 1) if actions.min() >= 0 else np.abs(actions)
            rows[-1]["abs_action_good"] = float(env_size[good].mean()) if good.any() else np.nan
            rows[-1]["saturated_good"] = float((env_size[good] > 0.95).mean()) if good.any() else np.nan
    table = pd.DataFrame(rows)
    table.to_csv(HERE.parent / "data" / "candidate_replays.csv", index = False)
    pd.set_option("display.width", 250, "display.max_columns", 20)
    print(table.round(3).to_string(index = False))


if __name__ == "__main__":
    main()
