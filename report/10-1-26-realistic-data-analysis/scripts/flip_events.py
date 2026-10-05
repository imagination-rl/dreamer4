"""The real flips of each RL run, from its replay: when they happen and what comes before and after them.

For each full real episode: the loop (64 episodes per loop, in replay order), the first upright-to-flipped step, the
flipped step fraction, the return, the speed and saturation in the 30 steps before the first flip, and the reward per
step after it. Flipped: |wrapped root angle| > pi / 2.
Output: report/10-1-26-realistic-data-analysis/data/flip_events.csv (one row per episode), flip_blocks.csv
Usage:  python report/10-1-26-realistic-data-analysis/scripts/flip_events.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from memmap_replay_buffer import ReplayBuffer

from train import offline_pretraining as off

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
G = "runs/gym_imagination/"
RUNS = {
    "real_noreh_s42": G + "pretrain_rl_realistic_data_s42", "real_noreh_s4": G + "pretrain_rl_realistic_data_s4",
    "real_anchor_s42": G + "pretrain_rl_anchor_realistic_data_s42", "real_anchor_s4": G + "pretrain_rl_anchor_realistic_data_s4",
    "s40_noreh_s42": G + "offline_pretrained_a2_scaled40_no_rehearsal_s42", "s40_noreh_s4": G + "offline_pretrained_a2_scaled40_no_rehearsal_s4",
    "local928_s42": "runs/local_9-28-26/attnpool_on_s42", "local928_s4": "runs/local_9-28-26/attnpool_on_s4",
    "local928_s1": "runs/local_9-28-26/attnpool_on_s1",
}
T, BEFORE, EPISODES_PER_LOOP = 1000, 30, 64
OUT = HERE.parent / "data"


def main():
    rows = []
    for name, folder in RUNS.items():
        replay = ReplayBuffer.from_folder(str(REPO / folder / "replay_buffer"), read_only = True)
        meta = replay.meta_data
        n = len(meta["episode_lens"])
        real = ~np.asarray(meta["is_from_world_model"], dtype = bool)
        full = np.flatnonzero((np.asarray(meta["episode_lens"]) >= T) & real)
        obs = np.asarray(replay.data["critic_state"][:n, :T])[full]
        rewards = np.asarray(replay.data["rewards"][:n, :T])[full]
        actions = np.asarray(replay.data["actions_continuous"][:n, :T])[full]
        size = np.abs(2 * actions - 1) if actions.min() >= 0 else np.abs(actions)
        flipped = np.abs(off._wrap_angle(obs[..., off.HALF_CHEETAH_ROOT_ANGLE_INDEX])) > np.pi / 2
        vx = obs[..., off.HALF_CHEETAH_X_VELOCITY_INDEX]
        for i, episode in enumerate(full):
            onset = np.flatnonzero(~flipped[i, :-1] & flipped[i, 1:])
            first = int(onset[0]) + 1 if len(onset) else -1
            lo = max(first - BEFORE, 0)
            rows.append(dict(
                run = name, episode = int(episode), loop = int(real[:episode].sum() // EPISODES_PER_LOOP),
                ret = rewards[i].sum(), flipped_frac = flipped[i].mean(), first_flip = first, flips = len(onset),
                saturated = (size[i] > 0.95).mean(), vx = vx[i].mean(),
                vx_before = vx[i, lo:first].mean() if first > 0 else np.nan,
                saturated_before = (size[i, lo:first] > 0.95).mean() if first > 0 else np.nan,
                reward_after = rewards[i, first:].mean() if first > 0 else np.nan,
                reward_upright = rewards[i][~flipped[i]].mean() if (~flipped[i]).any() else np.nan,
            ))
    ep = pd.DataFrame(rows)
    ep.to_csv(OUT / "flip_events.csv", index = False)
    ep["block"] = ep.loop // 10 * 10
    ep["has_flip"] = ep.first_flip > 0
    blocks = ep.groupby(["run", "block"]).agg(
        episodes = ("ret", "size"), ret = ("ret", "mean"), flip_episode_frac = ("has_flip", "mean"),
        flipped_frac = ("flipped_frac", "mean"), first_flip_median = ("first_flip", lambda s: s[s > 0].median()),
        vx_before = ("vx_before", "mean"), saturated_before = ("saturated_before", "mean"), saturated = ("saturated", "mean"),
        reward_after = ("reward_after", "mean"), reward_upright = ("reward_upright", "mean"),
    ).reset_index()
    blocks.to_csv(OUT / "flip_blocks.csv", index = False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(blocks[blocks.block < 70].round(3).to_string(index = False))


if __name__ == "__main__":
    main()
