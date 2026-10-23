"""Effect of a change of the scaled40 data mix, without a new collection: remove modes, then compute the statistics.

For each variant: the |a|-speed correlation (the confound), the tokenizer normalization of the forward velocity
(mean and std of obs index 8 over all steps), the latent resolution of 0-3 m/s against the current data, the step
fraction at 0-3 m/s (the online range at loop 100), and the upright-to-flipped transitions.

Output: report/9-30-26-pretrain-analysis/data/data_mix.csv
Usage:  python report/9-30-26-pretrain-analysis/scripts/data_mix.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from train import offline_pretraining as off

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = REPO / "runs" / "offline_data" / "HalfCheetah-v4" / "halfcheetah_v4_mix_scaled40_5p5m"
VX, ANGLE = off.HALF_CHEETAH_X_VELOCITY_INDEX, off.HALF_CHEETAH_ROOT_ANGLE_INDEX
FLIPPED = np.pi / 2


def noise_of(dataset):
    for name in ("noise_std", "noise_stds", "expert_noise_std"):
        if hasattr(dataset, name):
            return np.asarray(getattr(dataset, name))
    return None


def stats(name, dataset, keep, base_std):
    velocity = dataset.obs[keep, :-1, VX]
    size = np.abs(dataset.actions[keep]).mean(-1)
    angle = np.abs(off._wrap_angle(dataset.obs[keep, :, ANGLE]))
    flipped = angle > FLIPPED
    flips = (~flipped[:, :-1] & flipped[:, 1:]).sum()
    std = float(dataset.obs[keep, :, VX].std())
    online = (velocity > 0) & (velocity < 3)
    return dict(
        variant = name, episodes = int(keep.sum()),
        corr_abs_action_velocity = float(np.corrcoef(size.reshape(-1), velocity.reshape(-1))[0, 1]),
        vx_mean = float(dataset.obs[keep, :, VX].mean()), vx_std = std,
        resolution_vs_current = base_std / std if base_std else 1.0,
        frac_0_3_ms = float(online.mean()),
        mean_abs_action_0_3_ms = float(size[online].mean()),
        frac_above_8_ms = float((velocity > 8).mean()),
        flipped_step_frac = float(flipped.mean()), upright_to_flipped = int(flips),
    )


def main():
    dataset = off.load_offline_dataset(DATA)
    mode = np.asarray(dataset.mode)
    ids = off.OFFLINE_MODE_IDS
    noise = noise_of(dataset)
    everything = np.ones(dataset.num_episodes, dtype = bool)
    base_std = float(dataset.obs[:, :, VX].std())

    variants = {
        "current scaled40": everything,
        "no expert": mode != ids["expert"],
        "no expert, no expert_scaled": (mode != ids["expert"]) & (mode != ids["expert_scaled"]),
        "only past_runs + random": (mode == ids["past_runs"]) | (mode == ids["random"]),
    }
    if noise is not None:
        low_noise = (mode == ids["expert_noise"]) & (noise <= 0.2)
        variants["no expert, no expert_noise std <= 0.2"] = (mode != ids["expert"]) & ~low_noise

    rows = [stats(name, dataset, keep, base_std) for name, keep in variants.items()]
    for name, mode_id in ids.items():
        keep = mode == mode_id
        if keep.any():
            rows.append(stats(f"mode {name}", dataset, keep, base_std))

    table = pd.DataFrame(rows)
    table.to_csv(HERE.parent / "data" / "data_mix.csv", index = False)
    pd.set_option("display.width", 250, "display.max_columns", 20)
    print("noise field:", None if noise is None else np.unique(noise))
    print(table.round(3).to_string(index = False))


if __name__ == "__main__":
    main()
