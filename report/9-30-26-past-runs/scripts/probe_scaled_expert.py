"""Speed and action size of the CleanRL TD3 expert with its actions scaled by a factor (closed loop).

The question: which scales give moderate actions (as the online policy: mean |a| about 0.4, 0-1 % at the bound)
at high speed (2-10 m/s)? 16 episodes of 1000 steps for each scale.

Output: report/9-30-26-past-runs/data/probe_scaled_expert.csv
Usage:  python report/9-30-26-past-runs/scripts/probe_scaled_expert.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from train import offline_pretraining as off
from train import train_gym

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EXPERT = next((REPO / "runs" / "offline_data" / "experts").rglob("*.cleanrl_model"))
# Obs index 1 is the root angle (HALF_CHEETAH_ROOT_ANGLE_INDEX); index 2 is a joint angle.
SCALES, EPISODES, VX, ANGLE = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0), 16, 8, 1
OUT = HERE.parent / "data" / "probe_scaled_expert.csv"


def main():
    actor = off.load_expert_actor(EXPERT, obs_dim = 17, action_dim = 6)
    rows = []
    for scale in SCALES:
        env = train_gym.make_env("HalfCheetah-v4", 0, vectorized = True, num_envs = EPISODES, env_vectorization_mode = "sync")
        obs, _ = env.reset(seed = 500)
        velocity, size, saturated, at_scale, flipped, returns = [], [], [], [], [], np.zeros(EPISODES)
        for _ in range(1000):
            action = np.clip(off.expert_actions(actor, obs.astype(np.float32), 6) * scale, -1., 1.)
            size.append(np.abs(action).mean()); saturated.append((np.abs(action) > 0.95).mean())
            # Bang-bang shape: the fraction of dims at 0.95 * scale or more (for scale 1, the same as saturated).
            at_scale.append((np.abs(action) >= 0.95 * min(scale, 1.)).mean())
            velocity.append(obs[:, VX]); flipped.append(np.abs(np.angle(np.exp(1j * obs[:, ANGLE]))) > np.pi / 2)
            obs, reward, *_ = env.step(action)
            returns += reward
        env.close()
        velocity = np.stack(velocity)
        rows.append(dict(scale = scale, mean_abs_action = float(np.mean(size)), saturated_frac = float(np.mean(saturated)), at_scale_frac = float(np.mean(at_scale)),
                         velocity_mean = float(velocity.mean()), velocity_p10 = float(np.percentile(velocity, 10)),
                         velocity_p90 = float(np.percentile(velocity, 90)), flipped_frac = float(np.mean(flipped)),
                         return_mean = float(returns.mean()), return_min = float(returns.min())))
        print(rows[-1], flush = True)
    OUT.parent.mkdir(parents = True, exist_ok = True)
    pd.DataFrame(rows).to_csv(OUT, index = False)
    print(pd.DataFrame(rows).round(3).to_string(index = False))


if __name__ == "__main__":
    main()
