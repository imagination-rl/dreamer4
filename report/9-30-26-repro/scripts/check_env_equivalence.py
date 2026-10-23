"""Sync vector env against GroupedAsyncVectorEnv (64 envs, 8 workers): same seed and actions -> same data?

Steps 1,100 times (past the 1,000-step truncation, so the autoreset path runs) and compares obs, rewards,
terminations, truncations, and the episode statistics.
Usage: python report/9-30-26-repro/scripts/check_env_equivalence.py
"""

import numpy as np

from train.train_gym import make_env, reset_env

if __name__ == "__main__":
    envs = {
        mode: make_env("HalfCheetah-v4", 42, num_envs = 64, parallel_envs = 8 if mode == "grouped" else None,
                       env_vectorization_mode = "async" if mode == "grouped" else "sync")
        for mode in ("sync", "grouped")
    }
    obs = {mode: reset_env(env, seed = 42) for mode, env in envs.items()}
    worst = abs(obs["sync"] - obs["grouped"]).max()
    rng = np.random.default_rng(0)
    counts = dict(truncations = 0, mismatched_steps = 0)
    for step in range(1100):
        actions = rng.uniform(-1, 1, size = (64, 6)).astype(np.float32)
        out = {mode: env.step(actions) for mode, env in envs.items()}
        (o1, r1, te1, tr1, _), (o2, r2, te2, tr2, _) = out["sync"], out["grouped"]
        diff = max(abs(o1 - o2).max(), abs(r1 - r2).max())
        worst = max(worst, diff)
        counts["truncations"] += int(tr1.sum())
        if diff > 0 or (te1 != te2).any() or (tr1 != tr2).any():
            counts["mismatched_steps"] += 1
    print("max abs difference over obs and rewards:", worst, counts)
    for env in envs.values():
        env.close()
