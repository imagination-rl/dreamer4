"""Are the flipped episodes of each loop chance at a slowly changing rate, or do they vary more than chance?

The count of flipped episodes (more than half of the steps flipped) of loop L is k of n = 64. If each episode flips
independently with the rate p of the policy, and p changes slowly, then k is binomial(64, p_L) with p_L from a centered
15-loop mean (loop L left out). The dispersion ratio is sum (k - n p)^2 / sum n p (1 - p); it is 1 for chance and above 1
when the rate itself jumps from loop to loop (for example after a policy update). The p-value comes from 2,000 simulated
binomial series with the same p_L. Also: the lag-1 autocorrelation of the residual k - n p (positive when bursts persist).
Windows: pair runs loops 20-174, base and arm runs loops 20-99.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    rng = np.random.default_rng(0)
    d = pd.read_csv(ROOT / "data" / "replay_loops.csv")
    d = d[~d.run.str.contains("pmpo|flat_lr")]
    rows = []
    for run, g in d.groupby("run"):
        g = g.sort_values("loop")
        g = g[g.loop >= 20]
        k = np.round(g.flipped_episode_frac.to_numpy() * 64)
        n = 64
        total = pd.Series(k).rolling(15, center = True, min_periods = 8).sum().to_numpy()
        count = pd.Series(np.ones_like(k)).rolling(15, center = True, min_periods = 8).sum().to_numpy()
        p = np.clip((total - k) / (count - 1) / n, 1e-3, 1 - 1e-3)
        ok = np.isfinite(p)
        k, p = k[ok], p[ok]
        if k.sum() < 10:
            continue
        ratio = ((k - n * p) ** 2).sum() / (n * p * (1 - p)).sum()
        sims = rng.binomial(n, p, size = (2000, len(p)))
        sim_ratio = ((sims - n * p) ** 2).sum(1) / (n * p * (1 - p)).sum()
        resid = (k - n * p) / np.sqrt(n * p * (1 - p))
        rows.append({"run": run, "loops": len(k), "flipped_eps": int(k.sum()), "dispersion": ratio,
                     "p_chance": float((sim_ratio >= ratio).mean()), "lag1_autocorr": float(np.corrcoef(resid[:-1], resid[1:])[0, 1])})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "data" / "dispersion.csv", index = False)
    pd.set_option("display.width", 160)
    print(out.round(3).to_string(index = False))
    print(f"\nmedian dispersion {out.dispersion.median():.2f}; runs with p < 0.05: {(out.p_chance < 0.05).sum()} of {len(out)}; "
          f"median lag-1 autocorrelation {out.lag1_autocorr.median():+.2f}")


if __name__ == "__main__":
    main()
