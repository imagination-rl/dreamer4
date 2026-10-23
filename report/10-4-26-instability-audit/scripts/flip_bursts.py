"""Loop-level event study: what changes in the loops before a burst of flipped episodes?

A burst starts at loop L when at least 15 % of the 64 episodes of loop L end flipped (more than half of their steps)
and the mean of loops L - 5 .. L - 1 is below 5 %. Runs: the 18 pair runs (loops 20-174) and the base and arm runs (loops 20-99).
For each measure the script compares loops L - 3 .. L - 1 with loops L - 10 .. L - 6 of the same run (a paired change), and
gives the same change for all loops of the same runs as the reference (the trend of the measure).
It also gives the within-run Spearman correlation of the flipped-episode fraction of loop L with each measure of loop L - 1,
after the removal of a 15-loop running mean from both (so the slow trends do not count).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
STAGE1 = ROOT.parent / "10-4-26-stage1-combinations" / "data" / "loop_metrics.csv"
TAGS = {
    "cleanrl_updated_ppo/approx_kl": "ppo_kl", "cleanrl_updated_ppo/clip_fraction": "clip_frac", "cleanrl_updated_ppo/entropy": "dream_entropy",
    "cleanrl_updated_ppo/explained_variance": "critic_ev", "extra/flow_loss_raw": "flow_loss", "world_model/reward_mae": "reward_mae",
    "imagination/dream_return": "dream_return", "world_model/grad_norm": "wm_grad_norm",
}
REPLAY = ["real_entropy_upright", "x_velocity", "saturation", "return_mean"]


def load():
    m = pd.read_csv(STAGE1)
    m = m[m.tag.isin(TAGS)].pivot_table(index = ["run", "loop"], columns = "tag", values = "value").rename(columns = TAGS).reset_index()
    r = pd.read_csv(ROOT / "data" / "replay_loops.csv")
    d = r.merge(m, on = ["run", "loop"], how = "left")
    d = d[~d.run.str.contains("pmpo|flat_lr")]
    return d.sort_values(["run", "loop"])


def main():
    d = load()
    measures = REPLAY + list(TAGS.values())
    events, reference = [], []
    for run, g in d.groupby("run"):
        g = g.set_index("loop")
        flip = g.flipped_episode_frac
        last = g.index.max()
        for L in range(20, last + 1):
            pre5 = flip.loc[L - 5:L - 1].mean()
            change = {k: g[k].loc[L - 3:L - 1].mean() - g[k].loc[L - 10:L - 6].mean() for k in measures}
            reference.append({"run": run, **change})
            if flip.loc[L] >= 0.15 and pre5 < 0.05:
                events.append({"run": run, "loop": L, "burst_frac": flip.loc[L], **change})
    events, reference = pd.DataFrame(events), pd.DataFrame(reference)
    events.to_csv(ROOT / "data" / "flip_bursts.csv", index = False)
    print(f"{len(events)} bursts in {events.run.nunique()} runs")
    print(events[["run", "loop", "burst_frac"]].to_string(index = False))
    print(f"\n{'measure':22s} {'change before burst':>20s} {'change, all loops':>18s} {'Wilcoxon p (bursts)':>20s} {'Mann-Whitney p':>15s}")
    summary = []
    for k in measures:
        e, ref = events[k].dropna(), reference[k].dropna()
        p_w = stats.wilcoxon(e).pvalue if len(e) > 5 else float("nan")
        p_mw = stats.mannwhitneyu(e, ref).pvalue
        print(f"{k:22s} {e.mean():20.4f} {ref.mean():18.4f} {p_w:20.3f} {p_mw:15.3f}")
        summary.append({"measure": k, "change_before_burst": e.mean(), "change_all_loops": ref.mean(), "p_burst_vs_all": p_mw})

    print("\nwithin-run Spearman: flipped-episode fraction of loop L against each measure of loop L - 1 (both detrended), pooled")
    rows = []
    for run, g in d.groupby("run"):
        g = g.set_index("loop").loc[20:]
        detrend = lambda x: x - x.rolling(15, center = True, min_periods = 5).mean()
        flip = detrend(g.flipped_episode_frac)
        for k in measures:
            x = detrend(g[k]).shift(1)
            ok = flip.notna() & x.notna()
            rows += [{"measure": k, "flip": f, "x": v} for f, v in zip(flip[ok], x[ok])]
    pooled = pd.DataFrame(rows)
    for k in measures:
        s = pooled[pooled.measure == k]
        rho, p = stats.spearmanr(s.x, s.flip)
        print(f"  {k:22s} rho {rho:+.3f} p {p:.4f} (n {len(s)})")
        next(row for row in summary if row["measure"] == k).update({"lag1_rho": rho, "lag1_p": p})
    pd.DataFrame(summary).to_csv(ROOT / "data" / "flip_bursts_summary.csv", index = False)


if __name__ == "__main__":
    main()
