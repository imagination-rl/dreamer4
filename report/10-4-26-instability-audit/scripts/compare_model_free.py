"""The world-model runs against the model-free CleanRL PPO runs (same Beta policy, same PPO settings, entropy weight 0).

One loop = one model-free iteration = 64,000 env steps and one PPO update of 10 epochs x 32 minibatches at lr 3e-4.
  1. Entropy at the same loop: the dream entropy (logged), the real entropy (replay), and the model-free entropy.
  2. Entropy at the same return: the real entropy at the first loop at which the 5-loop mean return reaches R.
  3. The size of each PPO update: approx_kl and clip fraction.
  4. Return stability: relative noise about a 9-loop mean, and loops more than 15 % below the 5-loop mean before them.
Reads ../data/replay_loops.csv, ../data/model_free.csv and the stage-1 loop metrics. Writes ../data/compare_*.csv.
"""

import collections
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAGE1 = ROOT.parent / "10-4-26-stage1-combinations" / "data" / "loop_metrics.csv"
MAIN = ["loss_normalization", "hl_gauss_value_loss", "norm_warmup15", "symexp_hl_gauss_actor_critic",
        "combo_cevalue_warmup15", "combo_symexp_warmup15", "combo_symlog_warmup15", "combo_rw4_block2", "combo_rw4_depth5", "combo_rw4_fullres"]
RETURNS = (500, 1000, 2000, 3000, 4000)


def group_of(run):
    return run[len("a2_grouped_"):].rsplit("_s", 1)[0] if run.startswith("a2_grouped_") else "model_free"


def load():
    metrics = pd.read_csv(STAGE1)
    metrics = metrics[metrics.tag.isin(["cleanrl_updated_ppo/entropy", "cleanrl_updated_ppo/approx_kl", "cleanrl_updated_ppo/clip_fraction",
                                        "rollout/average_return"])]
    wide = metrics.pivot_table(index = ["run", "loop"], columns = "tag", values = "value").reset_index()
    replay = pd.read_csv(ROOT / "data" / "replay_loops.csv")[["run", "loop", "real_entropy", "real_entropy_upright"]]
    wide = wide.merge(replay, on = ["run", "loop"], how = "left")
    # The dream entropy of loop L - 1 belongs to the policy that collected loop L.
    dream = wide[["run", "loop", "cleanrl_updated_ppo/entropy"]].copy()
    dream["loop"] += 1
    wide = wide.drop(columns = "cleanrl_updated_ppo/entropy").merge(dream, on = ["run", "loop"], how = "left")
    mf = pd.read_csv(ROOT / "data" / "model_free.csv").pivot_table(index = ["run", "loop"], columns = "tag", values = "value").reset_index()
    # The model-free entropy of iteration L is on the real states of iteration L, after the update; shift it as the dream entropy.
    mf_entropy = mf[["run", "loop", "cleanrl_updated_ppo/entropy"]].copy()
    mf_entropy["loop"] += 1
    mf = mf.drop(columns = "cleanrl_updated_ppo/entropy").merge(mf_entropy, on = ["run", "loop"], how = "left")
    mf["real_entropy"] = mf["cleanrl_updated_ppo/entropy"]
    data = pd.concat([wide, mf], ignore_index = True)
    data["group"] = data.run.map(group_of)
    return data[data.group.isin(MAIN + ["model_free"])]


def stability(ret):
    ret = np.asarray(ret)
    trend = np.convolve(ret, np.ones(9) / 9, mode = "valid")
    noise = (ret[4:-4] - trend).std() / max(ret.mean(), 1.)
    smooth = np.convolve(ret, np.ones(5) / 5, mode = "valid")
    drops = sum(1 for i in range(5, len(ret)) if ret[i] < 0.85 * smooth[i - 5])
    return noise, drops / len(ret)


def main():
    data = load()
    order = MAIN + ["model_free"]
    pd.set_option("display.width", 200)

    print("== 1. entropy at the same loop (mean over seeds); dream = cleanrl_updated_ppo/entropy of the collecting policy")
    rows = []
    for (lo, hi) in ((1, 25), (25, 50), (50, 75), (75, 100), (100, 125), (125, 150), (150, 175)):
        w = data[(data.loop >= lo) & (data.loop < hi)].groupby(["group", "run"])[["real_entropy", "cleanrl_updated_ppo/entropy", "rollout/average_return"]].mean()
        w = w.groupby("group").mean()
        for group, row in w.iterrows():
            rows.append({"window": f"{lo}-{hi - 1}", "group": group, "real_entropy": row.real_entropy,
                         "dream_entropy": row["cleanrl_updated_ppo/entropy"] if group != "model_free" else float("nan"),
                         "return": row["rollout/average_return"]})
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "data" / "compare_entropy_by_loop.csv", index = False)
    for value in ("real_entropy", "dream_entropy", "return"):
        print(f"\n-- {value}")
        print(table.pivot(index = "group", columns = "window", values = value).reindex(order).round(2 if value != "return" else 0).to_string())

    print("\n== 2. real entropy at the first loop at which the 5-loop mean return reaches R (mean over the runs that reach R; n runs)")
    rows = []
    for run, g in data.groupby("run"):
        g = g.sort_values("loop")
        smooth = g["rollout/average_return"].rolling(5).mean().to_numpy()
        for target in RETURNS:
            hit = np.nonzero(smooth >= target)[0]
            if len(hit):
                i = hit[0]
                rows.append({"run": run, "group": group_of(run), "R": target, "loop": int(g.loop.iloc[i]),
                             "real_entropy": float(g.real_entropy.iloc[max(0, i - 4):i + 1].mean())})
    matched = pd.DataFrame(rows)
    matched.to_csv(ROOT / "data" / "compare_entropy_by_return.csv", index = False)
    summary = matched.groupby(["group", "R"]).agg(entropy = ("real_entropy", "mean"), loop = ("loop", "mean"), n = ("run", "count")).reset_index()
    for value in ("entropy", "loop", "n"):
        print(f"\n-- {value}")
        print(summary.pivot(index = "group", columns = "R", values = value).reindex(order).round(2 if value == "entropy" else 0).to_string())

    print("\n== 3. size of each PPO update (mean over seeds)")
    rows = []
    for (lo, hi) in ((1, 25), (25, 50), (50, 100), (100, 175)):
        w = data[(data.loop >= lo) & (data.loop < hi)].groupby("group")[["cleanrl_updated_ppo/approx_kl", "cleanrl_updated_ppo/clip_fraction"]].mean()
        for group, row in w.iterrows():
            rows.append({"window": f"{lo}-{hi - 1}", "group": group, "approx_kl": row["cleanrl_updated_ppo/approx_kl"],
                         "clip_fraction": row["cleanrl_updated_ppo/clip_fraction"]})
    kl = pd.DataFrame(rows)
    kl.to_csv(ROOT / "data" / "compare_ppo_step.csv", index = False)
    for value in ("approx_kl", "clip_fraction"):
        print(f"\n-- {value}")
        print(kl.pivot(index = "group", columns = "window", values = value).reindex(order).round(3).to_string())

    print("\n== 4. return stability: relative noise / fraction of drop loops, by window, mean over seeds; model-free also at its matched-return window")
    rows = []
    for run, g in data.groupby("run"):
        ret = g.sort_values("loop")["rollout/average_return"].to_numpy()
        windows = {"50-99": (50, 100), "100-174": (100, 175), "175-299": (175, 300), "300-420": (300, 421)}
        for name, (lo, hi) in windows.items():
            if len(ret) >= hi:
                noise, drops = stability(ret[lo:hi])
                rows.append({"run": run, "group": group_of(run), "window": name, "rel_noise": noise, "drop_frac": drops, "mean_return": ret[lo:hi].mean()})
    stab = pd.DataFrame(rows)
    stab.to_csv(ROOT / "data" / "compare_stability.csv", index = False)
    print(stab.groupby(["group", "window"])[["rel_noise", "drop_frac", "mean_return"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
