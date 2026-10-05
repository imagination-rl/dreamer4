"""Figures of the instability audit.

  figures/entropy.png        real-state entropy of the world-model runs and of model-free PPO, against the loop and against the return
  figures/flipped_states.png real-state entropy on upright and on flipped states; return noise of all episodes and of upright episodes
  figures/policy_change.png  the change of the acting policy in one loop: PPO update (logged approx-KL) and world-model drift (probe)
  figures/counterfactual.png fraction of branches that flip, from the restored state before a recorded flip
  figures/primer.png         what an entropy value and a KL value look like for one action dim of a Beta policy
"""

import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
STAGE1 = ROOT.parent / "10-4-26-stage1-combinations" / "data" / "loop_metrics.csv"
PAIRS = ["combo_cevalue_warmup15", "combo_symexp_warmup15", "combo_symlog_warmup15", "combo_rw4_block2", "combo_rw4_depth5", "combo_rw4_fullres"]
LABEL = {"combo_cevalue_warmup15": "cevalue + warmup15", "combo_symexp_warmup15": "symexp + warmup15", "combo_symlog_warmup15": "symlog + warmup15",
         "combo_rw4_block2": "rw4 + block2", "combo_rw4_depth5": "rw4 + depth5", "combo_rw4_fullres": "rw4 + fullres",
         "loss_normalization": "base", "model_free": "model-free PPO"}
COLORS = ["#c0392b", "#8e44ad", "#e67e22", "#2c6fbb", "#16a085", "#b8860b"]
MF_COLOR, BASE_COLOR = "#222222", "#8e8e8e"


def tidy(ax):
    ax.grid(True, color = "#e6e6e6", lw = 0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def group_of(run):
    return run[len("a2_grouped_"):].rsplit("_s", 1)[0]


def smooth(x, n = 5):
    return np.convolve(x, np.ones(n) / n, mode = "valid")


def entropy_figure():
    replay = pd.read_csv(ROOT / "data" / "replay_loops.csv")
    replay["group"] = replay.run.map(group_of)
    returns = pd.read_csv(STAGE1)
    returns = returns[returns.tag == "rollout/average_return"].rename(columns = {"value": "ret"})[["run", "loop", "ret"]]
    replay = replay.merge(returns, on = ["run", "loop"])
    mf = pd.read_csv(ROOT / "data" / "model_free.csv").pivot_table(index = ["run", "loop"], columns = "tag", values = "value").reset_index()
    fig, axes = plt.subplots(1, 2, figsize = (14, 5.2))
    for group, color in [("loss_normalization", BASE_COLOR)] + list(zip(PAIRS, COLORS)):
        g = replay[replay.group == group].groupby("loop")[["real_entropy", "ret"]].mean()
        axes[0].plot(g.index[4:], smooth(g.real_entropy.to_numpy()), color = color, lw = 1.6, label = LABEL.get(group, group))
        axes[1].plot(smooth(g.ret.to_numpy()), smooth(g.real_entropy.to_numpy()), color = color, lw = 1.6, label = LABEL.get(group, group))
    m = mf.groupby("loop")[["cleanrl_updated_ppo/entropy", "rollout/average_return"]].mean()
    m = m[m.index < 430]
    axes[0].plot(m.index[4:], smooth(m["cleanrl_updated_ppo/entropy"].to_numpy()), color = MF_COLOR, lw = 2.2, ls = "--", label = "model-free PPO (2 seeds)")
    axes[1].plot(smooth(m["rollout/average_return"].to_numpy()), smooth(m["cleanrl_updated_ppo/entropy"].to_numpy()), color = MF_COLOR, lw = 2.2, ls = "--",
                 label = "model-free PPO (2 seeds)")
    axes[0].set_xlim(0, 180)
    axes[0].set_xlabel("loop (64,000 env steps; one PPO update of 10 x 32 minibatches)")
    axes[1].set_xlabel("real return (5-loop mean)")
    axes[1].set_xlim(-400, 9500)
    for ax in axes:
        ax.set_ylabel("policy entropy on real states (nats, + 6 log 2)")
        ax.axhline(6 * np.log(2), color = "#bbbbbb", lw = 0.8, ls = ":")
        tidy(ax)
    axes[0].set_title("Entropy against env steps", loc = "left", fontweight = "bold")
    axes[1].set_title("Entropy against the return it buys", loc = "left", fontweight = "bold")
    axes[1].legend(frameon = False, fontsize = 8, loc = "upper right")
    fig.tight_layout()
    fig.savefig(FIG / "entropy.png", dpi = 150)
    plt.close(fig)


def flipped_states_figure():
    replay = pd.read_csv(ROOT / "data" / "replay_loops.csv")
    replay["group"] = replay.run.map(group_of)
    late = replay[(replay.loop >= 100) & replay.group.isin(PAIRS)]
    fig, axes = plt.subplots(1, 2, figsize = (14, 5))
    x = np.arange(len(PAIRS))
    up = [late[late.group == g].groupby("run").real_entropy_upright.mean().mean() for g in PAIRS]
    fl = [late[late.group == g].groupby("run").real_entropy_flipped.mean().mean() for g in PAIRS]
    axes[0].bar(x - 0.2, up, 0.4, color = "#2c6fbb", label = "upright states")
    axes[0].bar(x + 0.2, fl, 0.4, color = "#c0392b", label = "flipped states")
    axes[0].axhline(6 * np.log(2), color = "#555555", lw = 1, ls = "--", label = "uniform policy (4.16)")
    axes[0].set_ylabel("policy entropy on real states, loops 100-174")
    axes[0].set_title("The policy is untrained on flipped states", loc = "left", fontweight = "bold")

    def noise(x):
        x = np.asarray(x)
        return (x[4:-4] - np.convolve(x, np.ones(9) / 9, "valid")).std() / max(x.mean(), 1.)

    allr = [np.mean([noise(g.return_mean) for _, g in late[late.group == p].groupby("run")]) for p in PAIRS]
    upr = [np.mean([noise(g.return_upright_eps.ffill()) for _, g in late[late.group == p].groupby("run")]) for p in PAIRS]
    axes[1].bar(x - 0.2, allr, 0.4, color = "#555555", label = "all 64 episodes")
    axes[1].bar(x + 0.2, upr, 0.4, color = "#2c6fbb", label = "upright episodes only (< 10 % flipped steps)")
    axes[1].set_ylabel("loop-to-loop return noise / mean return, loops 100-174")
    axes[1].set_title("Without the flipped episodes, the return is smooth", loc = "left", fontweight = "bold")
    for ax in axes:
        ax.set_xticks(x, [LABEL[g] for g in PAIRS], rotation = 20, ha = "right")
        ax.legend(frameon = False, fontsize = 8)
        tidy(ax)
    fig.tight_layout()
    fig.savefig(FIG / "flipped_states.png", dpi = 150)
    plt.close(fig)


def policy_change_figure():
    files = [f for f in glob.glob(str(ROOT / "data" / "probe" / "*__loop_*.csv"))]
    if not files:
        return
    probe = pd.concat([pd.read_csv(f) for f in files])
    probe = probe[(probe.measure == "drift_kl_full") & (probe.t >= 100)]
    drift = probe.groupby(["run", "checkpoint"]).upright_mean.mean().reset_index()
    metrics = pd.read_csv(STAGE1)
    kl = metrics[metrics.tag == "cleanrl_updated_ppo/approx_kl"]
    loop_of = {"loop_50.pt": 49, "loop_100.pt": 99, "loop_175.pt": 174}
    drift["ppo_kl"] = [kl[(kl.run == r) & (kl.loop.between(loop_of[c] - 4, loop_of[c]))].value.mean() for r, c in zip(drift.run, drift.checkpoint)]
    drift["group"] = drift.run.map(group_of)
    drift["label"] = [f"{LABEL.get(g, g)}\n{c[:-3]}" for g, c in zip(drift.group, drift.checkpoint)]
    summary = drift.groupby("label")[["upright_mean", "ppo_kl"]].mean()
    mf = pd.read_csv(ROOT / "data" / "model_free.csv")
    mf_kl = mf[(mf.tag == "cleanrl_updated_ppo/approx_kl") & (mf.loop.between(40, 175))].value.mean()
    fig, ax = plt.subplots(figsize = (max(8, 0.9 * len(summary) + 3), 5))
    x = np.arange(len(summary))
    ax.bar(x - 0.2, summary.ppo_kl, 0.4, color = "#2c6fbb", label = "PPO update of one loop (logged approx-KL, dream states)")
    ax.bar(x + 0.2, summary.upright_mean, 0.4, color = "#c0392b", label = "world-model drift of one loop (probe KL, upright real states)")
    ax.axhline(mf_kl, color = MF_COLOR, ls = "--", lw = 1.2, label = f"model-free PPO update (approx-KL {mf_kl:.3f})")
    ax.set_xticks(x, summary.index, fontsize = 7.5, rotation = 25, ha = "right")
    ax.set_ylabel("KL(before || after) of the acting policy")
    ax.set_title("How far the policy moves in one loop (mean over seeds)", loc = "left", fontweight = "bold")
    ax.legend(frameon = False, fontsize = 8)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(FIG / "policy_change.png", dpi = 150)
    plt.close(fig)


def counterfactual_figure():
    files = [f for f in glob.glob(str(ROOT / "data" / "counterfactual" / "*__loop_*.csv"))]
    if not files:
        return
    cf = pd.concat([pd.read_csv(f) for f in files])
    cf = cf[cf["mode"] != "entropy"]
    fig, ax = plt.subplots(figsize = (8, 4.8))
    offsets = sorted(cf.offset.unique(), reverse = True)
    x = np.arange(len(offsets))
    for i, (mode, color, label) in enumerate((("recorded", "#555555", "recorded actions (check)"), ("sample", "#c0392b", "16 new policy samples"),
                                              ("mean", "#2c6fbb", "mean action (no noise)"))):
        values = [cf[(cf["mode"] == mode) & (cf.offset == o)].flip_frac.mean() for o in offsets]
        ax.bar(x + (i - 1) * 0.27, values, 0.27, color = color, label = label)
    ax.set_xticks(x, [f"{o} steps before the onset" for o in offsets])
    ax.set_ylabel("fraction of branches flipped at onset + 60")
    ax.set_ylim(0, 1)
    ax.set_title(f"Simulator branches from the state before a recorded flip ({cf.groupby(['run', 'loop', 'episode']).ngroups} onsets)",
                 loc = "left", fontweight = "bold", fontsize = 10)
    ax.legend(frameon = False, fontsize = 8)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(FIG / "counterfactual.png", dpi = 150)
    plt.close(fig)


def primer_figure():
    from scipy import optimize, stats as st
    x = np.linspace(-1, 1, 801)

    def beta_with(mean, total_entropy):
        # the concentration that gives the per-dim entropy total_entropy / 6 on the [-1, 1] scale
        target = total_entropy / 6
        f = lambda log_c: st.beta(mean * np.exp(log_c), (1 - mean) * np.exp(log_c)).entropy() + np.log(2.) - target
        c = np.exp(optimize.brentq(f, np.log(2.0001), np.log(1e5)))
        return st.beta(mean * c, (1 - mean) * c)

    fig, axes = plt.subplots(1, 2, figsize = (14, 4.6))
    for total, color in ((4.16, "#8e8e8e"), (2.0, "#2c6fbb"), (0.0, "#16a085"), (-3.0, "#e67e22"), (-6.0, "#c0392b")):
        dist = st.uniform(0, 1) if total > 4.1 else beta_with(0.6, total)
        axes[0].plot(x, dist.pdf((x + 1) / 2) / 2, color = color, lw = 2,
                     label = f"total entropy {total:+.1f} nats: spread about {2 * np.exp(total / 6 - np.log(2)):.2f} wide" + (" (uniform)" if total > 4.1 else ""))
    axes[0].set_xlabel("action of one joint (env scale, -1 to +1)")
    axes[0].set_ylabel("probability density")
    axes[0].set_title("Entropy: how spread out the policy's actions are", loc = "left", fontweight = "bold")
    axes[0].legend(frameon = False, fontsize = 8)
    base = beta_with(0.6, -3.0)
    sd = base.std() * 2
    axes[1].plot(x, base.pdf((x + 1) / 2) / 2, color = "#222222", lw = 2.2, label = "policy before the update")
    for k, color in ((0.05, "#2c6fbb"), (0.2, "#c0392b")):
        # a shift of the mean that gives KL = k for this dim alone (Gaussian approximation: shift = sqrt(2 k) sd)
        shift = np.sqrt(2 * k) * sd
        moved = beta_with(0.6 + shift / 2, -3.0)
        kl_exact = float(np.sum(base.pdf((x[1:-1] + 1) / 2) * (np.log(base.pdf((x[1:-1] + 1) / 2)) - np.log(moved.pdf((x[1:-1] + 1) / 2))))) * (x[1] - x[0]) / 2
        axes[1].plot(x, moved.pdf((x + 1) / 2) / 2, color = color, lw = 2, label = f"after: KL {kl_exact:.2f} (mean moved {shift / sd:.2f} of the spread)")
    axes[1].set_xlabel("action of one joint (env scale, -1 to +1)")
    axes[1].set_title("KL: how far the policy moved", loc = "left", fontweight = "bold")
    axes[1].legend(frameon = False, fontsize = 8)
    for ax in axes:
        ax.set_xlim(-1, 1)
        tidy(ax)
    fig.tight_layout()
    fig.savefig(FIG / "primer.png", dpi = 150)
    plt.close(fig)


def main():
    FIG.mkdir(parents = True, exist_ok = True)
    plt.rcParams.update({"font.size": 9.5, "font.family": "DejaVu Sans"})
    primer_figure()
    entropy_figure()
    flipped_states_figure()
    policy_change_figure()
    counterfactual_figure()


if __name__ == "__main__":
    main()
