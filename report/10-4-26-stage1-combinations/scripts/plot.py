"""Figures of the stage-1 report, from ../data/loop_metrics.csv (written by extract.py).

  figures/returns.png      real return of each pair (3 seeds and their mean), its two arms and the base (loops 0-99)
  figures/flipped.png      fraction of flipped episodes of each run
  figures/diagnostics.png  critic, policy, dream and world-model measures, mean of the seeds of each pair
"""

import collections
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
PAIRS = [
    ("cevalue_warmup15", ("hl_gauss_value_loss", "norm_warmup15"), "cevalue + warmup15"),
    ("symlog_warmup15", ("symlog_hl_gauss_value_head", "norm_warmup15"), "symlog + warmup15"),
    ("symexp_warmup15", ("symexp_hl_gauss_actor_critic", "norm_warmup15"), "symexp + warmup15"),
    ("rw4_block2", ("reward_weight4", "block2_attn_res"), "rw4 + block2"),
    ("rw4_depth5", ("reward_weight4", "no_attn_pool_depth5"), "rw4 + depth5"),
    ("rw4_fullres", ("reward_weight4", "full_attn_res"), "rw4 + fullres"),
]
SEEDS = (4, 42, 1)
SEED_COLORS = {4: "#c0392b", 42: "#2c6fbb", 1: "#27884a"}
PAIR_COLORS = ["#c0392b", "#e67e22", "#8e44ad", "#2c6fbb", "#16a085", "#b8860b"]
ARM_STYLES = ["--", ":"]
BASE_COLOR = "#8e8e8e"


def load():
    data = collections.defaultdict(dict)
    with open(ROOT / "data" / "loop_metrics.csv") as file:
        for row in csv.DictReader(file):
            data[(row["run"], row["tag"])][int(row["loop"])] = float(row["value"])
    return data


def curve(data, run, tag):
    points = data.get((run, tag))
    if not points:
        return None, None
    loops = np.array(sorted(points))
    return loops, np.array([points[loop] for loop in loops])


def mean_curve(data, runs, tag, smooth = 1):
    curves = [curve(data, run, tag) for run in runs]
    curves = [c for c in curves if c[0] is not None]
    if not curves:
        return None, None
    length = min(len(c[1]) for c in curves)
    loops = curves[0][0][:length]
    mean = np.mean([c[1][:length] for c in curves], axis = 0)
    if smooth > 1 and length >= smooth:
        mean = np.convolve(mean, np.ones(smooth) / smooth, mode = "valid")
        loops = loops[smooth - 1:]
    return loops, mean


def tidy(ax):
    ax.grid(True, color = "#e6e6e6", lw = 0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def returns_figure(data):
    fig, axes = plt.subplots(2, 3, figsize = (17, 8.5), sharex = True, sharey = True)
    for ax, (pair, arms, title) in zip(axes.flat, PAIRS):
        base_loops, base_mean = mean_curve(data, [f"a2_grouped_loss_normalization_s{s}" for s in (4, 42, 1, 2)], "rollout/average_return")
        ax.plot(base_loops, base_mean, color = BASE_COLOR, lw = 1.6, label = "base (4 seeds)")
        for arm, style in zip(arms, ARM_STYLES):
            loops, mean = mean_curve(data, [f"a2_grouped_{arm}_s{s}" for s in SEEDS], "rollout/average_return")
            ax.plot(loops, mean, color = "#333333", ls = style, lw = 1.3, label = f"{arm} (arm)")
        for seed in SEEDS:
            loops, values = curve(data, f"a2_grouped_combo_{pair}_s{seed}", "rollout/average_return")
            ax.plot(loops, values, color = SEED_COLORS[seed], lw = 0.8, alpha = 0.75, label = f"pair s{seed}")
        loops, mean = mean_curve(data, [f"a2_grouped_combo_{pair}_s{s}" for s in SEEDS], "rollout/average_return", smooth = 5)
        ax.plot(loops, mean, color = "black", lw = 2.2, label = "pair mean (5-loop)")
        ax.axvline(100, color = "#bbbbbb", lw = 0.8)
        ax.set_title(title, loc = "left", fontsize = 10.5, fontweight = "bold")
        ax.legend(frameon = False, fontsize = 7.5, loc = "upper left")
        tidy(ax)
    for ax in axes[1]:
        ax.set_xlabel("loop")
    for ax in axes[:, 0]:
        ax.set_ylabel("real return")
    fig.suptitle("Stage 1: real return of each pair (thin: seeds; black: mean), its two arms (dashed, dotted; 100 loops) and the base (grey)",
                 x = 0.01, ha = "left", fontweight = "bold", fontsize = 12)
    fig.tight_layout(rect = (0, 0, 1, 0.95))
    fig.savefig(FIGURES / "returns.png", dpi = 150)
    plt.close(fig)


def flipped_figure(data):
    fig, axes = plt.subplots(2, 3, figsize = (17, 7), sharex = True, sharey = True)
    for ax, (pair, _, title) in zip(axes.flat, PAIRS):
        for seed in SEEDS:
            loops, values = curve(data, f"a2_grouped_combo_{pair}_s{seed}", "audit/rollout/flipped_frac")
            ax.plot(loops, values, color = SEED_COLORS[seed], lw = 0.9, label = f"s{seed}")
        ax.set_title(title, loc = "left", fontsize = 10.5, fontweight = "bold")
        ax.set_ylim(0, 1)
        ax.legend(frameon = False, fontsize = 8, loc = "upper left")
        tidy(ax)
    for ax in axes[1]:
        ax.set_xlabel("loop")
    for ax in axes[:, 0]:
        ax.set_ylabel("flipped fraction of the 64 episodes")
    fig.suptitle("Stage 1: fraction of flipped episodes in each loop", x = 0.01, ha = "left", fontweight = "bold", fontsize = 12)
    fig.tight_layout(rect = (0, 0, 1, 0.95))
    fig.savefig(FIGURES / "flipped.png", dpi = 150)
    plt.close(fig)


def diagnostics_figure(data):
    panels = [
        ("cleanrl_updated_ppo/explained_variance", "Critic explained variance (dreams)", 5),
        ("cleanrl_updated_ppo/entropy", "Policy entropy", 5),
        ("audit/rollout/x_velocity", "Real forward velocity (m/s)", 5),
        ("audit/closed_loop/reward_gap_per_step", "Dream minus true reward per step (each 10 loops)", 1),
        ("audit/world_model/one_step_x_velocity_mae", "One-step velocity error of the world model (each 10 loops)", 1),
        ("audit/rollout/action_saturation_frac", "Fraction of saturated actions", 5),
    ]
    fig, axes = plt.subplots(2, 3, figsize = (17, 8.5), sharex = True)
    for ax, (tag, title, smooth) in zip(axes.flat, panels):
        loops, mean = mean_curve(data, [f"a2_grouped_loss_normalization_s{s}" for s in (4, 42, 1, 2)], tag, smooth)
        if loops is not None:
            ax.plot(loops, mean, color = BASE_COLOR, lw = 2.2, label = "base (100 loops)")
        for (pair, _, label), color in zip(PAIRS, PAIR_COLORS):
            loops, mean = mean_curve(data, [f"a2_grouped_combo_{pair}_s{s}" for s in SEEDS], tag, smooth)
            if loops is not None:
                ax.plot(loops, mean, color = color, lw = 1.5, marker = "o" if smooth == 1 else None, ms = 2.5, label = label)
        ax.set_title(title, loc = "left", fontsize = 10, fontweight = "bold")
        tidy(ax)
    for ax in axes[1]:
        ax.set_xlabel("loop")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc = "lower center", ncol = 7, frameon = False, fontsize = 9)
    fig.suptitle("Stage 1: critic, policy, dreams and world model (mean of the 3 seeds of each pair)", x = 0.01, ha = "left", fontweight = "bold", fontsize = 12)
    fig.tight_layout(rect = (0, 0.05, 1, 0.95))
    fig.savefig(FIGURES / "diagnostics.png", dpi = 150)
    plt.close(fig)


def main():
    data = load()
    FIGURES.mkdir(parents = True, exist_ok = True)
    plt.rcParams.update({"font.size": 9.5, "font.family": "DejaVu Sans"})
    returns_figure(data)
    flipped_figure(data)
    diagnostics_figure(data)


if __name__ == "__main__":
    main()
