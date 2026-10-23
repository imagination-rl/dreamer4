"""Figures of the report, from ../data/loop_metrics.csv and ../data/run_summary.csv (written by extract.py).

  figures/returns.png      real return of each arm against the base, both seeds
  figures/speed.png        loop at which the 5-loop mean return passes 0 and 1,000, and the return at loops 95-99
  figures/world_model.png  reward error, raw flow loss, and critic explained variance of each arm (mean of the seeds)
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

ARMS = [
    ("wm_pretrain640", "640 world-model steps before loop 0"),
    ("reward_weight4", "reward loss weight 4"),
    ("norm_warmup15", "plain sum before loop 15"),
    ("pmpo", "PMPO"),
    ("hl_gauss_value_loss", "HL-Gauss value loss"),
    ("symlog_hl_gauss_value_head", "symlog HL-Gauss value head"),
    ("symexp_hl_gauss_actor_critic", "symexp + HL-Gauss critics"),
    ("no_attn_pool", "no AttnPool (ff 7)"),
    ("full_attn_res", "full AttnRes (ff 7)"),
    ("block2_attn_res", "block AttnRes 2 (ff 7)"),
]
BASE_COLOR, ARM_COLOR, PLAIN_COLOR = "#8e8e8e", "#c0392b", "#2c6fbb"


def load():
    data = collections.defaultdict(dict)
    with open(ROOT / "data" / "loop_metrics.csv") as file:
        for row in csv.DictReader(file):
            data[(row["run"], row["tag"])][int(row["loop"])] = float(row["value"])
    with open(ROOT / "data" / "run_summary.csv") as file:
        summary = {row["run"]: row for row in csv.DictReader(file)}
    return data, summary


def series(data, run, tag):
    points = data.get((run, tag))
    if not points:
        return None, None
    loops = np.array(sorted(points))
    return loops + 1, np.array([points[loop] for loop in loops])


def run_name(arm, seed):
    return f"a2_grouped_{'loss_normalization' if arm == 'base' else arm}_s{seed}"


def tidy(ax):
    ax.grid(True, color = "#e6e6e6", lw = 0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def returns_figure(data):
    fig, axes = plt.subplots(2, 5, figsize = (17, 7), sharex = True, sharey = True)
    for ax, (arm, title) in zip(axes.flat, ARMS):
        for seed, style in ((4, "-"), (42, "--")):
            for run, color, width, label in (
                (run_name("base", seed), BASE_COLOR, 1.2, f"base s{seed}"),
                (run_name(arm, seed), ARM_COLOR, 1.8, f"arm s{seed}"),
            ):
                loops, values = series(data, run, "rollout/average_return")
                if loops is not None:
                    ax.plot(loops, values, color = color, ls = style, lw = width, label = label)
        ax.set_title(title, loc = "left", fontsize = 9.5, fontweight = "bold")
        ax.set_xlim(1, 100)
        tidy(ax)
    for ax in axes[1]:
        ax.set_xlabel("loop")
    for ax in axes[:, 0]:
        ax.set_ylabel("real return")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc = "lower center", ncol = 4, frameon = False)
    fig.suptitle("Real return of each arm (red) against the grouped-normalization base (grey); solid seed 4, dashed seed 42",
                 x = 0.01, ha = "left", fontweight = "bold", fontsize = 12)
    fig.tight_layout(rect = (0, 0.05, 1, 0.95))
    fig.savefig(FIGURES / "returns.png", dpi = 150)
    plt.close(fig)


def speed_figure(summary):
    arms = [("base", "base (grouped normalization)")] + ARMS + [("plain A2", "plain A2 (no normalization)")]
    fig, axes = plt.subplots(1, 3, figsize = (17, 6.2), sharey = True)
    y = np.arange(len(arms))
    for ax, key, title in (
        (axes[0], "loop_at_0", "Loop at which the 5-loop mean return passes 0"),
        (axes[1], "loop_at_1000", "Loop at which it passes 1,000 (none: 105)"),
        (axes[2], "return_last5", "Return at loops 95-99"),
    ):
        for offset, seed, marker in ((-0.15, 4, "o"), (0.15, 42, "s")):
            values = []
            for arm, _ in arms:
                run = f"audit_a2_next_state_reward_target_local_s{seed}" if arm == "plain A2" else run_name(arm, seed)
                row = summary.get(run)
                value = row.get(key) if row else None
                values.append(float(value) if value not in (None, "") else (105. if row and key != "return_last5" else np.nan))
            colors = [BASE_COLOR if arm == "base" else PLAIN_COLOR if arm == "plain A2" else ARM_COLOR for arm, _ in arms]
            ax.scatter(values, y + offset, c = colors, marker = marker, s = 42, label = f"seed {seed}", zorder = 3)
        ax.set_title(title, loc = "left", fontsize = 10, fontweight = "bold")
        tidy(ax)
    axes[0].set_yticks(y, [label for _, label in arms])
    axes[0].invert_yaxis()
    axes[0].legend(frameon = False, loc = "lower right")
    fig.suptitle("Speed and final return of each arm (circle seed 4, square seed 42)", x = 0.01, ha = "left", fontweight = "bold", fontsize = 12)
    fig.tight_layout(rect = (0, 0, 1, 0.94))
    fig.savefig(FIGURES / "speed.png", dpi = 150)
    plt.close(fig)


def world_model_figure(data):
    panels = [
        ("world_model/reward_mae", "Reward error on replay (MAE)", False),
        ("extra/flow_loss_raw", "Flow loss, raw (log scale)", True),
        ("cleanrl_updated_ppo/explained_variance", "Critic explained variance on dreams", False),
    ]
    groups = [("base", "base")] + ARMS
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 3, figsize = (17, 5.6), sharex = True)
    for ax, (tag, title, log) in zip(axes, panels):
        for index, (arm, label) in enumerate(groups):
            curves = [series(data, run_name(arm, seed), tag)[1] for seed in (4, 42)]
            curves = [curve for curve in curves if curve is not None]
            if not curves:
                continue
            length = min(len(curve) for curve in curves)
            mean = np.mean([curve[:length] for curve in curves], axis = 0)
            if length >= 5:
                mean = np.convolve(mean, np.ones(5) / 5, mode = "valid")
            color = "#222222" if arm == "base" else cmap(index % 10)
            ax.plot(np.arange(len(mean)) + 5, mean, color = color, lw = 2.2 if arm == "base" else 1.3, label = label)
        ax.set_title(title, loc = "left", fontsize = 10, fontweight = "bold")
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("loop")
        ax.set_xlim(1, 100)
        tidy(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc = "lower center", ncol = 6, frameon = False, fontsize = 8.6)
    fig.suptitle("World model and critic (mean of the two seeds, 5-loop mean)", x = 0.01, ha = "left", fontweight = "bold", fontsize = 12)
    fig.tight_layout(rect = (0, 0.1, 1, 0.94))
    fig.savefig(FIGURES / "world_model.png", dpi = 150)
    plt.close(fig)


def main():
    data, summary = load()
    FIGURES.mkdir(parents = True, exist_ok = True)
    plt.rcParams.update({"font.size": 9.5, "font.family": "DejaVu Sans"})
    returns_figure(data)
    speed_figure(summary)
    world_model_figure(data)


if __name__ == "__main__":
    main()
