"""Figures of the report, from ../data/loop_metrics.csv and ../data/run_summary.csv (written by extract.py).

  figures/returns.png        real return for each loop
  figures/world_model.png    world-model quality
  figures/actor_critic.png   actor and critic behavior
  figures/stability.png      best return against the largest drop, all compared runs
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

GROUPED_S4 = "a2_grouped_loss_normalization_s4"
GROUPED_S42 = "a2_grouped_loss_normalization_s42"

# run: (label, color, line style, line width)
A2_FAMILY = {
    GROUPED_S4: ("grouped norm s4", "#c0392b", "-", 2.6),
    GROUPED_S42: ("grouped norm s42", "#c0392b", "--", 2.0),
    "audit_a2_next_state_reward_target_local_s4": ("A2 local s4", "#2c6fbb", "-", 1.5),
    "audit_a2_next_state_reward_target_local_s42": ("A2 local s42", "#2c6fbb", "--", 1.2),
    "audit_a2_next_state_reward_target2": ("A2 H100 s4", "#6aa5d8", "-", 1.5),
    "audit_a2_next_state_reward_target": ("A2 H100 s42", "#6aa5d8", "--", 1.2),
    "a2_reduced_reward_transformer_gradient2": ("A2 reward grad 0.01 s4 (H100)", "#1a9a6c", "-", 1.5),
    "a2_reduced_reward_transformer_gradient": ("A2 reward grad 0.01 s42 (H100)", "#1a9a6c", "--", 1.2),
    "a2_loss_normalization_s4": ("separate norm s4", "#8e8e8e", "-", 1.5),
}
OTHER_FAMILY = {
    GROUPED_S4: ("grouped norm s4", "#c0392b", "-", 2.6),
    GROUPED_S42: ("grouped norm s42", "#c0392b", "--", 2.0),
    "symlog_hl_gauss_value_head2": ("symlog HL-Gauss, PMPO s4 (H100)", "#7b4fb5", "-", 1.5),
    "symlog_hl_gauss_value_head": ("symlog HL-Gauss, PMPO s42 (H100)", "#7b4fb5", "--", 1.2),
    "closer_ppo_symlog_hl_gauss_value_head2": ("symlog HL-Gauss, closer PPO s4 (H100)", "#e08a00", "-", 1.5),
    "closer_ppo_symlog_hl_gauss_value_head": ("symlog HL-Gauss, closer PPO s42 (H100)", "#e08a00", "--", 1.2),
    "closer_ppo_next_state_reward_symlog_value_head2": ("symlog HL-Gauss, A2 s4 (H100)", "#a0522d", "-", 1.5),
    "closer_ppo_next_state_reward_symlog_value_head": ("symlog HL-Gauss, A2 s42 (H100)", "#a0522d", "--", 1.2),
    "offline_pretrained_a2_scaled40_no_rehearsal_s4": ("pretrained scaled40 s4 (local)", "#555555", "-", 1.5),
    "offline_pretrained_a2_scaled40_no_rehearsal_s42": ("pretrained scaled40 s42 (local)", "#555555", "--", 1.2),
}
GROUP_COLORS = {
    "grouped norm": "#c0392b",
    "separate norm": "#8e8e8e",
    "A2 local": "#2c6fbb",
    "A2 H100": "#6aa5d8",
    "A2 reward grad 0.01 (H100)": "#1a9a6c",
    "symlog HL-Gauss, PMPO (H100)": "#7b4fb5",
    "symlog HL-Gauss, closer PPO (H100)": "#e08a00",
    "symlog HL-Gauss, A2 (H100)": "#a0522d",
    "local pretrained": "#b9b9b9",
}


def load():
    data = collections.defaultdict(dict)
    with open(ROOT / "data" / "loop_metrics.csv") as file:
        for row in csv.DictReader(file):
            data[(row["run"], row["tag"])][int(row["loop"])] = float(row["value"])
    with open(ROOT / "data" / "run_summary.csv") as file:
        summary = list(csv.DictReader(file))
    return data, summary


def series(data, run, tag):
    points = data.get((run, tag))
    if not points:
        return None, None
    loops = np.array(sorted(points))
    return loops + 1, np.array([points[loop] for loop in loops])


def style(ax, title, ylabel = None, log = False):
    ax.set_title(title, loc = "left", fontweight = "bold", fontsize = 10)
    if ylabel:
        ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.set_xlim(1, 100)
    ax.grid(True, color = "#e6e6e6", lw = 0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def draw(ax, data, family, tag, smooth = 1):
    for run, (label, color, line_style, width) in family.items():
        loops, values = series(data, run, tag)
        if loops is None:
            continue
        if smooth > 1 and len(values) >= smooth:
            values = np.convolve(values, np.ones(smooth) / smooth, mode = "valid")
            loops = loops[smooth - 1:]
        marker = "o" if len(loops) <= 12 else None
        ax.plot(loops, values, color = color, ls = line_style, lw = width, label = label, marker = marker, ms = 3, zorder = 3 if "grouped" in label else 2)


def legend_below(fig, ax, columns):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc = "lower center", ncol = columns, frameon = False, fontsize = 8.6)


def main():
    data, summary = load()
    FIGURES.mkdir(parents = True, exist_ok = True)
    plt.rcParams.update({"font.size": 9.5, "font.family": "DejaVu Sans"})

    # 1. returns
    fig, axes = plt.subplots(1, 2, figsize = (14.5, 5.6), sharey = True)
    for ax, family, title in (
        (axes[0], A2_FAMILY, "A2 settings, from scratch"),
        (axes[1], OTHER_FAMILY, "Against the symlog HL-Gauss value head runs and the best local pretrained runs"),
    ):
        draw(ax, data, family, "rollout/average_return")
        style(ax, title, "real return (mean of 64 episodes)" if ax is axes[0] else None)
        ax.set_xlabel("loop")
        ax.legend(loc = "upper left", frameon = False, fontsize = 8.2)
    fig.suptitle("Real return for each loop (first 100 loops of each run)", x = 0.01, ha = "left", fontweight = "bold", fontsize = 12.5)
    fig.tight_layout(rect = (0, 0, 1, 0.95))
    fig.savefig(FIGURES / "returns.png", dpi = 160)
    plt.close(fig)

    # 2. world model
    panels = [
        ("extra/flow_loss_raw", "Flow loss, raw (replay windows)", True, 1),
        ("audit/world_model/one_step_norm_mae", "One-step state error (normalized MAE, log scale)", True, 1),
        ("audit/closed_loop/state_norm_mae_h8", "Dream state error after 8 steps (log scale)", True, 1),
        ("audit/world_model/reward_head_corr", "Reward head correlation with the true reward", False, 1),
        ("audit/world_model/ranking_spearman", "Action ranking (Spearman, dream against simulator)", False, 1),
        ("audit/closed_loop/reward_gap_per_step", "Dream reward minus true reward, for each step", False, 1),
    ]
    fig, axes = plt.subplots(2, 3, figsize = (14.5, 8.2), sharex = True)
    for ax, (tag, title, log, smooth) in zip(axes.flat, panels):
        draw(ax, data, A2_FAMILY, tag, smooth)
        style(ax, title, log = log)
    axes[1, 2].axhline(0., color = "#444444", lw = 0.8)
    for ax in axes[1]:
        ax.set_xlabel("loop")
    legend_below(fig, axes[0, 0], 5)
    fig.suptitle("World model: grouped normalization against the A2 runs", x = 0.01, ha = "left", fontweight = "bold", fontsize = 12.5)
    fig.tight_layout(rect = (0, 0.07, 1, 0.95))
    fig.savefig(FIGURES / "world_model.png", dpi = 160)
    plt.close(fig)

    # 3. actor and critic
    panels = [
        ("cleanrl_updated_ppo/entropy", "Policy entropy (closer PPO)", 1),
        ("cleanrl_updated_ppo/approx_kl", "Policy change of one PPO update (approx. KL)", 3),
        ("cleanrl_updated_ppo/clip_fraction", "PPO clip fraction", 3),
        ("cleanrl_updated_ppo/explained_variance", "Critic explained variance on dreams", 3),
        ("audit/rollout/flipped_frac", "Fraction of real steps with a flipped body", 3),
        ("audit/rollout/control_cost_per_step", "Control cost for each real step", 3),
    ]
    family = {**A2_FAMILY, **{run: OTHER_FAMILY[run] for run in ("closer_ppo_next_state_reward_symlog_value_head2", "closer_ppo_next_state_reward_symlog_value_head")}}
    fig, axes = plt.subplots(2, 3, figsize = (14.5, 8.2), sharex = True)
    for ax, (tag, title, smooth) in zip(axes.flat, panels):
        draw(ax, data, family, tag, smooth)
        style(ax, title)
    for ax in axes[1]:
        ax.set_xlabel("loop")
    legend_below(fig, axes[1, 2], 4)
    fig.suptitle("Actor and critic: grouped normalization against the A2 runs and the symlog HL-Gauss A2 runs", x = 0.01, ha = "left", fontweight = "bold", fontsize = 12.5)
    fig.tight_layout(rect = (0, 0.10, 1, 0.95))
    fig.savefig(FIGURES / "actor_critic.png", dpi = 160)
    plt.close(fig)

    # 4. stability of all runs
    fig, axes = plt.subplots(1, 2, figsize = (14.5, 5.8))
    seen = set()
    for row in summary:
        color = GROUP_COLORS[row["group"]]
        best, drawdown, noise = float(row["return_best5"]), float(row["max_drawdown5"]), float(row["noise_std_after_takeoff"])
        is_grouped = row["group"] == "grouped norm"
        kwargs = dict(color = color, s = 150 if is_grouped else 46, marker = "*" if is_grouped else "o", edgecolor = "#222222" if is_grouped else "white", linewidth = 0.8, zorder = 4 if is_grouped else 2)
        label = row["group"] if row["group"] not in seen else None
        seen.add(row["group"])
        axes[0].scatter(best, drawdown, label = label, **kwargs)
        axes[1].scatter(best, noise, **kwargs)
        if is_grouped:
            for ax, y in ((axes[0], drawdown), (axes[1], noise)):
                ax.annotate(row["label"], (best, y), textcoords = "offset points", xytext = (8, 6), fontsize = 8.6, color = color)
    for ax, title, ylabel in (
        (axes[0], "Largest drop of the 5-loop mean return below its earlier maximum", "largest drop"),
        (axes[1], "Loop-to-loop noise after the return passes 0 (std around a 9-loop mean)", "noise (std)"),
    ):
        ax.set_title(title, loc = "left", fontweight = "bold", fontsize = 10)
        ax.set_xlabel("best 5-loop mean return in 100 loops")
        ax.set_ylabel(ylabel)
        ax.grid(True, color = "#e6e6e6", lw = 0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc = "lower center", ncol = 5, frameon = False, fontsize = 8.6)
    fig.suptitle("Stability of each compared run (one point for each run, first 100 loops)", x = 0.01, ha = "left", fontweight = "bold", fontsize = 12.5)
    fig.tight_layout(rect = (0, 0.09, 1, 0.95))
    fig.savefig(FIGURES / "stability.png", dpi = 160)
    plt.close(fig)


if __name__ == "__main__":
    main()
