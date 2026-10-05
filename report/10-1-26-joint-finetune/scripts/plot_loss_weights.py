"""Loss weight of each term in the joint loss (fixed_full_transformer_finetuning), for each loop.

Reads the TensorBoard logs of queue 10-1-26b (entries 69-72) and writes:
  data/loss_terms.csv        per-loop mean of each term, for each run
  figures/loss_weights.png   the figure

Three views of "weight", all on a log scale:
  1. share of the summed loss (term * coefficient / sum of absolute terms)
  2. share of the transformer gradient (joint_diag/ norms, measured one time for each loop)
  3. the multiplier that the paper's RMS normalization applies (1 / running RMS)
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
LOG_ROOT = REPO / "runs" / "gym_imagination"

ARMS = {
    "From scratch": ["joint_finetune_scratch_s42", "joint_finetune_scratch_s4"],
    "Pretrained (a2_scaled40_s4)": ["joint_finetune_pretrained_s42", "joint_finetune_pretrained_s4"],
}

# name: (tag, coefficient in the sum). The reward loss sums over 8 multi-token distances; the tag holds the mean.
LOSS_TERMS = {
    "reward": ("world_model/reward_loss", 8.),
    "value": ("imagination/value_loss", 1.),
    "policy (PMPO + KL)": ("imagination/policy_loss", 1.),
    "terminal": ("world_model/terminal_loss", 1.),
    "flow": ("world_model/flow_loss", 1.),
    "bootstrap": ("world_model/shortcut_loss", 1.),
}

GRAD_TERMS = {
    "reward": "joint_diag/trunk_grad_norm_reward",
    "value": "joint_diag/trunk_grad_norm_value",
    "policy (PMPO + KL)": "joint_diag/trunk_grad_norm_policy",
    "flow + bootstrap": "joint_diag/trunk_grad_norm_dynamics",
}

COLORS = {
    "reward": "#c0392b",
    "value": "#2c6fbb",
    "policy (PMPO + KL)": "#1a9a6c",
    "terminal": "#8e8e8e",
    "flow": "#e08a00",
    "bootstrap": "#7b4fb5",
    "flow + bootstrap": "#e08a00",
}

RMS_BETA = 0.95  # LossNormalizer default
STEPS_PER_LOOP = 10


def scalars(acc, tag):
    return np.array([event.value for event in acc.Scalars(tag)])


def running_rms_weight(values):
    """The multiplier of LossNormalizer for each step: 1 / sqrt(EMA of the squared loss), read before the update.

    The EMA starts at the first value, not at 1 as in LossNormalizer. This removes the warm-up ramp of about 27 loops.
    """
    exp_avg_sq, weights = float(values[0]) ** 2, []
    for value in values:
        weights.append(1. / max(np.sqrt(exp_avg_sq), 1e-6))
        exp_avg_sq += (1. - RMS_BETA) * (value * value - exp_avg_sq)
    return np.array(weights)


def per_loop(values):
    loops = len(values) // STEPS_PER_LOOP
    return values[:loops * STEPS_PER_LOOP].reshape(loops, STEPS_PER_LOOP).mean(axis = 1)


def load_run(name):
    acc = EventAccumulator(str(LOG_ROOT / name), size_guidance = {"scalars": 0}).Reload()
    joint_steps = len(scalars(acc, "imagination/value_loss")) // STEPS_PER_LOOP * STEPS_PER_LOOP

    contribution, rms_weight = {}, {}
    for term, (tag, coefficient) in LOSS_TERMS.items():
        values = scalars(acc, tag)
        # World-model tags of a from-scratch run start with the 256 world-model-only steps of loop 0. Drop them.
        is_world_model_tag = tag.startswith("world_model/")
        offset = len(values) - len(scalars(acc, "imagination/value_loss")) if is_world_model_tag else 0
        contribution[term] = per_loop(np.abs(values[offset:offset + joint_steps]) * coefficient)
        rms_weight[term] = per_loop(running_rms_weight(values)[offset:offset + joint_steps])

    total = sum(contribution.values())
    loss_share = {term: value / total for term, value in contribution.items()}

    grad_norm = {term: scalars(acc, tag)[:joint_steps // STEPS_PER_LOOP] for term, tag in GRAD_TERMS.items()}
    grad_total = sum(grad_norm.values())
    grad_share = {term: value / grad_total for term, value in grad_norm.items()}

    return dict(contribution = contribution, loss_share = loss_share, grad_norm = grad_norm, grad_share = grad_share, rms_weight = rms_weight)


def geometric_mean(curves):
    length = min(len(curve) for curve in curves)
    return np.exp(np.mean([np.log(np.maximum(curve[:length], 1e-12)) for curve in curves], axis = 0))


def draw(ax, runs, key, terms):
    for term in terms:
        curves = [run[key][term] for run in runs]
        for curve in curves:
            ax.plot(np.arange(1, len(curve) + 1), curve, color = COLORS[term], lw = 0.7, alpha = 0.3)
        mean = geometric_mean(curves)
        ax.plot(np.arange(1, len(mean) + 1), mean, color = COLORS[term], lw = 1.9, label = term)

    ax.set_yscale("log")
    ax.set_xlim(1, 100)
    ax.set_xticks([1, 20, 40, 60, 80, 100])
    ax.grid(True, which = "major", color = "#e3e3e3", lw = 0.6)
    ax.grid(True, which = "minor", axis = "y", color = "#f1f1f1", lw = 0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    data = {arm: [load_run(name) for name in names] for arm, names in ARMS.items()}

    (OUT / "data").mkdir(parents = True, exist_ok = True)
    (OUT / "figures").mkdir(parents = True, exist_ok = True)

    with open(OUT / "data" / "loss_terms.csv", "w", newline = "") as file:
        writer = csv.writer(file)
        writer.writerow(["run", "loop", "view", "term", "value"])
        for arm, names in ARMS.items():
            for name, run in zip(names, data[arm]):
                for view, terms in run.items():
                    for term, curve in terms.items():
                        for loop, value in enumerate(curve, start = 1):
                            writer.writerow([name, loop, view, term, f"{value:.6g}"])

    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 3, figsize = (14.5, 7.4), sharex = True)

    columns = [
        ("loss_share", list(LOSS_TERMS), "Share of the summed loss"),
        ("grad_share", list(GRAD_TERMS), "Share of the transformer gradient"),
        ("rms_weight", list(LOSS_TERMS), "Multiplier under RMS normalization (1 / RMS)"),
    ]

    for row, (arm, runs) in enumerate(data.items()):
        for col, (key, terms, title) in enumerate(columns):
            ax = axes[row, col]
            draw(ax, runs, key, terms)
            if row == 0:
                ax.set_title(title, loc = "left", fontweight = "bold")
            if row == 1:
                ax.set_xlabel("loop")
            if col == 0:
                ax.set_ylabel(arm, fontweight = "bold")
            if col < 2:
                ax.set_ylim(top = 1.5)

    for col in range(3):
        handles, labels = axes[0, col].get_legend_handles_labels()
        axes[1, col].legend(handles, labels, loc = "upper center", bbox_to_anchor = (0.5, -0.17), ncol = 3, frameon = False, fontsize = 8.8)

    fig.suptitle(
        "Loss weight of each term in the joint loss, loops 1 to 100 (plain sum, no normalization)",
        x = 0.01, ha = "left", fontsize = 12.5, fontweight = "bold",
    )
    fig.text(
        0.01, 0.925,
        "Thick line: geometric mean of seeds 42 and 4. Thin lines: each seed. Loss values are the mean of the 10 joint steps of a loop. "
        "The gradient is measured one time for each loop.",
        fontsize = 8.8, color = "#555555",
    )
    fig.tight_layout(rect = (0, 0.0, 1, 0.925))
    fig.savefig(OUT / "figures" / "loss_weights.png", dpi = 170)

    # Summary numbers for the report text.
    for arm, runs in data.items():
        for name, run in zip(ARMS[arm], runs):
            loops = len(run["contribution"]["flow"])
            last = slice(max(loops - 10, 0), loops)
            line = [f"{name}: loops={loops}"]
            line += [f"{term}={run['contribution'][term][last].mean():.4g}" for term in LOSS_TERMS]
            line += [f"flow/bootstrap all={np.mean(run['contribution']['flow']) / np.mean(run['contribution']['bootstrap']):.1f}"]
            line += [f"share {term}={run['loss_share'][term][last].mean():.2e}" for term in LOSS_TERMS]
            line += [f"grad {term}={run['grad_share'][term][last].mean():.2e}" for term in GRAD_TERMS]
            line += [f"1/rms {term}={run['rms_weight'][term][last].mean():.3g}" for term in LOSS_TERMS]
            print("\n  ".join(line))


if __name__ == "__main__":
    sys.exit(main())
