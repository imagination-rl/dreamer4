"""Read the TensorBoard logs of the compared runs and write per-loop metrics and per-run summary numbers.

Output (in ../data):
  loop_metrics.csv   run, loop, tag, value          one row for each loop and tag (the mean of the steps of that loop)
  episodes.csv       run, loop, mean, std, p10, min  the 64 real episodes of each loop
  run_summary.csv    one row for each run: return, speed, and stability numbers over the first 100 loops

Usage: python extract.py            (8 worker processes)
"""

import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "data"
LOG_ROOT = REPO / "runs" / "gym_imagination"
MAX_LOOPS = 100
PRETRAIN_STEPS = 256

# name: (group, label)
RUNS = {
    "a2_grouped_loss_normalization_s4": ("grouped norm", "grouped norm s4"),
    "a2_grouped_loss_normalization_s42": ("grouped norm", "grouped norm s42"),
    "a2_loss_normalization_s4": ("separate norm", "separate norm s4"),
    "audit_a2_next_state_reward_target_local_s4": ("A2 local", "A2 local s4"),
    "audit_a2_next_state_reward_target_local_s42": ("A2 local", "A2 local s42"),
    "audit_a2_next_state_reward_target2": ("A2 H100", "A2 H100 s4"),
    "audit_a2_next_state_reward_target": ("A2 H100", "A2 H100 s42"),
    "a2_reduced_reward_transformer_gradient2": ("A2 reward grad 0.01 (H100)", "A2 reward grad 0.01 s4"),
    "a2_reduced_reward_transformer_gradient": ("A2 reward grad 0.01 (H100)", "A2 reward grad 0.01 s42"),
    "symlog_hl_gauss_value_head2": ("symlog HL-Gauss, PMPO (H100)", "symlog PMPO s4"),
    "symlog_hl_gauss_value_head": ("symlog HL-Gauss, PMPO (H100)", "symlog PMPO s42"),
    "closer_ppo_symlog_hl_gauss_value_head2": ("symlog HL-Gauss, closer PPO (H100)", "symlog PPO s4"),
    "closer_ppo_symlog_hl_gauss_value_head": ("symlog HL-Gauss, closer PPO (H100)", "symlog PPO s42"),
    "closer_ppo_next_state_reward_symlog_value_head2": ("symlog HL-Gauss, A2 (H100)", "symlog A2 s4"),
    "closer_ppo_next_state_reward_symlog_value_head": ("symlog HL-Gauss, A2 (H100)", "symlog A2 s42"),
    "offline_pretrained_a2_scaled40_no_rehearsal_s4": ("local pretrained", "scaled40 no-rehearsal s4"),
    "offline_pretrained_a2_scaled40_no_rehearsal_s42": ("local pretrained", "scaled40 no-rehearsal s42"),
    "offline_pretrained_a2_no_rehearsal_s4": ("local pretrained", "mix1m no-rehearsal s4"),
    "offline_pretrained_a2_no_rehearsal_s42": ("local pretrained", "mix1m no-rehearsal s42"),
    "offline_pretrained_a2_rehearsal_s4": ("local pretrained", "mix1m rehearsal s4"),
    "offline_pretrained_a2_rehearsal_s42": ("local pretrained", "mix1m rehearsal s42"),
    "offline_pretrained_a2_rehearsal2": ("local pretrained", "mix1m rehearsal (first) s4"),
    "offline_pretrained_a2_rehearsal": ("local pretrained", "mix1m rehearsal (first) s42"),
    "offline_pretrained_a2_entropy_rehearsal25_s4": ("local pretrained", "entropy rehearsal25 s4"),
    "offline_pretrained_a2_entropy_rehearsal25_s42": ("local pretrained", "entropy rehearsal25 s42"),
    "offline_pretrained_a2_bc_no_rehearsal_s4": ("local pretrained", "BC no-rehearsal s4"),
    "offline_pretrained_a2_bc_no_rehearsal_s42": ("local pretrained", "BC no-rehearsal s42"),
    "pretrain_rl_realistic_data_s4": ("local pretrained", "realistic s4"),
    "pretrain_rl_realistic_data_s42": ("local pretrained", "realistic s42"),
    "pretrain_rl_bangbang_scaled_s4": ("local pretrained", "bangbang s4"),
    "pretrain_rl_bangbang_scaled_s42": ("local pretrained", "bangbang s42"),
    "pretrain_rl_anchor_realistic_data_s4": ("local pretrained", "realistic + re-anchor s4"),
    "pretrain_rl_anchor_realistic_data_s42": ("local pretrained", "realistic + re-anchor s42"),
}

# Tags logged at each world-model step. A from-scratch run starts with PRETRAIN_STEPS steps before loop 0.
WORLD_MODEL_TAGS = [
    "extra/flow_loss_raw", "extra/shortcut_loss_raw", "extra/reward_loss_raw", "extra/terminal_loss_raw",
    "extra/dynamics_loss_normalized", "world_model/flow_loss", "world_model/shortcut_loss", "world_model/reward_loss",
    "world_model/loss", "world_model/reward_mae", "world_model/grad_norm", "world_model/trunk_grad_norm",
    "world_model/reward_head_grad_norm", "world_model/action_head_grad_norm",
]
# Tags logged at each imagination update (one for each loop with closer PPO, 3 for each loop with PMPO).
IMAGINATION_TAGS = [
    "imagination/dream_return", "imagination/action_std", "imagination/policy_loss", "imagination/value_loss",
    "imagination/loss", "imagination/grad_norm", "imagination/approx_kl_mean", "imagination/approx_kl_max",
    "imagination/prior_kl_mean", "imagination/policy_head_grad_norm", "imagination/value_head_grad_norm",
    "extra/policy_entropy", "extra/value_mse", "extra/value_excess_cross_entropy", "extra/value_target_entropy",
    "extra/advantage_abs_mean", "extra/importance_ratio_clip_frac", "extra/positive_advantage_frac",
    "critic/value_abs_error", "critic/value_mean", "critic/value_max", "critic/value_min", "critic/gae_return_mean",
    "critic/gae_return_max", "critic/gae_return_min", "critic/value_target_saturation_frac",
    "cleanrl_updated_ppo/approx_kl", "cleanrl_updated_ppo/old_approx_kl", "cleanrl_updated_ppo/clip_fraction",
    "cleanrl_updated_ppo/entropy", "cleanrl_updated_ppo/explained_variance", "cleanrl_updated_ppo/value_loss",
    "cleanrl_updated_ppo/policy_loss", "cleanrl_updated_ppo/grad_norm", "charts/learning_rate",
]
# Tags logged one time for each loop.
LOOP_TAGS = [
    "rollout/average_return", "audit/rollout/flipped_frac", "audit/rollout/x_velocity",
    "audit/rollout/action_saturation_frac", "audit/rollout/control_cost_per_step", "audit/rollout/reward_per_step",
    "audit/rollout/reward_per_step_first_half", "audit/rollout/reward_per_step_second_half", "audit/progress/elapsed_hours",
]
# Tags logged each 10 loops (the audit diagnostics). The row keeps the loop of the event.
AUDIT_TAGS = [
    "audit/closed_loop/state_norm_mae_h1", "audit/closed_loop/state_norm_mae_h8", "audit/closed_loop/state_norm_mae_h32",
    "audit/closed_loop/return_corr", "audit/closed_loop/reward_gap_per_step", "audit/closed_loop/dream_reward_per_step",
    "audit/closed_loop/true_reward_per_step", "audit/closed_loop/dream_above_true_frac",
    "audit/closed_loop/dream_flipped_at_end_frac", "audit/closed_loop/true_flipped_at_end_frac",
    "audit/closed_loop/prompt_at_current_speed_frac", "audit/world_model/one_step_norm_mae",
    "audit/world_model/one_step_x_velocity_mae", "audit/world_model/one_step_error_to_change_ratio",
    "audit/world_model/ranking_spearman", "audit/world_model/ranking_pairwise_sign_agreement",
    "audit/world_model/ranking_spread_ratio", "audit/world_model/reward_head_corr", "audit/world_model/reward_head_mae",
    "tokenizer/policy_recon_loss",
]


def values(acc, tag):
    return np.array([event.value for event in acc.Scalars(tag)])


def per_loop(series, loops, leading = 0):
    """Mean of the steps of each loop. The series has `leading` steps before loop 0 and the same count for each loop."""
    series = series[leading:]
    if loops == 0 or len(series) < loops:
        return None
    steps = len(series) // loops
    return series[:steps * loops].reshape(loops, steps).mean(axis = 1)


def rolling(series, window = 5):
    return np.convolve(series, np.ones(window) / window, mode = "valid")


def summary(returns, episodes_std):
    """Return, speed, and stability numbers of one return curve (one value for each loop)."""
    smooth = rolling(returns)

    def first_loop_at(threshold):
        hits = np.nonzero(smooth >= threshold)[0]
        # The window [i, i + 4] ends at loop i + 4.
        return int(hits[0]) + 4 if len(hits) else None

    running_max = np.maximum.accumulate(smooth)
    takeoff = first_loop_at(0.)
    after = returns[takeoff:] if takeoff is not None else np.array([])
    steps = np.diff(after) if len(after) > 1 else np.array([0.])
    # A residual from a 9-loop centered mean: loop-to-loop noise without the trend.
    if len(after) >= 9:
        trend = np.convolve(after, np.ones(9) / 9, mode = "valid")
        residual = after[4:-4] - trend
    else:
        residual = np.array([0.])
    return {
        "loops": len(returns),
        "return_last5": float(returns[-5:].mean()),
        "return_best5": float(smooth.max()),
        "return_mean_all": float(returns.mean()),
        "loop_at_0": takeoff,
        "loop_at_500": first_loop_at(500.),
        "loop_at_1000": first_loop_at(1000.),
        "max_drawdown5": float((running_max - smooth).max()),
        "last5_below_best5": float(smooth.max() - smooth[-1]),
        "noise_std_after_takeoff": float(residual.std()),
        "worst_single_loop_drop": float(-steps.min()),
        "loops_with_drop_over_200": int((steps < -200.).sum()),
        "slope_per_loop_after_takeoff": float(np.polyfit(np.arange(len(after)), after, 1)[0]) if len(after) > 5 else None,
        "episode_std_last10": float(np.mean(episodes_std[-10:])) if len(episodes_std) else None,
    }


def read_run(name):
    path = LOG_ROOT / name
    if not path.exists():
        return name, None
    acc = EventAccumulator(str(path), size_guidance = {"scalars": 0}).Reload()
    tags = set(acc.Tags()["scalars"])
    if "rollout/average_return" not in tags:
        return name, None

    returns = values(acc, "rollout/average_return")
    all_loops = len(returns)
    loops = min(all_loops, MAX_LOOPS)
    rows = []

    def add(tag, series):
        if series is None:
            return
        for loop, value in enumerate(series[:loops]):
            rows.append((name, loop, tag, float(value)))

    for tag in LOOP_TAGS:
        if tag in tags:
            add(tag, values(acc, tag))

    # A run from a pretrained checkpoint has no world-model steps before loop 0.
    world_model_count = len(values(acc, "world_model/loss"))
    leading = PRETRAIN_STEPS if (world_model_count - PRETRAIN_STEPS) % all_loops == 0 and world_model_count % all_loops != 0 else world_model_count % all_loops
    for tag in WORLD_MODEL_TAGS:
        if tag in tags:
            series = values(acc, tag)
            add(tag, per_loop(series, all_loops, leading if len(series) == world_model_count else len(series) % all_loops))

    for tag in IMAGINATION_TAGS:
        if tag in tags:
            series = values(acc, tag)
            add(tag, per_loop(series, all_loops, len(series) % all_loops))

    steps_per_loop = None
    for tag in AUDIT_TAGS:
        if tag in tags:
            events = acc.Scalars(tag)
            if tag.startswith("audit/"):
                # The event step is the env step. The first rollout tells the env steps of one loop.
                steps_per_loop = steps_per_loop or acc.Scalars("audit/progress/loop")[0].step
                for event in events:
                    loop = int(round(event.step / steps_per_loop)) - 1
                    if loop < loops:
                        rows.append((name, loop, tag, float(event.value)))
            else:
                for event in events:
                    if event.step < loops:
                        rows.append((name, int(event.step), tag, float(event.value)))

    episode_rows = []
    if "charts/episodic_return" in tags:
        episodes = values(acc, "charts/episodic_return")
        per = len(episodes) // all_loops
        if per > 0:
            grid = episodes[:per * all_loops].reshape(all_loops, per)[:loops]
            for loop, row in enumerate(grid):
                episode_rows.append((name, loop, float(row.mean()), float(row.std()), float(np.percentile(row, 10)), float(row.min())))

    stats = summary(returns[:loops], [row[3] for row in episode_rows])
    stats["world_model_steps_per_loop"] = (world_model_count - leading) // all_loops
    stats["leading_world_model_steps"] = leading
    if "audit/rollout/flipped_frac" in tags:
        flipped = values(acc, "audit/rollout/flipped_frac")[:loops]
        stats["flipped_frac_max"] = float(flipped.max())
        stats["flipped_frac_last10"] = float(flipped[-10:].mean())
    if "audit/progress/elapsed_hours" in tags:
        stats["hours"] = float(values(acc, "audit/progress/elapsed_hours")[:loops][-1])
    return name, (rows, episode_rows, stats)


def main():
    OUT.mkdir(parents = True, exist_ok = True)
    with ProcessPoolExecutor(8) as pool:
        results = dict(pool.map(read_run, RUNS))

    summaries = []
    with open(OUT / "loop_metrics.csv", "w", newline = "") as metrics_file, open(OUT / "episodes.csv", "w", newline = "") as episodes_file:
        metrics, episodes = csv.writer(metrics_file), csv.writer(episodes_file)
        metrics.writerow(["run", "loop", "tag", "value"])
        episodes.writerow(["run", "loop", "mean", "std", "p10", "min"])
        for name, (group, label) in RUNS.items():
            if results[name] is None:
                print("missing:", name, file = sys.stderr)
                continue
            rows, episode_rows, stats = results[name]
            metrics.writerows(rows)
            episodes.writerows(episode_rows)
            summaries.append({"run": name, "group": group, "label": label, **stats})

    keys = list(dict.fromkeys(key for row in summaries for key in row))
    with open(OUT / "run_summary.csv", "w", newline = "") as file:
        writer = csv.DictWriter(file, fieldnames = keys)
        writer.writeheader()
        writer.writerows(summaries)

    for row in summaries:
        print(
            f"{row['label'][:30]:30s} loops={row['loops']:3d} last5={row['return_last5']:6.0f} best5={row['return_best5']:6.0f} "
            f"at0={row['loop_at_0']} at500={row['loop_at_500']} at1000={row['loop_at_1000']} drawdown={row['max_drawdown5']:5.0f} "
            f"noise={row['noise_std_after_takeoff']:4.0f} drops>200={row['loops_with_drop_over_200']} worst_drop={row['worst_single_loop_drop']:5.0f} "
            f"slope={row['slope_per_loop_after_takeoff']} flip_max={row.get('flipped_frac_max')} wm_steps={row['world_model_steps_per_loop']} lead={row['leading_world_model_steps']}"
        )


if __name__ == "__main__":
    main()
