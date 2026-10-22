# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fire",
#     "pyyaml",
# ]
# ///

from __future__ import annotations

import json
import os
import re
import sys
import traceback
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Mapping, Sequence

import fire
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
GYM_LOG_ROOT = Path("runs/gym_imagination")

# Closer CleanRL PPO on dreams (cleanrl_ppo_closer / 2, runs/audit_9-24-26/REPORT.md section 9).
# The ablations build on these settings, except the ablations that replace the objective.
CLEANRL_PPO_CLOSER = {
    "objective": "ppo",
    "cleanrl_updated_ppo": True,
    "cleanrl_ppo_closer": True,
    "old_reward_shifting": True,
    "align_generated_dream_transitions": True,
    "agent_learning_rate": 0.0003,
}

# Audit ablations from runs/audit_9-24-26/REPORT.md (section 6).
# Each arm is audit_control plus one change. All audit runs set audit_diagnostics,
# so the world-model quality logs are comparable. num_loops keeps the config value,
# so audit_control uses the learning-rate schedule of cleanrl_ppo_closer.
AUDIT_DIAGNOSTICS = {**CLEANRL_PPO_CLOSER, "audit_diagnostics": True}

# The reward target of audit_a2_next_state_reward_target / 2.
# align_generated_dream_transitions requires old_reward_shifting, so it is also false.
NEXT_STATE_REWARD_TARGET = {
    "old_reward_shifting": False,
    "align_generated_dream_transitions": False,
}

# The value head of closer_ppo_symlog_hl_gauss_value_head / 2, with its cross-entropy value loss.
SYMLOG_HL_GAUSS_VALUE_HEAD = {
    "use_symlog_hl_gauss_value_head": True,
    "value_range": [-10.0, 10.0],
    "cleanrl_ppo_closer_hl_gauss_value_loss": True,
}

# Offline data collection and pretraining (notes/pretraining_impl_plan.md, section 13).
# Run the collect and pretrain entries once, and alone: two runs must not write the same folder.
# The RL arms need the pretrained checkpoint of the pretrain entry.
OFFLINE_PRETRAIN_NAME = "a2_next_state_reward"

OFFLINE_COLLECT = {
    "offline_pretraining": True,
    "offline_stages": ["collect"],
}

# Replays of earlier online runs for the past_runs mode (report/9-30-26-pretrain-rootcause, report/9-30-26-past-runs).
# Only the offline-pretrained RL arms (entries 17-20 and 25-32). They cover good runs (no_rehearsal, rehearsal2,
# no_rehearsal_s42), slow runs (rehearsal_s4, no_rehearsal_s4), and runs that collapsed or stayed near random
# (rehearsal_s42, entropy_rehearsal25_s4, bc_no_rehearsal_s42 / s4). Each replay holds all loops of its run.
# The local repro runs (audit_a2_next_state_reward_target_local_s42 / s4) are not in this list: their replays are the
# held-out online states of the probes in report/9-30-26-repro and report/9-30-26-pretrain-rootcause.
PAST_RUN_REPLAYS = [f"runs/gym_imagination/{name}/replay_buffer" for name in (
    "offline_pretrained_a2_rehearsal", "offline_pretrained_a2_rehearsal2",
    "offline_pretrained_a2_no_rehearsal", "offline_pretrained_a2_no_rehearsal2",
    "offline_pretrained_a2_rehearsal_s42", "offline_pretrained_a2_rehearsal_s4",
    "offline_pretrained_a2_no_rehearsal_s42", "offline_pretrained_a2_no_rehearsal_s4",
    "offline_pretrained_a2_entropy_rehearsal25_s42", "offline_pretrained_a2_entropy_rehearsal25_s4",
    "offline_pretrained_a2_bc_no_rehearsal_s42", "offline_pretrained_a2_bc_no_rehearsal_s4",
)]

# Entries 33-43 use this dataset. It is archived in
# runs/offline_data/archive/HalfCheetah-v4/halfcheetah_v4_mix_past_runs_scaled_expert_5p5m (2026-09-30): only entry 35
# ran (stopped at step 12,000), and entry 34 ran out of GPU memory. Entries 44-54 use OFFLINE_DATA_SCALED40.
# The offline data with past runs and scaled expert actions: 5500 episodes. Less expert data and two new modes, so
# that action size and speed are less confounded (report/9-30-26-pretrain-rootcause, report/9-30-26-past-runs).
#   expert 750, expert_noise 1500 (the four noise stds of train/config.yaml), random 1000,
#   past_runs 1750: half poor or collapsed (return -150 to 500), half good (500 to 1500). The lower edge keeps
#     collapsed episodes of about 0 and leaves out the random first policy (about -290), which random already gives.
#   expert_scaled 500: the expert with its actions times 0.4, 0.5, 0.6, or 0.7 (moderate actions at 1.7 to 5.7 m/s).
# The fractions are the episode counts over 5500. The default mix in train/config.yaml (0.3 / 0.4 / 0.3) stays, so the
# entries that use halfcheetah_v4_mix_1m still reproduce. An earlier version is archived in
# runs/offline_data/archive/HalfCheetah-v4/halfcheetah_v4_mix_past_runs_1m (uniform past runs from 26 runs).
OFFLINE_DATA_EPISODES = {"expert": 750, "expert_noise": 1500, "random": 1000, "past_runs": 1750, "expert_scaled": 500}
OFFLINE_DATA_PAST_RUNS = {
    "offline_dataset_name": "halfcheetah_v4_mix_past_runs_scaled_expert_5p5m",
    "offline_data_steps": 1000 * sum(OFFLINE_DATA_EPISODES.values()),
    "offline_data_mix": {mode: count / sum(OFFLINE_DATA_EPISODES.values()) for mode, count in OFFLINE_DATA_EPISODES.items()},
    "offline_past_run_replays": PAST_RUN_REPLAYS,
    "offline_past_run_return_buckets": [[-150.0, 500.0], [500.0, 1500.0]],
    "offline_expert_action_scales": [0.4, 0.5, 0.6, 0.7],
}

# The offline data with 40 % scaled expert episodes (entries 44-54): 5500 episodes.
#   expert_scaled 2200 (40 %): the expert with its actions times a scale that changes every 100 steps (uniform in
#     [0.3, 0.8]), plus Gaussian noise of std 0.1 (half the action range). The review of OFFLINE_DATA_PAST_RUNS found
#     that one fixed scale for each episode gives 4 steady bang-bang gaits: larger scale => faster, with no change of
#     the scale from a given state. Changes of the scale in an episode give accelerations and slowdowns from the same
#     states, and the noise breaks the bang-bang shape. Scale 0.3 gives about 0.7 m/s, 0.8 gives about 7 m/s.
#   The other modes keep their ratios of OFFLINE_DATA_PAST_RUNS: expert 500, expert_noise 1000 (the four noise stds of
#   train/config.yaml), random 650, past_runs 1150 (half with return -150 to 500, half with 500 to 1500).
OFFLINE_DATA_SCALED40_EPISODES = {"expert": 500, "expert_noise": 1000, "random": 650, "past_runs": 1150, "expert_scaled": 2200}
OFFLINE_DATA_SCALED40 = {
    "offline_dataset_name": "halfcheetah_v4_mix_scaled40_5p5m",
    "offline_data_steps": 1000 * sum(OFFLINE_DATA_SCALED40_EPISODES.values()),
    "offline_data_mix": {
        mode: count / sum(OFFLINE_DATA_SCALED40_EPISODES.values()) for mode, count in OFFLINE_DATA_SCALED40_EPISODES.items()
    },
    "offline_past_run_replays": PAST_RUN_REPLAYS,
    "offline_past_run_return_buckets": [[-150.0, 500.0], [500.0, 1500.0]],
    "offline_expert_action_scales": [0.3, 0.8],
    "offline_expert_scale_change_steps": 100,
    "offline_expert_scaled_noise_std": 0.1,
}

# The pretrain signature includes old_reward_shifting, so the A2 arms need their own pretrained checkpoint.
OFFLINE_PRETRAIN_NEXT_STATE_REWARD = {
    **AUDIT_DIAGNOSTICS,
    **NEXT_STATE_REWARD_TARGET,
    "offline_pretraining": True,
    "offline_stages": ["pretrain"],
    "offline_pretrain_name": OFFLINE_PRETRAIN_NAME,
}

OFFLINE_PRETRAINED_NEXT_STATE_REWARD = {
    "offline_pretraining": True,
    "offline_stages": ["rl"],
    # pretrained.pt is a symlink to pretrained_step_12000.pt: the pretrain run stopped at step 16,448, and the
    # validation flow loss was flat after step 12,000.
    "offline_pretrained_checkpoint": f"runs/offline_pretrain/{OFFLINE_PRETRAIN_NAME}/pretrained.pt",
    # Half of each world-model batch comes from the offline replay.
    "offline_rehearsal_fraction": 0.5,
}

# Local 16 GB GPU settings (report/9-29-26-pretrain): world-model chunks of 64 windows, bf16 autocast for the
# world-model loss, and einsum attention in the attention pools. They change the rounding against the default path.
# The pretrain signature does not include them.
LOCAL_16GB_GPU = {
    "world_model_micro_batch_size": 64,
    "world_model_autocast_bf16": True,
    "attn_pool_einsum_attend": True,
}

# Local RL runs with the settings of the local A2 runs in runs/local_9-28-26/attnpool_on_s42 / s4:
# 100 of the 500 scheduled loops, no checkpoints. The envs use the config default
# (async, parallel_envs worker processes), which gives the same data as the sync envs of those runs.
# Run two of these in parallel (--max_workers=2): two runs at micro-batch 16 take 19.3 s per loop each
# (9.65 s per loop in total, 6.3 GB), one run at micro-batch 64 takes 11.1 s per loop (6.7 GB).
# Two runs at micro-batch 32 need about 9 GB.
LOCAL_RL_100_LOOPS = {
    **LOCAL_16GB_GPU,
    "world_model_micro_batch_size": 32,
    "stop_after_loops": 100,
    # A checkpoint copies the replay as a dense 9.9 GB folder (the live replay uses 3.9 GB on disk).
    "checkpoint_every": 0,
}

# The 9-29-26 queue (scripts/queue_9-29-26.sh) runs two lanes, one for each seed. Each lane runs its pretrain
# entry, then the four RL entries that load that pretrain, then the audit reproduction. Two runs share the GPU at all times.
# Two pretrain runs with different offline_pretrain_name values write different folders.
QUEUE_SEEDS = (42, 4)
QUEUE_PRETRAIN_NAME = "a2_next_state_reward_s{seed}"
# 100 loops as the other local runs, with checkpoints at loops 50 and 100, as entries 17 and 18.
# Each checkpoint copies the replay as a dense 9.9 GB folder on /data.
LOCAL_RL_QUEUE = {**LOCAL_RL_100_LOOPS, "checkpoint_every": 50}


def queue_ablations():
    """Entries 21-32, seeds 42 and 4: audit reproduction, pretrain, and four RL arms from that pretrain."""

    audit, pretrain, rehearsal, no_rehearsal, entropy, behavior_cloning = [], [], [], [], [], []

    for seed in QUEUE_SEEDS:
        pretrain_name = QUEUE_PRETRAIN_NAME.format(seed = seed)
        audit_name = f"audit_a2_next_state_reward_target_local_s{seed}"
        # The step 12,000 checkpoint, as for entries 17-20. The world model trains online in all RL arms.
        pretrained_rl = {
            "seed": seed,
            **AUDIT_DIAGNOSTICS,
            **NEXT_STATE_REWARD_TARGET,
            **OFFLINE_PRETRAINED_NEXT_STATE_REWARD,
            "offline_pretrained_checkpoint": f"runs/offline_pretrain/{pretrain_name}/pretrained_step_12000.pt",
        }

        audit.append({
            "run_name": audit_name,
            "run_details": "prev=[audit_a2_next_state_reward_target], change=[local reproduction of the H100 run: 100 of 500 loops, local 16 GB GPU settings]",
            "seed": seed,
            **AUDIT_DIAGNOSTICS,
            **NEXT_STATE_REWARD_TARGET,
            **LOCAL_RL_QUEUE,
        })
        pretrain.append({
            "run_name": f"offline_pretrain_a2_next_state_reward_s{seed}",
            "run_details": "prev=[offline_collect_halfcheetah_v4_mix_1m], change=[pretrain the tokenizer and the A2 world model on the offline data only]",
            "seed": seed,
            **OFFLINE_PRETRAIN_NEXT_STATE_REWARD,
            **LOCAL_16GB_GPU,
            # Two pretrain runs share the GPU: micro-batch 32 needs 3.6 GB, micro-batch 64 needs 7 GB.
            # The batch stays at world_model_batch_size windows; only the chunk size changes.
            "world_model_micro_batch_size": 32,
            "offline_pretrain_name": pretrain_name,
        })
        # The settings of entries 17 and 18 (0.5 rehearsal, checkpoints at loops 50 and 100).
        rehearsal.append({
            "run_name": f"offline_pretrained_a2_rehearsal_s{seed}",
            "run_details": f"prev=[{audit_name}], change=[start from pretrain {pretrain_name}; 0.5 offline rehearsal in world-model batches]",
            **pretrained_rl,
            **LOCAL_RL_QUEUE,
        })
        # The settings of entries 19 and 20 (no rehearsal, no checkpoints).
        no_rehearsal.append({
            "run_name": f"offline_pretrained_a2_no_rehearsal_s{seed}",
            "run_details": f"prev=[offline_pretrained_a2_rehearsal_s{seed}], change=[no offline rehearsal in world-model batches]",
            **pretrained_rl,
            **LOCAL_RL_100_LOOPS,
            "offline_rehearsal_fraction": 0.0,
        })
        # The suspected cause of the no-rehearsal collapse: the policy entropy fell early, and the near-deterministic
        # policy did not recover from the flipped gait. This arm adds an entropy bonus, with 0.25 rehearsal.
        entropy.append({
            "run_name": f"offline_pretrained_a2_entropy_rehearsal25_s{seed}",
            "run_details": f"prev=[offline_pretrained_a2_rehearsal_s{seed}], change=[PPO entropy bonus 0.01; 0.25 offline rehearsal]",
            **pretrained_rl,
            **LOCAL_RL_100_LOOPS,
            "offline_rehearsal_fraction": 0.25,
            # The entropy weight of the other objectives in train/config.yaml (policy_entropy_weight).
            "cleanrl_ppo_entropy_weight": 0.01,
        })
        # The paper's phase-2 technique in the RL loop: each world-model update also trains the policy head by BC on
        # the replay actions, with the world-model optimizer, so the policy head follows the change of its input
        # features. The flag does not support rehearsal, so the control is the no-rehearsal arm (one change).
        # Checkpoints at loops 50 and 100 permit a later check of the policy drift.
        behavior_cloning.append({
            "run_name": f"offline_pretrained_a2_bc_no_rehearsal_s{seed}",
            "run_details": f"prev=[offline_pretrained_a2_no_rehearsal_s{seed}], change=[BC of the policy head on the replay actions in each world-model update (paper phase 2 in the RL loop)]",
            **pretrained_rl,
            **LOCAL_RL_QUEUE,
            "offline_rehearsal_fraction": 0.0,
            "behavior_cloning_in_world_model_updates": True,
        })

    return [*audit, *pretrain, *rehearsal, *no_rehearsal, *entropy, *behavior_cloning]


PAST_RUNS_PRETRAIN_NAME = "a2_past_runs_scaled_expert_s{seed}"
REDUCED_REWARD_GRADIENT_PRETRAIN_NAME = "a2_past_runs_scaled_expert_reward_grad_s{seed}"

# The setting of a2_reduced_reward_transformer_gradient / 2: the reward-loss gradients that enter the shared
# transformer are scaled by 0.01. The reward heads still train at full rate.
REDUCED_REWARD_GRADIENT = {
    "scale_reward_gradient_into_transformer": True,
    "reward_gradient_into_transformer_scale": 0.01,
}


def past_runs_queue_ablations():
    """Entries 34-39, seeds 42 and 4: pretrain on the data of entry 33, then RL with and without rehearsal.

    Each entry differs from its entry in queue_ablations() (23-24, 25-26, 27-28) only by the offline data.
    """

    pretrain, rehearsal, no_rehearsal = [], [], []

    for seed in QUEUE_SEEDS:
        pretrain_name = PAST_RUNS_PRETRAIN_NAME.format(seed = seed)
        # The step 12,000 checkpoint, as for entries 25-28. The world model trains online in both RL arms.
        pretrained_rl = {
            "seed": seed,
            **AUDIT_DIAGNOSTICS,
            **NEXT_STATE_REWARD_TARGET,
            **OFFLINE_PRETRAINED_NEXT_STATE_REWARD,
            "offline_pretrained_checkpoint": f"runs/offline_pretrain/{pretrain_name}/pretrained_step_12000.pt",
            # The rl stage does not read the dataset (rehearsal reads the latent replay of the pretrain folder).
            # The name only records the data of the pretrain in the run settings.
            "offline_dataset_name": OFFLINE_DATA_PAST_RUNS["offline_dataset_name"],
        }

        pretrain.append({
            "run_name": f"offline_pretrain_a2_past_runs_scaled_expert_s{seed}",
            "run_details": f"prev=[offline_pretrain_a2_next_state_reward_s{seed}], change=[offline data {OFFLINE_DATA_PAST_RUNS['offline_dataset_name']}]",
            "seed": seed,
            **OFFLINE_PRETRAIN_NEXT_STATE_REWARD,
            **LOCAL_16GB_GPU,
            # Two pretrain runs share the GPU, as entries 23 and 24.
            "world_model_micro_batch_size": 32,
            "offline_dataset_name": OFFLINE_DATA_PAST_RUNS["offline_dataset_name"],
            "offline_pretrain_name": pretrain_name,
        })
        # The settings of entries 25 and 26 (0.5 rehearsal, checkpoints at loops 50 and 100).
        rehearsal.append({
            "run_name": f"offline_pretrained_a2_past_runs_scaled_expert_rehearsal_s{seed}",
            "run_details": f"prev=[offline_pretrained_a2_rehearsal_s{seed}], change=[start from pretrain {pretrain_name}]",
            **pretrained_rl,
            **LOCAL_RL_QUEUE,
        })
        # The settings of entries 27 and 28 (no rehearsal, no checkpoints).
        no_rehearsal.append({
            "run_name": f"offline_pretrained_a2_past_runs_scaled_expert_no_rehearsal_s{seed}",
            "run_details": f"prev=[offline_pretrained_a2_no_rehearsal_s{seed}], change=[start from pretrain {pretrain_name}]",
            **pretrained_rl,
            **LOCAL_RL_100_LOOPS,
            "offline_rehearsal_fraction": 0.0,
        })

    return [*pretrain, *rehearsal, *no_rehearsal]


def reduced_reward_gradient_queue_ablations():
    """Entries 40-43, seeds 42 and 4: entries 34-35 and 38-39 with the reward gradient into the transformer times 0.01.

    The pretrain signature includes the flag, so an RL run with the flag cannot load the pretrain of entries 34-35.
    The flag is on in both stages: pretrain (40-41) and RL without rehearsal (42-43).
    """

    pretrain, no_rehearsal = [], []

    for seed in QUEUE_SEEDS:
        pretrain_name = REDUCED_REWARD_GRADIENT_PRETRAIN_NAME.format(seed = seed)

        pretrain.append({
            "run_name": f"offline_pretrain_a2_past_runs_scaled_expert_reward_grad_s{seed}",
            "run_details": f"prev=[offline_pretrain_a2_past_runs_scaled_expert_s{seed}], change=[reward-loss gradients into the shared transformer times 0.01]",
            "seed": seed,
            **OFFLINE_PRETRAIN_NEXT_STATE_REWARD,
            **LOCAL_16GB_GPU,
            # Two pretrain runs share the GPU, as entries 34 and 35.
            "world_model_micro_batch_size": 32,
            "offline_dataset_name": OFFLINE_DATA_PAST_RUNS["offline_dataset_name"],
            "offline_pretrain_name": pretrain_name,
            **REDUCED_REWARD_GRADIENT,
        })
        # The settings of entries 38 and 39 (no rehearsal, no checkpoints).
        no_rehearsal.append({
            "run_name": f"offline_pretrained_a2_past_runs_scaled_expert_reward_grad_no_rehearsal_s{seed}",
            "run_details": f"prev=[offline_pretrained_a2_past_runs_scaled_expert_no_rehearsal_s{seed}], change=[start from pretrain {pretrain_name}; reward-loss gradients into the shared transformer times 0.01]",
            "seed": seed,
            **AUDIT_DIAGNOSTICS,
            **NEXT_STATE_REWARD_TARGET,
            **OFFLINE_PRETRAINED_NEXT_STATE_REWARD,
            "offline_pretrained_checkpoint": f"runs/offline_pretrain/{pretrain_name}/pretrained_step_12000.pt",
            # The rl stage does not read the dataset. The name only records the data of the pretrain.
            "offline_dataset_name": OFFLINE_DATA_PAST_RUNS["offline_dataset_name"],
            **LOCAL_RL_100_LOOPS,
            "offline_rehearsal_fraction": 0.0,
            **REDUCED_REWARD_GRADIENT,
        })

    return [*pretrain, *no_rehearsal]


def scaled40_queue_ablations():
    """Entries 45-54, seeds 42 and 4, on the data of entry 44 (OFFLINE_DATA_SCALED40).

    45-46 pretrain, 47-48 RL with 0.5 rehearsal, 49-50 RL without rehearsal, 51-52 pretrain with the reward gradient
    into the transformer times 0.01, 53-54 RL without rehearsal from 51-52. The same set as entries 34-43; only the
    offline data differs. All RL runs load the step 12,000 checkpoint, as entries 25-28. The pretrains stop at step
    12,000 (pretrained.pt) instead of 20,000: the world-model learning rate is constant, so the step 12,000 weights
    do not depend on the step count, and each pretrain takes about 70 minutes less.
    """

    data_name = OFFLINE_DATA_SCALED40["offline_dataset_name"]
    pretrain, rehearsal, no_rehearsal, reward_pretrain, reward_rl = [], [], [], [], []

    for seed in QUEUE_SEEDS:
        names = {"": f"a2_scaled40_s{seed}", "_reward_grad": f"a2_scaled40_reward_grad_s{seed}"}

        def pretrained_rl(tag):
            return {
                "seed": seed,
                **AUDIT_DIAGNOSTICS,
                **NEXT_STATE_REWARD_TARGET,
                **OFFLINE_PRETRAINED_NEXT_STATE_REWARD,
                "offline_pretrained_checkpoint": f"runs/offline_pretrain/{names[tag]}/pretrained.pt",
                # The rl stage does not read the dataset. The name only records the data of the pretrain.
                "offline_dataset_name": data_name,
            }

        def pretrain_entry(tag, extra):
            return {
                "run_name": f"offline_pretrain_a2_scaled40{tag}_s{seed}",
                "run_details": f"prev=[offline_pretrain_a2_past_runs_scaled_expert{tag}_s{seed}], change=[offline data {data_name}]",
                "seed": seed,
                **OFFLINE_PRETRAIN_NEXT_STATE_REWARD,
                **LOCAL_16GB_GPU,
                # Two runs share the GPU. The validation runs in chunks of this size too.
                "world_model_micro_batch_size": 32,
                "offline_dataset_name": data_name,
                "offline_pretrain_name": names[tag],
                "offline_world_model_steps": 12000,
                **extra,
            }

        pretrain.append(pretrain_entry("", {}))
        # The settings of entries 25 and 26 (0.5 rehearsal, checkpoints at loops 50 and 100).
        rehearsal.append({
            "run_name": f"offline_pretrained_a2_scaled40_rehearsal_s{seed}",
            "run_details": f"prev=[offline_pretrained_a2_rehearsal_s{seed}], change=[start from pretrain {names['']} (data {data_name})]",
            **pretrained_rl(""),
            **LOCAL_RL_QUEUE,
        })
        # The settings of entries 27 and 28 (no rehearsal, no checkpoints).
        no_rehearsal.append({
            "run_name": f"offline_pretrained_a2_scaled40_no_rehearsal_s{seed}",
            "run_details": f"prev=[offline_pretrained_a2_no_rehearsal_s{seed}], change=[start from pretrain {names['']} (data {data_name})]",
            **pretrained_rl(""),
            **LOCAL_RL_100_LOOPS,
            "offline_rehearsal_fraction": 0.0,
        })
        # The pretrain signature includes scale_reward_gradient_into_transformer, so these RL runs need their own pretrain.
        reward_pretrain.append(pretrain_entry("_reward_grad", REDUCED_REWARD_GRADIENT))
        reward_rl.append({
            "run_name": f"offline_pretrained_a2_scaled40_reward_grad_no_rehearsal_s{seed}",
            "run_details": f"prev=[offline_pretrained_a2_scaled40_no_rehearsal_s{seed}], change=[start from pretrain {names['_reward_grad']}; reward-loss gradients into the shared transformer times 0.01]",
            **pretrained_rl("_reward_grad"),
            **LOCAL_RL_100_LOOPS,
            "offline_rehearsal_fraction": 0.0,
            **REDUCED_REWARD_GRADIENT,
        })

    return [*pretrain, *rehearsal, *no_rehearsal, *reward_pretrain, *reward_rl]


# The offline data without expert episodes (entries 55-61, report/9-30-26-pretrain-analysis section 3): 7500 episodes.
#   expert 0 and expert_scaled 0. The expert modes cover 10-12 m/s (expert) or give moderate actions at speed with
#     no flips (expert_scaled). The local online policies run at 0-3 m/s and flip.
#   expert_noise 1500 (20 %) with the noise stds 0.2, 0.4, and 0.8 (no 0.1: those episodes are expert data at 10.6 m/s).
#   random 1125 (15 %).
#   past_runs 4875 (65 %) from all local runs that have a replay: the 12 pretrained arms of PAST_RUN_REPLAYS (good,
#     slow, collapsed, and BC runs), the 4 scaled40 arms (entries 47-50, two of them collapsed into the flipped gait),
#     the 12 online-only runs of runs/local_9-28-26 (attention-pool, AttnRes, and depth-5 variants), and the 3 short
#     runs of runs/local_9-29-26 (early loops only). The local repro runs (entries 21-22) stay out: their replays hold
#     the online states of the probes. The smoke run stays out.
#   Four equal return buckets. [-500, 0) holds the first random policies and the flipped-gait episodes, [0, 500)
#     the slow and partly flipped episodes, [500, 1500) and [1500, 4000) the running gaits at 1-2.7 m/s.
#     In each bucket, FLIP_EPISODE_FRACTION of the episodes come from the episodes in which the cheetah flips.
#     report/9-30-26-pretrain-analysis/scripts/realistic_past_runs.py gives the diversity of the selection.
OFFLINE_DATA_REALISTIC_REPLAYS = [
    *PAST_RUN_REPLAYS,
    *[f"runs/gym_imagination/offline_pretrained_a2_scaled40_{name}/replay_buffer" for name in (
        "rehearsal_s42", "rehearsal_s4", "no_rehearsal_s42", "no_rehearsal_s4",
    )],
    *[f"runs/local_9-28-26/{name}/replay_buffer" for name in (
        "attnpool_on_s42", "attnpool_on_s4", "attnpool_on_s1", "attnpool_off_s42", "attnpool_off_s4", "attnpool_off_s1",
        "attnpool_off_depth5_s42", "attnpool_off_depth5_s4", "attnres_full_s42", "attnres_full_s4",
        "attnres_block2_s42", "attnres_block2_s4",
    )],
    *[f"runs/local_9-29-26/{name}/replay_buffer" for name in ("long_depth5_s1", "long_pools_s1", "long_res_full_s1")],
]
# The share of each return bucket from episodes with an upright-to-flipped transition. It gives 1.5 times the flipped
# steps of an unweighted selection from the same buckets (report/9-30-26-pretrain-analysis/scripts/realistic_past_runs.py).
FLIP_EPISODE_FRACTION = 0.45
OFFLINE_DATA_REALISTIC_EPISODES = {"expert_noise": 1500, "random": 1125, "past_runs": 4875}
OFFLINE_DATA_REALISTIC = {
    "offline_dataset_name": "halfcheetah_v4_mix_realistic_7p5m",
    "offline_data_steps": 1000 * sum(OFFLINE_DATA_REALISTIC_EPISODES.values()),
    "offline_data_mix": {
        mode: count / sum(OFFLINE_DATA_REALISTIC_EPISODES.values()) for mode, count in OFFLINE_DATA_REALISTIC_EPISODES.items()
    },
    "offline_expert_noise_stds": [0.2, 0.4, 0.8],
    "offline_past_run_replays": OFFLINE_DATA_REALISTIC_REPLAYS,
    "offline_past_run_return_buckets": [[-500.0, 0.0], [0.0, 500.0], [500.0, 1500.0], [1500.0, 4000.0]],
    "offline_past_run_flip_episode_fraction": FLIP_EPISODE_FRACTION,
}

# Two runs share the 16 GB GPU. A run that starts on a free GPU caches up to 14.7 GB (entry 49), and the second run then
# runs out of memory (entry 50). Two runs together need about 13.5 GB (entries 47 and 48), so each run gets at most 48 %.
SHARED_GPU_MEMORY = {"cuda_memory_fraction": 0.48}

# The head re-anchor of reports/9-28-26 (item D1): after each world-model phase, fit the policy and value heads to
# their outputs before that phase, on real windows of the newest rollout (200 Adam steps, lr 3e-4).
HEAD_REANCHOR = {"reanchor_heads_after_world_model_updates": True}


def realistic_data_queue_ablations():
    """Entries 56-61, seeds 42 and 4, on the data of entry 55 (OFFLINE_DATA_REALISTIC).

    56-57 pretrain (12,000 steps, as entries 45-46), 58-59 RL without rehearsal (the settings of entries 49-50),
    60-61 the same RL runs with the head re-anchor after each world-model phase.
    """

    data_name = OFFLINE_DATA_REALISTIC["offline_dataset_name"]
    pretrain, rl, anchor = [], [], []

    for seed in QUEUE_SEEDS:
        pretrain_name = f"realistic_data_s{seed}"
        pretrained_rl = {
            "seed": seed,
            **AUDIT_DIAGNOSTICS,
            **NEXT_STATE_REWARD_TARGET,
            **OFFLINE_PRETRAINED_NEXT_STATE_REWARD,
            "offline_pretrained_checkpoint": f"runs/offline_pretrain/{pretrain_name}/pretrained.pt",
            # The rl stage does not read the dataset. The name only records the data of the pretrain.
            "offline_dataset_name": data_name,
            **LOCAL_RL_100_LOOPS,
            "offline_rehearsal_fraction": 0.0,
            **SHARED_GPU_MEMORY,
        }
        pretrain.append({
            "run_name": f"pretrain_realistic_data_s{seed}",
            "run_details": f"prev=[offline_pretrain_a2_scaled40_s{seed}], change=[offline data {data_name}: no expert or scaled-expert episodes, 70 % past runs of all local runs]",
            "seed": seed,
            **OFFLINE_PRETRAIN_NEXT_STATE_REWARD,
            **LOCAL_16GB_GPU,
            # Two runs share the GPU. The validation runs in chunks of this size too.
            "world_model_micro_batch_size": 32,
            "offline_dataset_name": data_name,
            "offline_pretrain_name": pretrain_name,
            "offline_world_model_steps": 12000,
            **SHARED_GPU_MEMORY,
        })
        rl.append({
            "run_name": f"pretrain_rl_realistic_data_s{seed}",
            "run_details": f"prev=[offline_pretrained_a2_scaled40_no_rehearsal_s{seed}], change=[start from pretrain {pretrain_name} (data {data_name})]",
            **pretrained_rl,
        })
        anchor.append({
            "run_name": f"pretrain_rl_anchor_realistic_data_s{seed}",
            "run_details": f"prev=[pretrain_rl_realistic_data_s{seed}], change=[re-anchor the policy and value heads after each world-model phase (reports/9-28-26 D1)]",
            **pretrained_rl,
            **HEAD_REANCHOR,
        })

    return [*pretrain, *rl, *anchor]


# The offline data with bang-bang and scaled expert episodes (entries 62-68, report/10-1-26-realistic-data-analysis
# section 7): 7500 episodes. The realistic data (entry 55) had saturated actions only at speed (expert_noise above 3 m/s),
# and its pretrains rated bang-bang actions at or above the online gait.
#   random 750 (10 %).
#   bang_bang 1125 (15 %): actions at the bounds, with mean hold steps 1, 4, or 16 for each episode. Saturated actions at
#     low speed, and flips from upright states (each episode flips, then stays flipped for 70-90 % of its steps).
#   expert_noise 1125 (15 %) with the noise stds 0.2, 0.4, and 0.8.
#   past_runs 3000 (40 %): the replays, return buckets, and flip share of OFFLINE_DATA_REALISTIC.
#   expert_scaled 1500 (20 %): as OFFLINE_DATA_SCALED40, but the scale range starts at 0.2, not 0.3. Scales below 0.3
#     (about 1 of 6 draws) give small actions at 0-0.7 m/s; scales of 0.3-0.4 give 0.7-1.7 m/s, the speeds at which the
#     online policies flip. The scale changes every 100 steps, so most episodes hold heavily scaled segments.
OFFLINE_DATA_BANGBANG_EPISODES = {"random": 750, "bang_bang": 1125, "expert_noise": 1125, "past_runs": 3000, "expert_scaled": 1500}
OFFLINE_DATA_BANGBANG = {
    "offline_dataset_name": "halfcheetah_v4_mix_bangbang_scaled_7p5m",
    "offline_data_steps": 1000 * sum(OFFLINE_DATA_BANGBANG_EPISODES.values()),
    "offline_data_mix": {
        mode: count / sum(OFFLINE_DATA_BANGBANG_EPISODES.values()) for mode, count in OFFLINE_DATA_BANGBANG_EPISODES.items()
    },
    "offline_expert_noise_stds": [0.2, 0.4, 0.8],
    "offline_bang_bang_hold_steps": [1.0, 4.0, 16.0],
    "offline_past_run_replays": OFFLINE_DATA_REALISTIC_REPLAYS,
    "offline_past_run_return_buckets": [[-500.0, 0.0], [0.0, 500.0], [500.0, 1500.0], [1500.0, 4000.0]],
    "offline_past_run_flip_episode_fraction": FLIP_EPISODE_FRACTION,
    "offline_expert_action_scales": [0.2, 0.8],
    "offline_expert_scale_change_steps": 100,
    "offline_expert_scaled_noise_std": 0.1,
}


def bangbang_data_queue_ablations():
    """Entries 63-68, seeds 42 and 4, on the data of entry 62 (OFFLINE_DATA_BANGBANG).

    63-64 pretrain (12,000 steps), 65-66 RL without rehearsal, 67-68 the same RL runs with the head re-anchor.
    The RL runs save a checkpoint every 10 loops, so the probes can read the world model during RL. Only the newest
    checkpoint keeps its replay snapshot (a dense 9.9 GB folder).
    """

    data_name = OFFLINE_DATA_BANGBANG["offline_dataset_name"]
    pretrain, rl, anchor = [], [], []

    for seed in QUEUE_SEEDS:
        pretrain_name = f"bangbang_scaled_s{seed}"
        pretrained_rl = {
            "seed": seed,
            **AUDIT_DIAGNOSTICS,
            **NEXT_STATE_REWARD_TARGET,
            **OFFLINE_PRETRAINED_NEXT_STATE_REWARD,
            "offline_pretrained_checkpoint": f"runs/offline_pretrain/{pretrain_name}/pretrained.pt",
            # The rl stage does not read the dataset. The name only records the data of the pretrain.
            "offline_dataset_name": data_name,
            **LOCAL_RL_100_LOOPS,
            "checkpoint_every": 10,
            "offline_rehearsal_fraction": 0.0,
            **SHARED_GPU_MEMORY,
        }
        pretrain.append({
            "run_name": f"pretrain_bangbang_scaled_s{seed}",
            "run_details": f"prev=[pretrain_realistic_data_s{seed}], change=[offline data {data_name}: 15 % bang-bang and 20 % scaled expert (scale 0.2-0.8) episodes, 40 % past runs]",
            "seed": seed,
            **OFFLINE_PRETRAIN_NEXT_STATE_REWARD,
            **LOCAL_16GB_GPU,
            # Two runs share the GPU. The validation runs in chunks of this size too.
            "world_model_micro_batch_size": 32,
            "offline_dataset_name": data_name,
            "offline_pretrain_name": pretrain_name,
            "offline_world_model_steps": 12000,
            **SHARED_GPU_MEMORY,
        })
        rl.append({
            "run_name": f"pretrain_rl_bangbang_scaled_s{seed}",
            "run_details": f"prev=[pretrain_rl_realistic_data_s{seed}], change=[start from pretrain {pretrain_name} (data {data_name}); checkpoint every 10 loops]",
            **pretrained_rl,
        })
        anchor.append({
            "run_name": f"pretrain_rl_anchor_bangbang_scaled_s{seed}",
            "run_details": f"prev=[pretrain_rl_bangbang_scaled_s{seed}], change=[re-anchor the policy and value heads after each world-model phase (reports/9-28-26 D1)]",
            **pretrained_rl,
            **HEAD_REANCHOR,
        })

    return [*pretrain, *rl, *anchor]


# Full-transformer finetuning in the RL loop (fixed_full_transformer_finetuning, paper section 3.3 and its footnote).
# Each of the 10 steps of a loop sums the world-model loss on replay windows and the RL loss on dreams, and one
# optimizer updates the transformer and the heads. The RL loss needs the policy-prior KL, so the objective is PMPO
# (the train/config.yaml defaults: KL weight 0.3, entropy weight 0.01), not the closer PPO of the other local runs.
# The dream forward with the transformer gradient runs in chunks of 64 dreams (two runs on a 16 GB GPU; chunks of 128
# ran out of memory under the 48 % cap).
JOINT_FINETUNE = {
    "objective": "pmpo",
    "fixed_full_transformer_finetuning": True,
    "fixed_full_transformer_finetuning_steps": 10,
    "fixed_full_transformer_finetuning_dream_micro_batch_size": 64,
}

# The pretrain of the best local RL run (offline_pretrained_a2_scaled40_no_rehearsal_s4: return 1,925 at loops 95-99).
# It also has the best action ranking of all pretrains (report/10-1-26-realistic-data-analysis section 4.3).
BEST_PRETRAIN_NAME = "a2_scaled40_s4"


def joint_finetune_queue_ablations():
    """Entries 69-72, seeds 42 and 4: RL with the world-model loss and the RL loss in each update (JOINT_FINETUNE).

    69-70 from scratch (the settings of the local reproduction runs, entries 21-22), 71-72 from the pretrain
    BEST_PRETRAIN_NAME without rehearsal (the settings of entries 49-50). Both seeds of 71-72 load the same pretrained
    checkpoint, so only the RL seed differs. Checkpoints at loops 50 and 100.
    """

    scratch, pretrained = [], []
    common = {
        # The world-model quality logs of the other local runs. The objective stays PMPO.
        "audit_diagnostics": True,
        **NEXT_STATE_REWARD_TARGET,
        **LOCAL_RL_QUEUE,
        **SHARED_GPU_MEMORY,
        **JOINT_FINETUNE,
    }

    for seed in QUEUE_SEEDS:
        scratch.append({
            "run_name": f"joint_finetune_scratch_s{seed}",
            "run_details": f"prev=[audit_a2_next_state_reward_target_local_s{seed}], change=[fixed_full_transformer_finetuning: 10 joint steps for each loop (world-model loss + PMPO RL loss, one optimizer); objective PMPO in place of closer PPO]",
            "seed": seed,
            **common,
        })
        pretrained.append({
            "run_name": f"joint_finetune_pretrained_s{seed}",
            "run_details": f"prev=[offline_pretrained_a2_scaled40_no_rehearsal_s{seed}], change=[fixed_full_transformer_finetuning: 10 joint steps for each loop (world-model loss + PMPO RL loss, one optimizer); objective PMPO in place of closer PPO; both seeds start from pretrain {BEST_PRETRAIN_NAME}]",
            "seed": seed,
            **OFFLINE_PRETRAINED_NEXT_STATE_REWARD,
            "offline_pretrained_checkpoint": f"runs/offline_pretrain/{BEST_PRETRAIN_NAME}/pretrained.pt",
            # The rl stage does not read the dataset. The name only records the data of the pretrain.
            "offline_dataset_name": OFFLINE_DATA_SCALED40["offline_dataset_name"],
            "offline_rehearsal_fraction": 0.0,
            **common,
        })

    return [*scratch, *pretrained]


# RMS normalization of the world-model losses (paper section 3) on the A2 settings.
# The earlier runs (loss_normalization, frozen_loss_normalization) used the PMPO settings of modded_better_baseline and
# gave the flow loss and the bootstrap loss one RMS statistic each. The bootstrap loss is 13 to 16 times smaller than
# the flow loss, so the two got equal weight, and the flow loss rose to the level of an untrained model.
LOSS_NORMALIZATION = {"use_dynamics_loss_normalization": True}
# One RMS statistic for the sum of the flow loss and the bootstrap loss (eq. 7 is one loss term).
GROUPED_LOSS_NORMALIZATION = {**LOSS_NORMALIZATION, "group_dynamics_loss_normalization": True}


def loss_normalization_queue_ablations():
    """Entries 73-74, seed 4: the local A2 run (entry 22) with RMS loss normalization, live statistics.

    73 groups the flow loss and the bootstrap loss under one statistic. 74 gives each its own statistic, as the earlier
    loss_normalization run. The comparison without normalization is audit_a2_next_state_reward_target_local_s4 (entry 22).
    """

    seed = 4
    common = {
        "seed": seed,
        **AUDIT_DIAGNOSTICS,
        **NEXT_STATE_REWARD_TARGET,
        **LOCAL_RL_QUEUE,
        **SHARED_GPU_MEMORY,
    }
    prev = f"prev=[audit_a2_next_state_reward_target_local_s{seed}]"

    return [
        {
            "run_name": f"a2_grouped_loss_normalization_s{seed}",
            "run_details": f"{prev}, change=[RMS loss normalization with live statistics; one statistic for the flow loss and the bootstrap loss together]",
            **common,
            **GROUPED_LOSS_NORMALIZATION,
        },
        {
            "run_name": f"a2_loss_normalization_s{seed}",
            "run_details": f"{prev}, change=[RMS loss normalization with live statistics; the flow loss and the bootstrap loss have one statistic each, as the earlier loss_normalization run]",
            **common,
            **LOSS_NORMALIZATION,
        },
    ]


# Changes of a2_grouped_loss_normalization_s4 / s42 (entries 73 and 77), one for each pair of entries.
# 1. Faster start (report/10-1-26-grouped-normalization, section 7): the reward model of the grouped runs stays at the
#    mean reward for 25 loops, so the return passes 0 at loops 30-32 (plain A2: 18-20).
# 2. The objective: PMPO with the train/config.yaml defaults (KL weight 0.3 to the policy prior, learning rate 6e-4).
# 3. The critic: HL-Gauss cross-entropy or MSE value loss, and the value heads of the H100 runs
#    symlog_hl_gauss_value_head and symexp_hl_gauss_actor_critic. The base head is the raw HL-Gauss head with the
#    closer PPO MSE loss on its expected value.
# 4. The depth mixing of the transformer: AttnPool (the base), no pool, full AttnRes, block AttnRes of 2 sublayers.
#    The arms without a pool use a feedforward expansion of 7 (base 4), so that their transformer has the parameter
#    count of the base transformer: 1,207,864 with AttnPool, 1,206,840 with no pool, 1,208,632 with AttnRes.
EQUAL_PARAMETER_FEEDFORWARD = {"use_attn_pool": False, "dynamics_ff_expansion_factor": 7}
GROUPED_NORMALIZATION_CHANGES = [
    ("wm_pretrain640", "640 world-model steps before loop 0 (base 256); the reward model left the mean after about 580 steps",
        {"pretrain_world_model_combined_steps": 640}),
    ("reward_weight4", "reward loss weight 4 in the world-model loss; under normalization the reward multiplier is only 0.38",
        {"world_model_reward_loss_weight": 4.0}),
    ("norm_warmup15", "plain loss sum (no RMS normalization) before loop 15, then grouped normalization; the RMS statistics update from the start",
        {"dynamics_loss_normalization_start_loop": 15}),
    ("pmpo", "PMPO objective with the config defaults (policy prior KL 0.3, learning rate 6e-4) in place of closer PPO",
        {"objective": "pmpo", "cleanrl_updated_ppo": False, "cleanrl_ppo_closer": False, "agent_learning_rate": 0.0006}),
    ("hl_gauss_value_loss", "HL-Gauss cross-entropy value loss on the base raw HL-Gauss value head, in place of MSE",
        {"cleanrl_ppo_closer_hl_gauss_value_loss": True}),
    ("symlog_hl_gauss_value_head", "symlog HL-Gauss value head with its cross-entropy value loss (closer_ppo_symlog_hl_gauss_value_head)",
        SYMLOG_HL_GAUSS_VALUE_HEAD),
    ("symexp_hl_gauss_actor_critic", "symexp two-hot value head beside a raw HL-Gauss critic for GAE (symexp_hl_gauss_actor_critic), both with cross-entropy losses",
        {"use_symexp_value_head": True, "use_hl_gauss_actor_critic": True, "value_range": [-10.0, 10.0], "cleanrl_ppo_closer_hl_gauss_value_loss": True}),
    ("no_attn_pool", "no AttnPool and no AttnRes, feedforward expansion 7 for an equal parameter count",
        EQUAL_PARAMETER_FEEDFORWARD),
    ("full_attn_res", "full AttnRes in place of AttnPool, feedforward expansion 7 for an equal parameter count",
        {**EQUAL_PARAMETER_FEEDFORWARD, "attn_res": "full"}),
    ("block2_attn_res", "block AttnRes (2 sublayers for each block) in place of AttnPool, feedforward expansion 7 for an equal parameter count",
        {**EQUAL_PARAMETER_FEEDFORWARD, "attn_res": "block", "attn_res_block_size": 2}),
]


def grouped_normalization_queue_ablations():
    """Entries 78-97: entries 73 / 77 with one change each (GROUPED_NORMALIZATION_CHANGES), seed 4 then seed 42."""

    base = loss_normalization_queue_ablations()[0]
    return [
        {
            **base,
            "run_name": f"a2_grouped_{name}_s{seed}",
            "run_details": f"prev=[a2_grouped_loss_normalization_s{seed}], change=[{change}]",
            "seed": seed,
            **overrides,
        }
        for name, change, overrides in GROUPED_NORMALIZATION_CHANGES
        for seed in (4, 42)
    ]


def grouped_normalization_more_seeds_queue_ablations():
    """Entries 98-108: more seeds for entries 73 / 77 and 78-97 (report/10-2-26-grouped-ablations, section 8).

    98-99 the base with seeds 1 and 2, 100-108 each arm except PMPO with seed 1.
    """

    base = loss_normalization_queue_ablations()[0]
    base_seeds = [
        {
            **base,
            "run_name": f"a2_grouped_loss_normalization_s{seed}",
            "run_details": f"prev=[a2_grouped_loss_normalization_s4], change=[seed {seed}]",
            "seed": seed,
        }
        for seed in (1, 2)
    ]
    arm_seeds = [
        {
            **base,
            "run_name": f"a2_grouped_{name}_s1",
            "run_details": f"prev=[a2_grouped_loss_normalization_s1], change=[{change}]",
            "seed": 1,
            **overrides,
        }
        for name, change, overrides in GROUPED_NORMALIZATION_CHANGES
        if name != "pmpo"
    ]
    return [*base_seeds, *arm_seeds]


# Learning-rate arms of the grouped-normalization base (entries 109-114). The closer PPO learning rate decays linearly
# from agent_learning_rate (3e-4) to 0 over num_loops (500), so it is 2.4e-4 at loop 100 of these 100-loop runs.
# The world-model learning rate is constant in all runs.
GROUPED_NORMALIZATION_LEARNING_RATE_CHANGES = [
    ("flat_lr", "constant closer PPO learning rate 3e-4 (no linear decay)", {"cleanrl_updated_ppo_constant_learning_rate": True}),
    ("flat_lr340", "constant closer PPO learning rate 3.4e-4 (13 % higher, no linear decay)",
        {"cleanrl_updated_ppo_constant_learning_rate": True, "agent_learning_rate": 0.00034}),
]


def grouped_normalization_learning_rate_queue_ablations():
    """Entries 109-114: the learning-rate arms of the grouped-normalization base, seeds 4, 42 and 1."""

    base = loss_normalization_queue_ablations()[0]
    return [
        {
            **base,
            "run_name": f"a2_grouped_{name}_s{seed}",
            "run_details": f"prev=[a2_grouped_loss_normalization_s{seed}], change=[{change}]",
            "seed": seed,
            **overrides,
        }
        for name, change, overrides in GROUPED_NORMALIZATION_LEARNING_RATE_CHANGES
        for seed in (4, 42, 1)
    ]


# Entries 115-117: no AttnPool, and a deeper transformer in place of the wider feedforward of no_attn_pool. Depth 5 (base 3)
# with time_block_every 2 gives 2 time layers (base 1) and 1,208,980 transformer parameters (AttnPool base 1,207,864).
DEEPER_NO_ATTN_POOL = {"use_attn_pool": False, "depth": 5}


def grouped_normalization_deeper_queue_ablations():
    """Entries 115-117: the grouped-normalization base without AttnPool and with depth 5, seeds 4, 42 and 1."""

    base = loss_normalization_queue_ablations()[0]
    return [
        {
            **base,
            "run_name": f"a2_grouped_no_attn_pool_depth5_s{seed}",
            "run_details": f"prev=[a2_grouped_no_attn_pool_s{seed}], change=[depth 5 (2 time layers) in place of feedforward expansion 7, for an equal parameter count without AttnPool]",
            "seed": seed,
            **DEEPER_NO_ATTN_POOL,
        }
        for seed in (4, 42, 1)
    ]


# Combinations of the one-change arms of the grouped-normalization base (report/10-3-26-combinations). Each arm in a
# combination improves on the base with p < 0.04 (return at loops 95-99, t test with the seed variance pooled over all
# arms; 3 seeds against 4). Stage 1 combines two arms. The pairs follow the order of the effects: each value arm with
# the start-speed arm, and each attention arm with the reward-weight arm. Stage 2 waits for the stage-1 results.
#   cevalue    hl_gauss_value_loss           (+999 at loops 95-99, +364 mean return of loops 0-99)
#   symlog     symlog_hl_gauss_value_head    (+946, +310)
#   symexp     symexp_hl_gauss_actor_critic  (+874, +228)
#   warmup15   norm_warmup15                 (+718, +541)
#   block2     block2_attn_res               (+610, +292)
#   fullres    full_attn_res                 (+525, +338; the Welch test gives p 0.14, because of the largest seed spread)
#   rw4        reward_weight4                (+449, +225)
#   depth5     no_attn_pool_depth5           (+435, +252)
# The three value arms are alternatives (one value loss), and the three attention arms are alternatives.
# wm_pretrain640 (+627, +839; it moves the loop at 0 from 30-34 to 8-10) is not in stage 1: it has the largest drops of
# all arms (6 loops with a drop over 200 on average). Entries 142-144 add it to symexp + warmup15 (not in queue 10-4-26).
# All combinations run 175 loops (stop_after_loops; num_loops stays 500, so the schedules of loops 0-99 do not change)
# and save one checkpoint and replay snapshot, at loop 175.
COMBINATION_PARTS = {
    "cevalue": "hl_gauss_value_loss",
    "symlog": "symlog_hl_gauss_value_head",
    "symexp": "symexp_hl_gauss_actor_critic",
    "warmup15": "norm_warmup15",
    "rw4": "reward_weight4",
    "block2": "block2_attn_res",
    "fullres": "full_attn_res",
    "depth5": "no_attn_pool_depth5",
    "wmpre640": "wm_pretrain640",
}
COMBINATION_PAIRS = [
    ("cevalue", "warmup15"),
    ("rw4", "block2"),
    ("symlog", "warmup15"),
    ("symexp", "warmup15"),
    ("rw4", "depth5"),
    ("rw4", "fullres"),
]
COMBINATION_SEEDS = (4, 42, 1)
COMBINATION_LOOPS = {"stop_after_loops": 175, "checkpoint_every": 175}


def combination_run_name(parts, seed):
    return f"a2_grouped_combo_{'_'.join(parts)}_s{seed}"


def combination_entries(combinations, stage):
    """The base (entry 73) with the arms of each combination, 175 loops, one entry for each seed of COMBINATION_SEEDS."""

    base = loss_normalization_queue_ablations()[0]
    changes = {name: (change, overrides) for name, change, overrides in GROUPED_NORMALIZATION_CHANGES}
    changes["no_attn_pool_depth5"] = ("no AttnPool, depth 5 (2 time layers) for an equal parameter count", DEEPER_NO_ATTN_POOL)
    entries = []
    for parts in combinations:
        arms = [COMBINATION_PARTS[part] for part in parts]
        overrides = {}
        for arm in arms:
            clash = set(overrides) & set(changes[arm][1])
            if clash:
                raise ValueError(f"the arms of {parts} set the same keys: {sorted(clash)}")
            overrides.update(changes[arm][1])
        for seed in COMBINATION_SEEDS:
            entries.append({
                **base,
                **COMBINATION_LOOPS,
                "run_name": combination_run_name(parts, seed),
                "run_details": (
                    f"prev=[{', '.join(f'a2_grouped_{arm}_s{seed}' for arm in arms)}], "
                    f"change=[{stage} combination, 175 loops: {'; '.join(changes[arm][0] for arm in arms)}]"
                ),
                "seed": seed,
                **overrides,
            })
    return entries


def grouped_normalization_combination_queue_ablations():
    """Entries 118-135: stage 1 of the combinations of the significant arms (COMBINATION_PAIRS), seeds 4, 42 and 1.

    Each run is 175 loops.
    """

    return combination_entries(COMBINATION_PAIRS, "stage 1")


# Stage 2 (report/10-3-26-combinations, section 4; report/10-4-26-stage1-combinations, section 8).
# Only symexp + warmup15 is better than each of its two arms with p < 0.04 (pooled t test, return at loops 95-99:
# +439 against symexp, +595 against warmup15). No attention pair is better than its arms. By the rule of the request,
# the better pair gets the better arm of the other pairs. The attention arms do not differ with p < 0.04 alone
# (block2 1,915 and depth5 1,740 at loops 95-99, p about 0.26), so both are run: block2 has the highest return alone,
# and depth5 gives the most accurate world model (action ranking 0.54 in rw4 + depth5), which limits the fast pairs.
# The third combination adds the world-model pretraining arm (wm_pretrain640, the TODO above) to the best pair.
# Queue 10-4-26 does not run it (entries 142-144): its pretrain steps train at loop 0 on the 64 episodes of the
# untrained policy. A pretraining test uses the precollected dataset (offline_pretraining) instead.
COMBINATION_STAGE2 = [
    ("symexp", "warmup15", "block2"),
]
COMBINATION_STAGE2B = [
    ("symexp", "warmup15", "depth5"),
    ("symexp", "warmup15", "wmpre640"),
]


def grouped_normalization_combination_stage2_queue_ablations():
    """Entries 136-138: stage 2 of the combinations (COMBINATION_STAGE2), seeds 4, 42 and 1, 175 loops."""

    return combination_entries(COMBINATION_STAGE2, "stage 2")


def grouped_normalization_combination_stage2b_queue_ablations():
    """Entries 139-144: stage 2 with depth5 in place of block2, and the best pair with wm_pretrain640 (COMBINATION_STAGE2B)."""

    return combination_entries(COMBINATION_STAGE2B, "stage 2")


# Sweeps of the best stage-1 pair (symexp + warmup15, entries 127-129, the reference at 175 loops). One value changes in
# each entry. report/10-4-26-stage1-combinations gives the reasons:
#   agent_learning_rate         the closer PPO learning rate (linear decay over num_loops = 500)
#   world_model_learning_rate   the world model falls behind the fast policy (one-step velocity error 0.45 against 0.16)
#   cleanrl_ppo_entropy_weight  the policy entropy falls to -4 to -5 and saturated actions predict the flips
COMBINATION_SWEEP_PAIR = ("symexp", "warmup15")
COMBINATION_SWEEPS = [
    ("lr150", "closer PPO learning rate 1.5e-4 (base 3e-4)", {"agent_learning_rate": 0.00015}),
    ("lr600", "closer PPO learning rate 6e-4 (base 3e-4)", {"agent_learning_rate": 0.0006}),
    ("ent001", "closer PPO entropy weight 0.001 (base 0)", {"cleanrl_ppo_entropy_weight": 0.001}),
    ("ent01", "closer PPO entropy weight 0.01 (base 0)", {"cleanrl_ppo_entropy_weight": 0.01}),
    ("wmlr300", "world-model learning rate 3e-4 (base 6e-4)", {"world_model_learning_rate": 0.0003}),
    ("wmlr1200", "world-model learning rate 1.2e-3 (base 6e-4)", {"world_model_learning_rate": 0.0012}),
]


def grouped_normalization_combination_sweep_queue_ablations():
    """Entries 145-162: COMBINATION_SWEEPS on symexp + warmup15, seeds 4, 42 and 1, 175 loops."""

    pair_entries = combination_entries([COMBINATION_SWEEP_PAIR], "sweep")
    entries = []
    for name, change, overrides in COMBINATION_SWEEPS:
        for entry in pair_entries:
            clash = {key for key in overrides if key in entry and entry[key] != loss_normalization_queue_ablations()[0].get(key)}
            if clash:
                raise ValueError(f"the sweep {name} changes keys that the pair sets: {sorted(clash)}")
            seed = entry["seed"]
            entries.append({
                **entry,
                "run_name": combination_run_name((*COMBINATION_SWEEP_PAIR, name), seed),
                "run_details": f"prev=[{combination_run_name(COMBINATION_SWEEP_PAIR, seed)}], change=[sweep, 175 loops: {change}]",
                **overrides,
            })
    return entries


# Pretraining on the precollected dataset (queue 10-4-26). It replaces wm_pretrain640 (entries 142-144), whose steps
# train at loop 0 on the 64 episodes of the untrained policy. Each run does two offline stages in one process: the
# pretrain stage trains the tokenizer and the world model on OFFLINE_DATA_SCALED40 (12,000 steps; the data and the length
# of the pretrain of the best local pretrained run, offline_pretrained_a2_scaled40_no_rehearsal_s4), then the rl stage runs
# 175 online loops without rehearsal (rehearsal hurt in every dataset, report/9-30-26-pretrain-analysis). No new data is
# collected and no older checkpoint is loaded: the older pretrains have use_dynamics_loss_normalization=false.
# warmup15 is left out: train_gym rejects dynamics_loss_normalization_start_loop on the offline path, and the pretraining
# gives the same head start. The reference runs are symexp + warmup15 (entries 127-129) and the symexp arm (100 loops).
COMBINATION_PRETRAIN_PART = "symexp"
COMBINATION_PRETRAIN = {
    "offline_pretraining": True,
    "offline_stages": ["pretrain", "rl"],
    "offline_dataset_name": OFFLINE_DATA_SCALED40["offline_dataset_name"],
    "offline_world_model_steps": 12000,
    "offline_rehearsal_fraction": 0.0,
}


def grouped_normalization_offline_pretrain_queue_ablations():
    """Entries 163-165: symexp + offline pretraining on OFFLINE_DATA_SCALED40, seeds 4, 42 and 1, 175 online loops."""

    entries = []
    for entry in combination_entries([(COMBINATION_PRETRAIN_PART,)], "pretraining"):
        seed = entry["seed"]
        entries.append({
            **entry,
            "run_name": combination_run_name((COMBINATION_PRETRAIN_PART, "scaled40pre"), seed),
            "run_details": (
                f"prev=[a2_grouped_symexp_hl_gauss_actor_critic_s{seed}, {combination_run_name(COMBINATION_SWEEP_PAIR, seed)}], "
                f"change=[pretrain the tokenizer and the world model on {COMBINATION_PRETRAIN['offline_dataset_name']} "
                f"(12,000 steps), then 175 online loops without rehearsal; no warmup15]"
            ),
            "offline_pretrain_name": f"a2_grouped_symexp_scaled40_s{seed}",
            **COMBINATION_PRETRAIN,
        })
    return entries


# The offline data from the runs of the stage-1 combinations (entries 118-135, report/10-4-26-pretrain-data): 6000
# episodes, all past_runs. The lessons of the earlier datasets:
#   - Expert and expert + noise data put saturated actions only at high speed. The pretrained world models learned
#     "large action means fast", rated bang-bang actions above the online gait, and the RL runs locked into the flipped
#     gait (report/9-30-26-pretrain-rootcause, report/10-1-26-realistic-data-analysis). This data has no expert modes.
#   - The earlier past runs were slow (at most 1.6-2.7 m/s), so fast data came only from the expert modes. The 18 runs of
#     175 loops reach 2.4-4.3 m/s at loops 170-174 (p90 of an episode up to 6.2 m/s) with the actions of the policy.
#   - Expert data (10-12 m/s) made the frozen pretrained tokenizer compress the velocity (vx std 3.9-5.1 against 0.74
#     online). This data has the speeds of the online runs.
#   - Six settings and three seeds give different gaits: the rw4 runs keep moderate actions at speed (|a|-speed
#     correlation -0.2 to 0.1), the fast value runs use larger actions at speed (0.3 to 0.6).
#   - Five equal return buckets cover the whole learning curve: the first near-uniform policies and the flipped gait
#     (below 0), the flip phase (0-1000), and the running gaits at 1-2, 2-3, and 3-6 thousand.
#   - In each bucket, 30 % of the episodes have an upright-to-flipped transition (about half recover within 10-18 steps).
#     The share of 0.45 in the realistic data gave mostly the flipped steady state (early flips); the flips of the fast
#     buckets come late in the episode (median step 680-740), so they add onsets and recoveries with few flipped steps.
#   - The replays of entries 136-165 (queue 10-4-26) are not in the data: the probes of these pretrains use their states.
OWN_RUN_REPLAYS = [f"runs/gym_imagination/{combination_run_name(pair, seed)}/replay_buffer" for pair in COMBINATION_PAIRS for seed in COMBINATION_SEEDS]
OWN_RUN_EPISODES = 6000
OFFLINE_DATA_OWN_RUNS = {
    "offline_dataset_name": "halfcheetah_v4_own_runs_6m",
    "offline_data_steps": 1000 * OWN_RUN_EPISODES,
    "offline_data_mix": {"expert": 0.0, "expert_noise": 0.0, "random": 0.0, "past_runs": 1.0},
    "offline_past_run_replays": OWN_RUN_REPLAYS,
    "offline_past_run_return_buckets": [[-1000.0, 0.0], [0.0, 1000.0], [1000.0, 2000.0], [2000.0, 3000.0], [3000.0, 6000.0]],
    "offline_past_run_flip_episode_fraction": 0.3,
    "offline_data_seed": 20261005,
}
# Queue 10-5-26: one process collects the data (if it does not exist), pretrains, and runs RL. The data exists before the
# queue starts (collected on 2026-10-04), so two lanes do not write it at the same time.
COMBINATION_OWN_RUNS_PRETRAIN = {
    **COMBINATION_PRETRAIN,
    **OFFLINE_DATA_OWN_RUNS,
    "offline_stages": ["collect", "pretrain", "rl"],
}


def grouped_normalization_own_runs_pretrain_queue_ablations():
    """Entries 166-168: symexp + data collection and pretraining on OFFLINE_DATA_OWN_RUNS, seeds 4, 42 and 1, 175 loops.

    The same settings as entries 163-165, except the data.
    """

    entries = []
    for entry in grouped_normalization_offline_pretrain_queue_ablations():
        seed = entry["seed"]
        entries.append({
            **entry,
            "run_name": combination_run_name((COMBINATION_PRETRAIN_PART, "ownrunspre"), seed),
            "run_details": (
                f"prev=[{entry['run_name']}], change=[collect {OFFLINE_DATA_OWN_RUNS['offline_dataset_name']} (6000 "
                f"episodes of the 18 stage-1 runs, no expert data) and pretrain on it (12,000 steps)]"
            ),
            "offline_pretrain_name": f"a2_grouped_symexp_own_runs_s{seed}",
            **COMBINATION_OWN_RUNS_PRETRAIN,
        })
    return entries


# Flip-onset sampling (report/10-4-26-instability-audit: the return drops are flips; from the same start states 3-8 % of
# the dreams end flipped and 9-41 % of the real rollouts; the policy is near uniform on the states where the body tips).
# In the late loops about 25 % of the episodes have an onset, so about 5 % of the 200-step world-model windows hold one.
# A fraction of 0.25 gives about 29 % (6 times more), and 75 % of each batch stays the normal windows.
#   flipwm25      world-model capability: the world model sees 6 times more real flip transitions.
#   flipprompt25  actor and critic: a quarter of the dreams start 1-32 steps before a real flip, so the policy and value
#                 heads train on the tipping states.
FLIP_ONSET_SWEEPS = [
    ("flipwm25", "a quarter of the world-model windows hold a real flip onset", {"world_model_flip_onset_window_fraction": 0.25}),
    ("flipprompt25", "a quarter of the imagination prompts end 1-32 steps before a real flip onset", {"imagination_flip_onset_prompt_fraction": 0.25}),
]


def grouped_normalization_flip_onset_queue_ablations():
    """Entries 169-174: FLIP_ONSET_SWEEPS on symexp + warmup15, seeds 4, 42 and 1, 175 loops."""

    entries = []
    for name, change, overrides in FLIP_ONSET_SWEEPS:
        for entry in combination_entries([COMBINATION_SWEEP_PAIR], "flip onsets"):
            seed = entry["seed"]
            entries.append({
                **entry,
                "run_name": combination_run_name((*COMBINATION_SWEEP_PAIR, name), seed),
                "run_details": f"prev=[{combination_run_name(COMBINATION_SWEEP_PAIR, seed)}], change=[175 loops: {change}]",
                **overrides,
            })
    return entries


def grouped_normalization_own_runs_flip_pretrain_queue_ablations():
    """Entries 175-177: entries 166-168 with flipwm25 in the pretrain and the rl stage (the data has 4011 flip onsets)."""

    entries = []
    for entry in grouped_normalization_own_runs_pretrain_queue_ablations():
        seed = entry["seed"]
        entries.append({
            **entry,
            "run_name": combination_run_name((COMBINATION_PRETRAIN_PART, "ownrunspre", "flipwm25"), seed),
            "run_details": f"prev=[{entry['run_name']}], change=[a quarter of the world-model windows (pretrain and online) hold a real flip onset]",
            "offline_pretrain_name": f"a2_grouped_symexp_own_runs_flipwm25_s{seed}",
            "world_model_flip_onset_window_fraction": 0.25,
        })
    return entries


# Entries 163-165 (symexp + scaled40 pretraining) with the arms of a2_grouped_combo_rw4_depth5 (entries 130-132): reward
# loss weight 4 and depth 5 without AttnPool. The arms apply in the pretrain stage and in the rl stage, so each seed
# trains its own pretrain (the scaled40 pretrains of entries 163-165 have depth 3 and reward weight 1).
COMBINATION_PRETRAIN_RW4_DEPTH5_PARTS = (COMBINATION_PRETRAIN_PART, "rw4", "depth5")


def grouped_normalization_rw4_depth5_pretrain_queue_ablations():
    """Entries 178-180: entries 163-165 with rw4 + depth5, seeds 4, 42 and 1, 175 online loops."""

    entries = []
    for entry in combination_entries([COMBINATION_PRETRAIN_RW4_DEPTH5_PARTS], "pretraining"):
        seed = entry["seed"]
        entries.append({
            **entry,
            "run_name": combination_run_name((*COMBINATION_PRETRAIN_RW4_DEPTH5_PARTS, "scaled40pre"), seed),
            "run_details": (
                f"prev=[{combination_run_name((COMBINATION_PRETRAIN_PART, 'scaled40pre'), seed)}, "
                f"{combination_run_name(('rw4', 'depth5'), seed)}], "
                f"change=[scaled40 pretraining (12,000 steps) and 175 online loops with reward loss weight 4 and "
                f"depth 5 without AttnPool, in the pretrain and the rl stage]"
            ),
            "offline_pretrain_name": f"a2_grouped_symexp_rw4_depth5_scaled40_s{seed}",
            **COMBINATION_PRETRAIN,
        })
    return entries


# The opt-in model flags of 2026-10-09 on entries 175-177 (symexp + own-runs pretraining + flipwm25). One change in each
# entry; the flag applies in the pretrain and the rl stage, so each seed trains its own pretrain.
#   kvlb1000   the KV cache of the dynamics transformer keeps the last 1000 steps. An env rollout reaches 1001 steps
#              (1000 + the bootstrap step), so only the bootstrap step of each episode loses step 0. Imagination (40
#              steps) does not change.
#   smallterm  the terminal ensemble members have width model_dim and one hidden layer (4.75 M -> 0.27 M parameters;
#              the world model 8.6 M -> 3.6 M).
OWN_RUNS_FLIP_PRETRAIN_PARTS = (COMBINATION_PRETRAIN_PART, "ownrunspre", "flipwm25")
MODEL_FLAG_SWEEPS = [
    ("kvlb1000", "the KV cache keeps the last 1000 time steps", {"limit_kv_cache_lookback": True, "kv_cache_lookback": 1000}),
    ("smallterm", "terminal ensemble members of width model_dim (dim -> dim -> dim -> 1)", {"small_terminal_head": True}),
]


def grouped_normalization_own_runs_flip_pretrain_model_flag_queue_ablations():
    """Entries 181-186: MODEL_FLAG_SWEEPS on entries 175-177, seeds 4, 42 and 1, 175 online loops."""

    entries = []
    for name, change, overrides in MODEL_FLAG_SWEEPS:
        for entry in grouped_normalization_own_runs_flip_pretrain_queue_ablations():
            seed = entry["seed"]
            entries.append({
                **entry,
                "run_name": combination_run_name((*OWN_RUNS_FLIP_PRETRAIN_PARTS, name), seed),
                "run_details": f"prev=[{entry['run_name']}], change=[pretrain and rl stage: {change}]",
                "offline_pretrain_name": f"a2_grouped_symexp_own_runs_flipwm25_{name}_s{seed}",
                **overrides,
            })
    return entries


# Agent-token readout and the RL policy loss in the world-model step (report/10-10-26-rl-gradient-agent-token) on
# entries 190-192 (175-177 + kvlb200smallterm). All four arms set policy_all_tokens false (the policy reads the agent
# token only). policy_all_tokens is a key of the pretrain signature, so each seed trains its own pretrain, as in entries
# 181-192.
#   agenttok       the agent-token readout alone (the control)
#   agenttokrl001  + rl_policy_loss_in_world_model_updates, RL trunk gradient norm = 0.01 x world-model gradient norm
#                  (the share that the reward MTP loss on the agent token gets now)
#   agenttokrl01   + the same with 0.1
#   agenttokrl03   + the same with 0.3
AGENT_TOKEN_RL_SWEEPS = [
    ("agenttok", "policy reads the agent token only", {"policy_all_tokens": False}),
    *[
        (
            f"agenttokrl{label}",
            f"policy reads the agent token only; RL policy loss in the world-model step, trunk grad ratio {ratio}",
            {"policy_all_tokens": False, "rl_policy_loss_in_world_model_updates": True, "rl_policy_loss_trunk_grad_ratio": ratio},
        )
        for label, ratio in (("001", 0.01), ("01", 0.1), ("03", 0.3))
    ],
]


def grouped_normalization_own_runs_flip_pretrain_agent_token_rl_queue_ablations():
    """Entries 193-204: AGENT_TOKEN_RL_SWEEPS on entries 190-192 (kvlb200smallterm), seeds 4, 42 and 1, 175 online loops."""

    entries = []
    for name, change, overrides in AGENT_TOKEN_RL_SWEEPS:
        for entry in grouped_normalization_own_runs_flip_pretrain_kv_lookback_200_queue_ablations():
            if not entry.get("small_terminal_head"):  # keep the kvlb200smallterm entries (190-192)
                continue
            seed = entry["seed"]
            entries.append({
                **entry,
                "run_name": combination_run_name((*OWN_RUNS_FLIP_PRETRAIN_PARTS, "kvlb200smallterm", name), seed),
                "run_details": f"prev=[{entry['run_name']}], change=[{change}]",
                "offline_pretrain_name": f"a2_grouped_symexp_own_runs_flipwm25_kvlb200smallterm_{name}_s{seed}",
                **overrides,
            })
    return entries


# The joint step (fixed_full_transformer_finetuning) with the agent-token readout on entries 193-195 (agenttok). Each
# joint step sums the world-model loss on replay and the closer-PPO loss (policy + 0.5 value, one pass over one new
# batch of 512 dreams) and updates the transformer and the heads with one optimizer. It replaces the 13 world-model
# steps and the 3 PPO steps of each loop. The RL gradient reaches the transformer only through the agent token (other
# tokens cannot attend to it). All arms load the pretrain of entries 193-195 of the same seed (offline_stages [rl]),
# so the seeds of the arms are paired with each other and with entries 193-195.
#   agenttokshared  the two-phase agent-token readout (entries 193-195) on the shared pretrain (the control)
#   jointrl<r>      the joint step, RL transformer gradient norm = r x world-model gradient norm
#                   (fixed_full_transformer_finetuning_trunk_grad_ratio; the rule of rl_policy_loss_trunk_grad_ratio)
#   jointrl01hlr    jointrl01 with the head learning rate x10 (3e-3): the heads get 10 updates for each loop, not 960
JOINT_AGENT_TOKEN = {
    "fixed_full_transformer_finetuning": True,
    "fixed_full_transformer_finetuning_steps": 10,
    # Two runs share the 16 GB GPU (as rl_policy_loss_dream_micro_batch_size in entries 196-204).
    "fixed_full_transformer_finetuning_dream_micro_batch_size": 64,
}
JOINT_AGENT_TOKEN_SWEEPS = [
    ("agenttokshared", "two-phase agent-token readout (entries 193-195) on the shared pretrain", {}),
    *[
        (
            f"jointrl{label}",
            f"joint step, RL transformer gradient ratio {ratio}",
            {**JOINT_AGENT_TOKEN, "fixed_full_transformer_finetuning_trunk_grad_ratio": ratio},
        )
        for label, ratio in (("001", 0.01), ("01", 0.1), ("04", 0.4), ("1", 1.0))
    ],
    (
        "jointrl01hlr",
        "joint step, RL transformer gradient ratio 0.1, head learning rate 3e-3",
        {**JOINT_AGENT_TOKEN, "fixed_full_transformer_finetuning_trunk_grad_ratio": 0.1, "agent_learning_rate": 3e-3},
    ),
]


def grouped_normalization_own_runs_flip_pretrain_joint_agent_token_queue_ablations():
    """Entries 205-222: JOINT_AGENT_TOKEN_SWEEPS on entries 193-195 (agenttok), seeds 4, 42 and 1, 175 online loops."""

    entries = []
    for name, change, overrides in JOINT_AGENT_TOKEN_SWEEPS:
        for entry in grouped_normalization_own_runs_flip_pretrain_agent_token_rl_queue_ablations():
            if entry["run_name"] != combination_run_name((*OWN_RUNS_FLIP_PRETRAIN_PARTS, "kvlb200smallterm", "agenttok"), entry["seed"]):
                continue
            seed = entry["seed"]
            pretrain_name = entry.pop("offline_pretrain_name")
            entries.append({
                **entry,
                "run_name": combination_run_name((*OWN_RUNS_FLIP_PRETRAIN_PARTS, "kvlb200smallterm", name), seed),
                "run_details": f"prev=[{entry['run_name']}], change=[pretrain of {entry['run_name']}; {change}]",
                "offline_stages": ["rl"],
                "offline_pretrained_checkpoint": f"runs/offline_pretrain/{pretrain_name}/pretrained.pt",
                **overrides,
            })
    return entries


# Fixes of the joint step (report/10-17-26-joint-step-fix) on the pretrains of entries 193-195, seeds paired with entries
# 205-222. The joint runs 208-222 failed in the heads: one optimizer step for each new dream batch (importance ratio 1,
# so the PPO clip never acts) moved the policy entropy about 0.25 nats in each step, and the value head got 10
# updates for each loop (960 in the two-phase loop).
#   jointrl0       the joint step of 208-219 with trunk grad ratio 0 (the heads only): the control that the 10-16
#                  report lacks
#   jointppo0      the fixed joint step, heads only: 12 joint steps for each loop, one batch of 512 dreams for each 4
#                  steps (4 minibatches of 128 dreams, so the clip acts), target KL stop, value head lr x10
#   jointppo001    jointppo0 + the RL policy gradient in the transformer, ratio 0.01
#   jointppo01     jointppo0 + the RL policy gradient in the transformer, ratio 0.1
#   slotshared001  the RL policy loss in the world-model step (entries 196-198, ratio 0.01) on the shared pretrain:
#                  the heads train in the two-phase PPO phase
JOINT_PPO = {
    **JOINT_AGENT_TOKEN,
    "fixed_full_transformer_finetuning_steps": 12,
    "fixed_full_transformer_finetuning_dream_minibatches": 4,
    "fixed_full_transformer_finetuning_target_kl": 0.03,
    "fixed_full_transformer_finetuning_value_lr_scale": 10.,
}
JOINT_FIX_SWEEPS = [
    (
        "jointrl0",
        "joint step, trunk grad ratio 0 (heads only)",
        {**JOINT_AGENT_TOKEN, "fixed_full_transformer_finetuning_trunk_grad_ratio": 0.},
    ),
    (
        "jointppo0",
        "joint step with dream reuse (12 steps, 4 minibatches), target KL 0.03, value lr x10, trunk grad ratio 0",
        {**JOINT_PPO, "fixed_full_transformer_finetuning_trunk_grad_ratio": 0.},
    ),
    (
        "jointppo001",
        "joint step with dream reuse (12 steps, 4 minibatches), target KL 0.03, value lr x10, trunk grad ratio 0.01",
        {**JOINT_PPO, "fixed_full_transformer_finetuning_trunk_grad_ratio": 0.01},
    ),
    (
        "jointppo01",
        "joint step with dream reuse (12 steps, 4 minibatches), target KL 0.03, value lr x10, trunk grad ratio 0.1",
        {**JOINT_PPO, "fixed_full_transformer_finetuning_trunk_grad_ratio": 0.1},
    ),
    (
        "slotshared001",
        "RL policy loss in the world-model step, ratio 0.01 (entries 196-198)",
        {"rl_policy_loss_in_world_model_updates": True, "rl_policy_loss_trunk_grad_ratio": 0.01},
    ),
]


def grouped_normalization_own_runs_flip_pretrain_joint_fix_queue_ablations():
    """Entries 223-237: JOINT_FIX_SWEEPS on the pretrains of entries 193-195, seeds 4, 42 and 1, 175 online loops."""

    entries = []
    for name, change, overrides in JOINT_FIX_SWEEPS:
        for entry in grouped_normalization_own_runs_flip_pretrain_joint_agent_token_queue_ablations():
            if entry["run_name"] != combination_run_name((*OWN_RUNS_FLIP_PRETRAIN_PARTS, "kvlb200smallterm", "agenttokshared"), entry["seed"]):
                continue
            seed = entry["seed"]
            entries.append({
                **entry,
                "run_name": combination_run_name((*OWN_RUNS_FLIP_PRETRAIN_PARTS, "kvlb200smallterm", name), seed),
                "run_details": f"prev=[{entry['run_name']}], change=[{change}]",
                **overrides,
            })
    return entries


# Head re-anchor (report/10-17-26-head-reanchor-eval) on the arms of entries 205-237, the pretrains of entries 193-195,
# seeds paired with them. A world-model update moves the head inputs, so the acting policy changes without a head
# update (0.71 nats for each world-model phase in agenttokshared, 1.75 in slotshared001, at loop 175).
#   jointppo0ra        jointppo0 + the per-step head re-anchor (fixed_full_transformer_finetuning_reanchor_steps):
#                      after each joint step, the heads fit the updated heads on the inputs before the step
#   jointppo001ra      jointppo001 + the per-step head re-anchor (it also removes the policy change of the RL
#                      transformer gradient on the dreams of the batch; the gradient still shapes the transformer)
#   agenttokra         agenttokshared + the head re-anchor after each world-model phase (reports/9-28-26 D1)
#   slotshared001ra    slotshared001 + the head re-anchor after each world-model phase
# 50 Adam steps at lr 1e-3 on the 512 dreams of the batch: on the trained jointppo0 s4 model the drift of one joint step
# (0.04-1.8 nats) falls to 0.004-0.018, in 0.62 s (200 steps at lr 3e-4: 0.001-0.012, in 2.1 s).
JOINT_REANCHOR_STEPS = 50
JOINT_REANCHOR_LEARNING_RATE = 1e-3
REANCHOR_SWEEPS = [
    (
        "jointppo0ra",
        f"jointppo0 + head re-anchor after each joint step ({JOINT_REANCHOR_STEPS} Adam steps, lr {JOINT_REANCHOR_LEARNING_RATE:g})",
        {**JOINT_PPO, "fixed_full_transformer_finetuning_trunk_grad_ratio": 0., "fixed_full_transformer_finetuning_reanchor_steps": JOINT_REANCHOR_STEPS,
         "fixed_full_transformer_finetuning_reanchor_learning_rate": JOINT_REANCHOR_LEARNING_RATE},
    ),
    (
        "jointppo001ra",
        f"jointppo001 + head re-anchor after each joint step ({JOINT_REANCHOR_STEPS} Adam steps, lr {JOINT_REANCHOR_LEARNING_RATE:g})",
        {**JOINT_PPO, "fixed_full_transformer_finetuning_trunk_grad_ratio": 0.01, "fixed_full_transformer_finetuning_reanchor_steps": JOINT_REANCHOR_STEPS,
         "fixed_full_transformer_finetuning_reanchor_learning_rate": JOINT_REANCHOR_LEARNING_RATE},
    ),
    (
        "agenttokra",
        "two-phase agent-token readout + head re-anchor after each world-model phase",
        {"reanchor_heads_after_world_model_updates": True},
    ),
    (
        "slotshared001ra",
        "RL policy loss in the world-model step, ratio 0.01 + head re-anchor after each world-model phase",
        {"rl_policy_loss_in_world_model_updates": True, "rl_policy_loss_trunk_grad_ratio": 0.01, "reanchor_heads_after_world_model_updates": True},
    ),
]


def grouped_normalization_own_runs_flip_pretrain_reanchor_queue_ablations():
    """Entries 238-249: REANCHOR_SWEEPS on the pretrains of entries 193-195, seeds 4, 42 and 1, 175 online loops."""

    entries = []
    for name, change, overrides in REANCHOR_SWEEPS:
        for entry in grouped_normalization_own_runs_flip_pretrain_joint_agent_token_queue_ablations():
            if entry["run_name"] != combination_run_name((*OWN_RUNS_FLIP_PRETRAIN_PARTS, "kvlb200smallterm", "agenttokshared"), entry["seed"]):
                continue
            seed = entry["seed"]
            entries.append({
                **entry,
                "run_name": combination_run_name((*OWN_RUNS_FLIP_PRETRAIN_PARTS, "kvlb200smallterm", name), seed),
                "run_details": f"prev=[{entry['run_name']}], change=[{change}]",
                **overrides,
            })
    return entries


# The tokenizer fix (report/10-20-26-tokenizer-ablations) on the two best arms of this family: kvlb200smallterm
# (entries 190-191; the policy reads all tokens) and agenttokrl001 (entries 196-197; the policy reads the agent token
# only), seeds 4 and 42. The tokenizer trains for 20,000 steps (not 2,000) with a cosine decay of the learning rate, on
# the balanced data mix (random, past runs, scaled expert, noisy expert, expert, recent policy) in place of the
# own-runs episodes. The world model still pretrains on the own-runs data. Each run trains its own pretrain.
TOKENIZER_FIX = {
    "tokenizer_cosine_lr_schedule": True,
    "pretrain_tokenizer_steps": 20000,
    "offline_tokenizer_observations": "runs/offline_data/HalfCheetah-v4/tokenizer_mix_balanced_5p7m/observations.npy",
}
TOKENIZER_FIX_SEEDS = (4, 42)


def grouped_normalization_own_runs_flip_pretrain_tokenizer_fix_queue_ablations():
    """Entries 250-253: TOKENIZER_FIX on kvlb200smallterm (190-191) and agenttokrl001 (196-197), seeds 4 and 42."""

    kvlb200smallterm = combination_run_name((*OWN_RUNS_FLIP_PRETRAIN_PARTS, "kvlb200smallterm"), 0).rsplit("_s", 1)[0]
    bases = [
        entry for entry in grouped_normalization_own_runs_flip_pretrain_kv_lookback_200_queue_ablations()
        if entry.get("small_terminal_head") and entry["seed"] in TOKENIZER_FIX_SEEDS
    ] + [
        entry for entry in grouped_normalization_own_runs_flip_pretrain_agent_token_rl_queue_ablations()
        if entry["run_name"].startswith(f"{kvlb200smallterm}_agenttokrl001_") and entry["seed"] in TOKENIZER_FIX_SEEDS
    ]
    entries = []
    for entry in bases:
        seed = entry["seed"]
        entries.append({
            **entry,
            "run_name": entry["run_name"].removesuffix(f"_s{seed}") + f"_tokfix_s{seed}",
            "run_details": (
                f"prev=[{entry['run_name']}], change=[tokenizer: 20,000 steps with cosine decay on the balanced data mix "
                f"(tokenizer_mix_balanced_5p7m)]"
            ),
            "offline_pretrain_name": entry["offline_pretrain_name"].removesuffix(f"_s{seed}") + f"_tokfix_s{seed}",
            **TOKENIZER_FIX,
        })
    return entries


# KV cache lookback 200 on entries 175-177, alone and with the small terminal head. The flags apply in the pretrain and
# the rl stage, so each seed trains its own pretrain, as in entries 181-186.
#   kvlb200            the KV cache keeps the last 200 time steps (kvlb1000 keeps 1000)
#   kvlb200smallterm   kvlb200 + small_terminal_head
KV_LOOKBACK_200_SWEEPS = [
    ("kvlb200", "the KV cache keeps the last 200 time steps", {"limit_kv_cache_lookback": True, "kv_cache_lookback": 200}),
    (
        "kvlb200smallterm",
        "the KV cache keeps the last 200 time steps; terminal ensemble members of width model_dim",
        {"limit_kv_cache_lookback": True, "kv_cache_lookback": 200, "small_terminal_head": True},
    ),
]


def grouped_normalization_own_runs_flip_pretrain_kv_lookback_200_queue_ablations():
    """Entries 187-192: KV_LOOKBACK_200_SWEEPS on entries 175-177, seeds 4, 42 and 1, 175 online loops."""

    entries = []
    for name, change, overrides in KV_LOOKBACK_200_SWEEPS:
        for entry in grouped_normalization_own_runs_flip_pretrain_queue_ablations():
            seed = entry["seed"]
            entries.append({
                **entry,
                "run_name": combination_run_name((*OWN_RUNS_FLIP_PRETRAIN_PARTS, name), seed),
                "run_details": f"prev=[{entry['run_name']}], change=[pretrain and rl stage: {change}]",
                "offline_pretrain_name": f"a2_grouped_symexp_own_runs_flipwm25_{name}_s{seed}",
                **overrides,
            })
    return entries


# Full-transformer finetuning with the settings of the A2 runs (closer CleanRL PPO, not PMPO). Each of the 10 steps of a
# loop sums the world-model loss on replay windows and the closer PPO loss on one new batch of 512 dreams (one pass,
# value loss weight 0.5, no policy prior). The A2 runs do 960 PPO minibatch updates of the heads for each loop.
JOINT_FINETUNE_A2 = {
    "fixed_full_transformer_finetuning": True,
    "fixed_full_transformer_finetuning_steps": 10,
    "fixed_full_transformer_finetuning_dream_micro_batch_size": 64,
}


def joint_finetune_a2_queue_ablations():
    """Entries 75-76, seed 4: the local A2 run (entry 22) with full-transformer finetuning (JOINT_FINETUNE_A2).

    75 is the comparison run: the plain sum of the losses. 76 adds RMS loss normalization with one statistic for the
    flow loss and the bootstrap loss together (GROUPED_LOSS_NORMALIZATION). The entries 69-72 used PMPO.
    """

    seed = 4
    common = {
        "seed": seed,
        **AUDIT_DIAGNOSTICS,
        **NEXT_STATE_REWARD_TARGET,
        **LOCAL_RL_QUEUE,
        **SHARED_GPU_MEMORY,
        **JOINT_FINETUNE_A2,
    }

    return [
        {
            "run_name": f"joint_finetune_a2_s{seed}",
            "run_details": f"prev=[audit_a2_next_state_reward_target_local_s{seed}], change=[fixed_full_transformer_finetuning: 10 joint steps for each loop (world-model loss + closer PPO loss, one optimizer)]",
            **common,
        },
        {
            "run_name": f"joint_finetune_a2_grouped_loss_normalization_s{seed}",
            "run_details": f"prev=[joint_finetune_a2_s{seed}], change=[RMS loss normalization of the world-model losses with live statistics; one statistic for the flow loss and the bootstrap loss together]",
            **common,
            **GROUPED_LOSS_NORMALIZATION,
        },
    ]


# Each comparison uses seeds 42 and 4.
ABLATIONS = [
    {
        "run_name": "a2_reduced_reward_transformer_gradient",
        "run_details": "prev=[audit_a2_next_state_reward_target], change=[scale reward-loss gradients into the shared transformer by 0.01]",
        "seed": 42,
        **AUDIT_DIAGNOSTICS,
        **NEXT_STATE_REWARD_TARGET,
        "scale_reward_gradient_into_transformer": True,
        "reward_gradient_into_transformer_scale": 0.01,
    },
    {
        "run_name": "a2_reduced_reward_transformer_gradient2",
        "run_details": "prev=[audit_a2_next_state_reward_target2], change=[scale reward-loss gradients into the shared transformer by 0.01]",
        "seed": 4,
        **AUDIT_DIAGNOSTICS,
        **NEXT_STATE_REWARD_TARGET,
        "scale_reward_gradient_into_transformer": True,
        "reward_gradient_into_transformer_scale": 0.01,
    },
    {
        "run_name": "audit_a1_deeper_dynamics",
        "run_details": "prev=[audit_control], change=[A1: dynamics depth 3 (S-T-S) -> 5 (S-T-S-T-S)]",
        "seed": 42,
        **AUDIT_DIAGNOSTICS,
        "audit_deeper_dynamics": True,
    },
    {
        "run_name": "audit_a1_deeper_dynamics2",
        "run_details": "prev=[audit_control2], change=[A1: dynamics depth 3 (S-T-S) -> 5 (S-T-S-T-S)]",
        "seed": 4,
        **AUDIT_DIAGNOSTICS,
        "audit_deeper_dynamics": True,
    },
    {
        "run_name": "audit_a3_quarter_envs_per_loop",
        "run_details": "prev=[audit_control], change=[A3: 16 envs per loop; same updates per loop, replay episodes, and env-step budget]",
        "seed": 42,
        **AUDIT_DIAGNOSTICS,
        "audit_quarter_envs_per_loop": True,
    },
    {
        "run_name": "audit_a3_quarter_envs_per_loop2",
        "run_details": "prev=[audit_control2], change=[A3: 16 envs per loop; same updates per loop, replay episodes, and env-step budget]",
        "seed": 4,
        **AUDIT_DIAGNOSTICS,
        "audit_quarter_envs_per_loop": True,
    },
    {
        "run_name": "world_model_policy_consistency",
        "run_details": "prev=[cleanrl_ppo_closer], change=[constrain real-context policy KL during world-model updates]",
        "seed": 42,
        **CLEANRL_PPO_CLOSER,
        "preserve_policy_during_world_model_updates": True,
        "world_model_policy_consistency_loss_weight": 1.0,
    },
    {
        "run_name": "world_model_policy_consistency2",
        "run_details": "prev=[cleanrl_ppo_closer2], change=[constrain real-context policy KL during world-model updates]",
        "seed": 4,
        **CLEANRL_PPO_CLOSER,
        "preserve_policy_during_world_model_updates": True,
        "world_model_policy_consistency_loss_weight": 1.0,
    },
    # aligned_ppo_batch_reuse and aligned_reinforce_* replace the PPO mechanics or the objective.
    # cleanrl_updated_ppo rejects reuse_ppo_dream_batch and objectives other than ppo,
    # so these ablations do not use CLEANRL_PPO_CLOSER.
    {
        "run_name": "aligned_ppo_batch_reuse",
        "run_details": "prev=[aligned_dream_transitions], change=[reuse each PPO dream for four epochs of four trajectory minibatches with exact-KL rollback]",
        "seed": 42,
        "objective": "ppo",
        "align_generated_dream_transitions": True,
        "reuse_ppo_dream_batch": True,
        "ppo_reuse_epochs": 4,
        "ppo_reuse_num_minibatches": 4,
        "ppo_reuse_max_exact_kl": 0.02,
        "agent_learning_rate": 0.0001,
        "normalize_advantages": True,
    },
    {
        "run_name": "aligned_ppo_batch_reuse2",
        "run_details": "prev=[aligned_dream_transitions2], change=[reuse each PPO dream for four epochs of four trajectory minibatches with exact-KL rollback]",
        "seed": 4,
        "objective": "ppo",
        "align_generated_dream_transitions": True,
        "reuse_ppo_dream_batch": True,
        "ppo_reuse_epochs": 4,
        "ppo_reuse_num_minibatches": 4,
        "ppo_reuse_max_exact_kl": 0.02,
        "agent_learning_rate": 0.0001,
        "normalize_advantages": True,
    },
    {
        "run_name": "aligned_reinforce_horizon32",
        "run_details": "prev=[reinforce_baseline], change=[align generated transitions for the 31-transition horizon control]",
        "seed": 42,
        "objective": "reinforce",
        "align_generated_dream_transitions": True,
        "imagination_batch_size": 512,
        "imagination_horizon": 32,
        "imagination_train_steps": 3,
    },
    {
        "run_name": "aligned_reinforce_horizon32_2",
        "run_details": "prev=[reinforce_baseline2], change=[align generated transitions for the 31-transition horizon control]",
        "seed": 4,
        "objective": "reinforce",
        "align_generated_dream_transitions": True,
        "imagination_batch_size": 512,
        "imagination_horizon": 32,
        "imagination_train_steps": 3,
    },
    {
        "run_name": "aligned_reinforce_horizon9",
        "run_details": "prev=[aligned_reinforce_horizon32], change=[use eight learned transitions with the same actor-sample count]",
        "seed": 42,
        "objective": "reinforce",
        "align_generated_dream_transitions": True,
        "imagination_batch_size": 1984,
        "imagination_horizon": 9,
        "imagination_train_steps": 3,
    },
    {
        "run_name": "aligned_reinforce_horizon9_2",
        "run_details": "prev=[aligned_reinforce_horizon32_2], change=[use eight learned transitions with the same actor-sample count]",
        "seed": 4,
        "objective": "reinforce",
        "align_generated_dream_transitions": True,
        "imagination_batch_size": 1984,
        "imagination_horizon": 9,
        "imagination_train_steps": 3,
    },
    {
        # Collect once. The collect stage runs on a CPU node.
        "run_name": "offline_collect_halfcheetah_v4_mix_1m",
        "run_details": "prev=[], change=[collect 1 M offline transitions: 0.3 expert, 0.4 expert + noise, 0.3 random]",
        **OFFLINE_COLLECT,
    },
    {
        # A 1:30 job can stop before the last step. Submit this entry again to continue from the last step checkpoint.
        "run_name": "offline_pretrain_a2_next_state_reward",
        "run_details": "prev=[offline_collect_halfcheetah_v4_mix_1m], change=[pretrain the tokenizer and the A2 world model on the offline data only]",
        "seed": 42,
        **OFFLINE_PRETRAIN_NEXT_STATE_REWARD,
        **LOCAL_16GB_GPU,
    },
    {
        "run_name": "offline_pretrained_a2_rehearsal",
        "run_details": "prev=[audit_a2_next_state_reward_target], change=[start from the offline pretrained tokenizer and world model; 0.5 offline rehearsal in world-model batches]",
        "seed": 42,
        **AUDIT_DIAGNOSTICS,
        **NEXT_STATE_REWARD_TARGET,
        **OFFLINE_PRETRAINED_NEXT_STATE_REWARD,
        **LOCAL_RL_100_LOOPS,
    },
    {
        "run_name": "offline_pretrained_a2_rehearsal2",
        "run_details": "prev=[audit_a2_next_state_reward_target2], change=[start from the offline pretrained tokenizer and world model; 0.5 offline rehearsal in world-model batches]",
        "seed": 4,
        **AUDIT_DIAGNOSTICS,
        **NEXT_STATE_REWARD_TARGET,
        **OFFLINE_PRETRAINED_NEXT_STATE_REWARD,
        **LOCAL_RL_100_LOOPS,
    },
    {
        # This arm measures the value of rehearsal.
        "run_name": "offline_pretrained_a2_no_rehearsal",
        "run_details": "prev=[offline_pretrained_a2_rehearsal], change=[no offline rehearsal in world-model batches]",
        "seed": 42,
        **AUDIT_DIAGNOSTICS,
        **NEXT_STATE_REWARD_TARGET,
        **OFFLINE_PRETRAINED_NEXT_STATE_REWARD,
        **LOCAL_RL_100_LOOPS,
        "offline_rehearsal_fraction": 0.0,
    },
    {
        "run_name": "offline_pretrained_a2_no_rehearsal2",
        "run_details": "prev=[offline_pretrained_a2_rehearsal2], change=[no offline rehearsal in world-model batches]",
        "seed": 4,
        **AUDIT_DIAGNOSTICS,
        **NEXT_STATE_REWARD_TARGET,
        **OFFLINE_PRETRAINED_NEXT_STATE_REWARD,
        **LOCAL_RL_100_LOOPS,
        "offline_rehearsal_fraction": 0.0,
    },
    # 21-22 audit reproductions, 23-24 pretrain, 25-26 rehearsal 0.5, 27-28 no rehearsal,
    # 29-30 entropy bonus with rehearsal 0.25, 31-32 BC in world-model updates without rehearsal (seeds 42, 4).
    *queue_ablations(),    # 33: collect the offline data with past runs and scaled expert actions (5500 episodes, OFFLINE_DATA_PAST_RUNS).
    {
        "run_name": "offline_collect_halfcheetah_v4_mix_past_runs_scaled_expert_5p5m",
        "run_details": "prev=[offline_collect_halfcheetah_v4_mix_1m], change=[5500 episodes: expert 750, expert + noise 1500, random 1000, 1750 from the replays of the 12 pretrained RL arms (half with return -150 to 500, half with 500 to 1500), and 500 expert episodes with actions times 0.4 to 0.7]",
        **OFFLINE_COLLECT,
        **OFFLINE_DATA_PAST_RUNS,
    },
    # 34-35 pretrain on that data, 36-37 RL with 0.5 rehearsal, 38-39 RL without rehearsal (seeds 42, 4).
    *past_runs_queue_ablations(),
    # 40-41 pretrain with the reward gradient into the transformer times 0.01, 42-43 RL from it without rehearsal.
    *reduced_reward_gradient_queue_ablations(),
    # 44: collect the offline data with 40 % scaled expert episodes (OFFLINE_DATA_SCALED40).
    {
        "run_name": "offline_collect_halfcheetah_v4_mix_scaled40_5p5m",
        "run_details": "prev=[offline_collect_halfcheetah_v4_mix_past_runs_scaled_expert_5p5m], change=[5500 episodes: expert 500, expert + noise 1000, random 650, past runs 1150, and 2200 (40 %) expert episodes with actions times a scale in [0.3, 0.8] that changes every 100 steps, plus noise std 0.1]",
        **OFFLINE_COLLECT,
        **OFFLINE_DATA_SCALED40,
    },
    # 45-46 pretrain, 47-48 RL 0.5 rehearsal, 49-50 RL no rehearsal, 51-52 pretrain with reward gradient x 0.01,
    # 53-54 RL no rehearsal from 51-52 (seeds 42, 4).
    *scaled40_queue_ablations(),
    # 55: collect the offline data without expert episodes (OFFLINE_DATA_REALISTIC).
    {
        "run_name": "offline_collect_halfcheetah_v4_mix_realistic_7p5m",
        "run_details": "prev=[offline_collect_halfcheetah_v4_mix_scaled40_5p5m], change=[7500 episodes: no expert or scaled-expert episodes; expert + noise 1500 (std 0.2, 0.4, 0.8), random 1125, and 4875 past runs from 31 local replays in 4 equal return buckets from -500 to 4000, 45 % of each bucket from episodes with a flip]",
        **OFFLINE_COLLECT,
        **OFFLINE_DATA_REALISTIC,
    },
    # 56-57 pretrain, 58-59 RL no rehearsal, 60-61 RL no rehearsal + head re-anchor (seeds 42, 4).
    *realistic_data_queue_ablations(),
    # 62: collect the offline data with bang-bang and scaled expert episodes (OFFLINE_DATA_BANGBANG).
    {
        "run_name": "offline_collect_halfcheetah_v4_mix_bangbang_scaled_7p5m",
        "run_details": "prev=[offline_collect_halfcheetah_v4_mix_realistic_7p5m], change=[7500 episodes: random 750, bang-bang 1125 (hold 1, 4, 16 steps), expert + noise 1125 (std 0.2, 0.4, 0.8), past runs 3000 (as entry 55), and 1500 expert episodes with actions times a scale in [0.2, 0.8] that changes every 100 steps, plus noise std 0.1]",
        **OFFLINE_COLLECT,
        **OFFLINE_DATA_BANGBANG,
    },
    # 63-64 pretrain, 65-66 RL no rehearsal, 67-68 RL no rehearsal + head re-anchor (seeds 42, 4).
    *bangbang_data_queue_ablations(),
    # 69-70 joint finetuning from scratch, 71-72 joint finetuning from the pretrain a2_scaled40_s4 (seeds 42, 4).
    *joint_finetune_queue_ablations(),
    # 73 A2 with grouped loss normalization, 74 A2 with separate loss normalization (seed 4).
    *loss_normalization_queue_ablations(),
    # 75 A2 with joint finetuning, 76 A2 with joint finetuning and grouped loss normalization (seed 4).
    *joint_finetune_a2_queue_ablations(),
    # 77: entry 73 (A2 with grouped loss normalization) with seed 42.
    {
        **loss_normalization_queue_ablations()[0],
        "run_name": "a2_grouped_loss_normalization_s42",
        "run_details": "prev=[audit_a2_next_state_reward_target_local_s42], change=[RMS loss normalization with live statistics; one statistic for the flow loss and the bootstrap loss together]",
        "seed": 42,
    },
    # 78-97: entries 73 / 77 (A2 with grouped loss normalization) with one change each, seeds 4 and 42.
    *grouped_normalization_queue_ablations(),
    # 98-99: the base (entry 73) with seeds 1 and 2. 100-108: entries 78-97 except PMPO, seed 1.
    *grouped_normalization_more_seeds_queue_ablations(),
    # 109-111 constant closer PPO learning rate 3e-4, 112-114 constant closer PPO learning rate 3.4e-4 (seeds 4, 42, 1).
    *grouped_normalization_learning_rate_queue_ablations(),
    # 115-117: no AttnPool, depth 5 in place of the wider feedforward (seeds 4, 42, 1).
    *grouped_normalization_deeper_queue_ablations(),
    # 118-135: stage 1 of the combinations of the significant arms (six pairs), 175 loops (seeds 4, 42, 1).
    *grouped_normalization_combination_queue_ablations(),
    # 136-138: stage 2 of the combinations (symexp + warmup15 + block2), 175 loops (seeds 4, 42, 1).
    *grouped_normalization_combination_stage2_queue_ablations(),
    # 139-141: symexp + warmup15 + depth5, 142-144: symexp + warmup15 + wm_pretrain640 (seeds 4, 42, 1, 175 loops).
    *grouped_normalization_combination_stage2b_queue_ablations(),
    # 145-162: sweeps of symexp + warmup15 (COMBINATION_SWEEPS), seeds 4, 42, 1, 175 loops.
    *grouped_normalization_combination_sweep_queue_ablations(),
    # 163-165: symexp + offline pretraining on the precollected scaled40 data, 175 online loops (seeds 4, 42, 1).
    *grouped_normalization_offline_pretrain_queue_ablations(),
    # 166-168: symexp + collection and pretraining on the data of the 18 stage-1 runs (seeds 4, 42, 1, 175 loops).
    *grouped_normalization_own_runs_pretrain_queue_ablations(),
    # 169-171 flipwm25, 172-174 flipprompt25: flip-onset sampling on symexp + warmup15 (seeds 4, 42, 1, 175 loops).
    *grouped_normalization_flip_onset_queue_ablations(),
    # 175-177: 166-168 with flipwm25 in the pretrain and the rl stage (seeds 4, 42, 1, 175 loops).
    *grouped_normalization_own_runs_flip_pretrain_queue_ablations(),
    # 178-180: 163-165 with rw4 + depth5 in the pretrain and the rl stage (seeds 4, 42, 1, 175 loops).
    *grouped_normalization_rw4_depth5_pretrain_queue_ablations(),
    # 181-183 kvlb1000, 184-186 smallterm: the opt-in model flags on 175-177 (seeds 4, 42, 1, 175 loops).
    *grouped_normalization_own_runs_flip_pretrain_model_flag_queue_ablations(),
    # 187-189 kvlb200, 190-192 kvlb200smallterm: KV cache lookback 200 alone and with the small terminal head on 175-177
    # (seeds 4, 42, 1, 175 loops).
    *grouped_normalization_own_runs_flip_pretrain_kv_lookback_200_queue_ablations(),
    # 193-195 agenttok, 196-198 agenttokrl001, 199-201 agenttokrl01, 202-204 agenttokrl03: agent-token readout and the
    # RL policy loss in the world-model step on 190-192, kvlb200smallterm (seeds 4, 42, 1, 175 loops). They were entries
    # 187-198 (on 175-177) until 2026-10-12; no run of them finished.
    *grouped_normalization_own_runs_flip_pretrain_agent_token_rl_queue_ablations(),
    # 205-207 agenttokshared, 208-210 jointrl001, 211-213 jointrl01, 214-216 jointrl04, 217-219 jointrl1, 220-222
    # jointrl01hlr: the joint step with the agent-token readout, on the pretrains of 193-195 (seeds 4, 42, 1, 175 loops).
    *grouped_normalization_own_runs_flip_pretrain_joint_agent_token_queue_ablations(),
    # 223-225 jointrl0, 226-228 jointppo0, 229-231 jointppo001, 232-234 jointppo01, 235-237 slotshared001: fixes of the
    # joint step and the slot design on the pretrains of 193-195 (seeds 4, 42, 1, 175 loops).
    *grouped_normalization_own_runs_flip_pretrain_joint_fix_queue_ablations(),
    # 238-240 jointppo0ra, 241-243 jointppo001ra, 244-246 agenttokra, 247-249 slotshared001ra: the head re-anchor on
    # the joint step, the two-phase control and the slot, on the pretrains of 193-195 (seeds 4, 42, 1, 175 loops).
    *grouped_normalization_own_runs_flip_pretrain_reanchor_queue_ablations(),
    # 250-251 kvlb200smallterm_tokfix (seeds 4, 42), 252-253 agenttokrl001_tokfix (seeds 4, 42): the tokenizer fix
    # (20,000 steps, cosine decay, balanced data mix); each run trains its own pretrain.
    *grouped_normalization_own_runs_flip_pretrain_tokenizer_fix_queue_ablations(),
]

# Slurm CLI runs require -N and override this programmatic default.
# This list uses 1-based indexing; it is not 0 indexed.
ABLATIONS_TO_RUN = list(range(1, len(ABLATIONS) + 1))

RUN_PARALLEL = True
MAX_PARALLEL_WORKERS: int | None = 2
CONTINUE_ON_ERROR = False
PARALLEL_LAUNCH_DELAY_SECONDS = 60


@dataclass(frozen = True)
class AblationSpec:
    index: int
    label: str
    overrides: dict[str, Any]
    train_kwargs: dict[str, Any]
    run_log_dir: str | None


@dataclass(frozen = True)
class AblationResult:
    index: int
    label: str
    run_name: str
    run_log_dir: str | None
    success: bool
    elapsed_seconds: float
    error: str | None = None


def ensure_repo_root_on_sys_path():
    repo_root = str(REPO_ROOT)

    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def load_baseline_config(config_path: str | Path = DEFAULT_CONFIG_PATH):
    config_path = Path(config_path)

    with config_path.open("r", encoding = "utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise TypeError(f"{config_path} must contain a top-level mapping")

    if "run_name" not in config:
        raise KeyError(f"{config_path} must define run_name")

    if "run_details" not in config:
        raise KeyError(f"{config_path} must define run_details")

    return config


def format_override_value(value: Any):
    if isinstance(value, str):
        return value

    return json.dumps(value, sort_keys = True, separators = (",", ":"))


def format_ablation_label(overrides: Mapping[str, Any]):
    if len(overrides) == 0:
        return "baseline"

    return ",".join(
        f"{key}={format_override_value(value)}"
        for key, value in sorted(overrides.items())
    )


def slugify(value: str):
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "baseline"


def build_run_details(
    base_details: str,
    *,
    index: int,
    label: str,
    overrides: Mapping[str, Any],
    extra_details: str | None,
):
    parts = [str(base_details).strip()] if str(base_details).strip() else []
    parts.append(f"Ablation {index}: {label}")
    parts.append(f"Overrides: {json.dumps(dict(sorted(overrides.items())), sort_keys = True)}")

    if extra_details:
        parts.append(str(extra_details).strip())

    return "\n\n".join(part for part in parts if part)


def plan_run_log_dir(train_kwargs: Mapping[str, Any]):
    checkpoint_path = train_kwargs.get("checkpoint_path")

    if checkpoint_path:
        return None

    return str(GYM_LOG_ROOT / str(train_kwargs["run_name"]))


def plan_checkpoint_folder(train_kwargs: dict[str, Any]):
    """Give each new ablation run its own checkpoint directory."""
    if train_kwargs.get("checkpoint_path"):
        return

    checkpoint_folder = train_kwargs.get("checkpoint_folder")

    if checkpoint_folder is None:
        return

    run_name = str(train_kwargs["run_name"])
    checkpoint_folder = Path(str(checkpoint_folder))

    if checkpoint_folder.name != run_name:
        checkpoint_folder /= run_name

    train_kwargs["checkpoint_folder"] = str(checkpoint_folder)


def merge_small_model_config(
    baseline: Mapping[str, Any],
    overrides: Mapping[str, Any],
):
    """Merge state-only overrides without dropping unspecified defaults."""

    merged = deepcopy(dict(baseline))
    merged.update(overrides)

    if "small_model" not in overrides:
        return merged

    base_small_model = baseline.get("small_model")
    override_small_model = overrides["small_model"]
    if not isinstance(base_small_model, Mapping):
        raise TypeError("baseline small_model must be a mapping")
    if not isinstance(override_small_model, Mapping):
        raise TypeError("small_model ablation override must be a mapping")

    merged_small_model = deepcopy(dict(base_small_model))
    merged_small_model.update(override_small_model)
    merged["small_model"] = merged_small_model
    return merged


def select_ablation_indices(
    ablation_count: int,
    indices: Sequence[int] | None = None,
):
    selected_indices = list(ABLATIONS_TO_RUN if indices is None else indices)

    if len(selected_indices) == 0:
        raise ValueError("ABLATIONS_TO_RUN must contain at least one ablation index")

    seen = set()

    for index in selected_indices:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(
                "ABLATIONS_TO_RUN must contain only integer indices; indices are 1-based"
            )

        if index < 1 or index > ablation_count:
            raise IndexError(
                f"ablation index {index} is out of range; choose values from 1 to "
                f"{ablation_count} (indices are 1-based)"
            )

        if index in seen:
            raise ValueError(f"ablation index {index} was selected more than once")

        seen.add(index)

    return selected_indices


def normalize_cli_ablation_indices(
    ablation_indices: int | Sequence[int],
):
    if isinstance(ablation_indices, int):
        return [ablation_indices]

    if isinstance(ablation_indices, (str, bytes)):
        raise TypeError("ablation_indices must be an integer or a sequence of integers")

    return list(ablation_indices)


def build_ablation_specs(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    ablations: Sequence[Mapping[str, Any]] | None = None,
    ablation_indices: Sequence[int] | None = None,
):
    baseline = load_baseline_config(config_path)
    raw_ablations = list(ABLATIONS if ablations is None else ablations)

    if len(raw_ablations) == 0:
        raise ValueError(
            "ABLATIONS is empty. Edit train/ablations.py and add one or more override dicts, "
            "for example ABLATIONS = [{'depth': 8}]"
        )

    if ablations is not None and ablation_indices is None:
        selected_indices = list(range(1, len(raw_ablations) + 1))
    else:
        selected_indices = select_ablation_indices(len(raw_ablations), ablation_indices)
    allowed_override_keys = set(baseline.keys())
    specs = []

    for index in selected_indices:
        ablation = raw_ablations[index - 1]

        if not isinstance(ablation, Mapping):
            raise TypeError(f"ablation #{index} must be a mapping, got {type(ablation).__name__}")

        overrides = dict(ablation)
        unknown_keys = sorted(set(overrides) - allowed_override_keys)

        if unknown_keys:
            raise KeyError(
                f"ablation #{index} contains keys not present in {config_path}: {unknown_keys}"
            )

        run_name = str(overrides.get("run_name", "")).strip()
        run_details = str(overrides.get("run_details", "")).strip()

        if not run_name:
            raise KeyError(
                f"ablation #{index} must define a non-empty run_name because each ablation needs "
                "its own log directory under runs/gym_imagination/<run_name>"
            )

        if not run_details:
            raise KeyError(f"ablation #{index} must define a non-empty run_details")

        label = run_name

        train_kwargs = merge_small_model_config(baseline, overrides)
        train_kwargs["run_name"] = run_name
        train_kwargs["log_dir"] = str(GYM_LOG_ROOT)
        plan_checkpoint_folder(train_kwargs)
        train_kwargs["run_details"] = build_run_details(
            str(baseline["run_details"]),
            index = index,
            label = label,
            overrides = {key: value for key, value in overrides.items() if key not in ("run_name", "run_details")},
            extra_details = run_details,
        )

        specs.append(
            AblationSpec(
                index = index,
                label = label,
                overrides = dict(overrides),
                train_kwargs = train_kwargs,
                run_log_dir = plan_run_log_dir(train_kwargs),
            )
        )

    return specs


def run_train_gym(train_kwargs: Mapping[str, Any]):
    ensure_repo_root_on_sys_path()
    from train.train_gym import main as train_gym_main

    return train_gym_main(**dict(train_kwargs))


def log_ablation_error(
    spec: AblationSpec,
    error: str,
    *,
    source: str,
):
    """Write a failure traceback before the launcher waits for other runs."""
    print(
        f"\n[ablation {spec.index}] {source} for {spec.train_kwargs['run_name']}:\n{error}",
        file = sys.stderr,
        flush = True,
    )


def failed_ablation_result(
    spec: AblationSpec,
    *,
    elapsed_seconds: float,
    error: str,
    source: str,
):
    log_ablation_error(spec, error, source = source)

    return AblationResult(
        index = spec.index,
        label = spec.label,
        run_name = str(spec.train_kwargs["run_name"]),
        run_log_dir = spec.run_log_dir,
        success = False,
        elapsed_seconds = elapsed_seconds,
        error = error,
    )


def execute_ablation(spec: AblationSpec, *, dry_run: bool):
    start = perf_counter()

    print(f"[ablation {spec.index}] starting {spec.label}", flush = True)
    print(f"[ablation {spec.index}] run_name={spec.train_kwargs['run_name']}", flush = True)

    if spec.run_log_dir:
        print(f"[ablation {spec.index}] log_dir={spec.run_log_dir}", flush = True)

    if spec.overrides:
        print(
            f"[ablation {spec.index}] overrides={json.dumps(spec.overrides, sort_keys = True)}",
            flush = True,
        )
    else:
        print(f"[ablation {spec.index}] overrides={{}}", flush = True)

    try:
        if not dry_run:
            run_train_gym(spec.train_kwargs)
    except Exception:
        return failed_ablation_result(
            spec,
            elapsed_seconds = perf_counter() - start,
            error = traceback.format_exc(),
            source = "worker error",
        )

    return AblationResult(
        index = spec.index,
        label = spec.label,
        run_name = str(spec.train_kwargs["run_name"]),
        run_log_dir = spec.run_log_dir,
        success = True,
        elapsed_seconds = perf_counter() - start,
    )


def execute_ablation_in_subprocess(spec: AblationSpec, dry_run: bool):
    return execute_ablation(spec, dry_run = dry_run)


def print_planned_ablations(specs: Sequence[AblationSpec], *, config_path: str | Path):
    print(f"loaded {len(specs)} ablation(s) from {Path(config_path)}")

    for spec in specs:
        print(
            f"- [{spec.index}] {spec.label} -> run_name={spec.train_kwargs['run_name']} "
            f"overrides={json.dumps(spec.overrides, sort_keys = True)}"
        )


def summarize_results(results: Sequence[AblationResult]):
    ordered = sorted(results, key = lambda result: result.index)
    success_count = sum(result.success for result in ordered)

    print("\nablation summary:")
    for result in ordered:
        status = "ok" if result.success else "failed"
        print(
            f"- [{result.index}] {status} {result.run_name} "
            f"elapsed={result.elapsed_seconds:.2f}s"
        )

    print(f"completed {success_count}/{len(ordered)} ablation(s) successfully")


def writes_offline_artifacts(train_kwargs: Mapping[str, Any]):
    stages = train_kwargs.get("offline_stages") or ()
    stages = [stages] if isinstance(stages, str) else stages
    return bool(train_kwargs.get("offline_pretraining")) and any(stage in ("collect", "pretrain") for stage in stages)


def run_ablation_specs(
    specs: Sequence[AblationSpec],
    *,
    parallel: bool,
    max_workers: int | None,
    continue_on_error: bool,
    dry_run: bool,
):
    if len(specs) == 0:
        return []

    if max_workers is not None and max_workers < 1:
        raise ValueError("max_workers must be at least 1 when provided")

    # The collect and pretrain offline stages write shared folders under runs/.
    offline_writers = [spec.label for spec in specs if writes_offline_artifacts(spec.train_kwargs)]
    if parallel and len(specs) > 1 and offline_writers:
        raise ValueError(
            f"run the offline collect and pretrain stages alone, not in parallel with other runs: {offline_writers}"
        )

    results: list[AblationResult] = []

    def run_one_in_fresh_process(spec: AblationSpec):
        with ProcessPoolExecutor(max_workers = 1, mp_context = get_context("spawn")) as executor:
            future = executor.submit(execute_ablation_in_subprocess, spec, dry_run)

            try:
                return future.result()
            except Exception:
                return failed_ablation_result(
                    spec,
                    elapsed_seconds = 0.,
                    error = traceback.format_exc(),
                    source = "worker process error",
                )

    if not parallel or len(specs) == 1:
        for spec in specs:
            result = execute_ablation(spec, dry_run = dry_run) if dry_run else run_one_in_fresh_process(spec)
            results.append(result)

            if not result.success and not continue_on_error:
                break

        return results

    worker_count = max_workers or min(len(specs), os.cpu_count() or 1)
    mp_context = get_context("spawn")

    # Queued runs wait for a free worker. Start each run in a fresh process, so no CUDA or compile state carries over.
    fresh_process_kwargs = dict(max_tasks_per_child = 1) if sys.version_info >= (3, 11) else dict()

    with ProcessPoolExecutor(max_workers = worker_count, mp_context = mp_context, **fresh_process_kwargs) as executor:
        future_to_spec = {}

        for index, spec in enumerate(specs):
            if index > 0:
                sleep(PARALLEL_LAUNCH_DELAY_SECONDS)

            future_to_spec[executor.submit(execute_ablation_in_subprocess, spec, dry_run)] = spec

        for future in as_completed(future_to_spec):
            spec = future_to_spec[future]

            try:
                result = future.result()
            except Exception:
                result = failed_ablation_result(
                    spec,
                    elapsed_seconds = 0.,
                    error = traceback.format_exc(),
                    source = "worker process error",
                )

            results.append(result)

    return results


def main(
    ablation_indices: int | Sequence[int],
    config_path: str = str(DEFAULT_CONFIG_PATH),
    parallel: bool = RUN_PARALLEL,
    max_workers: int | None = MAX_PARALLEL_WORKERS,
    continue_on_error: bool = CONTINUE_ON_ERROR,
    dry_run: bool = False,
):
    specs = build_ablation_specs(
        config_path = config_path,
        ablation_indices = normalize_cli_ablation_indices(ablation_indices),
    )
    print_planned_ablations(specs, config_path = config_path)

    results = run_ablation_specs(
        specs,
        parallel = parallel,
        max_workers = max_workers,
        continue_on_error = continue_on_error,
        dry_run = dry_run,
    )

    summarize_results(results)

    failures = [result for result in results if not result.success]
    if failures:
        first_failure = failures[0]

        if first_failure.error:
            print("\nfirst failure traceback:\n")
            print(first_failure.error)

        raise SystemExit(1)

    return [asdict(result) for result in sorted(results, key = lambda result: result.index)]


if __name__ == "__main__":
    fire.Fire(main)
