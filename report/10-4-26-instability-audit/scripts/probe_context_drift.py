"""Checkpoint probe: context length and world-model drift of the acting policy on real states.

For one run and one checkpoint (the state after loop L - 1, so the policy of the loop-L episodes):
  1. The acting policy on 32 real episodes of the replay, with the full cached context of interact_with_env
     (up to 1000 frames). The probe reproduces the stored log-probs (checked: |diff| < 0.01 nats).
  2. The same policy with a short context: only the last W frames (W = 8, 40, 200), with a new cache.
     PPO trains the policy on dreams with an 8-frame real prompt and 32 dream steps (contexts of 8 to 40 frames).
     The world model trains on windows of 200 frames.
  3. World-model drift: one loop of world-model training (13 steps, batch 128 x 200 frames, the run's optimizer
     state) on a scratch replay of 512 episodes of the run (latents from the frozen tokenizer). The policy heads
     do not change. The probe measures KL(before || after) of the acting policy on the same real states.
     The run's logged cleanrl_updated_ppo/approx_kl is the PPO change of one loop (on dream states), for scale.

Usage: python probe_context_drift.py RUN CHECKPOINT_NAME EPISODE_LOOP OUT_CSV
Writes one row for each evaluation time and measure.
"""

import copy
import csv
import pickle
import sys
import tempfile
import time

import numpy as np
import torch
from memmap_replay_buffer.replay_buffer import ReplayBuffer

from probe_common import (STEPS, acting_policy, beta, build, encode, entropy, kl, log_prob, replay_arrays, run_dir)
from train import train_gym
from train.compile_runtime import WorldModelTrainingLoss

EVAL_T = (8, 40, 100, 200, 400, 600, 800, 999)
# The first three jobs also used W = 8 and 200. Their context KL was below 0.01 (W = 200) and the windows
# of 8 and 200 frames took about half of the policy time, so the later jobs use the dream-like W = 40 only.
WINDOWS = (40,)
NUM_EPISODES = 32
SCRATCH_EPISODES = 512
# Gradient accumulation in micro-batches of 8 windows (the run used 32) gives the same gradient with less CPU memory.
# The loss-normalization EMA updates from the first micro-batch only, so it sees 8 rows, not 32.
PROBE_MICRO_BATCH = 8
WM_STEPS = 13


def scratch_replay(run, tokenizer, folder, rng):
    source = run_dir(run) / "replay_buffer"
    config = pickle.load(open(source / "metadata.pkl", "rb"))
    count = int(np.load(source / "num_episodes.state.npy"))
    episodes = np.sort(rng.choice(count, size = min(SCRATCH_EPISODES, count), replace = False))
    replay = ReplayBuffer(folder, max_episodes = len(episodes), max_timesteps = config["max_timesteps"], fields = config["fields"],
                          meta_fields = config["meta_fields"], circular = True)
    data_fields = [f for f in config["fields"] if f != "latents"]
    arrays = {f: np.load(source / f"{f}.data.npy", mmap_mode = "r") for f in data_fields}
    metas = {f: np.load(source / f"{f}.data.meta.npy", mmap_mode = "r") for f in config["meta_fields"]}
    lens = np.load(source / "episode_lens.data.meta.npy", mmap_mode = "r")
    for e in episodes:
        n = int(lens[e])
        data = {f: torch.from_numpy(np.asarray(arrays[f][e, :n])) for f in data_fields}
        latents = encode(tokenizer, np.asarray(arrays["critic_state"][e, :n])[None])[0]  # (n, 1, 4, 32)
        data["latents"] = latents
        meta = {f: torch.from_numpy(np.asarray(metas[f][e])) for f in config["meta_fields"]}
        replay.store_episode(**data, **meta)
    return replay


def policies(world_model, latents, actions, rewards):
    full = acting_policy(world_model, latents, actions, rewards, keep = set(EVAL_T))
    window = {}
    for w in WINDOWS:
        for t in EVAL_T:
            if t + 1 >= w:
                window[(w, t)] = acting_policy(world_model, latents, actions, rewards, start = t + 1 - w, stop = t + 1, keep = {t})[t]
    return full, window


def main():
    run, checkpoint_name, episode_loop, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    k, tokenizer, world_model, package = build(run, run_dir(run) / "checkpoints" / checkpoint_name)
    episodes = np.arange(episode_loop * 64, episode_loop * 64 + NUM_EPISODES)
    d = replay_arrays(run, episodes)
    latents = encode(tokenizer, d["critic_state"])
    actions = torch.from_numpy(d["actions_continuous"]).float()
    rewards = torch.from_numpy(d["rewards"]).float()
    flipped = np.abs(np.angle(np.exp(1j * d["critic_state"][..., 1]))) > np.pi / 2  # (b, T)

    start = time.time()
    full, window = policies(world_model, latents, actions, rewards)
    check = max(float(np.abs(log_prob(beta(world_model, full[t][:, 0]), actions[:, t]).numpy() - d["log_probs_continuous"][:, t].sum(-1)).max()) for t in EVAL_T)
    print(f"{run} {checkpoint_name}: policies {time.time() - start:.0f} s; max |log-prob - stored| {check:.4f} (exact only for the collecting checkpoint)")

    # one loop of world-model training, the policy heads fixed
    trained = copy.deepcopy(world_model)
    trained.apply_loss_normalization = True
    optimizer = train_gym.make_optimizer(trained.world_model_parameters(), lr = k["world_model_learning_rate"], weight_decay = k["optimizer_weight_decay"],
                                         use_muon = k["use_muon_optimizer"], muon_params = trained.muon_parameters())
    optimizer.load_state_dict(package["world_optimizer"])
    heads_before = {n: p.detach().clone() for n, p in trained.named_parameters() if any(p is q for q in trained.agent_parameters())}
    with tempfile.TemporaryDirectory(dir = sys.argv[5] if len(sys.argv) > 5 else None) as folder:
        replay = scratch_replay(run, tokenizer, folder, rng)
        start = time.time()
        trained.train()
        train_gym.train_world_model(
            trained, optimizer, replay, world_model_loss_fn = WorldModelTrainingLoss(trained), world_model_loss_timing = None,
            world_model_step_timing = None, steps = WM_STEPS, batch_size = k["world_model_batch_size"],
            sequence_length = k["world_model_train_sequence_length"], static_batch_shape = False, max_grad_norm = k["max_grad_norm"],
            global_step = 0, writer = None, training_profiler = None, shortcut_train_mode = "combined", desc = "probe world model",
            micro_batch_size = PROBE_MICRO_BATCH, autocast_bf16 = False)
        del replay
        trained.eval()
        print(f"  world-model loop {time.time() - start:.0f} s")
    for n, p in trained.named_parameters():
        if n in heads_before:
            assert torch.equal(p, heads_before[n]), f"head {n} changed"
    full_after, window_after = policies(trained, latents, actions, rewards)

    rows = []
    for t in EVAL_T:
        p_full = beta(world_model, full[t][:, 0])
        flip_t = torch.from_numpy(flipped[:, t])
        base = {"run": run, "checkpoint": checkpoint_name, "episode_loop": episode_loop, "t": t, "flipped_frac": float(flip_t.float().mean())}
        measures = {"entropy_full": entropy(p_full), "drift_kl_full": kl(p_full, beta(trained, full_after[t][:, 0]))}
        for w in WINDOWS:
            if (w, t) in window:
                p_w = beta(world_model, window[(w, t)][:, 0])
                measures[f"entropy_w{w}"] = entropy(p_w)
                measures[f"context_kl_w{w}"] = kl(p_full, p_w)
                measures[f"drift_kl_w{w}"] = kl(p_w, beta(trained, window_after[(w, t)][:, 0]))
        for name, values in measures.items():
            upright = values[~flip_t]
            rows.append({**base, "measure": name, "mean": float(values.mean()), "median": float(values.median()),
                         "upright_mean": float(upright.mean()) if len(upright) else float("nan")})
    with open(out_path, "w", newline = "") as file:
        writer = csv.DictWriter(file, fieldnames = list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
