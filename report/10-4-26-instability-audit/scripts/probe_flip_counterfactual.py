"""Simulator counterfactual of real flips: is a flip bad luck (exploration noise) or a property of the gait?

The HalfCheetah observation holds the full simulator state except the x position, which has no effect on the dynamics.
So the probe restores the exact state of a recorded episode at a branch step t_s before the flip onset t0, and runs:
  recorded  the recorded actions (a check of the restore: the branch must flip at about t0)
  sample    16 branches with new samples of the policy's actions (the policy's own noise)
  mean      1 branch with the mean of the Beta policy (no noise)
The policy reads the recorded episode up to t_s (teacher forcing, as in probe_common.acting_policy) and then the branch.
Each branch runs to t0 + 60. The probe also records the analytic entropy and the surprise -log pi(a|s) of the
recorded actions on the steps before t0, to split the high pre-onset surprise into "uncertain policy" and "improbable action".

Usage: python probe_flip_counterfactual.py RUN CHECKPOINT_NAME FIRST_LOOP-LAST_LOOP OUT_CSV [MAX_ONSETS]
The onsets come from the episodes of these loops, newest first. Only the episodes of the loop after the checkpoint
come from the checkpoint's own policy; the episodes of the loops before it come from a policy a few updates older.
"""

import copy
import csv
import sys

import gymnasium as gym
import numpy as np
import torch

from probe_common import beta, build, encode, entropy, log_prob, policy_embed_of, replay_arrays, run_dir

BRANCHES = 16
OFFSETS = (100, 30, 10)
AFTER = 60
HOLD = 50


def flipped(obs):
    return np.abs(np.angle(np.exp(1j * obs[..., 1]))) > np.pi / 2


def onset(flip):
    run_length = np.convolve(flip.astype(np.int32), np.ones(HOLD, dtype = np.int32), "valid")
    hits = np.nonzero(run_length == HOLD)[0]
    return int(hits[0]) if len(hits) else -1


class Actor:
    """The cached acting loop of interact_with_env, one step at a time, for a batch of branches."""

    def __init__(self, world_model, tokenizer):
        self.world_model, self.tokenizer = world_model, tokenizer
        self.time_cache, self.last_action, self.last_reward = None, None, None

    @torch.no_grad()
    def policy(self, obs):
        latents = encode(self.tokenizer, obs[:, None].astype(np.float32))
        wm = self.world_model
        _, (embeds, self.time_cache) = wm(
            latents = latents, signal_levels = wm.max_steps - 1, step_sizes = wm.max_steps // 4,
            rewards = self.last_reward, continuous_actions = self.last_action, time_cache = self.time_cache,
            latent_is_noised = True, latent_has_view_dim = True, return_pred_only = True, return_intermediates = True)
        return beta(wm, policy_embed_of(wm, embeds)[:, 0])

    def record(self, native_action, reward):
        self.last_action = torch.as_tensor(native_action, dtype = torch.float32)[:, None]
        self.last_reward = torch.as_tensor(reward, dtype = torch.float32)[:, None]


def run_branch(actor, d, e, t_s, t_end, rng):
    """From a copy of the teacher-forced actor at step t_s. Row 0: the recorded actions; row 1: the mean action; rows 2 ..: new samples."""
    n = 2 + BRANCHES
    states, actions = d["critic_state"][e], d["actions_continuous"][e]
    envs = [gym.make("HalfCheetah-v4") for _ in range(n)]
    for env in envs:
        env.reset(seed = int(rng.integers(1 << 30)))
        env.unwrapped.set_state(np.concatenate([[0.], states[t_s][:8]]).astype(np.float64), states[t_s][8:].astype(np.float64))
    obs = np.repeat(states[t_s][None], n, 0).astype(np.float64)
    for t in range(t_s, t_end):
        dist = actor.policy(obs)
        native = dist.sample().numpy()
        native[0] = actions[t]
        native[1] = dist.mean[1].numpy()
        out = [env.step(np.clip(2. * a - 1., -1., 1.)) for env, a in zip(envs, native)]
        obs = np.stack([o[0] for o in out])
        actor.record(native, np.array([o[1] for o in out], dtype = np.float32))
    for env in envs:
        env.close()
    end = flipped(obs)  # a branch counts as flipped when it is flipped at t0 + AFTER
    return {"recorded": end[:1], "mean": end[1:2], "sample": end[2:]}


def teacher_forced_pass(world_model, tokenizer, d, e, t0, rng, branch_offsets):
    """One teacher-forced pass over frames 0 .. t0 - 1 of a recorded episode, with 2 + BRANCHES identical rows.
    It records the entropy and the surprise of row 0 on the last 250 steps, and at each branch step t0 - offset it runs
    the branches from a copy of the actor. The copy keeps the time cache, so the prefix runs only once."""
    n = 2 + BRANCHES if branch_offsets else 1
    actor = Actor(world_model, tokenizer)
    states, actions, rewards = d["critic_state"][e], d["actions_continuous"][e], d["rewards"][e]
    ent, sur, branches = [], [], {}
    branch_steps = {t0 - offset: offset for offset in branch_offsets}
    for t in range(t0):
        if t in branch_steps:
            branches[branch_steps[t]] = run_branch(copy.deepcopy(actor), d, e, t, t0 + AFTER, rng)
        dist = actor.policy(np.repeat(states[t][None], n, 0))
        if t >= t0 - 250:
            a = torch.from_numpy(actions[t][None]).float()
            ent.append(float(entropy(dist)[0]))
            sur.append(float(-log_prob(dist, a)[0] + 6 * np.log(2.)))
        actor.record(np.repeat(actions[t][None], n, 0), np.repeat(rewards[t:t + 1][None], n, 0)[:, 0])
    return np.array(ent), np.array(sur), branches


def main():
    run, checkpoint, loops, out_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    first_loop, last_loop = (int(x) for x in loops.split("-"))
    max_onsets = int(sys.argv[5]) if len(sys.argv) > 5 else 6
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    _, tokenizer, world_model, _ = build(run, run_dir(run) / "checkpoints" / checkpoint)
    # newest loop first; e indexes the episodes of all the loops
    d = replay_arrays(run, np.concatenate([np.arange(loop * 64, loop * 64 + 64) for loop in range(last_loop, first_loop - 1, -1)]))
    episode_loop_of = lambda e: last_loop - e // 64
    flips = [(e, onset(flipped(d["critic_state"][e]))) for e in range(len(d["critic_state"]))]
    cases = [(e, t0) for e, t0 in flips if t0 >= 300 and t0 + AFTER <= 1000][:max_onsets]
    upright = [e for e, t0 in flips[:64] if t0 < 0 and not flipped(d["critic_state"][e]).any()]
    print(f"{run} {checkpoint} loops {loops}: {len(cases)} onsets, {len(upright)} upright episodes in loop {last_loop}")
    rows = []

    for e, t0 in cases + [(e, 700) for e in upright[:2]]:
        kind = "onset" if (e, t0) in cases else "upright"
        ent, sur, branches = teacher_forced_pass(world_model, tokenizer, d, e, t0, rng, OFFSETS if kind == "onset" else ())
        common = {"run": run, "checkpoint": checkpoint, "loop": episode_loop_of(e), "episode": e % 64, "t0": t0, "kind": kind}
        rows.append({**common, "mode": "entropy", "offset": 0, "flip_frac": float("nan"),
                     "entropy_pre": ent[-30:].mean(), "entropy_base": ent[:100].mean(),
                     "surprise_pre": sur[-30:].mean(), "surprise_base": sur[:100].mean()})
        for offset in OFFSETS if kind == "onset" else ():
            for mode, end_flipped in branches[offset].items():
                rows.append({**common, "mode": mode, "offset": offset, "flip_frac": float(end_flipped.mean()),
                             "entropy_pre": float("nan"), "entropy_base": float("nan"), "surprise_pre": float("nan"), "surprise_base": float("nan")})
            print(f"  episode {e} t0 {t0} offset {offset}: " + ", ".join(f"{r['mode']} {r['flip_frac']:.2f}" for r in rows[-3:]), flush = True)

    with open(out_path, "w", newline = "") as file:
        writer = csv.DictWriter(file, fieldnames = list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
