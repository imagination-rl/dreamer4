"""Which actions does each world model reward? Open-loop dreams against MuJoCo for four action sources.

From the same online start states (repro s4 replay, loops 45-54: upright, about 1.3 m/s), 32 steps of each source:
  online     the replay actions (the gait of the online policy)
  expert     the CleanRL TD3 expert, closed loop in MuJoCo (its actions then go open loop into the dream)
  bang-bang  random actions at the action bounds
  zero       all actions 0
Each model gets the 8 real prompt frames (encoded by its own tokenizer) and then the source actions. The dream reward
per step is compared with the MuJoCo reward per step of the same actions. The question: does the model rank the
sources as MuJoCo does, and which source does it favor?

Output: report/9-30-26-pretrain-rootcause/data/probe_action_sources.csv
Usage:  python report/9-30-26-pretrain-rootcause/scripts/probe_action_sources.py [--cuda] [--model 'label|run|checkpoint' ...] [--out csv]
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "reports" / "9-28-26" / "scripts"))

import numpy as np
import pandas as pd
import torch
from memmap_replay_buffer import ReplayBuffer

from probe_common import build, seed_all
from train import offline_pretraining as off
from train.audit_ablations import AuditDiagnostics, _numpy, env_to_native_actions, native_to_env_actions

G = REPO / "runs" / "gym_imagination"
P_DIR = REPO / "runs" / "offline_pretrain"
REPLAY = G / "audit_a2_next_state_reward_target_local_s4" / "checkpoints" / "replay_loop_100"
EXPERT = next((REPO / "runs" / "offline_data" / "experts").rglob("*.cleanrl_model"))
MODELS = [
    ("pretrained s4 step 12000 (RL start)", "audit_a2_next_state_reward_target2", P_DIR / "a2_next_state_reward_s4" / "pretrained_step_12000.pt"),
    ("pretrained s42 step 12000 (RL start)", "audit_a2_next_state_reward_target", P_DIR / "a2_next_state_reward_s42" / "pretrained_step_12000.pt"),
    ("rehearsal 0.5 s4 loop 50", "audit_a2_next_state_reward_target2", G / "offline_pretrained_a2_rehearsal_s4" / "checkpoints" / "loop_50.pt"),
    ("BC s4 loop 50 (pretrained, no rehearsal)", "audit_a2_next_state_reward_target2", G / "offline_pretrained_a2_bc_no_rehearsal_s4" / "checkpoints" / "loop_50.pt"),
    ("repro s4 loop 50", "audit_a2_next_state_reward_target2", G / "audit_a2_next_state_reward_target_local_s4" / "checkpoints" / "loop_50.pt"),
    ("H100 s4 loop 50", "audit_a2_next_state_reward_target2", G / "audit_a2_next_state_reward_target2" / "checkpoints" / "loop_50.pt"),
]
PROMPT, HORIZON, WINDOWS, LOOPS = 8, 32, 256, (45, 55)
ACTION_RANGE = (-1., 1.)
OUT = HERE.parent / "data" / "probe_action_sources.csv"


def expert_rollouts(audit: AuditDiagnostics, actor, start_obs: np.ndarray):
    env = audit.sim()
    actions = np.zeros((len(start_obs), HORIZON, 6), dtype = np.float32)
    for b, obs in enumerate(start_obs):
        audit.restore(env, obs)
        current = obs.astype(np.float32)
        for t in range(HORIZON):
            actions[b, t] = off.expert_actions(actor, current[None], 6)[0]
            current, *_ = env.step(actions[b, t])
            current = current.astype(np.float32)
    return actions



def models_from_argv(default):
    """--model 'label|run|checkpoint' (repeatable) replaces MODELS. run gives the architecture config."""

    given = [sys.argv[i + 1] for i, arg in enumerate(sys.argv) if arg == "--model"]
    return [tuple(value.split("|")) for value in given] if given else default


def out_from_argv(default):
    return Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else default


def main():
    device = torch.device("cuda" if "--cuda" in sys.argv else "cpu")
    rng = np.random.default_rng(0)
    replay = ReplayBuffer.from_folder(REPLAY, read_only = True)
    lengths = np.asarray(replay.episode_lens)
    episodes = rng.integers(64 * LOOPS[0], 64 * LOOPS[1], size = WINDOWS)
    starts = np.array([rng.integers(0, lengths[e] - PROMPT - HORIZON) for e in episodes])
    obs = np.stack([replay.data["critic_state"][e, s:s + PROMPT] for e, s in zip(episodes, starts)]).astype(np.float32)
    native = np.stack([replay.data["actions_continuous"][e, s:s + PROMPT - 1 + HORIZON] for e, s in zip(episodes, starts)]).astype(np.float32)
    start_obs = obs[:, -1]

    audit = AuditDiagnostics(
        env_name = "HalfCheetah-v4", obs_dim = 17, action_range = ACTION_RANGE, prompt_length = PROMPT, horizon = HORIZON,
        generate_steps = 4, use_time_cache = True, old_reward_shifting = False, every_env_steps = 1, seed = 0, device = device,
    )
    actor = off.load_expert_actor(EXPERT, obs_dim = 17, action_dim = 6)
    margin = 1e-3
    sources = {
        "online": native_to_env_actions(native[:, PROMPT - 1:], ACTION_RANGE),
        "expert": expert_rollouts(audit, actor, start_obs),
        "bang-bang": np.sign(rng.uniform(-1, 1, size = (WINDOWS, HORIZON, 6))).astype(np.float32) * (1 - margin),
        "zero": np.zeros((WINDOWS, HORIZON, 6), dtype = np.float32),
    }
    sources = {name: np.clip(actions, -1 + margin, 1 - margin) for name, actions in sources.items()}
    true = {name: audit.resimulate(start_obs, actions)[0] for name, actions in sources.items()}
    dream_slice, true_slice = audit.aligned_reward_slices(HORIZON)

    rows = []
    for label, run, checkpoint in models_from_argv(MODELS):
        seed_all(0)
        _, tokenizer, world_model, _, _ = build(run, str(checkpoint), device = device)
        with torch.no_grad():
            prompt_latents = tokenizer.encode(torch.from_numpy(obs).to(device)).float()
        for name, env_actions in sources.items():
            # Prompt actions 0..P-2 are real; action P-1 is the first action of the source (as in the audit).
            actions = np.concatenate([native[:, :PROMPT - 1], env_to_native_actions(env_actions, ACTION_RANGE)], axis = 1)
            with torch.no_grad():
                generated = world_model.generate(
                    PROMPT + HORIZON, num_steps = 4, batch_size = WINDOWS, return_decoded_video = False,
                    return_agent_actions = False, return_rewards_per_frame = True, return_terminals = False,
                    use_time_cache = True, prompt_latents = prompt_latents,
                    prompt_continuous_actions = torch.from_numpy(actions).to(device),
                )
            dream = _numpy(generated.rewards)[:, dream_slice].mean(-1)
            real = true[name][:, true_slice].mean(-1)
            rows.append(dict(model = label, source = name, dream_reward_per_step = float(dream.mean()),
                             true_reward_per_step = float(real.mean()), gap = float((dream - real).mean()),
                             corr = float(np.corrcoef(dream, real)[0, 1])))
            print(rows[-1], flush = True)

    audit.close()
    table = pd.DataFrame(rows)
    out = out_from_argv(OUT)
    out.parent.mkdir(parents = True, exist_ok = True)
    table.to_csv(out, index = False)
    pd.set_option("display.width", 250)
    print(table.round(3).to_string(index = False))
    # Dream and true reward of each source relative to the online actions of the same model.
    wide = table.pivot(index = "model", columns = "source", values = ["dream_reward_per_step", "true_reward_per_step"])
    for kind in ("dream_reward_per_step", "true_reward_per_step"):
        print("\n", kind, "minus online")
        print(wide[kind].sub(wide[kind]["online"], axis = 0).round(3).to_string())


if __name__ == "__main__":
    main()
