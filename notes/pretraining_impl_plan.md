# Offline data collection and pretraining: implementation plan

Status: implemented on 2026-09-27. Open questions are in `notes/pretrain_impl_questions.md`. Plan date: 2026-09-26.

## 1. Goal

Add an optional offline path to `train/train_gym.py`. A flag enables the path. The default path does not change.

The path has three stages. You can select each stage separately:

1. **collect**: download an expert policy if it is not on disk. Use the expert to collect expert, expert + noise, and random data. Save the data to disk.
2. **pretrain**: train the tokenizer and the world model (WM) on the offline data only. Save a pretrained checkpoint.
3. **rl**: load the pretrained checkpoint and run the current online loop (Gym rollouts, WM updates, imagination RL).

This order follows the Dreamer 4 paper: tokenizer training, then dynamics training on offline data, then imagination RL. Section 12 lists the paper parts that stay out of scope.

## 2. Current behavior (the parts that change)

| Step | Location | Current behavior |
|---|---|---|
| Tokenizer data | `train/train_gym.py:2902` | `collect_random_observations()` collects 16,384 random-policy observations. `train_tokenizer()` fits the normalization and trains 2,000 steps. |
| WM pretraining | `train/train_gym.py:3330` | After the first rollout (64 episodes of the untrained policy), `train_world_model()` runs `pretrain_world_model_combined_steps` (256) steps on that replay. |
| Replay | `train/train_gym.py:3257` | The loop stores each episode from `interact_with_env()` in a memmap `ReplayBuffer`. The buffer holds tokenizer latents, not raw observations only. |
| Checkpoint | `train/train_gym.py:2440`, `3076` | `save_checkpoint()` stores the tokenizer, the WM, and both optimizers. `checkpoint_path` resumes the loop. |

REPORT section 10.7 (`runs/audit_9-24-26/REPORT.md`) gives the design constraints that this plan uses:

- The WM accuracy follows its data. The pretraining data must be broad.
- Keep offline data in the online WM batches (rehearsal). Without rehearsal, a narrow online distribution erases the broad knowledge.
- Fit the tokenizer normalization on the pretraining data. The random-data statistics saturate at high speed (F5).
- Count the offline env steps, or call the result an offline-to-online setting.
- More random-data WM pretraining did not help (old repo `MoreWMPretrain_LateBootstrap`). Random data does not cover the states that matter.

## 3. Design overview

```
offline_pretraining: true
offline_stages: [collect, pretrain, rl]      # any ordered subset

collect  -> runs/offline_data/<env>/<dataset>/        raw episodes + manifest.json
pretrain -> runs/offline_pretrain/<name>/pretrained.pt tokenizer + WM + signature
            runs/offline_pretrain/<name>/offline_replay/  latent memmap replay (for rehearsal)
rl       -> the current loop, with the pretrained tokenizer and WM
```

Each stage writes a complete artifact and a manifest. A later stage reads only the artifact. Thus you can run the stages in separate jobs:

- Run **collect** on a CPU node. It needs MuJoCo and a small MLP only.
- Run **pretrain** once on an H100 or GH200 node.
- Run many **rl** ablations from one pretrained checkpoint.

Put the new helper code in a new module `train/offline_pretraining.py`. This follows the pattern of `train/audit_ablations.py`: `train_gym.py` imports the helpers, and `main()` keeps the model construction. This plan does not add a second trainer.

## 4. Config keys

Add these keys to `train/config.yaml` and to the `main()` signature. Each key is inactive while `offline_pretraining` is false.

```yaml
# Offline data collection and pretraining. Inactive unless offline_pretraining is true.
offline_pretraining: false
# Ordered subset of [collect, pretrain, rl].
offline_stages: [collect, pretrain, rl]
# Expert policy source (CleanRL TD3, HalfCheetah-v4).
offline_expert_repo: cleanrl/HalfCheetah-v4-td3_continuous_action-seed1
offline_expert_revision: 85477545d6db77ae3c7ca8e4e0b48f515d4c1432
offline_expert_filename: td3_continuous_action.cleanrl_model
offline_expert_cache_dir: runs/offline_data/experts
# A local expert file. It replaces the download when set.
offline_expert_path: null
offline_expert_min_return: 8000.0
# Data collection.
offline_data_dir: runs/offline_data
offline_dataset_name: halfcheetah_v4_mix_1m
offline_data_steps: 1000000
offline_data_mix: {expert: 0.3, expert_noise: 0.4, random: 0.3}
# Gaussian noise std in units of half the action range. Each expert_noise episode samples one value.
offline_expert_noise_stds: [0.1, 0.2, 0.4, 0.8]
offline_data_seed: 10042
offline_data_num_envs: 64
# Pretraining.
offline_pretrain_dir: runs/offline_pretrain
offline_pretrain_name: null            # null: use run_name
offline_world_model_steps: 20000
offline_world_model_finetune_steps: 0
offline_validation_fraction: 0.05
offline_pretrain_checkpoint_every: 2000
# RL stage.
offline_pretrained_checkpoint: null    # null: use the checkpoint of the pretrain stage in this process
offline_load_world_optimizer: true
offline_rehearsal_fraction: 0.25
```

## 5. Stage 1: collect

### 5.1 Expert source

Use `cleanrl/HalfCheetah-v4-td3_continuous_action-seed1` on the Hugging Face Hub. Reasons:

- It uses `HalfCheetah-v4`, the same env version as `train/config.yaml`.
- The model card reports a mean return of 10,122 ± 120.
- The file `td3_continuous_action.cleanrl_model` (871,773 bytes) is a plain `torch.save((actor, qf1, qf2))` tuple of state dicts. It loads with `torch.load(weights_only = True)`.
- The actor is a 3-layer MLP: `fc1` (17 → 256), `fc2` (256 → 256), `fc_mu` (256 → 6), `tanh`, and the buffers `action_scale` and `action_bias`. A 20-line module in `train/offline_pretraining.py` reproduces it. No new dependency is necessary (no `stable-baselines3`, no `huggingface_hub`).

Alternatives (not the default):

| Model | Env | Reported return | Cost |
|---|---|---|---|
| `farama-minari/HalfCheetah-v5-SAC-expert` | v5 | 11,448 | SB3 zip format. v5 differs from v4. |
| `sb3/sac-HalfCheetah-v3`, `sb3/td3-HalfCheetah-v3` | v3 | 9,564 / 9,709 | SB3 zip format. v3 differs from v4. |
| `sm_cleanrl_updated_PPO` from this repo (`train/train_model_free.py`) | v4 | about 6,000 | Local training. It covers speeds between the online policy and the expert. |

Download procedure (`download_expert()`):

1. Build the URL `https://huggingface.co/<repo>/resolve/<revision>/<filename>`. Pin the revision.
2. If `<cache_dir>/<repo>/<revision>/<filename>` exists, and its sha256 agrees with the `.sha256` file next to it, use it.
3. Otherwise, download with `urllib.request` to a temporary file in the same directory. Compute the sha256. Rename the file atomically. Write the `.sha256` file.
4. If `offline_expert_path` is set, skip steps 1 to 3 and use that file.

Validation (`validate_expert()`): run 5 deterministic episodes. If the mean return is below `offline_expert_min_return`, raise an error. This check finds a wrong file, a wrong observation order, or an env mismatch before the collection starts.

### 5.2 Behavior policies

| Mode | Action | Purpose |
|---|---|---|
| `expert` | `actor(obs)` | High-speed states (x-velocity near 10), far above the online policy. |
| `expert_noise` | `clip(actor(obs) + N(0, (σ · action_scale)²))`. Each episode samples σ from `offline_expert_noise_stds`. | Intermediate speeds, stumbles, and recoveries. The σ = 0.8 episodes reach nose-down and flipped states. |
| `random` | `env.action_space.sample()` | The current tokenizer data. Low speed, low control structure. |

Each mode gets `round(offline_data_steps · fraction / 1000)` episodes. HalfCheetah truncates at 1,000 steps and never terminates, so each episode has 1,000 transitions. The default gives 300 expert, 400 expert + noise, and 300 random episodes.

Use `make_env()` (`train/train_gym.py:369`) with `offline_data_num_envs` sync envs. Seed the envs with `offline_data_seed`. Keep this seed different from the RL seeds, so that the RL stage does not see the same initial states.

### 5.3 Storage format

Store raw data, not latents. The tokenizer does not exist at collect time.

```
runs/offline_data/HalfCheetah-v4/<offline_dataset_name>/
  obs.npy           float32 [E, 1001, 17]   obs_0 .. obs_1000 (the last row is the bootstrap observation)
  actions.npy       float32 [E, 1000, 6]    env-range actions
  rewards.npy       float32 [E, 1000]
  terminated.npy    bool    [E]
  truncated.npy     bool    [E]
  lens.npy          int32   [E]             transitions per episode
  mode.npy          int8    [E]             0 expert, 1 expert_noise, 2 random
  noise_std.npy     float32 [E]
  manifest.json
```

1,000,000 transitions need about 100 MB. Use `np.load(mmap_mode = "r")` to read.

`manifest.json` holds: the env name, the gymnasium and MuJoCo versions, the expert repo, revision, and sha256, the collection config, and per-mode statistics. The statistics are: the return mean and std, the x-velocity percentiles (obs index 8), and the flipped fraction. For the flipped fraction, reuse `HALF_CHEETAH_ROOT_ANGLE_INDEX` and `_wrap_angle` from `train/audit_ablations.py`.

Write the data to `<dir>.tmp`, then rename the folder. If the folder exists with a manifest that agrees with the config, skip the stage. If the manifest does not agree, raise an error. Do not overwrite data.

## 6. Stage 2: pretrain

### 6.1 Tokenizer

At `train/train_gym.py:2902`, replace the random observations with all offline observations if the pretrain stage is selected. `train_tokenizer()` then fits `obs_mean` and `obs_std` on the offline data and trains `pretrain_tokenizer_steps` steps. Keep the tokenizer hyperparameters of the config.

Log the reconstruction loss per mode on 5 % held-out episodes. The expert states dominate the variance. Check that the loss on random-mode states stays low, because the online policy starts there.

### 6.2 Offline latent replay

After the tokenizer training, encode the offline episodes and store them in a memmap replay at `<pretrain_dir>/<name>/offline_replay/`. The layout must agree row by row with `interact_with_env()` (`dreamer4/dreamer4.py:7082`), because `train_world_model()` and the rehearsal read both buffers the same way.

| Field | Rows for a truncated episode | Value |
|---|---|---|
| `latents` | 1,001 | `tokenizer(obs_t, return_latents = True)` |
| `critic_state` | 1,001 | `obs_t` |
| `actions_continuous` | 1,001 | Native Beta range [0, 1]: `(a_env - low) / (high - low)`, the inverse of `rescale_for_env()`. Clamp to [1e-4, 1 - 1e-4]. The last row is 0 (the pad of `interact_with_env()`). |
| `rewards` | 1,001 | `r_t` for the transition from `obs_t` with `a_t`. The last row is 0. |
| `log_probs_continuous`, `values` | 1,001 | 0. The WM loss does not read them. They keep the schema equal to the online replay. |
| `lens` | meta | 1,001 (1,000 transitions + 1 bootstrap row) |
| `is_truncated`, `terminals` | meta | true, false |
| `is_from_world_model` | meta | false |
| `episode_return` | meta | the sum of `rewards` |

Reuse `AuditDiagnostics.to_native_actions()` (`train/audit_ablations.py:215`) for the action conversion. Move it to a shared function if both modules use it.

Build the buffer with `Experience.create_memmap_replay_buffer()` and `store_episode()`, as the loop does (`train/train_gym.py:3257`). Set `max_episodes` to the number of offline episodes and `circular = False`. Write the held-out episodes to a second buffer `offline_replay_val/`.

The reward target does not depend on the stored layout. `old_reward_shifting` changes only how the WM loss reads the rewards. Thus one offline replay serves both reward targets. The pretrained WM does depend on it (section 6.4).

### 6.3 WM training

Call the current `train_world_model()` (`train/train_gym.py:1357`) with the offline replay:

1. `offline_world_model_finetune_steps` steps with `shortcut_train_mode = "finetune"` (default 0, as `pretrain_world_model_finetune_steps` today).
2. `offline_world_model_steps` steps with `shortcut_train_mode = "combined"` and `update_loss_ema = True`.

Use the `world_optimizer`, `compile_runtime.world_model_loss`, and batch shape of `main()`. Thus the compiled graphs of the RL stage do not change.

Every `offline_pretrain_checkpoint_every` steps:

- Compute the WM loss terms on the validation buffer (no gradient). Log them as `offline_pretrain/val/*`.
- Save `pretrained_step_<n>.pt`. A restart of the pretrain stage continues from the last file.

Do not train the policy and value heads in this stage. `world_model_parameters()` already excludes them.

### 6.4 Pretrained checkpoint

`<pretrain_dir>/<name>/pretrained.pt` holds:

- `tokenizer`: the tokenizer state dict, which includes `obs_mean` and `obs_std`.
- `world_model`: the WM state dict without the agent keys (see below).
- `world_optimizer`: the optimizer state.
- `signature`: the config values that decide the tokenizer shape, the WM shape, and the WM targets.
- `dataset_manifest`, `world_model_steps`, `offline_env_steps`.

Signature keys: `env_name`, `obs_dim`, `num_latent_tokens`, `dim_latent`, `model_dim`, `attn_heads`, `attn_dim_head`, the effective `depth` (after the audit overrides), `time_block_every`, `final_special_cross_attn`, `old_state_tokenizer`, `old_dynamics_ff`, `old_embedding_init`, `old_continuous_action_encoder`, `add_action_embed_to_agent_token`, `continuous_dist_type`, `policy_all_tokens`, `agent_predicts_state`, all `reward_*` keys, `old_reward_head`, `old_terminal_head`, `old_reward_shifting`, `prob_shortcut_train`, `use_paper_aligned_shortcut_objective`, `use_dynamics_loss_normalization`, and `scale_reward_gradient_into_transformer`.

Agent keys: derive the parameter names from `world_model.agent_parameters()`. Add the buffers of the same modules and of the value encoder. The RL stage loads the WM with `strict = False`. Then it asserts that each missing and each unexpected key is an agent key. Thus one pretrained WM serves RL runs with different value heads (for example `use_symlog_hl_gauss_value_head`).

`old_reward_shifting` is in the signature. A2 (`false`) and the control (`true`) thus need two pretrained checkpoints.

## 7. Stage 3: rl

Changes in `main()`:

1. If `offline_pretrained_checkpoint` is set, or the pretrain stage ran in this process, skip `collect_random_observations()` and `train_tokenizer()`.
2. Load the tokenizer. Load the WM without the agent keys. If `offline_load_world_optimizer` is true, load the world optimizer state with `load_optimizer_state_if_compatible()`.
3. Compare the signature with the current config. If a key differs, raise a `ValueError` that names the key.
4. Set `did_world_model_pretrain = True`. The in-loop 256 pretraining steps on the first rollout do not run. Record this in `hyperparameters["offline_effective_overrides"]`.
5. Keep all other parts of the loop.

### 7.1 Rehearsal

Add two optional arguments to `train_world_model()`: `rehearsal_replay` and `rehearsal_fraction`. If both are set:

- The offline part of each batch is `n_off = round(batch_size · rehearsal_fraction)`. The online part is `batch_size - n_off`. The default gives 32 offline and 96 online windows.
- Build a second `ReplayDatasetTimeWindow` on the offline replay with the same `window_length`.
- Join the two batch dicts key by key along dim 0 (`concat_replay_batches()`). The two schemas are equal (section 6.2), so each key exists in both.
- The total batch shape stays `batch_size`. The static compile shapes do not change.

Open the offline replay with `ReplayBuffer.from_folder(path, read_only = True)`. Two parallel ablation runs can then share it.

The imagination prompts continue to come from the online replay only. The dreams then start from states that the current policy visits. A later test can add offline prompts (REPORT 11.5, S4).

### 7.2 Env-step accounting

`env_step` stays the count of online steps. All current plots and comparisons stay valid. Also:

- Record `offline_env_steps` in the hyperparameters.
- Log `charts/episodic_return_offline_inclusive` at `env_step + offline_env_steps`. Use this tag for sample-efficiency comparisons with model-free PPO.

### 7.3 Resume

A loop checkpoint (`checkpoint_path`) already holds the tokenizer and the WM. On resume, do not load the pretrained checkpoint again. Only reopen the rehearsal replay. The resume defect of REPORT section 11 (the step counters reset) also applies here. Fix it before long runs.

## 8. Stage dispatch in `main()`

```
validate offline keys
if "collect" in stages:
    collect_offline_dataset(...)                  # before the CUDA check
    if stages == ["collect"]: return
build device, tokenizer, world model, optimizers, compile runtime (current code)
if "pretrain" in stages:
    pretrain_offline(tokenizer, world_model, world_optimizer, compile_runtime, ...)
    if "rl" not in stages: return
if "rl" in stages:
    load_offline_pretrained(...) unless the pretrain stage ran in this process
    current loop, with the rehearsal arguments
```

Place the collect branch before `require_cuda`, so that a collect-only run works on a CPU node.

Validation errors:

- `offline_pretraining` and `small_model_only` together.
- `offline_stages` is not an ordered subset of `[collect, pretrain, rl]`.
- The `rl` stage without the pretrain stage and without `offline_pretrained_checkpoint`.
- `offline_pretrained_checkpoint` together with the `pretrain` stage.
- The `pretrain` stage together with `checkpoint_path`.
- `offline_rehearsal_fraction` outside [0, 1).
- `offline_data_mix` fractions that do not add up to 1.

## 9. Orchestration

1. Collect once (CPU node):
   `python -m train.train_gym --offline_pretraining=True --offline_stages='[collect]' --run_name=offline_collect --run_details=...`
2. Pretrain once per value of `old_reward_shifting` (GPU node, SLURM through `scripts/train_slurm.sh`). A 1:30 job limit applies. The step checkpoints of section 6.3 let a second job continue.
3. RL ablations: add one dict to `train/ablations.py` and combine it with `**`, as the current audit dicts do:

   ```python
   OFFLINE_PRETRAINED_NEXT_STATE_REWARD = {
       "offline_pretraining": True,
       "offline_stages": ["rl"],
       "offline_pretrained_checkpoint": "runs/offline_pretrain/<name>/pretrained.pt",
       "offline_rehearsal_fraction": 0.25,
   }
   ```

Do not run the collect or the pretrain stage inside parallel ablation runs. Two runs can then not write the same folder.

## 10. Files

| File | Change |
|---|---|
| `train/offline_pretraining.py` (new) | `CleanRLTD3Actor`, `download_expert()`, `validate_expert()`, `collect_offline_dataset()`, `load_offline_dataset()`, `build_offline_replay()`, `pretrain_offline()`, `pretrain_signature()`, `agent_state_keys()`, `load_offline_pretrained()`, `concat_replay_batches()` |
| `train/train_gym.py` | New keyword arguments, stage dispatch, tokenizer data branch, pretrained load, rehearsal arguments in `train_world_model()`, offline-inclusive return tag. |
| `train/config.yaml` | The keys of section 4, with `offline_pretraining: false`. |
| `train/ablations.py` | The dict and two seeds per arm, after the first pretrained checkpoint exists. |
| `tests/test_offline_pretraining.py` (new) | Section 11. |
| `AGENTS.md` | Update the Project Goal paragraph. It says that the project does not reproduce the offline pipeline. |

No new package dependency.

## 11. Tests

Add only these tests. Each test protects a defect that is hard to see in the training curves.

1. **Replay layout parity.** Run `interact_with_env()` for a short episode on a tiny CPU model. Build an offline episode from the same observations, actions, and rewards. Assert equal field names, dtypes, shapes, and row alignment (reward t with action t and latent t, and the bootstrap row).
2. **Action round trip.** Env actions → native → `rescale_for_env()` gives the same actions. Native actions stay inside (0, 1).
3. **Expert loader.** Save a random CleanRL-format tuple to `tmp_path`. Load it with `offline_expert_path`. The output agrees with a reference forward pass. This test uses no network.
4. **Agent-key filter.** Load a pretrained WM into a WM with a different value head. The WM parameters agree. The agent heads keep their new initialization.
5. **Signature check.** A different `old_reward_shifting` raises a `ValueError`.
6. **Rehearsal split.** For batch 128 and fraction 0.25, the batch has 32 offline and 96 online windows, and the keys agree.
7. **Three-stage CPU smoke.** Run collect, pretrain, and rl with a tiny config and a local fake expert (`offline_expert_min_return` very low). Use the pattern of `test_main_logs_audit_diagnostics_with_all_audit_flags` in `tests/test_audit_ablations.py`.

The current tests cover the default path. Run `pytest tests/test_train_gym.py tests/test_audit_ablations.py tests/test_train_ablations.py` to confirm that the default path does not change.

## 12. Out of scope

- **Behavior cloning.** The paper finetunes the policy head on task data before the imagination RL. Expert BC makes a comparison of WM pretraining impossible, because the policy then starts near the expert. Add it later as a separate flag if necessary.
- **Unlabeled data.** The paper trains the dynamics on video without actions for the most part. Here, all offline data has actions.
- **Offline prompts** for the imagination (section 7.1).

## 13. First experiment

- Dataset: 1 M transitions, mix 0.3 / 0.4 / 0.3.
- Pretrain: tokenizer 2,000 steps, WM 20,000 steps. Measure the time per WM step with `collect_timings`, then set the steps for one job.
- Base: A2 (`audit_a2_next_state_reward_target`), the best arm at 5 M env steps (1,564–1,605).
- Arms, two seeds each (42 and 4):
  1. A2 + offline pretraining, rehearsal 0.25.
  2. A2 + offline pretraining, rehearsal 0. This arm measures the value of rehearsal.
- Metrics: the online env steps to a return of 1,000 (A2: 3.2 / 3.8 M), the return at 5 M, the maximum flipped share, and `audit/world_model/*` at the first diagnostic point.
- Decision rule: keep the offline path if both seeds pass 1,000 at least 1 M online env steps before A2, match or pass A2 at 5 M, and stay below a flipped share of 0.2.

## 14. Risks

| Risk | Check or mitigation |
|---|---|
| The expert states set the tokenizer normalization. The online states near zero speed then use a small part of the latent range. | Section 6.1 per-mode reconstruction loss. `tokenizer/policy_recon_loss` in the RL stage. |
| The value range. The expert reward is about 10 per step, so the discounted value is near 1,000. The MSE value head range is [-1200, 1200]. | The pretrain stage does not train the value head. Check the range only if the RL policy reaches expert speeds. The symlog value head has no such limit. |
| Loss of plasticity after long pretraining (primacy bias). | Rehearsal. Compare the step checkpoints (for example 10 k against 20 k steps) if the RL stage learns slowly. |
| The WM capacity spreads over a broad distribution. The one-step error on the online states can rise. | `audit/world_model/one_step_norm_mae` at the first diagnostic point, against A2. |
| The gymnasium version drops `HalfCheetah-v4`. | Record the versions in the manifest. `validate_expert()` fails before the collection. |
