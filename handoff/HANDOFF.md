# Handoff: dreamer4 dynamics model on HalfCheetah (state 2026-09-27)

## 0. How to use this document

- This document gives facts, measurements, code locations, and tools. It does not rank causes.
- Section 11 lists open questions. Their order has no meaning.
- `runs/audit_9-27-26/REPORT.md` holds the interpretations of the previous agent. Read it only if you want those opinions. This handoff repeats all of its measurements without the conclusions.
- All paths are relative to the repo root `/work/nvme/bguz/mravichandran/proj/dreamer4lucid` unless they start with `/`.
- "Norm MAE" in this document always means `mean(|pred − true| / std(true))`. It is computed per state dimension over a test set, then averaged over the 17 HalfCheetah observation dimensions.

## 1. Environment and constraints

| Item | Fact |
|---|---|
| Node | CPU-only login node. No CUDA. Training runs go to H100 nodes (`scripts/train_slurm.sh`). |
| Time limit | Thread-intensive processes on the login node stop after about 25 minutes. The user does not want work-arounds such as chunk-and-resume. Size each experiment to finish in one process under 25 minutes. |
| Thread limit | Parallel torch processes hit `libgomp: Thread creation failed`. Use `OMP_NUM_THREADS=8` and `torch.set_num_threads(8)`. Run one heavy process at a time. |
| Python | `source .venv/bin/activate`. `uv` manages the environment. matplotlib is not installed; do not change the project environment without asking. |
| Process kill | `pkill -f <pattern>` also matches the calling shell and kills it. Kill by PID. |
| Prior instructions from the user | The previous task said: do not read `notes/` or other audits' report content (bias). Scripts under `runs/audit_9-24-26/scripts/` were allowed. Ask the user if this rule still applies. |
| Style | `AGENTS.md` requires ASD-STE100 style for written text. |
| Pending item | A comment in `train/ablations.py` (`reduced_reward_transformer_gradient` entries) says: alert the user that this ablation must run with the best settings from previous ablations combined with this change. The entries now use `CLEANRL_PPO_CLOSER`. Section 7.9 has a measurement that relates to this ablation. |

CPU speed references (8 threads, one process alone): world-model forward+backward at batch 32×64 ≈ 1.4 s. Dream generation of 9 frames for 512 windows ≈ 20–40 s. A 2,000-step MLP (batch 256) ≈ 35 s.

## 2. Code map (dynamics model focus)

| What | Where |
|---|---|
| `DynamicsWorldModel.__init__` | `dreamer4/dreamer4.py:5413` |
| `latents_to_spatial_tokens`, `to_latent_pred` | `dreamer4/dreamer4.py:5674–5693` |
| signal / step-size embeddings, `prob_shortcut_train` default | `dreamer4/dreamer4.py:5754–5764` |
| main `AxialSpaceTimeTransformer` construction | `dreamer4/dreamer4.py:6224` |
| `AxialSpaceTimeTransformer` class (layers, attention pools, no time window) | `dreamer4/dreamer4.py:3138` (forward at 3326) |
| `Rotary1D` (time positions, offset = cache length) | `dreamer4/dreamer4.py:1945` |
| `ramp_weight` (0.9τ + 0.1) | `dreamer4/dreamer4.py:998` |
| `get_times_from_signal_level` (τ = level / 64) | `dreamer4/dreamer4.py:6579` |
| `embed_agent_actions` (reward/terminal action conditioning, unused with `old_reward_head`) | `dreamer4/dreamer4.py:6515` |
| `interact_with_env` (real rollout, KV cache from step 0) | `dreamer4/dreamer4.py:6642` |
| `learn_from_experience` (PPO / PMPO / REINFORCE, value losses) | `dreamer4/dreamer4.py:7105` |
| `generate` / `_generate_optimized` (dream sampler) | `dreamer4/dreamer4.py:7768` / `7862` |
| `forward` (tokens, noise, shortcut, all losses) | `dreamer4/dreamer4.py:8545` |
| token packing and `get_prediction` | `dreamer4/dreamer4.py:~9010–9135` |
| flow / shortcut loss | `dreamer4/dreamer4.py:~9150–9290` |
| reward / terminal loss (`old_reward_shifting`, MTP) | `dreamer4/dreamer4.py:~9320–9470` |
| total loss composition | `dreamer4/dreamer4.py:~9705–9760` |
| `StateTokenizer` (tanh latents) | `dreamer4/dreamer4.py:4191` |
| `ObservationTokenizer` (normalization wrapper) | `train/train_gym.py:242` |
| random-observation tokenizer data, `train_tokenizer` | `train/train_gym.py:466`, `1279` |
| `train_world_model` (batch load, clip groups, optimizer step) | `train/train_gym.py:1439` |
| `world_model_clip_groups` | `train/train_gym.py:752` |
| `make_optimizer` (MuonAdamAtan2, parameter-order note) | `train/train_gym.py:568` |
| `trim_prompt_from_dream`, `align_dream_transition_rewards` | `train/train_gym.py:791`, `879` |
| `train_agent_in_imagination` (prompts, generate, PPO call) | `train/train_gym.py:2071` |
| `train_cleanrl_updated_ppo_dream` (closer PPO: 10 epochs × 32 minibatches) | `train/train_gym.py:1741` |
| `main` (model construction, loop order) | `train/train_gym.py:2733`; world-model construction in the loop section after tokenizer setup |
| `WorldModelTrainingLoss`, `ImaginationGenerateRollout` | `train/compile_runtime.py:~228`, `256` |
| Audit flags and in-run diagnostics | `train/audit_ablations.py` |
| Ablation list, `CLEANRL_PPO_CLOSER`, shared dicts | `train/ablations.py` |
| Offline pretraining path (not run; `runs/offline_data` does not exist) | `train/offline_pretraining.py`, flags in `train/config.yaml` |
| State-only model-free trainer (`sm_*` runs) | `train/train_model_free.py` |

## 3. Dynamics model: exact configuration in the recent runs

Values from `train/config.yaml` plus the run overrides. Verify per run with `scripts/common.py:hparams(run)`.

### 3.1 Sizes

| Item | Value |
|---|---|
| Observation | 17-dim HalfCheetah state. `obs[0:8] = qpos[1:]` (`obs[1]` = root angle), `obs[8:17] = qvel` (`obs[8]` = x-velocity). |
| Action | 6-dim, env range [−1, 1]. Beta policy; stored actions are in [0, 1] (`native = (env + 1) / 2`). |
| Tokenizer latent | 4 tokens × 32 dims, tanh-bounded. |
| `model_dim` / heads / head dim | 128 / 4 / 32 |
| `depth` / `time_block_every` | 3 / 2 → layer 1 spatial, layer 2 temporal, layer 3 spatial. One temporal attention layer. |
| `max_steps` (K_max) | 64 → 6 step sizes (log2 0…5 used; `num_step_sizes_log2 = 6`) |
| `num_register_tokens` | 1 |
| `final_special_cross_attn` | false |
| `attn_softclamp_value` | 50 |
| Value residual, attention pools | on (`value_residual = True`, `use_attn_pool = True`) |

Parameter counts (`audit_control` checkpoint, total 8.80 M):

| Module | Parameters |
|---|---|
| `to_agent_embed_terminal_pred` (8-head MTP ensemble of 128→512→512→512→1 MLPs) | 5,341,833 |
| `policy_head` (input 7 tokens × 128 = 896 with `policy_all_tokens`) | 1,248,768 |
| `transformer` (the dynamics trunk) | 1,008,906 |
| `value_head` | 723,711 |
| `to_agent_embed_reward_pred` (8 MTP heads × 255 bins) | 294,912 |
| `action_embedder` | 49,920 |
| `terminal_action_embedder` | 49,920 |
| `to_latent_state_terminal_pred` | 37,793 |
| `to_terminal_action_embed` | 33,152 |
| `latents_to_spatial_tokens` (Linear 32→128) | 4,224 |
| `to_latent_pred` (RMSNorm + LinearNoBias 128→32) | 4,224 |
| `signal_levels_embed` | 4,096 |
| others (embeddings, critic state embedder) | < 3,000 |

### 3.2 Tokens per frame (8)

`pack([flow_token, space_tokens(4), proprio(0), state_pred(0), registers(1), action_tokens(1), reward_tokens(0), aug(0), agent_tokens(1)])`.

- `flow_token` = concat(signal-level embedding, step-size embedding), each 64 dims.
- `space_tokens` = Linear(32→128) of the (noised) latent tokens.
- `action_tokens` at frame t hold the action a_{t−1} (shift by one; frame 0 gets zeros). See `forward`, "maybe create the action tokens".
- `reward_tokens` are empty (`add_reward_embed_to_agent_token = False`).
- One special token (the agent). The agent attends to all tokens. Other tokens do not attend to the agent.
- Spatial layers: full attention among the 8 tokens of one frame (agent masked as above).
- Temporal layer: each token position attends causally to the same position in earlier frames (`b t s d → b s t d`). Rotary positions start at 0 and grow with the KV cache. There is no window.
- Output: `to_latent_pred` on the 4 spatial-token outputs → predicted clean latent (x-prediction, `pred_orig_latent = True`).

### 3.3 Train objective (default path: `use_paper_aligned_shortcut_objective = False`, `use_loss_normalization = False`)

- Diffusion forcing: each frame gets an independent signal level. τ = level/64 ∈ {0, …, 63/64}. The input is never fully clean (maximum τ = 0.984).
- `noised = noise.lerp(latents, τ)` with `noise ~ N(0, 1)`. The latents lie in [−1, 1].
- Per batch: with probability `prob_shortcut_train` = 1 − 1/6 ≈ 0.833 the batch is a shortcut batch. A shortcut batch samples one step size d = 2^k (k ∈ 1…5) per sequence, and signal levels are rounded down to multiples of d. Otherwise d = 1 (log2 = 0) and levels are uniform.
- Flow loss: MSE(pred, clean latent) × ramp(τ) = 0.9τ + 0.1. It is computed on every batch, also on shortcut batches.
- Shortcut loss (shortcut batches only): two no-grad half-steps give the target velocity. The loss is in v-space and weighted by (1 − τ)². It is not ramp-weighted in this path.
- Masking: `masked_mean` over `lens` (all 200 frames valid for normal windows).
- Inference (dreams) uses K = 4 (d = 16, log2 = 4). Only about 17% of train batches use d = 1. The share of batches with d = 16 is (5/6)·(1/5) ≈ 17%.
- Total loss = `flow + shortcut + Σ_MTP(agent reward CE) + Σ(latent-state reward CE, off with old_reward_head) + Σ_MTP(agent terminal BCE) + latent terminal BCE + …` (weights 1). Reward and terminal terms are summed over the 8 MTP heads. The logged `world_model/reward_loss` is the MTP mean, so the value in the total is about 8× the logged value.
- Reward head (`old_reward_head = True`): `to_agent_embed_reward_pred` reads the agent token only. The agent token has no separate action embedding for the reward; it sees actions through the action tokens.
  - `old_reward_shifting = True`: agent token at frame t+1 predicts `rewards[t]` = r(s_t, a_t).
  - `old_reward_shifting = False` (A2 runs): agent token at frame t predicts `rewards[t+1]`. At a window that ends at the episode end, the last target is the zero pad of the bootstrap frame.
- Reward encoder: HL-Gauss, 255 bins, range [−20, 20], sigma/bin 2.0.
- Terminal: HalfCheetah has no terminals. Terminal loss is constant at 0.0560 (label smoothing ε = 0.01). Its trunk gradient is about 1e-9 (section 7.9).

### 3.4 Optimizer and clip

- `MuonAdamAtan2`, lr 6e-4, weight decay 0.01. Muon parameters: attention `to_v`, `to_out`, and feed-forward weights of the trunk. All other world parameters use Adam-atan2.
- World parameters = all parameters minus agent parameters (policy head, action unembed, value head). The world optimizer therefore updates the trunk that the policy and value heads read.
- Gradient clip per group with max norm 1: `reward_head`, `action_head`, `terminal_head`, …, and `trunk` (everything else, including `transformer`, `latents_to_spatial_tokens`, `to_latent_pred`, and the embeddings). Logged `world_model/trunk_grad_norm` is 0.3–12 after the first steps, so the trunk update is usually clipped.
- The saved `world_optimizer` state in the existing checkpoints does not load into a new optimizer (parameter-order mismatch, see the note in `make_optimizer`).

### 3.5 Generation (dreams) — `_generate_optimized` with `forward_for_generate = self.forward`

- Prompt: 8 real replay frames (latents + 8 actions). Prompt frames get no context noise.
- For each new frame: start from N(0, 1). Run K = 4 denoise passes at levels 0, 16, 32, 48. Then run one extra pass at level 63 on the denoised frame to read the agent / policy embeddings (`take_extra_step`).
- Context frames are labeled with level 63 (τ = 0.984). Generated context frames are mixed with 10% fixed noise (`latents.lerp(noise, 0.1)`), and their label stays at level 63.
- Action for frame T: `decoded_actions[:, :T]` → the action token of frame T is a_{T−1}. For the first generated frame this is the last prompt action (the replay behavior action). Policy actions start at the first generated frame.
- Reward at frame T (old head) = agent token of frame T → with old shifting this is the reward of a_{T−1}. `align_generated_dream_transitions = True` shifts dream rewards by one index so reward k pairs with action k.
- `latents.clamp_(-1, 1)` runs once at the end of generation, not per frame.
- `return_terminals = False` in the imagination wrapper, so dream terminals are not used.
- Timing (H100, compiled): 4 × 1.6 s per loop for 4 × 512 dreams of 40 frames.

### 3.6 Tokenizer

- `StateTokenizer`, depth 2, dim 256, `attn_every = 0` (no time attention; per-frame MLP-like). Output tanh.
- Trained once for 2,000 steps on 16,384 random-policy observations. The normalization mean and std come from the same observations. It is then frozen for the whole run. Replay stores the latents from this frozen tokenizer.

## 4. Data conventions and files on disk

### 4.1 Replay buffer (`runs/gym_imagination/<run>/replay_buffer`, memmap)

| Field | Shape (capacity) | Notes |
|---|---|---|
| `latents` | (16384, 1010, 1, 4, 32) | frozen-tokenizer latents |
| `critic_state` | (16384, 1010, 17) | raw observations |
| `actions_continuous` | (16384, 1010, 6) | native [0, 1] |
| `rewards` | (16384, 1010) | env reward |
| `log_probs_continuous`, `values` | … | behavior log-prob per action dim, value |
| `episode_lens` | meta | 1001 per episode (1000 steps + 1 bootstrap frame) |

- Index t: `latents[t]` = z(s_t), `actions[t]` = a_t taken at s_t, `rewards[t]` = r(s_t, a_t). Index 1000 is the bootstrap state; its reward and action are zero pads.
- Episodes are stored in collection order. Loop k (0-based) owns episodes `[64k, 64k + 64)`. Capacity (16384) is larger than any run, so the replay holds the full history, and prompts and world-model windows sample uniformly over all history.
- Checkpoint `loop_N.pt` is saved after loop index N−1. The rollout of loop index N used exactly these weights. Episodes `[64N, 64N + 64)` are therefore held-out on-policy data for that checkpoint.
- Checkpoints exist every 25 loops (`runs/gym_imagination/<run>/checkpoints/loop_{25,50,75,100,…}.pt`, about 106 MB each). Keys: `loop, tokenizer, world_model, world_optimizer, agent_optimizer` (some newer ones also have `progress`).
- 70 runs have replay buffers. The `sm_*` runs have none.

### 4.2 Extracted data from this audit (`runs/audit_9-27-26/data/`)

| File | Content |
|---|---|
| `tb_all.pkl` (306 MB) | `{run: {"scalars": {tag: (steps, values, walltimes)}, "text": {tag: str}}}` for all 92 runs. Hyperparameters: `text["run/hyperparameters/text_summary"]` (JSON in backticks). Ablation rationale: `text["run/details/text_summary"]`. |
| `flip_scan.pkl` | per run, per loop: `ret`, `flip_time_frac`, `ever_flip_frac`, `xvel`, `upright_reward`, `flipped_reward` |
| `run_catalog.md` | one row per run (section 10) |
| `e1…e9_*.txt` | raw outputs of the experiments in section 7 |

## 5. Loop structure and budgets (recent runs)

| Quantity | Value |
|---|---|
| Envs × steps per loop | 64 × 1000 = 64,000 env steps |
| World-model updates per loop | 13 at batch 128 × 200 = 332,800 frames → about 5.2 sampled frames per env step |
| World-model pretrain | 256 combined steps on the first rollout only (random-init policy) |
| World-model updates per full run | 1,600–3,000 (for example `audit_control`: 1,803 at 7.7 M env steps) |
| World-model updates in the first 1 M env steps | about 460 (256 + 16 loops × 13) |
| Imagination (closer PPO) per loop | 4 chunks × 512 dreams × (8 prompt + 32) frames → about 63,500 learnable transitions (31 per dream), 10 epochs × 32 minibatches |
| Model-free reference (`sm_cleanrl_updated_PPO`) per iteration | 64 × 1000 real transitions, 10 epochs × 32 minibatches, `NormalizeObservation`, `NormalizeReward` + clip 10 |
| Wall time per loop (`audit_control`, H100) | about 45 s: rollout 14.3 s (env 3.6 s, model 10.7 s), world model 13 × 1.07 s (forward 0.54 s, backward 0.40 s), imagination about 10 s, audit diagnostics about 6 s every 10 loops |
| Run length | Slurm jobs of about 1.5 h → 7–13 M env steps |

## 6. Results overview (episodic return)

| Run | @1M | @3M | @5M | @10M | final | env steps |
|---|---|---|---|---|---|---|
| sm_cleanrl_updated_PPO (state-only PPO, real data) | 595 | 3896 | 5346 | 7522 | 9266 | 27.7 M |
| sm_cleanrl_updated_PPO2 | 1017 | 4201 | 5806 | 7780 | 9363 | 27.5 M |
| closer_ppo_symlog_hl_gauss_value_head | −108 | 800 | 1385 | 1831 | 1813 | 6.7 M |
| closer_ppo_symlog_hl_gauss_value_head2 | −36 | 709 | 1212 | 963 | 1003 | 6.5 M |
| audit_a2_next_state_reward_target | −112 | 930 | 1319 | 2318 | 1983 | 9.3 M |
| audit_a2_next_state_reward_target2 | −152 | 802 | 1477 | 1856 | 1794 | 7.9 M |
| closer_ppo_next_state_reward_symlog_value_head | −49 | 765 | 1717 | 1758 | 1721 | 6.6 M |
| closer_ppo_next_state_reward_symlog_value_head2 | −21 | 440 | 961 | 754 | 765 | 6.7 M |
| audit_control | −188 | −81 | −25 | 593 | 595 | 7.7 M |
| audit_control2 | −170 | 727 | 417 | −54 | −66 | 7.8 M |
| cleanrl_ppo_closer | −164 | 47 | 359 | 554 | 526 | 7.8 M |
| cleanrl_ppo_closer2 | −147 | 206 | 1311 | 26 | 9 | 7.8 M |
| symlog_hl_gauss_value_head2 (PMPO path) | −227 | 412 | 439 | 1435 | 2371 | 13.7 M |
| symexp_hl_gauss_actor_critic | −247 | −145 | 713 | 1731 | 1912 | 10.7 M |

"@X" = mean of logged episodes near X env steps; after the run end it repeats the last value. Section 10 has all runs. User-provided context (not verified here): standard SAC reaches about 15k and CleanRL PPO more than 10k in far fewer env steps.

Seed variance: over 24 seed pairs, the median |return difference| at 6.4 M env steps is 233 (mean 306).

## 7. Measurements

All offline experiments use saved checkpoints and replay on CPU. Scripts are in `runs/audit_9-27-26/scripts/` (section 12). Test data is the held-out on-policy loop of the checkpoint unless stated otherwise.

### 7.1 In-run audit diagnostics (MuJoCo-grounded; runs with `audit_diagnostics = true`)

Runs: `audit_control`/`2`, `audit_a2_next_state_reward_target`/`2`, `closer_ppo_next_state_reward_symlog_value_head`/`2`. Every 640k env steps, before the world model trains on the fresh rollout. Definitions are in `train/audit_ablations.py`:

- `one_step_norm_mae`: 128 windows, 8-frame prompt with real actions, generate 1 frame, decode, compare with the real next state. `one_step_copy_last_norm_mae` = copy-last baseline.
- `ranking_*`: 64 start states × 8 action sequences (real + 7 with N(0, 0.3) noise over 32 steps), open loop. Truth from MuJoCo resimulation from the exact state. Spearman of dream vs true 32-step return per start state; `spread_ratio` = std of dream returns / std of true returns.
- `closed_loop/*`: 64 replay prompts, the policy acts in the dream for 32 steps, the same actions are replayed in MuJoCo. `state_norm_mae_h{1,8,32}` compares decoded dream states with true states.

Value ranges over each run (first → last diagnostic):

| Metric | audit_control | audit_control2 | audit_a2 | audit_a2 2 | closer_next_state | closer_next_state 2 |
|---|---|---|---|---|---|---|
| one_step_norm_mae | 0.47 → 0.33 | 0.47 → 0.40 | 0.31 → 0.34 (min 0.30) | 0.37 → 0.28 | 0.29 → 0.35 | 0.38 → 0.32 |
| copy-last | 0.88 → 0.80 | 0.90 → 0.87 | 0.88 → 0.78 | 0.89 → 0.80 | 0.85 → 0.78 | 0.90 → 0.75 |
| one-step x-vel MAE | 0.36 → 0.21 | 0.31 → 0.30 | 0.22 → 0.32 | 0.31 → 0.15 | 0.28 → 0.31 | 0.34 → 0.33 |
| reward_head_corr | 0.87 → 0.96 | 0.91 → 0.97 | 0.59 → 0.98 | 0.48 → 0.97 | 0.31 → 0.98 | 0.56 → 0.97 |
| ranking_spearman | −0.06 → 0.30 | 0.14 → 0.07 | 0.04 → 0.09 (max 0.23) | 0.08 → 0.39 (max 0.48) | 0.07 → 0.34 (max 0.42) | 0.09 → 0.39 |
| ranking_spread_ratio | 0.27–0.73 | 0.26–0.63 | 0.15–0.60 | 0.02–0.42 | 0.06–0.50 | 0.11–0.49 |
| closed-loop state MAE h1 | 0.50 → 0.28 | 0.43 → 0.28 | 0.33 → 0.24 | 0.35 → 0.25 | 0.32 → 0.27 | 0.36 → 0.27 |
| closed-loop state MAE h8 | 0.64 → 0.38 | 0.52 → 0.35 | 0.44 → 0.38 | 0.46 → 0.38 | 0.43 → 0.37 | 0.52 → 0.35 |
| closed-loop state MAE h32 | 0.78 → 0.46 | 0.60 → 0.58 | 0.52 → 0.47 | 0.50 → 0.46 | 0.45 → 0.62 | 0.59 → 0.44 |
| dream − true reward/step (late) | −0.12 … +0.65 | +0.36 … +0.58 | +0.14 … +0.51 | +0.06 … +0.50 | +0.28 … +0.52 | 0.00 … +0.39 |
| prompt_at_current_speed_frac | 0.66–0.92 | 0.67–1.00 | 0.45–0.95 | 0.28–0.91 | 0.22–0.88 | 0.66–0.94 |

The tokenizer reconstruction loss on the fresh rollout (`tokenizer/policy_recon_loss`, in random-policy normalized units) rises from 0.000 to 0.001–0.006 over these runs.

### 7.2 E1 — world-model one-step error vs sampler settings (`e1_audit_control.txt`)

`audit_control`, checkpoint loop 100, held-out loop 100, 512 windows, frames t ∈ [64, 999):

| Prompt | Denoise steps | One-step norm MAE | Copy-last | x-vel MAE | Reward corr |
|---|---|---|---|---|---|
| 8 | 4 | 0.347 | 0.804 | 0.215 | 0.970 |
| 8 | 16 | 0.347 | 0.804 | 0.252 | 0.968 |
| 8 | 64 | 0.352 | 0.804 | 0.276 | 0.967 |
| 32 | 4 | 0.349 | 0.804 | 0.221 | 0.969 |
| 32 | 16 | 0.345 | 0.804 | 0.249 | 0.968 |
| 32 | 64 | 0.352 | 0.804 | 0.278 | 0.967 |

### 7.3 E2 — MLP references on the same data (`e2_audit_control.txt`)

Train: 2,000 random episodes of loops 0–99 (2 M transitions). Test: all 64,000 transitions of loop 100. MLP = 3 hidden layers × 256, SiLU, AdamW lr 1e-3, cosine schedule, batch 256. State MLP predicts the normalized delta. Latent MLP predicts the latent delta on the frozen latents and decodes with the frozen tokenizer.

| Predictor | Updates | Norm MAE | x-vel MAE |
|---|---|---|---|
| Copy last | – | 0.783 | – |
| State MLP | 500 | 0.147 | 0.183 |
| Latent MLP | 500 | 0.172 | – |
| State MLP | 2,000 | 0.110 | 0.119 |
| Latent MLP | 2,000 | 0.138 | – |
| State MLP | 8,000 | 0.078 | 0.082 |
| Latent MLP | 8,000 | 0.101 | – |
| Decode of the true next latent (tokenizer ceiling) | – | 0.013 | – |

The E1 world-model number (0.347) uses 512 windows and E2 uses all transitions of the same loop, with the same scale definition. The copy-last values differ slightly (0.804 vs 0.783) because of window sampling.

### 7.4 E8 — one-step action sensitivity vs MuJoCo (`e8_audit_control.txt`)

128 held-out states × 8 actions (real, plus 7 with N(0, 0.3) noise in env units, clipped). One MuJoCo step from the exact state gives the truth (resimulation reproduces the logged next state to 5e-5). World model: 8-frame real prompt, last action replaced. Latent MLP: 2,000 steps on 1,000 train episodes.

| Model | Norm MAE (all actions) | Action spread ratio (pred / true) | Action-effect corr (all dims) | Action-effect corr (x-vel) |
|---|---|---|---|---|
| World model | 0.350 | 1.09 | 0.80 | 0.59 |
| Latent MLP | 0.158 | 0.96 | 0.95 | 0.84 |

"Action effect" = prediction minus its mean over the 8 actions, per state. The world-model spread also contains the sampling noise of the diffusion sampler, because each action gets a separate noise sample.

### 7.5 E5 — flip onset prediction, open loop with the real actions (`e5_*.txt`)

A flip is |wrap(obs[1])| > π/2. For held-out episodes with a flip onset t_f, the 8-frame prompt ends k steps before t_f; the dream runs k + 16 frames with the real actions. "Predicted" = the decoded dream crosses π/2 in the window. Controls = random windows with no real flip in the window.

| Checkpoint (held-out loops) | Flipped steps in test | k=2 | k=8 | k=16 | False flips (k=2/8/16) | True max \|angle\| | Dream max \|angle\| (k=2/8/16) |
|---|---|---|---|---|---|---|---|
| audit_control2 loop 50 (50–74) | 16% | 0.00 | 0.00 | 0.01 | 0.00/0.00/0.00 | 1.95 | 1.15/1.10/0.89 |
| audit_control2 loop 75 (75–94) | 65% | 0.01 | 0.00 | 0.00 | 0.00/0.00/0.00 | 2.09 | 1.19/1.03/0.74 |
| audit_a2_next_state_reward_target loop 100 (100–144) | 27% | 0.18 | 0.00 | 0.00 | 0.00/0.00/0.00 | 2.87 | 1.23/0.72/0.56 |
| closer_ppo_next_state_reward_symlog_value_head2 loop 50 (50–74) | 49% | 0.99 | 0.96 | 0.34 | 0.02/0.08/0.18 | 2.25 | 2.24/2.19/1.11 |

96–128 flip windows per row. The held-out loops after the checkpoint loop were collected by later policies but were not seen by this checkpoint.

### 7.6 E6 — continued world-model training from a checkpoint (`e6_audit_control.txt`)

`audit_control` loop 100, fresh MuonAdamAtan2 (lr 6e-4, wd 0.01; the saved state does not load), batch 32 × 64 from loops 0–99, clip 1 on all world parameters. Evaluation: 256 held-out windows. Policy KL on 256 held-out 40-frame real contexts, heads frozen.

| Step | Flow loss | One-step norm MAE | Policy KL vs step 0 | KL vs previous report | Mean action shift |
|---|---|---|---|---|---|
| 0 | – | 0.339 | – | – | – |
| 13 | 0.0090 | 0.356 | 0.152 | 0.152 | 0.012 |
| 26 | 0.0096 | 0.344 | 0.169 | 0.735 | 0.013 |
| 50 | 0.0102 | 0.358 | 0.200 | 0.128 | 0.015 |
| 100 | 0.0096 | 0.367 | 0.523 | 0.333 | 0.019 |
| 200 | 0.0094 | 0.371 | 0.096 | 0.624 (vs 100) | 0.012 |
| 213 | 0.0099 | 0.354 | 0.498 | 0.624 | 0.017 |
| 226 | 0.0100 | 0.357 | 0.129 | 0.289 | 0.013 |

The batch has 1/12 of the frames of the run batch. Only one 226-step run fits in the 25-minute limit.

### 7.7 E7 — policy change split into trunk and heads (`e7_drift.txt`)

The policy on 512 held-out real 40-frame contexts, with parameters mixed from two checkpoints 25 loops apart. Agent parameters = `wm.agent_parameters()`.

| Run, loops | Total KL | Trunk-only KL | Heads-only KL | Old heads on new trunk KL[(B,B)‖(B,A)] | Trunk-only mean action shift | Value (A,A) / (B,A) / (A,B) / (B,B) |
|---|---|---|---|---|---|---|
| audit_control 75→100 | 1.839 | 0.126 | 1.470 | 0.947 | 0.014 | −0.86 / 1.78 / 2.57 / 1.93 |
| audit_a2 100→125 | 1.070 | 0.093 | 0.931 | 0.763 | 0.013 | 73.9 / 67.7 / 97.2 / 94.0 |
| closer_ppo_symlog_hl_gauss 75→100 | 2.439 | 0.242 | 1.737 | 1.172 | 0.019 | 123.9 / 123.5 / 170.8 / 171.3 |

For scale: the logged closer-PPO `approx_kl` per iteration is 0.01–0.03, and the policy entropy is low late in training.

### 7.8 E3 — frozen tokenizer on policy data (`e3_tokenizer.txt`)

400 random replay episodes per run, final checkpoint. Error normalized by the replay std.

| Quantity | audit_a2_next_state_reward_target | symlog_hl_gauss_value_head2 |
|---|---|---|
| Recon norm MAE, all / upright / flipped | 0.020 / 0.022 / 0.013 | 0.025 / 0.025 / 0.018 |
| x-vel abs error at vx < 0, [0,1), [1,2), [2,3), ≥ 3 | 0.006, 0.006, 0.012, 0.024, 0.138 | 0.005, 0.008, 0.019, 0.073, 0.241 |
| Latents with \|z\| > 0.99 / > 0.9 | 0.000 / 0.009 | 0.000 / 0.008 |
| Velocities ×1 / ×2 / ×4: recon norm MAE | 0.007 / 0.035 / 0.521 | 0.011 / 0.049 / 0.673 |
| Velocities ×4: x-vel abs error | 0.634 | 0.650 |
| Max \|z-score\| under the random-policy normalization, over 17 dims | 2.4–7.5 (x-vel 7.5) | 2.4–7.0 |

Random-policy obs std (normalization): `[0.15 0.95 0.25 0.27 0.27 0.33 0.30 0.29 0.74 0.76 1.57 5.57 6.74 6.94 6.35 6.66 6.06]`.

### 7.9 E9 — per-loss gradient on the dynamics trunk (`e9_audit_control.txt`)

`audit_control` loop 100, batch 32 × 64, before clipping. Two RNG seeds (one per shortcut / non-shortcut draw).

| Loss term | Value | \|g\| on `transformer.*` | cos with flow grad | \|g\| on `latents_to_spatial_tokens` | \|g\| on `to_latent_pred` |
|---|---|---|---|---|---|
| total | 18.97–19.05 | 3.84–4.11 | +0.39…+0.49 | 0.49–0.64 | 0.012–0.013 |
| flow | 0.0090–0.0092 | 0.027–0.029 | 1 | 0.0042–0.0045 | 0.0138–0.0146 |
| shortcut | 0.0012–0.0018 | 0.024–0.034 | +0.01…+0.36 | 0.0033–0.0050 | 0.0067–0.0078 (cos −0.43…−0.58) |
| reward (agent MTP sum) | 18.91–18.98 | 3.81–4.09 | +0.38…+0.48 | 0.49–0.64 | 0 |
| terminal (sum) | 0.056 | ≈ 8e-10 | – | ≈ 5e-11 | 0 |

The `trunk` clip group (max norm 1) includes all of these parameters.

### 7.10 E4 — long env context vs dream-length context (`e4_audit_control.txt`)

`audit_control` loop 100, 16 held-out episodes. The full-context forward reproduces the logged action log-probs (mean |diff| 0.0000, max 0.0001). Policy at env time t with the full context vs the last 40 or 9 frames:

| t | KL(full‖last 9) | Mean action shift | KL(full‖last 40) | Mean action shift | Value full / last 40 |
|---|---|---|---|---|---|
| 60 | 0.301 | 0.014 | 0.001 | 0.001 | 23.3 / 23.2 |
| 200 | 0.360 | 0.015 | 0.011 | 0.004 | 20.9 / 20.8 |
| 400 | 0.225 | 0.013 | 0.088 | 0.009 | 3.4 / 2.5 |
| 999 | 0.021 | 0.012 | 0.019 | 0.011 | −10.7 / −11.6 |

Mean KL between the full-context policy at adjacent frames: 5.3. The same data show reward per step falling within the episode (0.44 at t = 20 → −0.005 at t = 999).

### 7.11 Flip statistics from all replay buffers (`flip_scan.pkl`)

- Flipped = |wrap(obs[1])| > π/2, per step.
- 22 of 61 non-`sm` runs (≥ 30 loops) end with more than 60% of steps flipped in the last 1/6 of the run. In `oldcode_baseline`, `old_pmpo2`, both `tdmpc2_symexp_value_head`, both `symexp_value_head_baseline_init`, `aligned_dream_transitions`, and `a3_short_aligned_reinforce`, flipped reward per step (0.6–0.7) is higher than upright reward per step (0.2–0.35), with returns near 600–700.
- In the higher-return runs, the final upright reward per step is 1.7–2.9 and the flipped reward per step is −0.15 to −0.52.
- Flips build up within episodes. Example, `audit_a2_next_state_reward_target` loop 130, steps [0,100) / [100,250) / [250,500) / [500,750) / [750,1000): flipped 0.00 / 0.10 / 0.30 / 0.51 / 0.74, reward per step 2.41 / 2.79 / 2.10 / 1.38 / 0.63.
- Return drops (5-loop mean below 60% of a running peak ≥ 500), peak → after (flipped fraction):

| Run | Peak (flip) | After (flip) |
|---|---|---|
| audit_control2 | 1153 @63 (0.13) | 600 @71 (0.40) |
| cleanrl_ppo_closer2 | 1325 @75 (0.14) | 749 @80 (0.43) |
| closer_ppo_next_state_reward_symlog_value_head2 | 1069 @58 (0.35) | 637 @64 (0.60) |
| symexp_hl_gauss_actor_critic | 1254 @100 (0.01) | 696 @107 (0.38) |
| ppo_baseline2 | 964 @116 (0.01) | 500 @119 (0.32) |
| symexp_value_head | 679 @131 (0.02) | 396 @150 (0.27) |
| symexp_value_head2 | 526 @92 (0.00) | 313 @140 (0.18) |
| a3_short_aligned_reinforce2 | 864 @72 (0.00) | 433 @76 (0.26) |
| ppo_baseline | 1062 @90 (0.00) | 567 @102 (0.00) |
| a3_short_aligned_reinforce | 778 @143 (0.98) | 462 @148 (0.98) |

- In the in-run closed-loop diagnostics, the true flipped fraction at the dream end is often above the dreamed fraction (for example `closer_ppo_next_state_reward_symlog_value_head` at 5.12 M: 0.38 true vs 0.05 dream; `audit_control2` at 4.48 M: 0.12 vs 0.02).

### 7.12 World-model training curves (logged)

| Run | WM updates | flow loss at 0 / 5% / 20% / 60% / 100% of updates | reward MAE at 20% / 100% |
|---|---|---|---|
| audit_control | 1,803 | 0.322 / 0.035 / 0.012 / 0.0094 / 0.0076 | 0.42 / 0.26 |
| audit_a2_next_state_reward_target | 2,141 | 0.323 / 0.031 / 0.013 / 0.0084 / 0.0066 | 0.42 / 0.33 |
| closer_ppo_symlog_hl_gauss_value_head | 1,605 | 0.248 / 0.037 / 0.009 / 0.0087 / 0.0064 | 0.44 / 0.21 |
| modded_better_baseline | 1,842 | 0.322 / 0.035 / 0.007 / 0.0045 / 0.0045 | 0.40 / 0.22 |
| symlog_hl_gauss_value_head2 | 3,031 | 0.263 / 0.026 / 0.012 / 0.0065 / 0.0053 | 0.41 / 0.32 |

The shortcut loss stays at 0.0004–0.0009 after the first 5%. The terminal loss is 0.0560 from about 2% of updates onward.

## 8. RL-side facts (condensed)

- Closer PPO (`cleanrl_ppo_closer = True`): dreams are flattened into independent transitions. Advantages come from frozen GAE (γ = 0.99, λ = 0.95) over 31 learnable steps with the last value as bootstrap. Advantages are normalized per minibatch. Linear lr decay over `num_loops`. Policy entropy weight 0.
- Value path with `cleanrl_updated_ppo`: `value_clip = 0.2`, `clip_values = True`, scalar MSE on raw (not normalized) returns, unless `cleanrl_ppo_closer_hl_gauss_value_loss = True` (cross-entropy on HL-Gauss bins, no clip).
- Explained variance (`cleanrl_updated_ppo/explained_variance`), at 5% / 20% / 40% / 60% / 100% of training:
  - cleanrl_ppo_closer: 0.02 / 0.72 / 0.75 / 0.95 / 0.95
  - audit_control: −0.26 / 0.64 / 0.68 / 0.96 / 0.99
  - audit_a2: 0.00 / 0.53 / 0.85 / 0.71 / 0.94
  - closer_ppo_symlog_hl_gauss_value_head: 0.00 / 0.94 / 0.98 / 0.99 / 1.00
  - sm_cleanrl_updated_PPO: 0.92 / 0.89 / 0.92 / 0.94 / 0.94
- Clip fraction: dream runs 0.15–0.33; `sm_cleanrl_updated_PPO` 0.02–0.08. Approx KL: dream runs 0.01–0.03; `sm` ≤ 0.01.
- Policy entropy (CleanRL convention, includes log range): dream runs end near −1 to −2.8; `sm_cleanrl_updated_PPO` ends near −6.1.
- Policy inputs (`policy_all_tokens = True`): agent + 4 spatial + register + action tokens of the current frame from the shared trunk (896 dims), detached during RL.
- Env rollout: temporal KV cache from step 0 to 1000; rotary positions up to 1000. The world model trains on 200-frame windows. The policy trains on dream contexts of 9–40 frames.
- Dream prompts: 8-frame windows, uniform over all stored episodes and start positions.
- `old_reward_shifting` / `align_generated_dream_transitions` / A2 variants: see section 3.3 and 3.5. `audit_a2` vs `audit_control` at 6.4 M: 2020 / 1743 vs 142 / 15. With the symlog HL-Gauss value head: `closer_ppo_next_state_reward_symlog_value_head` 1887 / 479 vs `closer_ppo_symlog_hl_gauss_value_head` 1699 / 920.

## 9. Ablation flags that exist but were not run

- `audit_deeper_dynamics` (A1: depth 3 → 5, S-T-S-T-S).
- `audit_quarter_envs_per_loop` (A3: 16 envs, 4× loops and replay; same updates per loop).
- `preserve_policy_during_world_model_updates` (KL penalty on the real-context policy during world-model updates): `world_model_policy_consistency`/`2` in `ABLATIONS`.
- `scale_reward_gradient_into_transformer` (reward-loss gradient into the trunk × 0.01): `reduced_reward_transformer_gradient`/`2`. See the pending item in section 1 and E9.
- The offline pretraining stages (`offline_pretraining`, collect → pretrain → rl): no offline dataset exists on disk.
- `use_paper_aligned_shortcut_objective`, `use_dynamics_loss_normalization`, and `full_transformer_finetuning` were run earlier; see section 10.

## 10. Run catalog

Generated by `scripts/run_catalog.py` → `data/run_catalog.md`. Columns: env steps, world-model updates, flow loss and reward MAE over the last 10% of updates, return near 3 M and 6.4 M env steps, final return (last 5% of episodes), flipped fraction (last 1/6 of loops, from replay), and the change text from `run/details`. `nan` = not logged or no replay.

See `data/run_catalog.md` for the full table (89 runs). The `run/hyperparameters` text of each run in `tb_all.pkl` holds the exact configuration.

## 11. Open questions (no order)

- Which part of the dynamics path limits one-step accuracy: trunk depth (one temporal layer), width, the per-slot temporal attention, the per-frame noise of diffusion forcing on context frames, the share of batches at the inference step size, the loss weights, or the update count?
- How much of the trunk update comes from the reward loss in practice, after per-group clipping and the Muon / Adam-atan2 normalization? E9 measures the raw gradient only.
- Does the reward head need trunk gradients to reach its current accuracy (corr about 0.97)?
- Why does one checkpoint (`closer_ppo_next_state_reward_symlog_value_head2` loop 50) predict flips and three others do not? What in the data or the run differs?
- How much does the mismatch between context-noise treatment in train (random per-frame levels) and in dreams (level-63 label, 10% noise on generated frames, none on prompts) change the rollout error?
- How does error grow with dream length, per state dimension, and for fast vs slow states?
- How accurate is the world model on current-policy states vs old-policy states, given uniform sampling over the whole replay?
- What is the RL result with the true simulator in place of the world-model transition, with all else equal?
- Would the model-free reference also flip at the same rate? The `sm_*` runs keep no replay.
- How will the frozen tokenizer and its random-policy normalization behave at speeds of a 10k-return gait (about 4× current velocities)?
- The 5.3 M-parameter terminal head: does it cost enough compute to matter for the update budget?

## 12. Tools

### 12.1 Scripts of this audit (`runs/audit_9-27-26/scripts/`)

Run from that directory with `source ../../../.venv/bin/activate` and `OMP_NUM_THREADS=8`. Each file header gives its usage.

| Script | Purpose | Usage | CPU time (alone) |
|---|---|---|---|
| `common.py` | `hparams(run)`, `checkpoints(run)`, `build(run, ckpt)` → `(h, tokenizer, world_model, pkg)`, `replay(run)`, `episodes(rb, idx, t0, t1)` → `(latents, states, actions, rewards)` | import | – |
| `extract_tb.py` | all TB scalars and text → `data/tb_all.pkl` | `python extract_tb.py` | about 1 min (48 procs; not thread-heavy) |
| `flip_scan.py` | per-loop flip stats from all replays → `data/flip_scan.pkl` | `python flip_scan.py` | about 3 min |
| `run_catalog.py` | `data/run_catalog.md` | `python run_catalog.py` | seconds |
| `e1_wm_onestep.py` | WM one-step error vs denoise steps and prompt length | `RUN CKPT_LOOP` | about 5 min |
| `e2_mlp_dynamics.py` | state and latent MLP references | `RUN CKPT_LOOP steps...` | about 2.5 min for 2000 + 8000 |
| `e3_tokenizer.py` | tokenizer error by speed and flip, saturation, velocity scaling | `RUN` | about 1 min |
| `e4_context_length.py` | policy with full vs short context; log-prob sanity check | `RUN CKPT_LOOP [n_eps]` | about 2 min |
| `e5_flip_prediction.py` | open-loop flip-onset prediction | `RUN CKPT_LOOP LOOPS_AHEAD` | about 2.5 min |
| `e6_more_wm_training.py` | continued WM training + policy drift | `RUN CKPT_LOOP STEPS BATCH SEQ [lr]` | about 1.4 s/step at 32×64 |
| `e7_drift_decomposition.py` | trunk vs heads policy change | `RUN LOOP_A LOOP_B` | about 1 min |
| `e8_action_sensitivity.py` | one-step action effect vs MuJoCo | `RUN CKPT_LOOP` | about 3 min |
| `e9_loss_gradients.py` | per-loss gradient norms on trunk parameters | `RUN CKPT_LOOP [B T]` | about 1 min |

Notes for new experiments:
- `build()` mirrors the world-model construction in `train/train_gym.py:main`. If `main` gains new constructor arguments, update `common.py`.
- Replay `latents` already include the view axis (b, t, 1, 4, 32). Pass them to `wm(...)` as they are. For `wm.generate(prompt_latents=...)` pass (b, t, 1, 4, 32) float tensors.
- `wm.generate(P + H, num_steps=4, batch_size=B, return_agent_actions=False, prompt_latents=..., prompt_continuous_actions=<P + H actions>)` gives an open-loop dream with the given actions (the audit diagnostics use this pattern).
- MuJoCo exact restore: `env = gym.make("HalfCheetah-v4").unwrapped; env.reset(seed=0); env.set_state(np.r_[0., obs[:8]], obs[8:]); env.step(a_env)` (`a_env = 2·native − 1`).
- Use `torch.set_grad_enabled(False)` for evaluation scripts; `Beta` log-prob outputs otherwise keep grad.

### 12.2 Scripts of the previous audit (`runs/audit_9-24-26/scripts/`, code only)

`load_run.py` (older loader), `onestep.py`, `onestep_sweep.py`, `action_sens.py`, `exploit.py`, `discrim.py`, `ctx_eval.py`, `rew_profile.py`, `stability.py`, `tb_series.py`, `crossrun.py`, `config_distance.py`. The `.txt` and `.pkl` files there are results and interpretations of that audit.

### 12.3 TensorBoard tags (per run)

- Returns: `charts/episodic_return` (x = env step), `rollout/average_return` (x = loop).
- World model (x = WM update): `world_model/{loss, flow_loss, shortcut_loss, reward_loss, reward_mae, terminal_loss, grad_norm, trunk_grad_norm, reward_head_grad_norm, action_head_grad_norm}`, `extra/{flow,shortcut,reward,terminal,dynamics}_loss_raw`.
- Tokenizer: `tokenizer/recon_loss` (pretrain), `tokenizer/policy_recon_loss` (every 10 loops on the fresh rollout).
- Imagination: `imagination/{dream_return, dream_length, action_std, approx_kl_*, policy_loss, value_loss, grad_norm}`, `critic/{value_mean, gae_return_mean, value_abs_error, value_target_saturation_frac}`, `extra/{policy_entropy, advantage_*, positive_advantage_frac, value_mse}`.
- Closer PPO (x = env step): `cleanrl_updated_ppo/{explained_variance, approx_kl, clip_fraction, entropy, value_loss, policy_loss}`.
- Audit runs only (x = env step): `audit/world_model/*`, `audit/closed_loop/*`, `audit/rollout/*` (including `flipped_frac`, `x_velocity`, `reward_per_step_first_half/second_half`), `audit/env_step/*` (loop metrics re-logged against env steps), `audit/progress/*`.
- Timing: `timings/<phase>/<block>/{cpu,cuda}_milliseconds`.
- Text: `run/details`, `run/hyperparameters`.

### 12.4 Other paths

- Run logs, checkpoints, replays: `runs/gym_imagination/<run>/`.
- Older runs: `runs/halfcheetah_imagination/` (22 runs, not analyzed in this audit).
- Paper: `dreamer4.pdf` (`pdftotext -layout dreamer4.pdf out.txt` works on this node). Relevant parts: sections 3.2 (shortcut forcing, ramp weight, K = 4, context corruption τ_ctx = 0.1), 3.3 (agent tokens, MTP L = 8, imagination training, γ = 0.997), 3.4 (temporal attention every 4 layers, alternating batch lengths longer than the context).
