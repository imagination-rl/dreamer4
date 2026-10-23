# Dreamer4 Bug Audit (2026-10-05)

This report answers one question. Which bugs of medium priority or higher must you fix for the current config, and which must you fix before the 3D work?

- The current config is `train/config.yaml`, `train/ablations.py` and the October queues (10-1-26 to 10-5-26b).
- The tracker is https://github.com/imagination-rl/dreamer4/issues. It has 26 open bugs of medium, high or critical priority.
- The tracker line numbers refer to the older `train_halfcheetah_imagination_rl.py` layout. This report gives the current locations.
- No code changed. No GPU probes ran, because two queue runs used the GPU.

## Answer

**Current config**

- Fix the corrupt first dream frame in compiled runs (C1).
- Limit the real-rollout KV cache (C2, #56).
- Fix the step size that generation gives the context frames (C3).
- Retest the A2 reward target on the current base (C4, #53).

**Before 3D**

- Fix the terminal path as one change: B1 to B5 (#63, #62, #57 and two new bugs).
- Decide the critic input (B6, #60).
- Use the arrival-aligned reward target (B7, #53).
- Fix #22, and fix the trainers if 3D uses them (B9, B10).

**Tracker status**

- 3 of 26 are fixed in this fork: #50, #51, #55.
- #58 is not active.
- 8 are off the training path.
- The tracker does not have C1, C3, B4 or B5.

## Current Config

### C1. The first dream frame is wrong in every compiled run

**Severity:** High. **Source:** local probe 2026-10-01. It is not on the tracker.

- **Where:** `train/train_gym.py:4881` sets `eager_generate_without_time_cache = fixed_full_transformer_finetuning`. `train/compile_runtime.py:694` is the switch.
- **What:** `generate()` sends the 8 prompt frames and the first new frame through the compiled forward without a time cache.
  - On CUDA, the compiled forward gives wrong output for short multi-frame inputs.
  - The probable cause is the TRITON_DECODE flex kernel. Flex attention selects it for fewer than 128 queries.
- **Effect:**
  - The first latent has a mean absolute error of 0.12. A real step is 0.27.
  - The first reward and value correlate about 0.7 with the eager values.
  - All later frames start from this wrong frame.
  - Every October queue entry has this bug, except the joint-finetune runs.
- **Fix:** Add a flag that turns on `EagerWithoutTimeCacheCallable` without `fixed_full_transformer_finetuning`. Keep the default path unchanged. Before you use the flag, compare frame 1 of the compiled path and the eager path on the GPU.

### C2. Real rollouts keep an unbounded time KV cache

**Severity:** High. **Source:** tracker #56.

- **Where:** `interact_with_env`, `dreamer4/dreamer4.py:6776-7235`.
- **Effect:**
  - The acting policy reads up to 1,001 frames of context. World-model windows have 200 frames, and dreams have 40.
  - All data after step 200 comes from attention distances that training never shows.
  - The 10-4-26 instability audit found a small effect with the default attention. In the rw4 attention pairs, the mean KL was 0.03-0.29.
- **Fix:** Add a flag that keeps a sliding window of `world_model_train_sequence_length` frames.

### C3. Generation gives the context frames a step size that training never pairs with them

**Severity:** Medium. **Source:** `notes/diff_to_paper.md` item 2. It is not on the tracker.

- **Where:** The training sampler is at `dreamer4/dreamer4.py:8912-8938`. The generation step size is at `:8073`, and the context signal level is at `:8339`.
- **What:**
  - Training samples one step size for each batch and rounds the signal levels down to multiples of it.
  - Signal level 63 therefore occurs only with step size 1.
  - Generation gives the clean context frames signal level 63 with step size 16 (d = 1/4).
  - The paper samples d for each frame.
- **Effect:** In every dream forward pass, the context tokens carry a pair of conditions that training never shows. The dynamics and the reward, value and policy heads all read these tokens. I did not measure the size of the effect.
- **Fix:** Let `forward` take one step size for each frame. Behind a flag, give the context frames step size 1 and the new frame step size 16. Both pairs then occur in training. Measure the dream error with the flag and without it.

### C4. The A2 reward target hides the immediate effect of the action

**Severity:** High on the tracker. **Source:** tracker #53. It is a design choice in this fork.

- **Where:** `dreamer4/dreamer4.py:9475-9483`, with `old_reward_shifting: false`.
- **Effect:**
  - The reward of a_t does not enter the advantage of log π(a_t) directly.
  - The control cost of an action therefore gives no direct gradient to that action.
  - This agrees with the saturated actions and the entropy collapse in the logs. It does not prove that A2 causes them.
- **Fix:** The arrival-aligned target already exists (`old_reward_shifting: true` with align). Run it as one arm on the grouped-normalization base.

### C5. `critic_state_embedder` never trains

**Severity:** Medium on the tracker, with a low effect now. **Source:** tracker #60.

- **Where:**
  - `dreamer4/dreamer4.py:5855` creates it.
  - `agent_parameters()` at `:6581` leaves it out.
  - `:6957`, `:7168` and `:7678` use it.
  - `train/train_gym.py:4582` sets `dim_critic_state = obs_dim`.
- **Effect:** `generate()` sets no `critic_state`, so dream values and PPO do not read the embedder. Real-rollout values get a random linear projection of the raw state. Do not trust logged real-rollout values.
- **Fix:** For state runs, add a flag that drops `dim_critic_state`. For 3D, see B6.

### C6. Context noise has no effect

**Severity:** Low. **Source:** tracker #61 and `notes/diff_to_paper.md` item 3.

- **Effect:** Prompt frames get no noise. With the time cache, the transformer keeps only the last frame, so it discards the noised past tokens. The paper's τ_ctx = 0.1 does not apply. This matters more for images.

### Note: read the "symexp" arm as a cevalue arm with an extra head

- In the `symexp_hl_gauss_actor_critic` arm, GAE and the dreams read the raw HL-Gauss ±1200 critic (`dreamer4/dreamer4.py:8549`). The cevalue arm reads the same critic.
- The symexp head only adds a cross-entropy term to the value loss. That term takes a share of the gradient clip.
- The arm description in `train/ablations.py` says this. Run names and conclusions such as "symexp + warmup15 is best" hide it.

### Lower Items

- With `world_model_micro_batch_size` 32, the RMS loss statistics change between the chunks of one batch. Chunk 0 reads the old statistics, and later chunks read the updated ones.
- `reward_loss_weight` is a persistent buffer, so a pretrained state dict overwrites it. Entries 178-180 are consistent. This becomes a bug when pretraining and RL use different weights.

## Before 3D

`notes/3d_navigation_start.md` already lists these items: the compiled bug, terminals in dreams, the terminal head size, discrete actions, the policy readout, the goal input, the image path and the 7 upstream fixes. The items below are not in that list.

Fix B1 to B5 as one change, because each one changes how terminal and target frames line up.

### B1. Terminal labels land one state early

**Severity:** High for 3D. **Source:** tracker #63 (low on the tracker).

- **Where:** `interact_with_env` stores no terminal observation for terminated episodes (lens = T). `dreamer4/dreamer4.py:9558` puts the label at `lens - 2`.
- **Effect:** The terminal head learns termination from s_{T-1}, the state before the last action. In navigation, this is the state one step before the goal. Dreams then end before the arrival state.
- **Fix:** Store s_T for terminated episodes, as the bootstrap row does for truncated episodes. Put the label on s_T. Add a test with a short terminating episode and known lens.

### B2. Shifted-target losses use the source-frame mask

**Severity:** Medium. **Source:** tracker #62.

- **Where:** `loss_mask[:, :-1]` at `dreamer4/dreamer4.py:9442`.
  - `:9582`, `:9599`, `:9620` and `:9749` use it.
  - Under A2, the reward loss at `:9483` uses the same mask.
- **Effect:** In a padded window, the last valid frame predicts a padded target. HalfCheetah windows are never padded, because all episodes have 1,001 frames. Terminating 3D episodes give many padded windows.
- **Fix:** Behind a flag, use `loss_mask[:, 1:]` for one-step shifted targets.

### B3. The bootstrap frame becomes a reward-0 target

**Severity:** Medium. **Source:** tracker #57.

- **Where:** `replay_batch_to_experience` (`train/train_gym.py:1821-1832`) keeps the full `lens`. The reward loss is at `dreamer4/dreamer4.py:9483`.
- **Effect:** For a window that reaches a truncated end, the padded reward 0 of the bootstrap row is a valid target. Now this affects about 1 frame in 1 of 800 windows. With short 3D episodes and time limits, it occurs often.
- **Fix:** For the world-model loss, subtract `is_truncated` from `lens`, as `learn_from_experience` does.

### B4. Label smoothing ends dreams with no real terminal

**Severity:** High for 3D. **Source:** new. It is not on the tracker.

- **Where:** `dreamer4/dreamer4.py:9593-9594` sets ε = 1 − `gae_discount_factor` = 0.01. `:8492` samples a Bernoulli terminal.
- **Effect:**
  - The latent terminal head never predicts below about 0.01.
  - `generate()` samples a terminal from that probability at each step.
  - A 32-step dream then ends with no real terminal 1 − 0.99^32 ≈ 27 % of the time.
  - The current runs do not sample terminals (`return_terminals=False`), so only 3D has this bug.
- **Fix:** Sample from a head without label smoothing. Alternatively, weight the returns with the continuation probability c_t, as the paper does.

### B5. Dream lengths vary after an early terminal

**Severity:** Medium. **Source:** new. It is not on the tracker.

- **Where:** `dreamer4/dreamer4.py:8575-8576` stops when all rows are terminal. With terminals, preallocation is off.
- **Effect:** Closer PPO joins 4 dream chunks of 512. Chunks with different lengths can fail to join. I did not run this case.
- **Fix:** Keep a fixed dream length, and mask the steps after a terminal.

### B6. The privileged-critic plan cannot work as written

**Severity:** High for the 3D plan. **Source:** tracker #60 and the 3D notes.

- **Where:** The 3D notes give the critic the privileged state (`dim_critic_state`). `generate()` sets no `critic_state`, and the embedder never trains (C5).
- **Effect:** In dream training, the critic gets no privileged state. Real-rollout values add a projection that never trains.
- **Fix:** Choose one of these two options:
  - Option 1: predict the privileged state with the world model, and feed the prediction in dreams. Then add the embedder to the agent optimizer.
  - Option 2: keep the privileged state out of the critic.

### B7. A2 with a sparse goal reward gives the credit to the wrong action

**Severity:** High for 3D. **Source:** tracker #53.

- **Effect:** The 3D recipe keeps A2. With a sparse goal reward and termination, the action that reaches the goal gets no direct credit.
- **Fix:** For 3D, use the arrival-aligned target (`old_reward_shifting: true` with align).

### B8. The unbounded KV cache costs more with images

**Severity:** High. **Source:** tracker #56.

- **Effect:** KV memory grows with the episode length, and the context goes outside the training distribution.
- **Fix:** Use the C2 flag for 3D.

### B9. PPO reads PMPO masks that do not exist

**Severity:** Medium. **Source:** tracker #22 (high on the tracker).

- **Where:** `dreamer4/dreamer4.py:7400-7406` sets `pos_advantage_mask` and `neg_advantage_mask` only for PMPO. `:7494` reads them.
- **Effect:** PPO on bootstrap-padded real experience raises `UnboundLocalError`. The dream path does not take this branch now.
- **Fix:** Set both masks to `None` before the PMPO block. This fix only prevents a crash. It does not change any numbers on the default path.

### B10. The video and BC trainers have six open bugs

**Severity:** High if 3D uses them. **Source:** tracker #3, #4, #7, #15, #16, #17.

- **Where:** `dreamer4/trainers.py`. All six bugs are present in this fork.
- **Effect:**
  - #3: the trainers throw away `optim_kwargs` (`:513`, `:931`, `:1382`, `:1524`).
  - #7: the mutable `accelerate_kwargs` default leaks logger settings between trainers.
  - #15: the EMA model exists only on the main rank (`:548`, `:998`), so multi-GPU runs crash.
  - #4: the self-flow loss runs under `no_grad`, so its gradient is zero.
  - #16: `DreamTrainer` and `SimTrainer` assign to a read-only property (`:1398`, `:1541`).
  - #17: the default arguments of `BehaviorCloneTrainer` fail an assertion (`:851`, `:967`).
- **Fix:** If 3D uses these trainers, fix the six bugs. Also port the trainer fixes in `notes/upstream_dreamer4_commits.md`.

### Other 3D Items

- **Low:** `interact_with_env` expects `b c h w` images, but Gymnasium returns HWC.
- **If used:** #29 and #27 are present. They matter only if 3D uses these wrappers to view dreams or to record videos.
  - #29: `WorldModelEnv.step` returns the full terminal sequence (`dreamer4/env.py:502`).
  - #27: `RecordToFolderEnvWrapper` crashes on vector-env termination arrays (`dreamer4/env.py:181`).
- The step-size bug (C3) and the context-noise bug (C6) matter more for image quality than for state runs.

## Tracker Status

The table lists all 26 open bugs on the tracker with priority medium, high or critical. I checked each bug that touches the training path against the current code. "Not checked" means the bug is off the training path.

| Issue | Priority | Bug | State in this fork | Matters for |
| --- | --- | --- | --- | --- |
| #50 | Critical | Replay buffer is not circular | Fixed: `train_gym.py:5080` passes `circular = True` | — |
| #51 | Critical | Resume reads replay from a new log dir | Fixed: `resolve_log_dir` returns the checkpoint run dir | — |
| #55 | High | Symlog value decode in raw space | Fixed: `config.yaml:48-51`; only the symlog arm uses raw decode, by design | — |
| #56 | High | Unbounded real-rollout KV cache | Present | Now (C2), 3D (B8) |
| #53 | High | Reward convention hides the current action | Present (A2 choice) | Now (C4), 3D (B7) |
| #22 | High | PPO reads PMPO masks that do not exist | Present | 3D (B9) |
| #29 | High | WorldModelEnv returns the terminal sequence | Present | 3D, if used |
| #27 | High | Record wrapper crashes on vector envs | Present | 3D, if used |
| #16 | High | Trainers assign a read-only property | Present | 3D, if used (B10) |
| #15 | High | EMA model only on the main rank | Present | 3D multi-GPU (B10) |
| #4 | High | Self-flow loss under `no_grad` | Present | 3D, if self-flow is used (B10) |
| #62 | Medium | Shifted targets use the source mask | Present | 3D (B2) |
| #60 | Medium | `critic_state_embedder` not optimized | Present | Now, low (C5); 3D (B6) |
| #57 | Medium | Bootstrap frame as reward-0 target | Present | 3D (B3); negligible now |
| #58 | Medium | Beta log-probs not clamped in RL | Not active: `BetaDist` clips samples to [1e-5, 1−1e-5]; rollouts and dreams run in fp32 | Only RL on external actions at exactly 0 or 1 |
| #3 | Medium | `optim_kwargs` discarded | Present | 3D, if used (B10) |
| #7 | Medium | Mutable `accelerate_kwargs` default | Present | 3D, if used (B10) |
| #17 | Medium | BC trainer defaults fail an assertion | Present | 3D, if used (B10) |
| #18 | High | e2e test collects about 100k cases | Not checked | Off path (tests) |
| #28 | High | Snake PPO wrapper transposes CHW | Not checked | Off path |
| #31 | Medium | CLI ignores discrete action validation | Not checked | Off path |
| #30 | Medium | Replay inspector fails on vector fields | Not checked | Off path |
| #24 | Medium | LAPO rejects tuple action specs | Not checked | Off path |
| #20 | Medium | Snake e2e uses the wrong frame size | Not checked | Off path (tests) |
| #19 | Medium | Box2D missing from the test extra | Not checked | Off path (tests) |
| #9 | Medium | Wrong shape in `test_proprioception` | Not checked | Off path (tests) |

### Related Enhancements

- #52 (high), "tokenizer stays frozen on random-policy normalization", is still true. It agrees with the earlier finding that the frozen tokenizer limits the velocity precision.
- #45, #48 and #49 already exist as flags: `policy_all_tokens`, world-model pretraining and `fixed_full_transformer_finetuning`.

## Checked and Not Bugs

- **Beta actions at 0 or 1:** `BetaDist` clips each sample to [1e-5, 1−1e-5]. Imagination and rollouts stay in fp32, so the clip holds.
- **RMS normalization of the agent terminal BCE:** the eps clamp of 1e-6 limits the normalized gradient. The logs show `extra/terminal_loss_raw` at the floor H(0.01) = 0.056.
- **Spatial attention backend at micro-batch 32:** frame 0 agrees with eager. Time attention with 200 queries does not use the decode kernel.
- **Flip-onset window index:** `FlipOnsetTimeWindow` agrees with `ReplayDatasetTimeWindow.__getitem__`.

## Recommended Order

1. When the GPU is free, compare frame 1 of a compiled dream and an eager dream. Then add the C1 flag.
2. Add the C2 sliding-window flag.
3. Add the C3 per-frame step-size flag, and measure the dream error.
4. Run the arrival-aligned target (C4) as one arm on the current base.
5. For 3D, change the terminal path as one unit (B1-B5). Add a test with a short terminating episode.
6. Decide the critic input (B6), and set the 3D reward target (B7).
7. Before the image work, fix #22. If you use the trainers, fix them, and port the upstream fixes.

## Method and Limits

- **Sources:**
  - the full Dreamer 4 paper (`dreamer4.pdf`, all sections and footnotes);
  - the code on branch `progress` with the uncommitted changes of 2026-10-05;
  - `train/config.yaml`, `train/ablations.py` and the October queue scripts;
  - `notes/diff_to_paper.md` and `notes/3d_navigation_start.md`;
  - the GitHub issue tracker.
- The C1 numbers come from local probes on 2026-10-01. No new GPU probe ran for this report.
- The B5 failure mode and the size of the C3 effect are not measured.
- No code changed for this audit. Each fix above goes behind a flag, so the default path stays bitwise unchanged.
