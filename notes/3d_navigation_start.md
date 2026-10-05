# Start of the 3D Navigation Work: Task List (2026-10-05)

This file lists the tasks to start the dreamer4 agent on a custom 3D navigation environment with image observations. Do the tasks in the order of Section 12. Each task gives the place in the code and the check that ends it.

Sources:

- `report/10-5-26-perf-and-3d/REPORT.md`: optimizers, speed, and the change list for 3D.
- `report/10-5-26-full-finetune-design/REPORT.md`: policy readout and full-transformer finetuning with one optimizer.
- `notes/upstream_dreamer4_commits.md`: upstream fixes that the image path needs.
- `dreamer4.pdf`: sections 3.1-3.4, 4, appendix A.

Rules that stay in force (`AGENTS.md`, memory):

- The default HalfCheetah path must not change. A change that changes gradients goes behind a flag. Do a bitwise check against the current code after each change.
- Write outputs under `runs/`. Write reports in `report/<date-topic>/`.
- The local GPU is an RTX 5070 Ti with 16 GB. /data has about 117 GB free (2026-10-05).

## 1. Decisions to Make First

Write these decisions in the environment README before you write code. Each one changes the model or the replay.

| Decision | Options | Recommendation for the first runs |
| --- | --- | --- |
| Frame size | 64×64, 96×96, 128×128 RGB | 64×64. Memory Maze and the DreamerV3 3D tasks use 64×64. |
| Action space | discrete (forward, turn left, turn right, ...) or continuous (speed, turn rate) | Discrete, 4-6 actions. The paper uses a categorical head. The continuous path here uses Beta. |
| Goal | goal position vector, goal image, or a fixed goal for each level | Fixed goal for the first levels. Then a goal vector. A goal image comes last. |
| Reward | sparse (goal reached) or sparse + dense progress term | Sparse + a small progress term for the first levels. Remove the progress term later. |
| Episode end | goal reached (terminated), time limit (truncated) | Keep terminated and truncated apart. The dreams need the terminal flag (Section 4). |
| Episode length | 200-4,000 steps | 500 steps for the first levels. |
| Privileged critic input | none, or position + heading + goal | Give the critic the privileged state (`dim_critic_state`). The replay already stores `critic_state`. |
| Partial observation | goal visible from the start, or goal found by search | Visible goal first. Search and memory come later. |

## 2. Environment

1. Make the environment a Gymnasium env.
   - `observation_space`: `Box(0, 255, (H, W, 3), uint8)`.
   - Return `terminated` and `truncated` apart.
   - Put the privileged state, the success flag, and the goal in `info`.
2. Make a headless render (EGL or OSMesa). The env workers have no display.
3. Measure the env speed in frames per second for 1, 8, and 16 workers. Now the HalfCheetah rollout makes about 16,000 env steps per second (64 envs × 1,000 steps in 3.95 s). If the 3D env is much slower, the rollout controls the loop time.
4. Set seeds for each level and each episode. A replay of one seed must give the same frames.
5. Write a scripted expert (shortest path on the level map) and a scripted explorer (random walk with wall follow). Section 6 uses them for the data, and Section 10 uses them for the checks.
6. Make a ladder of levels:
   1. One open room, goal in view.
   2. A room with obstacles.
   3. A small maze, goal out of view.
   4. Several mazes, random goal positions (memory).
7. Optional: test the full pipeline on a known 3D env before the custom env. MiniWorld (Farama, Gymnasium API) and Memory Maze (64×64, 6 discrete actions) have published Dreamer results.

## 3. Upstream Fixes for Images

Port these commits from `notes/upstream_dreamer4_commits.md`. After each port, do a bitwise check of the HalfCheetah default path.

| Commit | Item | Why |
| --- | --- | --- |
| `bec034b5ab` | 1. VGG of the LPIPS loss in `eval()` with `requires_grad_(False)` | Without it, the VGG weights can get gradients. |
| `bec034b5ab` | 2. Video decoder with `num_latent_tokens` and `special_attend_only_itself = True` | The special attention of the decoder is wrong without it. |
| `bec034b5ab` | 3. `generate` removes the lone view dimension only when it does not return an `Experience` | Image dreams have the wrong shape without it. |
| `3291467c8f` | 1. `temp_eval(sample_model)` in `dreamer4/trainers.py`; torch-einops-utils >= 0.1.19 (local 0.1.7) | Without it, MAE masking stays off after the first evaluation. |
| `f1fbbb504f` | 2. `embeds.actor` and `embeds.critic` when `actor_depth` or `critic_depth` > 0 | Necessary before a run gives depth to the actor or critic. |
| `f1fbbb504f` | 1. sigreg mask broadcast | Necessary only if a run uses sigreg. |
| `cd30676b9d` | 3. Clamp the PPO log ratio to [-20, 20] | Optional. Add it behind a flag for the closer PPO. |

## 4. Fixes in the Current Code before Images

1. **Compiled short-sequence bug** (memory `compiled-short-sequence-bug`). The compiled forward gives wrong outputs for short multi-frame inputs on CUDA. In the default path, the first dream frame is corrupt. `EagerWithoutTimeCacheCallable` corrects it only for the joint flag. Fix it in the default path behind a flag. More spatial tokens can make the error larger.
2. **Terminals in dreams.** `ImaginationGenerateRollout` calls `generate(return_terminals = False)`. A navigation episode ends at the goal, so the dreams must end too: use the continuation flag of eq. 10 in the returns and in GAE.
3. **Terminal head size.** `to_agent_embed_terminal_pred` has 8 MLPs of width 512 (5.3 M parameters, 5× the transformer). Make the width and the MTP depth of this head configurable. Start with one small MLP.
4. **Discrete actions.**
   - The closer-PPO path is tested only with Beta actions.
   - `validate_offline_config` accepts only Beta.
   - Add a test for the categorical path: entropy, log-probabilities, the PPO ratio, the BC loss, and the offline stage.
5. **Policy readout.** Set `policy_all_tokens: false` for image runs. With 64 spatial tokens and dim 384, the flattened policy input has more than 25,000 values for each frame. The weights do not share across the image. Use one of these readouts:
   - several agent tokens for one agent. `num_agents` means separate agents. Thus this needs a new option: S agent tokens for each agent, pooled or concatenated at the head input.
   - an attentive probe: 1-4 learned queries read the final tokens of the current frame, then an MLP. `LearnedQueriesAttentionPool` and `final_special_cross_attn` are starting points.
6. **Goal input.** Give the goal to the agent tokens only (`num_tasks` for a fixed set of goals, or a new goal embedder for a goal vector). World tokens do not read agent tokens, so the goal cannot leak into the dynamics.

## 5. Image Path in `train/train_gym.py`

The trainer has only the state tokenizer (`ObservationTokenizer`, `obs_to_latents_fn`). It has no image path.

1. **Video tokenizer stage.**
   - Use `VideoTokenizer` (causal, patch 8) and the training loop of `VideoTokenizerTrainer`, or a small loop in `train_gym.py`.
   - Losses: MSE + 0.2 LPIPS, MAE with the drop probability p ~ U(0, 0.9), tanh bottleneck (paper section 3.1).
   - Bottleneck: 32-64 latent tokens × 16-32 dims. A small bottleneck lost the velocity in the HalfCheetah tokenizer (memory `pretrain-root-cause`).
   - Check: held-out PSNR and LPIPS, and decoded videos of moves and turns.
2. **`obs_to_latents_fn` for images.** Encode each new frame with the tokenizer time cache. Do not encode the full history at each step.
3. **Replay.**
   - Store the latents as 16-bit values.
   - Store the uint8 frames on disk for a re-encode after a tokenizer update.
   - Size: one 64×64×3 frame is 12 KB. One million frames are 12 GB. 32 × 32 latents in 16 bits are 2 KB for each frame.
   - The disk holds 117 GB. Delete the frames or latents of old runs before the first long image run.
4. **Env workers.**
   - Move `_grouped_env_worker` into a small module that does not import torch. Each worker now holds about 470 MB because of torch.
   - Send the frames through shared memory, not through pickled pipes.
5. **Config.** Add `observation_type: state | image` and the tokenizer keys to `train/config.yaml`. The default stays `state`.

## 6. Data and Pretraining

1. **Tokenizer data.** Train the tokenizer on diverse frames: the scripted explorer on many levels, random actions, and then replay frames. A tokenizer that saw only random-policy data compressed the velocity in HalfCheetah.
2. **Action-outcome confound.** Expert-only data confounded the action size with the speed (memory `pretrain-root-cause`, `realistic-data-rootcause`). Collect with several policies: random walk, scripted explorer, noisy expert. Before the pretrain, measure the correlation between the action and the outcome for each policy mode.
3. **Offline collection.** Extend `train/offline_pretraining.py` `collect` for image envs and discrete actions. Use the scripted expert in place of the TD3 expert.
4. **Pretrain stage.** Train the tokenizer, then the world model, on the offline data. Keep some offline windows in each world-model batch of the RL stage, as now.
5. **Rare events.** Goal reached, collisions, and stuck states are rare, like the flip onsets. Make `FlipOnsetTimeWindow` general: an event-window sampler for reward > 0, terminals, and collisions. Use it for the world-model windows and the dream prompts.

## 7. Model Size and Settings

| Item | HalfCheetah now | 3D starting point (16 GB GPU) |
| --- | --- | --- |
| Input | 17 states → 4 × 32 latents | 64×64 RGB, patch 8 → 64 patches |
| Tokenizer bottleneck | 4 × 32 | 32-64 × 16-32 |
| Spatial tokens of the dynamics | 4 | 16-64 |
| Dynamics | dim 128, depth 3, time layer every 2 | dim 384-512, depth 8-12, time layer every 4, GQA, 4-8 registers |
| Context | 200-frame windows | short and long batches in turn (for example 32 and 128 frames) |
| Start frames | none | 30 % of the batch as single images (paper section 4) |
| Attention | 9 tokens for each frame | SDPA flash attention for space; KV cache for time |
| Precision | bf16 for the world-model loss only | bf16 everywhere; float32 heads and log-probabilities |
| Memory | micro-batch 32 | micro-batch and selective activation checkpointing (`configure_sac`) |
| Value and reward | symexp two-hot, grouped normalization | the same; terminal head on |

## 8. Training Recipe

1. **Baseline first: two phases with a frozen trunk in RL**, as in the paper and the current A2 runs. Do not start with full finetuning.
2. **Loss scale.** Use one RMS normalizer for flow + bootstrap (the grouped normalizer). Use separate normalizers for reward, terminal, and BC. Separate flow and bootstrap normalizers broke the dynamics (memory `loss-normalization-failure`).
3. **Agent-token training without BC.** The agent token gets gradients only from the reward and terminal heads. For navigation, add these losses to the world-model step:
   - a value TD loss on real replay windows;
   - a policy-preservation KL to the policy snapshot on real replay states.
4. **One optimizer.** Use one `MuonAdamAtan2` with three groups:
   - Muon for the transformer hidden matrices;
   - AdamAtan2 for the other world-model parameters;
   - a head group with `agent_learning_rate`, beta2 0.999, and no weight decay.
5. **Full finetuning later.** Use the design of `report/10-5-26-full-finetune-design` Section 5:
   - a gradient-scale operator at the head inputs (α_π start 0.1);
   - K shared steps for each loop, with the trunk learning rate ≈ 6e-4 × 13 / K;
   - a target-KL stop.
   The untied agent stream (Section 6 of that report) is the lower-risk option for images.

## 9. Speed and Memory

1. **NVIDIA MPS is on** (since 2026-10-05, `scripts/mps.sh`). Run two lanes only under MPS.
2. **CUDA graph for the rollout step** with a static time KV cache and a length mask (flag F1 of the perf report). At 3D scale this is necessary, not optional.
3. **One generation batch** in place of chunks (flag F2), if the memory allows it.
4. **Preallocated rollout buffers** in place of `safe_cat` for each step (S3). The copy cost grows with the frame size.
5. **PPO metrics** only for the logged minibatch (S2).
6. **bf16** for generation and rollout (F5).
7. **Use the `timings/` section in TensorBoard** to find the slow phase after each change. Compare medians, not means.

## 10. Checks and Metrics

| Check | Method | Pass condition |
| --- | --- | --- |
| Tokenizer | PSNR and LPIPS on held-out frames; decoded video | Walls, goal, and turns are clear in the decoded frames |
| Action following | paper Figure 7: PSNR of action-conditioned generations against held-out video, for 16-64 steps | PSNR stays above a model that ignores the actions |
| Dreams | decode 8 dreams to video for each checkpoint | No drift into random frames in 64 steps |
| Terminals | dream terminal rate against the real goal rate | Rates agree within a factor of 2 |
| Agent-token probe | linear probe from the agent token to position, heading, and goal direction | The probe beats a probe on the action alone |
| Policy | success rate, SPL (success weighted by path length), return | Success rises above the scripted explorer on level 1 |
| Counterfactual probe | the flip-probe method: a start state and a set of actions, real against dreamed | The dreams follow the real outcome for the rare events |

## 11. Tests on HalfCheetah before 3D

These tests run on the current code and data. They answer readout questions before the image work. `report/10-5-26-full-finetune-design` Section 7 gives the details.

1. Linear probe of the agent token (pitch, angular velocity, vx) on saved checkpoints. CPU only.
2. `policy_all_tokens: false`, two phases as now. This measures the loss of the agent-token readout.
3. Step 2 + value TD on replay + preservation KL in the world-model step.
4. Step 3 + several agent tokens or an attentive probe.

Use three seeds for each arm. The run-to-run noise is about ±1,000 return.

## 12. Order of Work

1. Section 11, tests 1-2 (they run while the 3D env is built).
2. Section 1 decisions; Section 2 env, steps 1-5; level 1 only.
3. Section 3 upstream fixes; bitwise check of the default path.
4. Section 4 fixes 1-4 (compiled bug, dream terminals, terminal head, discrete actions), each behind a flag, each with a test.
5. Section 5 image path; tokenizer stage on explorer data; tokenizer checks of Section 10.
6. Section 4 fixes 5-6 (readout, goal input).
7. Section 6 data collection and pretrain; action-following and dream checks.
8. First RL run on level 1: two phases, frozen trunk, two seeds.
9. Section 9 speed flags F1, F2, F5 with equivalence checks.
10. Levels 2-4. Then Section 8 steps 3-5 (auxiliary losses, one optimizer, full finetuning).
