# Optimizers, Run Speed, and Changes for a 3D Image Environment (2026-10-05)

## 1. Optimizers in the Current Runs

The checkpoint `a2_grouped_combo_symexp_warmup15_s4/checkpoints/loop_175.pt` and `train/train_gym.py` (`make_optimizer`, `make_cleanrl_updated_ppo_optimizer`, `train_tokenizer`) give this split. Three optimizers exist.

| Optimizer | Parameters | Settings | When it runs |
| --- | --- | --- | --- |
| `AdamW` (tokenizer) | state tokenizer | lr 6e-4, fused | loop 0 only (2,000 steps). The tokenizer is then frozen. |
| `MuonAdamAtan2`, Muon group (18 tensors) | transformer `to_v`, `to_out`, feedforward `proj_in`, `proj_out` weights | lr 6e-4, momentum 0.95, Newton-Schulz 5 steps, update RMS 0.2·sqrt(max dim), weight decay 0.01 (coupled) | each world-model step |
| `MuonAdamAtan2`, AdamAtan2 group (114 tensors) | `to_q`, `to_k` (left out of Muon, as in Kimi K2), gates, norms, embeddings, attention pools, latent / reward / terminal heads | lr 6e-4, betas (0.9, 0.99), atan2 in place of eps, scale a = 1.27, weight decay 0.01 | each world-model step |
| `Adam` (agent, CleanRL) | policy head, action unembed, value heads (35 tensors) | lr 3e-4 with linear decay, betas (0.9, 0.999), eps 1e-5, no weight decay | each PPO minibatch |

- `MuonAdamAtan2` is one optimizer object with two update rules. Thus "one optimizer" for a joint update means one object and one step, with a rule for each parameter group.
- With `fixed_full_transformer_finetuning`, the heads become a third group of `MuonAdamAtan2` with the AdamAtan2 rule. Their update rule changes from Adam (beta2 0.999, eps 1e-5, no weight decay) to AdamAtan2 (beta2 0.99, weight decay 0.01). This was a second difference in the joint runs 75-76, beside the step count.
- Recommended single optimizer for joint training: one `MuonAdamAtan2` with three groups:
  - Muon: the transformer hidden matrices.
  - AdamAtan2: the other world-model parameters.
  - AdamAtan2 for the heads, with `lr = agent_learning_rate`, beta2 0.999 and weight decay 0. These keys can be set for each group.
- AdamAtan2 has no eps. An exact copy of CleanRL Adam for the heads needs a small optimizer class with a plain Adam group.

## 2. Run Speed

### 2.1 Measurements

Source: the `timings/` scalars in TensorBoard. The phase boundaries come from the wall times of the CPU timing events (`loop_phases.py`, `timings_compare.py` in the session scratchpad). Entry 157 ran alone on the GPU after 04:51, when lane b waited for the GPU break. Entry 145 shared the GPU for all of its run.

| Phase (median of loops 58-174) | Alone (157) | Shared (145) |
| --- | --- | --- |
| Rollout (64 envs × 1000 steps) | 3.95 s | 6.45 s |
| World model (13 steps) | 3.59 s | 7.00 s |
| Dream generation (4 chunks × 512 dreams) | 1.82 s | 3.75 s |
| PPO (10 epochs × 32 minibatches = 320 updates) | 1.45 s | 2.36 s |
| Loop | 10.85 s | 19.65 s |

- **Two runs on one GPU give only about 10 % more throughput than one run** (2 loops in 19.65 s, against 1 loop in 10.85 s). Without NVIDIA MPS, the GPU gives each process its own time slice. The kernels of the two processes do not run at the same time. The world-model step is GPU-bound (258 ms alone, 510 ms shared). Thus each run gets half of the GPU in that phase.
- The world-model step uses the GPU well: CUDA time equals wall time. The model is small (dim 128), so the work is elementwise and memory-bound. dcgmi shows the tensor cores at 3-9 %.
- The rollout, the generation, and PPO are launch-bound: many small kernels and Python overhead. A py-spy sample of one run (90 s) shows the main thread at 96 % of one core. In the rollout, the time goes to:
  - the `.cpu()` sync of the actions: 10 % of all samples (the wait for the GPU, which the other process holds);
  - the call of the compiled forward (guards and launch): 7.6 %;
  - the env step: 5.9 %.
- Rollout alone: 3.0 ms of model and orchestration for each env step (1,000 sequential steps).
- Host memory: each run has 8 env workers. Each worker holds about 470 MB of private memory, because the spawn start imports `train.train_gym` and thus torch: 7.5 GB for two runs. The swap holds 30 GB. A run in progress had 4.2 GB in swap.
- `/tmp` (RAM tmpfs) holds 292 nvcc files `tmpxft_*` from 2026-10-04 (6 GB). They are not from these runs. I did not delete them.

### 2.2 Speed-ups that do not change gradients (a bitwise check is still necessary)

| # | Change | Where | Expected gain |
| --- | --- | --- | --- |
| S1 | Run the two lanes under NVIDIA MPS (`nvidia-cuda-mps-control -d` before the queue starts). Kernels of the two processes then run together. | system | Up to 1.5-1.8× queue throughput (SM occupancy is 0.18-0.36). MPS on a GeForce card needs a test. |
| S2 | PPO: compute the clip fraction for each minibatch on the GPU, without `.item()`. Compute the other metrics (`agent_approx_kl_metrics`, about 15 `.item()` calls, a second head forward) only for the last minibatch. Closer mode logs only that one. | `train_cleanrl_updated_ppo_dream` | 30-50 % of the PPO phase (0.5-1 s for each loop shared) |
| S3 | Rollout: write into preallocated `[B, 1001, ...]` buffers in place of `safe_cat` for each step. Each step now copies all earlier steps: about 210 GB of GPU memory traffic for each rollout (the policy input alone is 64 × 896 floats for each step). | `interact_with_env` | 0.3-0.6 s for each loop |
| S4 | Env workers in a small module that imports only gymnasium and numpy | `_grouped_env_worker` | About 6.5 GB of host RAM for two runs; less swap |
| S5 | Batched replay windows: draw the starts in the same order (`torch.randint` for each item), then one fancy-index read for each field | `ReplayDatasetTimeWindow` use | Small (median load 7-8 ms); it removes the slow outliers |

### 2.3 Speed-ups that change numerics (behind flags, by the default-path rule)

| # | Change | Expected gain |
| --- | --- | --- |
| F1 | CUDA graph for one whole rollout step (tokenizer, forward, heads, sampling) with a static time KV cache of 1,001 frames and a length mask | Rollout 3.95 s → about 1.5 s alone. This is the largest single gain. |
| F2 | One generation of 2,048 dreams in place of 4 chunks of 512 (the RNG order changes) | Generation 1.8 s → about 0.6-0.8 s |
| F3 | CUDA graph or `torch.compile` for the closer-PPO minibatch step (static shapes) | PPO 1.45 s → about 0.5 s |
| F4 | No terminal head when the env cannot end an episode (HalfCheetah). The head has 5.3 M parameters (8 MLPs of width 512, one for each MTP distance), 5× the transformer. It gives about 0.7 TFLOP for each world-model step. | 5-10 % of the world-model step |
| F5 | bf16 for generation and rollout, with float32 heads and log-probabilities | Small at dim 128; large at 3D scale |

With S1-S3 and F1-F3, an estimate is 6-7 s for each loop alone, against 10.85 s.

## 3. Changes for a 3D Navigation Environment with Images

### 3.1 Necessary before the first image run

1. **Image path in `train/train_gym.py`.** It does not exist. The trainer has only the state tokenizer (`ObservationTokenizer`, `obs_to_latents_fn`), and the replay stores latents and `critic_state`. The image path needs:
   - a stage that pretrains the causal video tokenizer (MAE with p ~ U(0, 0.9), MSE + 0.2 LPIPS, tanh bottleneck);
   - `obs_to_latents_fn` with the tokenizer time cache;
   - a replay that stores the latents (16-bit) and, on disk, uint8 frames for a re-encode.
2. **Upstream fixes** (`notes/upstream_dreamer4_commits.md`): `bec034b5ab`, `3291467c8f` item 1, and `f1fbbb504f` item 2 if the actor or critic heads get depth. Then do a bitwise check of the state path.
3. **Policy readout.** Set `policy_all_tokens: false`. Use agent tokens (several) or an attentive probe (report/10-5-26-full-finetune-design).
4. **The compiled short-sequence bug.** In the default path, the first dream frame is wrong (`EagerWithoutTimeCacheCallable` corrects it only for the joint flag). Fix it in the default path before images. At more spatial tokens the error can be larger.
5. **Terminals in dreams.** `ImaginationGenerateRollout` calls `generate(return_terminals = False)`. A navigation episode ends at the goal, so the dreams must end too: the continuation flag of eq. 10. Replace the 8 × 512-wide terminal MLPs with a small head.
6. **Actions.** Discrete navigation actions need the categorical head (as in the paper). The offline path accepts only Beta (`validate_offline_config`), and the closer-PPO path is tested only with Beta. Test the discrete path before the runs.
7. **Goal input.** For goal-conditioned navigation, give the goal (task embedding or goal image) to the agent tokens only, as in the paper. World tokens do not read agent tokens, so the goal cannot leak into the dynamics.

### 3.2 Model and data

| Item | HalfCheetah now | 3D starting point (16 GB GPU) | Reason |
| --- | --- | --- | --- |
| Input | 17 states → 4 × 32 latents | 64×64 or 96×96 RGB, patch 8 → 64-144 patches | rendering and memory budget |
| Tokenizer bottleneck | 4 × 32 | 32-64 × 16-32, then reshape to N_z 16-64 tokens | paper: 512 × 16 → 256 × 32; a small bottleneck lost velocity here |
| Dynamics | dim 128, depth 3, time layer every 2 | dim 384-512, depth 8-12, time layer every 4, GQA, 4-8 registers | paper section 3.4 |
| Context | 200-frame windows | short and long batches in turn (for example 32 and 128 frames) | paper: T1/T2 alternation; memory for navigation |
| Start frames | none | 30 % of the batch as single images | paper section 4 |
| Spatial attention | 9 tokens | 64-256 tokens: flash attention (SDPA) for space; time attention with a KV cache | memory-bound at more tokens |
| Precision | bf16 world-model loss only | bf16 everywhere, float32 heads and log-probabilities | speed |
| Memory | micro-batch 32 | micro-batch, selective activation checkpointing (`configure_sac`) | 16 GB |

### 3.3 Training

- **Tokenizer data.** Here, a tokenizer frozen after random-policy data compressed the velocity. For 3D, pretrain the tokenizer on diverse frames (random and scripted exploration, then replay). Keep it frozen in the RL phase, or re-encode the replay after each tokenizer update.
- **Rare events.** Goal reached, collision and stuck states are rare, like the flip onsets. Make `FlipOnsetTimeWindow` general: an event-window sampler for reward > 0, terminals, and collisions. Use it for world-model windows and for dream prompts.
- **Pretraining data.** HalfCheetah showed that expert-only data confounds action size with speed. Collect data with varied policies (random walk, scripted explorer, noisy expert). Check the confound before the pretrain (action-outcome correlation).
- **Loss scale.** Use one RMS normalizer for flow + bootstrap, and normalizers for reward, terminal and BC. Use the gradient-scale operator for the RL terms (report/10-5-26-full-finetune-design).
- **One optimizer** with groups (section 1).
- **Speed (section 2).** At 3D scale, F1 (CUDA graph with a static KV cache) and bf16 become necessary. Image envs are slower to render: use a GPU renderer or more env workers. Send frames as uint8 through shared memory, not through pickled pipes.
- **Checks.**
  - Decode dreams to video.
  - Measure action following as in paper Figure 7: PSNR of action-conditioned generations against held-out video.
  - Use counterfactual probes (the flip probe is the model).
  - Check dream terminals against real terminals.

## 4. Sources

- Timings: TensorBoard `timings/` of entries 145, 157, 136, 139 (queue 10-4-26).
- py-spy sample: 90 s of process 2168278 (2,941 samples), `--nonblocking`.
- [CUDA MPS: when to use MPS](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html): without MPS, each process gets a time slice of the whole GPU.
- `dreamer4.pdf` sections 3.1-3.4, 4, appendix A.
