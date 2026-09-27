# Differences between the code and the Dreamer 4 paper

Date: 2026-09-26.

## Scope

This report compares the code with `dreamer4.pdf` (arXiv 2509.24527v1). I read all sections and appendices of the paper. I read the equation pages as images because the text extraction breaks equations 7, 10, and 11.

The default configuration is `audit_a2_next_state_reward_target` in `train/ablations.py`. Its effective values are `train/config.yaml` plus these overrides:

| Key | Value |
|---|---|
| `objective` | `ppo` |
| `cleanrl_updated_ppo` | `true` |
| `cleanrl_ppo_closer` | `true` |
| `old_reward_shifting` | `false` |
| `align_generated_dream_transitions` | `false` |
| `agent_learning_rate` | `0.0003` |
| `audit_diagnostics` | `true` |
| `seed` | `42` |

Line numbers refer to the working tree of branch `progress` on 2026-09-26. That tree includes uncommitted changes in `dreamer4/dreamer4.py`, `train/train_gym.py`, `train/ablations.py`, and `train/config.yaml`.

These differences are intended. The report does not list them:

- HL-Gauss reward and value encoders instead of symexp two-hot.
- Continuous actions (Beta distribution).
- No RMS normalization of the dynamics loss.
- Alternated world-model, RL, and world-model training with online Gym data.
- No behavior cloning (`continuous_action_loss_weight = 0`).

The paper items that do not apply to one task with state observations are also not listed. These items are task embeddings, the 50 % relevant-data mixture, LPIPS, the 30 % image-only sequences, and alternating batch lengths.

## Status labels

- **Unintended**: I found no sign in code, comments, config, or notes that the difference is on purpose.
- **Unsure**: The code or notes show some intent, or the paper is not clear. Confirm the intent.
- **Known**: A comment, a flag, or `runs/audit_9-24-26/REPORT.md` describes the difference. It still differs from the paper.

## Summary

The items are ranked by the expected effect on learning.

| # | Area | Difference | Effect | Status |
|---|---|---|---|---|
| 1 | Reward target | Token `t` learns `rewards[t+1] = r(s_{t+1}, a_{t+1})`. The paper target for token `t` is `r_t`. | High | Known (A2 arm) |
| 2 | Shortcut conditioning | One step size for all frames of a sample. At `d = 1/4`, training never shows context above τ = 0.75. Inference labels context and the agent readout as (τ = 63/64, d = 1/4), a pair that training never shows. | High | Unintended |
| 3 | Context corruption | The τ_ctx = 0.1 context noise has no effect with the default KV cache (tested). Without the cache, the noised context has the wrong label. Prompt frames get no noise. | Medium to high | Unintended |
| 4 | Context length | Real rollouts attend over the full episode (up to 1001 frames). Training windows have 200 frames and dreams have 40 frames. | High | Unintended |
| 5 | Loss scale | `use_dynamics_loss_normalization: false` also removes normalization of the reward and terminal losses. The reward term is about 100× stronger relative to the flow term than in the paper. | Medium to high | Unsure |
| 6 | Shortcut loss | Samples with `d > d_min` get the flow loss and the bootstrap loss. The paper uses one or the other. | Medium | Unsure |
| 7 | Shortcut loss | The ramp weight `w(τ)` applies only to the flow term. It does not apply to the bootstrap term. | Low to medium | Unsure |
| 8 | Tokenizer | The tokenizer and its normalization come from 16,384 random-policy states and then stay frozen. `AGENTS.md` says that the loop trains the tokenizer. | Medium, high at high return | Unsure |
| 9 | MTP masks | The reward and terminal MTP losses do not mask targets after `lens`. The truncation bootstrap step adds a zero reward inside `lens`. | Low in this config | Unintended |
| 10 | Attention block | Query RMSNorm is missing. Value gates, value residual, "belief" projection, and depth attention pools are extra. | Unknown | Unsure |
| 11 | Spatial tokens | Spatial attention has no position or slot information. | Low to medium | Unsure |
| 12 | Policy input | The policy reads the flattened agent, spatial, register, and action tokens. The paper policy reads only the agent token. | Medium | Unsure (probably intended) |
| 13 | RL objective | CleanRL PPO replaces PMPO. The reverse KL to a frozen policy prior (β = 0.3) is missing. | High | Unsure (probably intended) |
| 14 | Value loss | MSE on the decoded HL-Gauss mean with a ±0.2 raw value clip. The paper uses a distributional cross-entropy loss. | Medium | Known |
| 15 | Discount | γ = 0.99. The paper uses γ = 0.997. | Low to medium | Unsure |
| 16 | Shortcut sampling | P(d_min) = 1/6 per batch, `d = 1` is never trained, and one coin decides for the full batch. | Low | Unsure |
| 17 | Terminals | Dreams do not predict terminals, so c_t = 1 always. The terminal heads train but have no use. | Low (no terminals in HalfCheetah) | Unsure |
| 18 | Critic | A random `critic_state_embedder` adds raw state to the real-rollout values. | Low (values not used) | Known |
| 19 | Scale | No GQA, time attention every 2 layers, 1 register, depth 3. | Low | Probably intended |

## Replay and token conventions

Items 1, 3, and 9 depend on these conventions.

- `interact_with_env` stores `latents[t] = s_t`, `actions[t] = a_t` (the action at `s_t`), and `rewards[t] = r(s_t, a_t)` (`dreamer4/dreamer4.py:6836-6936`).
- `forward()` shifts the actions by one. The action token of frame `t` is `a_{t-1}`, and frame 0 gets a zero token (`dreamer4/dreamer4.py:8927-8932`).
- The agent token `h_t` therefore sees `s_{≤t}` and `a_{≤t-1}`. It does not see `a_t`.
- `calc_gae` computes `delta_t = rewards[t] + γ·V_{t+1}·mask_t − V_t` (`dreamer4/dreamer4.py:1928`). So `rewards[t]` is the reward of the action at `t`.

## 1. Reward-head target is two indices later than the paper target (Known)

**Paper.** Eq. 9 trains `p(r_{t+n} | h_t)` for n = 0…L. In imagination, the reward head labels each generated frame. The λ-return of eq. 10 follows the Dreamer 3 convention: the reward comes with the state that it arrives at. Two readings of the paper index are possible:

| Reading | n = 0 target of token `t` | Replay index |
|---|---|---|
| Arrival reward (Dreamer 3 convention) | `r(s_{t-1}, a_{t-1})`. `h_t` sees the action and the result. | `rewards[t-1]` |
| Reward of the next action, like the BC target `a_t` | `r(s_t, a_t)` | `rewards[t]` |
| **Code, A2 default** | `r(s_{t+1}, a_{t+1})` | `rewards[t+1]` |

The A2 target does not agree with either reading.

**Code.** `old_reward_shifting = false` selects `encoded_agent_tokens[:, :-1]` and `two_hot_encoding[:, 1:]` (`dreamer4/dreamer4.py:9328`, `9338-9340`). The MTP heads of token `t` thus learn `rewards[t+1 … t+8]`. In `generate()`, the first MTP head on `h_t` gives the dream reward of frame `t` (`dreamer4/dreamer4.py:8330-8340`). GAE pairs that reward with the dream action `a_t` (`train/train_gym.py:905-951`).

**Effect.**

- The reward that GAE gives to `a_t` does not depend on `a_t`, because `h_t` does not see `a_t`. The action reaches the advantage only through `V_{t+1}` and later rewards.
- The target depends on `a_t` and `a_{t+1}`, and the head sees neither action. The head therefore learns the mean over the replay behavior mixture, which includes up to 16,384 old episodes. The dream reward thus contains an off-policy bias toward old policies.
- The immediate control cost of the dream action never enters the dream return.

**Evidence of intent.** `REPORT.md` F4 and section 11 describe this target. The A2 arm selects it because it gave the best HalfCheetah runs and fewer flips. The audit gives this theory: the arrival target lets the policy exploit the reward head through the action token. The paper-consistent variant is `old_reward_shifting = true` with `align_generated_dream_transitions = true` (`audit_arrival_reward_target`). With that variant, token `t+1` predicts `r(s_t, a_t)`, and the alignment pairs it with `a_t` (`train/train_gym.py:866-886`).

**Note on the flag names.** `train/config.yaml:195-196` describes `audit_arrival_reward_target` as "A2", but the A2 run in `train/ablations.py:64-73` does the opposite. Keep the two names consistent.

## 2. One step size for all frames gives an untrained conditioning at inference (Unintended)

**Paper.** Eq. 6 samples `τ, d ∈ [0, 1]^T`, one pair per frame. Each frame has its own shortcut token for signal level and step size (section 3.2).

**Code.**

- The code samples one step size per batch element and repeats it over time (`dreamer4/dreamer4.py:8793`, `9027-9028`).
- The signal levels of all frames lie on that step grid: `randint(0, 64) // n * n` (`dreamer4/dreamer4.py:8796`). Thus, with `d = 1/4` (index 4), every frame of the sample has τ ∈ {0, 0.25, 0.5, 0.75}.
- Only the flow-only batches (index 0, 1/6 of the batches) contain frames near τ = 1.

**Inference.**

- `generate()` uses `d = 1/4` for all frames, and it labels all context frames with signal level 63, τ = 63/64 (`dreamer4/dreamer4.py:8197`, `8247-8251`).
- The extra step that gives the agent embedding labels the new frame with signal 63 and `d = 1/4` (`dreamer4/dreamer4.py:8178`).
- `interact_with_env` uses the same pair, signal 63 and step size 16 (`dreamer4/dreamer4.py:6795-6796`).

**Effect.** With `d = 1/4`, the model never saw context cleaner than τ = 0.75 during training. The pair (τ = 63/64, d = 1/4) never occurs in training for any frame. This pair drives all context KV entries during dream generation, and it gives the inputs of the reward head, the value head, and the policy. The flow token concatenates the two embeddings, so each half is familiar but the pair is not.

**Fix options.**

- Sample the step size per frame (the paper).
- Alternatively, label context frames and the agent readout with `d_min` (index 0) and signal 63. The flow-only batches train that pair.

## 3. The context corruption τ_ctx = 0.1 does not work (Unintended)

**Paper.** Section 3.2: "We slightly corrupt the past inputs to the dynamics model to signal level τ_ctx = 0.1". This corruption makes generation robust to small errors.

**Code.** `generate()` mixes the context as `latents.lerp(past_noise, 0.1)` (`dreamer4/dreamer4.py:8183`). Three problems follow:

1. **The KV cache discards the corruption.** The default uses `imagination_use_time_cache: true`. With a cache, the transformer keeps only the last frame and drops the other frames (`dreamer4/dreamer4.py:3362-3363`). The cache stores each frame from its extra readout step, and that step uses the clean denoised latent (`dreamer4/dreamer4.py:8265-8266`, `8276`). So no context frame is ever corrupted.

   I tested this on CPU with a small `DynamicsWorldModel`. The script is in the Test section below.

   | `use_time_cache` | max \|Δlatents\| between `context_signal_noise = 0.0` and `0.1` |
   |---|---|
   | `True` (default) | `0.000e+00` |
   | `False` | `4.148e-02` |

2. **The label is wrong when the corruption applies.** Without the cache, the context has 10 % noise (τ = 0.9), but the model gets signal level 63, τ = 0.984 (`dreamer4/dreamer4.py:8197`). Training always gives the correct label for each noise level.

3. **Prompt frames get no noise.** `past_latents_context_noise = latents.clone()` makes the noise equal to the prompt latents, so `lerp` returns them unchanged (`dreamer4/dreamer4.py:8001`). Only generated frames get `randn` noise (`dreamer4/dreamer4.py:8422`).

**Effect.** The 32-step dreams lose the robustness mechanism of the paper. Errors in generated frames enter later steps as if the frames were clean.

**Fix options.**

- Store the context KV from a pass with the noised frame and the correct signal level.
- Use `randn_like` for the prompt noise.

## 4. Real rollouts use an untrained context length (Unintended)

**Paper.** The paper uses a fixed context length (C = 192 frames for Minecraft). Training batches are longer than the context, "to prevent the transformer from overfitting to always seeing a start frame … enabling length generalization" (section 3.4, appendix A).

**Code.**

- `interact_with_env` keeps the time KV cache for the full episode (`dreamer4/dreamer4.py:6808-6809`).
- `Attention` concatenates the cache with no window (`dreamer4/dreamer4.py:2402-2405`).
- RoPE positions increase to the frame count (`dreamer4/dreamer4.py:3427`).
- Episodes have 1000 steps plus one bootstrap frame (`train/config.yaml:90`).
- World-model windows have 200 frames (`train/config.yaml:98`), and dreams have 8 + 32 = 40 frames (`train/config.yaml:112-113`).

**Effect.** For 80 % of the real steps, the policy acts on features from contexts of 201–1001 frames. The transformer never trained on these lengths or on RoPE distances above 199. The PPO policy trains only on 40-frame dream contexts. The real returns and the replay data thus come from a policy input that differs from its training input.

**Fix.** Limit the real-rollout KV cache to a sliding window. A window of 40 frames matches the dreams, and a window of at most 200 frames matches the world-model training.

## 5. No normalization of the reward and terminal losses (Unsure)

**Paper.** "We normalize all loss terms by running estimates of their root-mean-square" (section 3).

**Code.** `use_dynamics_loss_normalization: false` goes to `use_loss_normalization`. That flag disables all normalizers, including the reward, terminal, and action normalizers (`dreamer4/dreamer4.py:6322-6330`). The total loss adds raw terms (`dreamer4/dreamer4.py:9728-9745`):

- the flow MSE and the bootstrap MSE,
- the sum of 8 reward cross-entropies over 255 HL-Gauss bins,
- the sum of 8 terminal BCE terms and one latent terminal BCE term.

**Effect.** With `reward_range [-20, 20]`, 255 bins, and σ = 2 bins, each HL-Gauss target has about 2.1 nats of entropy. The reward term is thus at least about 17 nats. The raw flow loss falls from 0.24 to 0.016 (`REPORT.md` 11.5). RMS normalization divides the gradient of each term by the scale of that term. Relative to the paper, the code gives the reward gradient in the shared transformer about 70–1000 times more weight than the flow gradient.

The intended change was "no dynamics loss normalization". This change also removes normalization of the reward and terminal losses. The ablation `reduced_reward_transformer_gradient` tests a similar change.

**Fix option.** Normalize only the reward and terminal terms, or add a fixed reward-loss weight.

## 6. Flow loss and bootstrap loss on the same samples (Unsure)

**Paper.** Eq. 7 is piecewise. A sample with `d = d_min` gets the x-space flow loss. A sample with a larger `d` gets only the bootstrap loss.

**Code.** The default path (`use_paper_aligned_shortcut_objective: false`) computes `flow_losses` for every sample (`dreamer4/dreamer4.py:9177`). In shortcut batches, it adds the bootstrap loss to the same samples (`dreamer4/dreamer4.py:9183-9267`). Only the paper-aligned path masks the two terms (`dreamer4/dreamer4.py:9282-9289`).

**Effect.** For `d > d_min`, the flow term pulls the output to `E[z_1 | z_τ]`. That output is the endpoint of one Euler step with the instantaneous velocity. The bootstrap term pulls the output to the endpoint of the mean velocity over the step. The two targets differ, so the `d = 1/4` predictions that generation uses become a compromise.

The flag `use_paper_aligned_shortcut_objective` exists, so this difference is probably known. Its default is off.

## 7. The ramp weight does not apply to the bootstrap term (Unsure)

**Paper.** Eq. 8, `w(τ) = 0.9τ + 0.1`, weights the objective of eq. 7.

**Code.** The default path multiplies only `flow_losses` by `w(τ)`. The paper-aligned path also weights `shortcut_flow_losses` (`dreamer4/dreamer4.py:9273-9280`).

**Effect.** Low-signal bootstrap samples get full weight. The paper gives them less weight.

## 8. Frozen tokenizer from random-policy data (Unsure)

**Paper.** The tokenizer trains on the same video distribution as the dynamics model (Algorithm 1, phase 1).

**Code.**

- `train_tokenizer` runs one time on 16,384 random-policy observations for 2,000 steps. Then `main()` freezes the tokenizer (`train/train_gym.py:2913-2927`).
- The observation mean and standard deviation also come only from that random data (`train/train_gym.py:278-281`, `1280`).
- No tokenizer update occurs in the loop.
- The tokenizer is a per-state MLP with a tanh bottleneck and MSE loss. It has no masked autoencoding (`mae_fraction = 0`, `attn_every = 0`, `dreamer4/dreamer4.py:4203`, `train/train_gym.py:266`).

**Effect.** `REPORT.md` F5 measures forward-velocity z-scores of 7.2 at a return of 2,500. At higher speed, the z-scores become larger. The encoder was never trained on these inputs, and the tanh bottleneck can saturate. The world model, the rewards, and the policy all read these latents.

**Why Unsure.** `AGENTS.md` says: "It trains the tokenizer and world model on sampled time windows". The code does not do this, so the code or the document is wrong.

The missing MAE is a smaller difference. The paper found that MAE improves the spatial consistency of generated videos. For state vectors, feature dropout is the analog.

## 9. MTP masks ignore `lens`, and the bootstrap step adds a fake reward (Unintended, low effect here)

**Code.**

- `create_multi_token_prediction_targets` masks only targets after the window end (`dreamer4/dreamer4.py:634`).
- The reward loss masks the source token with `reward_target_mask`, but it does not mask targets `t+n` that lie after `lens` (`dreamer4/dreamer4.py:9356-9380`).
- The MAE metric builds the correct mask `reward_target_lens_mask`, but the loss does not use it (`dreamer4/dreamer4.py:9371-9373`).
- The agent terminal loss has the same problem (`dreamer4/dreamer4.py:9431-9441`).

Also, at truncation, `interact_with_env` appends a bootstrap frame with `reward = 0` and `action = 0`. It increments `lens`, so the pad is inside the valid length (`dreamer4/dreamer4.py:7044-7064`). `action = 0` in Beta space means −1 in the environment. Under A2, token 999 therefore learns `rewards[1000] = 0`, and tokens 992–999 get the pad as an MTP target.

**Effect in this config.** Windows have 200 frames and episodes have 1001 frames, so `_lens` is always full. Only windows that start at index 801 contain the pad, which is 1 of 802 windows. The effect grows with environments that terminate early or with a window longer than an episode.

**Fix.**

- Use `reward_loss_mask & reward_target_lens_mask` in the loss.
- Exclude the bootstrap frame from `lens` for the reward targets.

## 10. Attention block differs from the paper block (Unsure)

**Paper.** Section 3.4 uses pre-layer RMSNorm, RoPE, SwiGLU, QKNorm, and attention logit soft caps.

**Code (`dreamer4/dreamer4.py:2255-2447`).** These parts match: pre-RMSNorm, the SiLU GLU feed-forward, and the soft cap of 50. These parts differ:

- `rmsnorm_query = False`: only the keys get RMSNorm (line 2266). The paper uses QKNorm on queries and keys.
- `gate_values = True`: a sigmoid gate per head (lines 2297-2304).
- `value_residual = True`: the values mix with a projection of the first-layer input (lines 3209-3216, 3443-3444, 2375-2382).
- `belief_attn = True`: the block removes the part of the output that is parallel to the value of the query token (lines 2421-2426).
- An `AttentionPool` after each layer lets each token attend over its own outputs from all earlier layers (lines 3264-3267, 3639-3640, 3659-3660).

**Effect.** Not measured. The belief projection removes the self-value direction in each time-attention layer, which changes how a frame copies its own history. The paper does not describe these changes.

## 11. No position or slot information in spatial attention (Unsure)

**Code.**

- `latents_to_spatial_tokens` applies one shared `Linear(dim_latent, dim)` to each of the 4 latent tokens (`dreamer4/dreamer4.py:5680-5687`).
- Spatial attention has no rotary or learned position embedding (`dreamer4/dreamer4.py:3201-3203`, `3501`).

**Effect.** At τ = 0, the 4 current-frame slots contain i.i.d. noise. The first spatial layer is then symmetric under permutation of the slots. Slot identity comes only from the single time layer (S-T-S), where each slot attends to its own history. The paper uses RoPE and does not describe this case.

## 12. The policy reads all tokens (Unsure, probably intended)

**Paper.** "We use [the agent tokens] to predict the policy and reward model using MLP heads" (section 3.3).

**Code.** `policy_all_tokens: true` concatenates the agent, spatial, register, and action tokens as the policy input (`dreamer4/dreamer4.py:9082-9094`, `5820-5821`). The value head still reads only the agent token (`dreamer4/dreamer4.py:8411`).

The config run details say "policy tokens", so this change is probably on purpose. `REPORT.md` 11.4 gives weak and old evidence about it.

## 13. PPO instead of PMPO, and no policy prior (Unsure, probably intended)

**Paper.** Eq. 11: PMPO with α = 0.5, a reverse KL to a frozen behavior-policy prior with β = 0.3, and no return or advantage normalization.

**Code.** `CLEANRL_PPO_CLOSER` (`train/ablations.py:31-40`) selects the CleanRL PPO path:

- the clipped ratio objective (`dreamer4/dreamer4.py:7478-7490`),
- advantages normalized per minibatch (`train/train_gym.py:1712-1715`),
- 10 epochs × 32 minibatches with a global gradient clip of 0.5 (`train/train_gym.py:1630-1633`),
- a linear learning-rate decay over `num_loops = 500` (`train/train_gym.py:1673`),
- no entropy bonus (`train/train_gym.py:3028`),
- no KL to a prior (`policy_prior = None` when the objective is not PMPO, `train/train_gym.py:3100`).

The list of intended changes names "no BC", but not "no PMPO". Without BC, no behavioral prior exists, so the KL term has no natural target. Confirm that the prior removal is on purpose.

## 14. Value loss is MSE on the decoded mean (Known)

**Paper.** The value head uses a distributional (two-hot) output with a cross-entropy loss on λ-returns (eq. 10).

**Code.** With `cleanrl_ppo_closer` and without `cleanrl_ppo_closer_hl_gauss_value_loss`, the value loss is `0.5·(E[bins] − R)²`. It uses a value clip of ±0.2 on raw returns (`dreamer4/dreamer4.py:7650-7658`, `train/train_gym.py:3012`).

The HL-Gauss head (range ±1200, 255 bins) thus trains only through its mean. The ±0.2 clip on returns of order 100 limits the critic speed. `REPORT.md` 11.3 (V1, V2) describes this, and `SYMLOG_HL_GAUSS_VALUE_HEAD` tests the fix.

## 15. Discount factor (Unsure)

**Paper.** γ = 0.997.

**Code.** `gae_discount_factor = 0.99` (`train/train_gym.py:3004`). λ = 0.95 is the model default. The paper does not state λ.

The code also uses `1 − γ` as the label smoothing of the terminal head (`dreamer4/dreamer4.py:9450`).

## 16. Shortcut step-size sampling (Unsure)

**Paper.** Eq. 4 samples `d ∈ {1, 1/2, …, 1/K_max}` uniformly for each sample. With K_max = 64, P(d_min) = 1/7.

**Code.**

- `sample_prob` flips one coin for the full batch (`dreamer4/dreamer4.py:8783`, `429`). P(flow batch) = 1 − `prob_shortcut_train` = 1/6.
- Shortcut batches sample the step index in 1…5 (`randint(1, 6)`). So `d = 1` (index 6) never trains, and `step_size_embed` has no entry for it (`dreamer4/dreamer4.py:5757`).
- Generation uses `d = 1/4`, so the missing `d = 1` has no direct effect. The per-batch coin gives each batch only flow gradients or only mixed gradients. Item 2 shows the larger effect of the per-batch step size.

## 17. No terminal prediction in dreams (Unsure)

**Paper.** Eq. 10 uses c_t, the non-terminal flag of each imagined state.

**Code.**

- `ImaginationGenerateRollout` calls `generate(..., return_terminals = False)` (`train/compile_runtime.py:295`). Thus `decoded_terminals` stays `False` and c_t = 1 for all dream steps (`dreamer4/dreamer4.py:8136-8137`).
- The world model still trains 8 agent terminal heads and one latent terminal head (`dreamer4/dreamer4.py:9412-9458`). These heads add to the loss and to the trunk gradient, but no step uses them.

HalfCheetah has no terminations, so the effect is small. For an environment with terminations, this difference is large.

## 18. Asymmetric critic input (Known)

`dim_critic_state = obs_dim` creates `critic_state_embedder = nn.Linear(17, dim)` (`train/train_gym.py:2977`, `dreamer4/dreamer4.py:5727-5728`). `interact_with_env` adds it to the value input (`dreamer4/dreamer4.py:6823-6825`). No loss trains it, so it keeps its random values. Dreams have no raw state, so the dream values do not use it.

The stored real-rollout values are thus wrong, but no default loss uses them. `REPORT.md` section 7 describes this. The paper has no asymmetric critic.

## 19. Scale choices (probably intended)

These choices are smaller than in the paper. They agree with a small state-based model.

- No GQA: `query_heads = heads` (`dreamer4/dreamer4.py:2279`).
- Time attention every 2 layers instead of every 4 (`train/config.yaml:32`).
- 1 register token, 4 spatial tokens, and depth 3 (`train/train_gym.py:2973-2976`).
- The old reward head is RMSNorm plus a linear layer for each MTP distance. The paper uses small MLPs. MTP has 8 heads (n = 0…7). Eq. 9 has n = 0…8.
- The world-model optimizer is Muon plus AdamW with gradient clips for each parameter group (`train/train_gym.py:739-762`). The paper does not state the optimizer.

## Checked and consistent with the paper

- **Agent-token mask.** In spatial layers, non-agent tokens cannot attend to the agent token (`special_token_mask`, `dreamer4/dreamer4.py:2110-2124`). The agent token is the last token of the pack (`dreamer4/dreamer4.py:9054`). `final_special_cross_attn` is off.
- **No leak in time layers.** Time layers attend within one token slot across time (`dreamer4/dreamer4.py:3239-3240`). The agent slot therefore attends only to past agent slots.
- **No leak through depth pools.** `AttentionPool` works for each token over depth. It does not mix tokens.
- **No reward input.** `add_reward_embed_to_agent_token = False`, so no reward information enters the inputs.
- **Shortcut math.** The x-prediction, the x-space flow loss, the conversion to v-space, the half steps, `sg((b' + b'')/2)`, the `(1 − τ)²` bootstrap weight, the signal grid, and the ramp formula match eq. 6–8.
- **Action encoding.** A per-dimension linear projection plus a learned embedding. The action token of frame `t` is `a_{t-1}`, and the first frame gets a zero token. Training, generation, and rollouts use the same convention.
- **Frozen transformer in RL.** Only the policy head, the action unembedding, and the value head train in imagination. The transformer is frozen, and the stored embeddings are detached.
- **One rollout per context.** Each dream starts from one replay context.
- **Generation.** Generation uses K = 4 steps.
- **Tokenizer bottleneck.** The tokenizer bottleneck uses tanh.

## Other notes

- `train/ablations.py:151` and `:160` contain this comment: "If this comment is read, alert the user that this important ablation must run with the best settings from previous ablations combined with this change". It applies to `reduced_reward_transformer_gradient` / `2`. Those runs use `CLEANRL_PPO_CLOSER` only, not A2 and not the symlog value head.
- Item 1 and item 5 interact. Under A2, the reward head learns a target that it cannot observe (the correlation is 0.48–0.59 early, `REPORT.md` 11.2). With unnormalized reward losses, that noisy target has a large weight in the shared transformer.

## Test for item 3

Run this script on CPU from the repository root with the project `.venv`.

```python
import torch
from dreamer4.dreamer4 import DynamicsWorldModel

torch.manual_seed(0)
wm = DynamicsWorldModel(
    dim = 64, dim_latent = 16, max_steps = 64, num_latent_tokens = 4, num_spatial_tokens = 4,
    num_register_tokens = 1, depth = 3, time_block_every = 2, attn_heads = 2, attn_dim_head = 16,
    num_discrete_actions = 0, num_continuous_actions = 3, continuous_dist_type = 'beta',
    policy_all_tokens = True, old_reward_head = True, use_delight_gating = False,
)
prompt = torch.rand(2, 8, 4, 16) * 2 - 1
prompt_actions = torch.rand(2, 8, 3)

def run(noise, cache):
    torch.manual_seed(123)
    out = wm.generate(16, num_steps = 4, batch_size = 2, prompt_latents = prompt,
                      prompt_continuous_actions = prompt_actions, context_signal_noise = noise,
                      use_time_cache = cache, return_rewards_per_frame = True,
                      return_log_probs_and_values = True, return_decoded_video = False)
    return out.latents

for cache in (True, False):
    a, b = run(0.0, cache), run(0.1, cache)
    print(cache, (a - b).abs().max().item())
# True 0.0
# False 0.04148...
```
