# Policy Readout and Full-Transformer Finetuning with One Optimizer (2026-10-05)

## 1. Summary

- **Does the policy read all tokens?** Yes. `policy_all_tokens: true` (`train/config.yaml` line 41) is on in all queued runs. The policy MLP reads the flattened final tokens of the current frame: 1 agent + 4 spatial + 1 register + 1 action token = 7 × 128 = 896 inputs (`dreamer4/dreamer4.py` near line 9225). The value head and the reward head read only the agent token.
- **This does not scale.** With 256 spatial tokens and dim 1024 (paper scale), the first policy layer has 263 k inputs and about 1 B weights. The weights are tied to token positions, so they do not share across the image. Section 3 gives the details.
- **The paper's answer to "the agent token is not trained for the policy" is phase 2, not the readout.** Phase 2 trains the agent token with BC (8-step MTP) and reward, plus the dynamics loss. The transformer is then frozen for RL. Full finetuning is a footnote option ("small additional benefits").
- **Why the earlier joint runs failed (from the logs of entries 69-76):**
  1. The trunk did not get an RL signal. Reward was 96-98 % of the transformer gradient from scratch. Policy was below 0.05-3 %.
  2. The closer-PPO joint arm made 10 head updates per loop. The two-phase A2 runs make 960.
  3. The PMPO arm also changed the objective.
  4. With `policy_all_tokens`, the RL gradient goes straight into the final spatial tokens. The latent prediction head reads these same tokens.
- **Recommended design (Section 5):**
  - Use one optimizer with parameter groups, one loss, and one backward.
  - The policy and value read agent tokens only. Use several agent tokens or an attentive probe.
  - Set the trunk share of each loss with a gradient-scale operator at the head input, not with loss weights.
  - Add a policy-preservation KL on real replay states (the "policy prior" loss of the footnote, and the PPG clone loss). Add a value loss on real replay windows.
  - Use one shared step count for each loop, and a trunk learning rate that keeps the world-model change per loop constant.
- **Lower-risk alternative (Section 6):** untie the agent-token weights (Mixture-of-Transformers style). RL then trains the full depth of the agent stream and never changes the dynamics.

## 2. The Paper (all sections and footnotes)

| Item | Text of the paper | Consequence for this fork |
| --- | --- | --- |
| Agent tokens (3.3) | "agent tokens attend to themselves and all other modalities, no other modalities can attend back". The policy and reward heads are MLPs on the agent-token output. | The agent token is the paper's readout of the full frame. The policy does not read the spatial tokens directly. |
| Phase 2 (3.3, eq. 9) | BC and reward with MTP, L = 8. "we reuse the pretraining setting ... so the representations are noisy and we continue to apply the video prediction loss." | Phase 2 is a joint loss in one update: dynamics + BC + reward. The BC term trains the agent token for the policy. This fork has no BC phase. |
| Phase 3 (3.3) | "We only update the policy and value heads and keep the transformer frozen." Value: TD(λ), symexp two-hot. Policy: PMPO, α = 0.5, β = 0.3, reverse KL to a frozen BC prior. | The main recipe is head-only RL on a policy-trained agent token. |
| Footnote † (page 7) | "Finetuning the full transformer provides small additional benefits at higher computational cost. For that, the dynamics, policy prior, and reward losses need to be applied during imagination training to preserve their functions." | Full finetuning is optional. It needs three preservation losses. No optimizer, learning rate, weight, or schedule is given. |
| Loss normalization (3) | "we normalize all loss terms by running estimates of their root-mean-square (RMS)." | In this fork, separate flow/bootstrap normalizers broke the dynamics. The grouped normalizer works. The paper does not say how RL losses are normalized. |
| Footnote * (page 6) | The x-space and v-space MSE relation motivates the (1 − τ)² multiplier of the bootstrap term. | The flow and bootstrap terms form one loss. This agrees with the grouped normalizer. |
| Multi-task data (4.1) | BC loss only on relevant sequences; dynamics loss only on uniform sequences "to avoid optimistic generations". | The paper keeps the dynamics data away from task-filtered data. A joint RL loss must not train the dynamics on dreams. |
| Dataset (A) | 256 spatial tokens (Minecraft), 512 (robotics). | The flattened readout would have 256-512 tokens × dim inputs. |

## 3. Why the Flattened Readout Does Not Scale

- **Parameters:** first-layer weights = N_tokens × dim × 4·dim. Here: 896 × 512 = 0.46 M. At 256 tokens and dim 1024: 263 k × 4096 ≈ 1.1 B, more than half of the paper's dynamics model.
- **No weight sharing:** each spatial position has its own weights. A goal at the left of the image and a goal at the right need different weights. The attention of the agent token is invariant to position (with RoPE for relative order). The MLP is not.
- **Sample efficiency:** RL gives few bits for each update. A 1 B-weight readout trained only by PPO overfits or does not move.
- **History:** the flattened input is the last frame only. In this fork the agent token also reads the past through the time layers. For 3D navigation (partial observation, memory of a seen goal), the history path is necessary.
- **Interference under full finetuning:** the latent head reads the final spatial tokens (`to_latent_pred(space_tokens)`). The flattened policy sends the RL gradient to the same tensor. An agent-token readout reaches the world tokens only through the shared weights and the keys and values of attention.

## 4. Why the Agent Token Was a Weak Policy Input Here

- In this fork the agent token gets gradients only from the reward and terminal heads (and BC if `behavior_cloning_in_world_model_updates` is on). The HalfCheetah reward is x velocity minus 0.1·|a|². Thus a reward-only agent token needs only x velocity and the action.
- Hypothesis (not tested): a reward-only agent token does not encode pitch angle and angular velocity well. A policy on that token cannot see that the body tips, which agrees with the flip failures. A linear probe of the agent token on replay states (target: `obs[1]` and `obs[10]`) can test this in minutes on CPU with saved checkpoints.
- The paper fixes this with BC (phase 2). Online, with no expert data, other losses give the same shaping:
  - **Value on real data.** The value depends on all state that matters for future reward, and a flip ends future reward. PPG uses the value loss as its auxiliary objective for this reason. TD-MPC2 trains its encoder with reward and value (Q) losses, and detaches the policy loss.
  - **Policy distillation / preservation.** KL to the current policy on replay states (the PPG clone loss; MuZero and Muesli train the representation with policy targets).
  - **Policy gradient, scaled down.** High variance, so a small trunk share.

## 5. Recommended Design: One Optimizer, One Loss, Routed Gradients

### 5.1 Readout

- `policy_all_tokens: false`. The policy and value heads read agent tokens only.
- Use more readout capacity that does not grow with the token count. Options:
  - S_agent agent tokens (4-8). The head input is S_agent × dim.
  - An attentive probe: 1-4 learned queries cross-attend to the final tokens of the current frame, then an MLP. The cost is O(N) and the parameters are O(dim²). V-JEPA uses this readout on frozen video features. The repo has `final_special_cross_attn`, which is one such layer for the special tokens.

### 5.2 The loss of each step

Each optimizer step sums all terms. There is one backward and one optimizer.

| Term | Data | Gradient into the trunk | Purpose |
| --- | --- | --- | --- |
| Dynamics (flow + bootstrap, grouped normalizer) | replay windows | full | world model |
| Reward, terminal (MTP) | replay windows | full | paper's reward loss |
| Policy preservation: KL(π_snapshot ‖ π_θ) | replay states (real) | full | paper's "policy prior" loss; PPG clone loss; limits the world-model drift of the policy (3.3× the PPO step in the audit) |
| Value TD(λ) on real windows (optional) | replay windows | scaled (α_v) | trains the agent token for long-term state |
| PPO policy loss | dreams | scaled (α_π, start 0.1) | RL |
| Value loss | dreams | scaled (α_v) or 0 | critic |

- **Dreams carry no gradient into the dynamics.** The rollout runs without gradient. The RL loss reads dream frames as fixed inputs. Thus the RL loss cannot teach the model to make optimistic dreams. Only the dynamics loss on real data trains the generation. This is the same rule as the paper's dynamics-on-uniform-data rule.
- **The policy-preservation KL is the key term.** In the current two-phase loop, the 13 world-model steps move the acting policy 3.3× more than the PPO step, and no term limits this. In the joint loss, the same KL that keeps the policy also sends a policy-shaped gradient into the trunk through the agent token.

### 5.3 Balance at the trunk, not with loss weights

- A head gets gradient only from its own loss. Adam and Muon normalize the update size, so a loss weight changes the head update very little. The trunk gets the sum, so the relative scale decides who controls it.
- Use a gradient-scale operator at each head input: the forward pass is the identity, and the backward pass multiplies by α. This sets the trunk share of a loss and keeps the head learning rate.
- Choose α from the measured shares. `joint_diag/trunk_grad_norm_*` already logs them. Target: RL + value at about 10-30 % of the dynamics norm. The earlier runs had policy below 3 %.
- Alternative: normalize the incoming gradient at the head input by its running RMS, then multiply by α. This makes the share independent of the loss scale, also for PPO, where the loss value has no fixed scale.
- Caution with Muon (`use_muon_optimizer: true`): Muon orthogonalizes the update. A small gradient component in other directions can get a larger share of the update than of the gradient. Measure the share of the update, not only of the gradient.

### 5.4 Step count and learning rate

- With one optimizer, the trunk and the heads share the step count. The earlier joint arm used 10 steps, so the heads made 10 updates instead of 960.
- Plan: generate one dream batch for each loop with the loop-start snapshot. Store old log-probabilities, values, and GAE targets. Then run K joint steps (PPO epochs × minibatches, for example K = 64-128). Each step uses one dream minibatch and 1/K of the world-model windows of the loop.
- Keep the world-model change per loop fixed: trunk learning rate ≈ 6e-4 × 13 / K (Adam-like updates move about lr for each step). The heads keep `agent_learning_rate`. Parameter groups give these two rates in one optimizer.
- The PPO ratio uses the rollout log-probabilities. Thus the ratio and the clip include the drift that comes from the trunk, so the trust region covers the full policy. Add a target-KL stop on the total KL (dreams and replay states).

### 5.5 Rules carried over from the earlier work

- Rollout with the time cache, unclamped latents, and an eager first frame (`notes/fixed_full_transformer_finetuning.md`). The learner must read the rollout inputs.
- One normalizer for flow + bootstrap. Keep RL terms out of the RMS normalizer and use α instead.
- Do not change the objective and the update count in the same arm.

## 6. Lower-Risk Alternative: Untied Agent-Stream Weights

- Give the agent tokens their own QKV, MLP, and norm weights in each layer, as Mixture-of-Transformers does for each modality. Attention stays global: agent queries read the world tokens; world tokens still cannot read agent tokens.
- Detach the world-token keys and values when agent queries read them. Then:
  - The dynamics loss never touches the agent weights (no token reads the agent tokens).
  - RL, value, and reward losses train the full depth of the agent stream and never touch the dynamics.
  - No dynamics or reward preservation loss is necessary. One optimizer holds both disjoint sets.
- The cost is small: the agent stream has 1-8 tokens for each frame. In imagination, it reads the cached keys and values of the trunk.
- What it gives up: the world tokens do not adapt to the policy (for example, more velocity resolution). The paper calls this part "small additional benefits".
- For 3D navigation with images, this is the safest first step. True full finetuning is then the same loss with α > 0 on the shared weights.

## 7. Tests to Run on HalfCheetah before the 3D Environment

| Step | Change | Question |
| --- | --- | --- |
| 0 | Linear probe of the agent token (pitch, angular velocity, vx) on saved checkpoints | Does the reward-only agent token hold the flip state? |
| 1 | `policy_all_tokens: false`, two-phase as now | How much does the agent-token readout lose today? |
| 2 | 1 + value TD on replay + preservation KL in the world-model step | Does auxiliary shaping recover the loss of step 1? This step changes no optimizer. |
| 3 | 2 + 4 agent tokens or an attentive probe | Is readout capacity the limit? |
| 4 | Joint design of Section 5 with α_π = 0 | Is one optimizer with K steps equal to two phases? |
| 5 | 4 with α_π = 0.1, 0.3 | Does the RL gradient in the trunk help? |
| 6 | Section 6 (untied agent stream) | The scalable default for images. |

Use three seeds for each arm. The run-to-run noise is about ±1,000.

## 8. Sources and Limits of Transfer

- Dreamer 4 (Hafner, Yan, Lillicrap 2025), `dreamer4.pdf`, all sections, footnotes *, †, ‡.
- [Phasic Policy Gradient](https://arxiv.org/abs/2009.04416): the value as an auxiliary loss and a KL clone term in a shared network. Limit: model-free, value on real data, separate phases.
- [TD-MPC2](https://arxiv.org/abs/2310.16828): the encoder learns from dynamics, reward, and value; the policy loss is detached. Limit: small MLP latent, no tokens.
- [Mixture-of-Transformers](https://arxiv.org/abs/2411.04996): weights for each modality with global attention. Limit: pretraining efficiency, not RL.
- [V-JEPA 2](https://arxiv.org/abs/2506.09985): attentive probe with a learned query on frozen tokens. Limit: supervised probes, not RL.
- Earlier local results: `report/10-1-26-joint-finetune/`, `report/10-1-26-grouped-normalization/`, `report/10-4-26-instability-audit/`.
