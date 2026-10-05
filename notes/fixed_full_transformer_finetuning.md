# Full-transformer finetuning in the RL loop (`fixed_full_transformer_finetuning`)

## Paper basis

Section 3.3 (behavior cloning and reward model): "we reuse the pretraining setting with this additional loss function, so the representations are noisy and we continue to apply the video prediction loss." The update sums the dynamics loss (7) and the policy and reward loss (9).

Footnote on page 7: "Finetuning the full transformer provides small additional benefits at higher computational cost. For that, the dynamics, policy prior, and reward losses need to be applied during imagination training to preserve their functions."

The paper gives no optimizer, learning rate, or loss weights for this mode. It does not say how the prior works when the transformer moves.

## Design

The flag replaces the world-model phase and the imagination phase of a loop with one joint phase. Each step does this:

1. Compute the world-model loss on replay windows (dynamics, reward, terminal, noisy inputs). This is the loss of the normal world-model phase.
2. Generate one dream batch without gradient.
3. Run one clean forward over prompt and dream with the transformer gradient. Read the policy and value heads on the generated frames.
4. Compute the PMPO loss with the policy-prior KL, and the value loss.
5. Sum the losses. Run one backward, clip, and step the one optimizer.

The one optimizer is the world-model optimizer plus one group for the policy and value heads. No agent optimizer exists. The heads get the world-model learning rate, weight decay, and update rule.

The code is in `train/train_gym.py`: `JointImaginationLoss`, `FrozenModelPolicyPrior`, `dream_policy_inputs`, and the `extra_loss_fn` argument of `train_world_model`.

## Decisions (2026-10-01)

| Topic | Decision |
| --- | --- |
| Steps | `fixed_full_transformer_finetuning_steps` (10) joint steps for each loop. Each step has both losses. |
| Prior | A frozen copy of the transformer and the policy head, refreshed at the start of each loop. A frozen head alone moves with the transformer. |
| Loss scale | Sum with weight 1 (`fixed_full_transformer_finetuning_rl_loss_weight`). No RMS normalization. |
| Diagnostic | At the first step of each loop, log the transformer gradient norm of each loss term (`joint_diag/`). |
| Memory | Chunks are permitted (`world_model_micro_batch_size`, `fixed_full_transformer_finetuning_dream_micro_batch_size`). The gradient equals the one-pass gradient. |

## Conditions for equal learner and rollout inputs

The learner must read the inputs that the policy read in the rollout. Three conditions give this:

- The rollout uses the time cache. Then each frame reads clean past frames. Without the cache, the past frames have context noise.
- The rollout returns its latents before the clamp to (-1, 1) (`clamp_generated_latents = False`).
- On CUDA, the rollout uses the eager forward for the first generated frame (`EagerWithoutTimeCacheCallable`). See the next section.

The run logs `joint/learner_input_error`. It is 0 on CPU and about 0.02 on CUDA (TF32 rounding, input scale 2).

## Compiled forward on short sequences (CUDA)

A compiled world-model forward gives wrong outputs for a short sequence of two or more frames without a time cache. Frame 0 is correct. Each later frame is wrong (agent embedding error 3 to 8 at scale 2). The eager forward on CUDA agrees with the CPU forward.

The probable cause is the flex-attention decode kernel, which PyTorch selects for fewer than 128 queries. The world-model loss on windows of 200 steps is not affected.

The default path calls the compiled forward without a cache for the first generated frame of each dream. Thus the first dream frame is wrong in the default compiled path (first latent error 0.12 mean, a real step is 0.27). This flag corrects the frame for its own runs only. The default path is not changed.

## Closer PPO objective (added 2026-10-01)

The PMPO runs of queue 10-1-26b had a second difference from the A2 runs: the objective. The flag now also runs with `cleanrl_ppo_closer` (the A2 settings).

- Each joint step generates one new batch of 512 dreams and does one pass of the closer PPO loss over it. The loss is the policy loss plus 0.5 times the value loss (squared error with the value clip).
- The targets are the frozen GAE returns of the rollout. The advantages are normalized over all valid transitions of the batch.
- The importance ratio is 1 in each step, so the PPO clip has no effect. The 10 epochs of 32 minibatches of the A2 runs do not exist here. A2 does 960 head updates for each loop. The joint phase does 10 updates.
- No policy prior exists with this objective.
- The heads are one group of the one optimizer. The group has `agent_learning_rate` with the linear decay of closer PPO.

## Loss normalization (added 2026-10-01)

`use_dynamics_loss_normalization` normalizes the world-model loss terms. It does not normalize the RL loss terms. With `group_dynamics_loss_normalization`, the flow loss and the bootstrap loss share one RMS statistic of their sum, so the two keep their ratio. The default gives each of the two its own statistic. In the earlier `loss_normalization` runs, separate statistics made the bootstrap loss equal to the flow loss, and the flow loss rose to the level of an untrained model.

## Limits

- The flag requires `objective: pmpo` or `cleanrl_ppo_closer`.
- The dream forward with gradient is eager. The world-model loss is compiled.
- `full_transformer_finetuning` (two optimizers, two phases) stays as the older version.
