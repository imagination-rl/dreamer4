# Paper-informed full-transformer finetuning test

## Objective

Test whether imagination updates through the transformer improve this implementation.

Use the Dreamer 4 paper as a design reference. Do not attempt an exact reproduction.

Keep behavior cloning out of every run. This is the main experiment choice.

Keep the prior local and refreshed. A prior copied from an untrained random policy cannot replace the missing BC policy.

The current Gym setup remains online and state-based. The paper uses offline video, task conditioning, and a behavior-cloning phase.

## Why the current full-transformer run is not a clean test

The current full path has several confounds:

| Area | Current behavior | Consequence |
| --- | --- | --- |
| Transformer ownership | The world optimizer owns world-model parameters. The full agent optimizer also owns them. | Two optimizer states update the same tensors. |
| Policy data | The full ablation changes the imagination batch from 512 to 128. | The policy receives one quarter of the normal imagined states. |
| Representation | Generation uses context noise of 0.1 and a time cache. The full learner recomputes a clean sequence without the stored context noise. | The prior targets and current policy use different hidden states. |
| Preservation scale | The full loss adds raw dynamics and eight reward-head losses to PMPO and value losses. | The shared trunk receives a poorly controlled mixture of objectives. |
| Paper objective | The default uses HL-Gauss, legacy PMPO, and the non-paper shortcut sampler. | Paper-specific effects remain mixed with legacy choices. |

The preservation sequence is already 200 in the current config. Do not change it to 32.

The relevant code is in [train_gym.py](/work/nvme/bguz/mravichandran/proj/dreamer4lucid/train/train_gym.py:601), [train_gym.py](/work/nvme/bguz/mravichandran/proj/dreamer4lucid/train/train_gym.py:1231), and [compile_runtime.py](/work/nvme/bguz/mravichandran/proj/dreamer4lucid/train/compile_runtime.py:385).

## Required source changes

### 1. Give each shared parameter one optimizer owner

Use one optimizer object for full-transformer runs. Give it all unique world-model parameters, including policy and value heads.

Use this optimizer for both the world-model phase and the imagination phase.

Keep the current frozen-transformer optimizer path unchanged.

Preserve separate learning rates with parameter groups if needed. Give the shared trunk a lower rate than the policy and value heads for the first run.

Do not let two optimizer objects update one parameter. Add an assertion for disjoint ownership in the frozen path and single ownership in the full path.

Save and restore one optimizer state for the shared full path. Do not save two independent states for one parameter set.

### 2. Make the policy learner see the rollout representation

Store the context-noise tensors or a replayable random seed for each generated timestep.

Recompute the differentiable learner representation with the same context noise, signal levels, step size, and cache convention as generation.

Keep the rollout action and prior distributions detached. Recompute only the current policy and value outputs with gradients.

Add a parity test. With fixed model weights and random inputs, the generation and learner policy inputs must match within numerical tolerance.

Disable the time cache in this parity test. Re-enable it after the no-cache path matches.

### 3. Separate and control every full-transformer loss

Return separate policy, value, dynamics, reward, and prior-KL losses from the imagination call.

Log gradient norms for each loss on the shared trunk. Log pairwise cosine similarity between policy, value, and preservation gradients.

Add separate coefficients for preservation dynamics and preservation reward losses.

Choose coefficients from gradient norms, not scalar loss values.

Start the preservation coefficients near 0.1 of the current raw scale. Increase them only when the trunk drifts or replay prediction degrades.

Use RMS normalization only after this loss decomposition works. Update RMS statistics during world-model updates, then freeze them during each full imagination step.

Normalize each objective before its coefficient. Do not normalize a sum of unrelated MTP heads.

The paper uses RMS normalization for its multi-head world model. The current implementation uses separate normalizers and failed when active normalization kept late model gradients near one.

### 4. Keep long-context preservation and complete the applicable targets

Expose `preservation_batch_size` as a config value. Set it to 128 while the world-model batch stays 128.

Set `preservation_sequence_length` to 200. Match the normal world-model sequence length.

Preserve dynamics and reward functions during full updates. Preserve terminal prediction when the environment has real terminal events.

HalfCheetah has no useful terminal events, so terminal preservation has no expected effect in this task.

Pad or sample preservation batches to a fixed size when static compilation is active.

Do not give a short remainder batch a full optimizer update.

### 5. Make policy data sufficient for a large parameter update

Restore `imagination_batch_size` to 512 for the full path.

If memory prevents 512, use gradient accumulation with an effective batch of 512. Accumulate PMPO positive and negative statistics across microbatches.

Run a direct comparison with three imagination updates per loop. Then run twelve updates per loop as the higher-compute full-transformer test.

The current full run uses 128 samples and three updates. That schedule gives the shared trunk too little control data.

Use a separate shared-trunk learning rate. Start at one third of the world-model rate, while policy and value heads keep the current rate.

## Staged experiment settings

Run every stage with seeds 42 and 4. Use the same replay and environment settings for both seeds.

### Stage A: implementation control

Use these settings after the source fixes:

```yaml
full_transformer_finetuning: true
imagination_batch_size: 512
imagination_train_steps: 3
preservation_sequence_length: 200
preservation_batch_size: 128
use_dynamics_loss_normalization: false
use_paper_aligned_shortcut_objective: false
imagination_use_time_cache: false
```

Keep the known-good baseline choices for this control: `old_reward_shifting: true`, `old_pmpo: true`, `old_reward_head: true`, and `policy_all_tokens: true`.

Do not enable `old_value_head`, `old_embedding_init`, or `add_action_embed_to_agent_token` in this test.

Require both seeds to improve real environment return before adding paper-specific loss changes.

### Stage B: sufficient full-transformer compute

Keep Stage A settings and set `imagination_train_steps: 12`.

Use the same effective imagination batch and the same shared-trunk learning rate.

This stage tests whether the failed run lacked policy updates rather than model capacity.

### Stage C: paper-informed full-transformer run

Use Stage A or Stage B as the implementation gate. Treat Stage C as the paper-informed test.

Add one paper choice at a time, with paired seeds:

1. Enable corrected per-objective RMS normalization.
2. Enable the paper shortcut sampler and the `0.9 * tau + 0.1` weight for both flow and shortcut terms.
3. Replace HL-Gauss with the symexp two-hot reward and value encoders.
4. Use the paper PMPO sign partition, with zero advantages in the positive set.

Do not copy `value_range: [-1200, 1200]` into `SymExpTwoHot`. Its constructor exponentiates the range and overflows at those values.

Use a safe symlog-domain support, such as `[-10, 10]`, or add a numerically safe encoder configuration.

Stop a variant when either seed loses real return while dream return rises. That pattern indicates model exploitation.

## Acceptance criteria

Treat full-transformer adaptation as useful only when both seeds satisfy all conditions:

- Real return improves above the frozen-transformer baseline.
- Dream return and real return improve together.
- Critic value error stays bounded.
- Policy KL remains active without runaway growth.
- Shared-trunk gradient cosine does not show persistent policy-versus-preservation cancellation.
- Long-context replay losses do not increase after imagination updates.

The no-BC constraint remains active for every stage. A behavior-cloning loss or BC checkpoint invalidates this experiment.
