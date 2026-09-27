# Offline pretraining: implementation questions

Date: 2026-09-27. Plan: `notes/pretraining_impl_plan.md`. Code: `train/offline_pretraining.py`, `train/train_gym.py`, `train/ablations.py`.

Each item gives the decision in the code and the question to confirm. The items that need an answer before the first GPU run come first.

## 1. Decisions that need an answer before the first run

### 1.1 The CLI of plan section 9 does not work

`python -m train.train_gym --offline_pretraining=True ...` calls `fire.Fire(main)`. `main()` has many required keyword arguments and does not load `train/config.yaml`, so this command fails. The code does not add a config loader to `train_gym.py`.

Decision: `train/ablations.py` has two new entries, `offline_collect_halfcheetah_v4_mix_1m` and `offline_pretrain_a2_next_state_reward`. Run them one at a time:

- Collect (CPU node): `python train/ablations.py --ablation_indices=[N] --parallel=False`, where N is the index of the collect entry.
- Pretrain (GPU node): `scripts/train_slurm.sh sbatch 1:30:00 -N <index>`. If the job stops before the last step, submit the same index again. The run continues from the last step checkpoint.

`run_ablation_specs()` now raises an error if a parallel launch with two or more runs includes a collect or pretrain entry.

Question: Do you want a config-loading CLI in `train_gym.py` instead?

### 1.2 Muon optimizer order (this change also affects the default path)

`MuonAdamAtan2` builds its non-Muon parameter group with `list(set(params) - set(muon_params))`. That order changes from process to process. A saved optimizer state then loads into the wrong parameters. The first rl-only smoke run crashed with a shape error in `optimizer.step()`. If two parameters have the same shape, the load does not fail, and the state goes to the wrong parameter without an error.

Decision: `make_optimizer()` passes the non-Muon parameters in module order with `remove_muon_params_from_params = False`. The optimizer math does not change. `test_muon_optimizer_keeps_the_module_parameter_order` covers this.

Consequences:

- This defect also affected `checkpoint_path` resume on the default path. The fix repairs resume for new checkpoints.
- The gradient-clip "trunk" group follows the optimizer order. Thus the old code summed its global grad norm in a random order, and two identical old runs differed at float precision (42 of 59 logged tags). The new code gives bitwise-identical runs.
- Loop checkpoints saved before this change have the old, random order. Their world and agent optimizer states do not load correctly into a new process. Do not resume old runs with their optimizer state, or accept a reset of that state.

Decision (2026-09-27): keep the fix. Exact compatibility is not possible: the old order was not saved, and 155 of 165 non-Muon world-model parameters share a shape with another parameter. Comments in `make_optimizer()` and at the checkpoint load in `main()` say that compatibility is broken.

### 1.3 The RL arms exist before the pretrained checkpoint exists

Plan section 10 says to add the RL dicts "after the first pretrained checkpoint exists". The four RL arms are in `ABLATIONS` now, at the end of the list, so that the current indices do not change. Before the pretrain job completes, these arms stop at once with `FileNotFoundError`. `test_offline_ablation_arms_agree_with_their_pretrain_signature` checks that the RL arms and the pretrain entry have the same signature and checkpoint path.

Decision (2026-09-27): keep the RL arms.

### 1.4 Resume of loop checkpoints

Decision (2026-09-27): fix the step counters, the replay mismatch, and the optimizer-settings change. The random-number state is still not saved.

- A loop checkpoint holds `progress`: `env_step`, `wm_step`, `imagination_step`, and the audit schedule and elapsed time. A resume continues these values.
- Each loop checkpoint `loop_<n>.pt` has a replay copy `replay_loop_<n>` in the same folder. `save_resume_checkpoint` writes the copy, then the checkpoint, then removes older copies.
- A resume removes the replay folder of the run and copies the snapshot back. Episodes collected after the checkpoint are not used.
- Only the newest checkpoint keeps a snapshot, because each snapshot is a full replay copy (about 10 GB at the default config). An older checkpoint cannot resume.
- Checkpoints saved before this change have no `progress` and no snapshot. They cannot resume.
- A checkpoint holds the optimizer group settings from the config at construction. If the current config gives different settings, an `assert` stops the resume. Schedules and PPO backoff change `lr` in the loaded state, and the resume keeps those values.
- TensorBoard points that the crashed run logged after its last checkpoint stay in the event files. The resumed run logs the same steps again.

## 2. Decisions in the code that differ from the plan text or add to it

### 2.1 The tokenizer trains on the training episodes only

Plan section 6.1 says "all offline observations". The code uses the training episodes only (about 95 %). Thus the per-mode reconstruction loss on the held-out episodes is a true held-out value. Each episode gives all of its observations, including the final observation.

### 2.2 The validation split

The split is per mode. Each mode with n episodes holds out `round_half_up(n * offline_validation_fraction)` episodes. A mode with 2 or more episodes holds out at least 1 episode and keeps at least 1 episode for training. The split uses `offline_data_seed`, so each pretrain run of one dataset uses the same split. `<pretrain_dir>/<name>/split.json` records it.

### 2.3 Agent keys

`agent_state_keys()` uses `agent_parameters()` and the buffers of modules that hold only agent parameters, as the plan says. It also adds these keys:

- `value_encoder.*` and `actor_critic_value_encoder.*` (plan: "the value encoder").
- `critic_state_embedder.*`. This module feeds only the value head. `agent_clip_groups()` also treats it as an agent module.
- `ema_returns_mean` and `ema_returns_var`. These are return statistics of the agent loss.

Observation (a possible defect on the default path): `critic_state_embedder` is not in `agent_parameters()`. Thus it is in the world optimizer, but no world-model loss reads it. Unless `full_transformer_finetuning` is true, it stays at its random initialization for the full run.

Question: Is `critic_state_embedder` meant to train with the value head?

### 2.4 Signature keys

The signature has the keys of plan section 6.4. For "all `reward_*` keys", the code uses an explicit list. It includes `reward_gradient_into_transformer_scale`.

The signature does not include these keys:

- `agent_state_pred_loss_weight`. It changes the world-model loss only when `agent_predicts_state` is true.
- `use_muon_optimizer` and `world_model_learning_rate`. If the optimizer type differs, `load_optimizer_state_if_compatible()` prints a message and skips the state.

When the collect stage is not selected, the pretrain stage does not compare the dataset with the current `offline_*` collection keys. It uses the dataset at `offline_data_dir/<env>/offline_dataset_name`. The pretrained checkpoint stores the dataset manifest.

Question: Do you want a stricter check for any of these keys?

### 2.5 Offline env-step accounting

`offline_env_steps` is the number of transitions in the dataset, including the validation episodes. It does not include:

- The 5 expert validation episodes (5,000 env steps).
- The env steps that trained the expert (CleanRL TD3, about 1 M).

`charts/episodic_return_offline_inclusive` uses `env_step + offline_env_steps`.

Question: Do you want the expert validation steps or the expert training steps in this count?

### 2.6 Actions: Beta only

The offline path converts env actions to the native Beta range [0, 1] and clamps them to [1e-4, 1 - 1e-4]. `validate_offline_config()` raises an error unless `continuous_dist_type` is `beta`. `env_to_native_actions()` and `native_to_env_actions()` are now shared functions in `train/audit_ablations.py`. `AuditDiagnostics` uses them.

### 2.7 Pretrain restart and existing folders

- A pretrain restart continues in the same run directory if `split.json` exists and `pretrained.pt` does not exist. The FileExistsError for an existing run directory does not apply in that case. `split.json` is the first file that the stage writes, so a job that stops during tokenizer training can also continue.
- A restart from a step checkpoint requires that the checkpoint tokenizer is equal to `tokenizer.pt`. Otherwise the world model would continue on the latents of a different encoder.
- If `pretrained.pt` exists, the pretrain stage raises an error. It does not skip the stage.
- A restart loads `tokenizer.pt`. It also reuses each replay folder whose marker agrees with the dataset, the tokenizer file hash, and the episode list. Otherwise it builds the replay again.

### 2.8 Step checkpoints

The code keeps each `pretrained_step_<n>.pt`. These files use the format of `pretrained.pt`, so an RL arm can load one directly (for example 10 k against 20 k steps). Each file holds the tokenizer, the world model without agent keys, and the world optimizer state.

### 2.9 TensorBoard tags and step counters

- The pretrain stage logs world-model metrics under `offline_pretrain/train/*`. It logs validation metrics under `offline_pretrain/val/*`.
- The per-mode held-out tokenizer loss uses `offline_pretrain/val/tokenizer_recon_loss/<mode>` at step 0.
- In a `[pretrain, rl]` run, the RL stage starts `wm_step` at 0, as a normal run does. The prefixes prevent tag collisions.

### 2.10 Validation loss

The validation loss uses the eager `WorldModelTrainingLoss` in `combined` mode with `update_loss_ema = False` and no gradient. It uses 4 batches of `world_model_batch_size` windows. It restores the RNG state, so the training stream does not change. The eager path prevents a second compiled graph for the no-grad call. The values are noisy, because the shortcut objective samples signal levels.

### 2.11 Rounding

The episode counts per mode and the rehearsal batch split use round-half-up. Python `round()` rounds 0.5 to the even integer, so `round(2 * 0.25)` gives 0. The defaults are not affected: 300, 400, and 300 episodes, and 96 + 32 windows.

### 2.12 Collection details

- The expert validation episodes use `offline_data_seed + 1_000_000`. Episode i of the dataset uses seed `offline_data_seed + i`.
- The last env batch can be partial. The unused envs then run random episodes that the collector drops.
- `offline_data_num_envs` changes the data order, so it is part of the manifest comparison.
- `KNOWN_EXPERT_SHA256` pins the sha256 of the default expert file. A download with a different hash fails.

### 2.13 TF32 for all runs

Decision (2026-09-27): each run uses TF32 and `torch.set_float32_matmul_precision("high")` on CUDA when `allow_tf32` is true (the config value). `configure_matmul_precision()` in `train_gym.py` sets this for `main()` and for `train_model_free.run()`. Before, `cleanrl_ppo_closer` and the small-model CleanRL PPO path forced "highest" precision without TF32, and the CleanRL PMPO and generic small-model paths kept the PyTorch default ("highest"). Those runs are not bitwise comparable with earlier runs. `cudnn.deterministic` does not change.

### 2.14 In-process `[collect, pretrain, rl]`

The rl stage uses the models in memory. It does not load `pretrained.pt` again. The agent heads keep their initialization in both cases.

### 2.15 The loaded world optimizer keeps the configured hyperparameters

`Optimizer.load_state_dict()` also replaces lr, weight decay, and betas with the saved values. The signature does not include `world_model_learning_rate`, so `load_optimizer_moments()` loads only the per-parameter state (moments and step counts). It keeps the group settings of the current config. The pretrain restart uses the same function.

## 3. Adversarial review results (2026-09-27)

### 3.1 Default path

- Reference: the tree before this change, rebuilt from the session file backups.
- Three CPU variants ran in both trees: the baseline config, an audit arm (A2 reward target, PPO, deeper dynamics, diagnostics), and policy consistency. Each run had 3 loops, checkpoints, and tokenizer evaluation.
- With only the Muon parameter order fixed in the old tree, all logged scalars (1,451 points) and all final checkpoint tensors (629) are bitwise identical to the new tree.
- The only default-path changes are the Muon order (section 1.2) and the new `offline_*` keys in the `run/hyperparameters` TensorBoard text.
- The GPU compile path was not compared. The changed code gives the same batch to the compiled loss (`repeat_batch_to_size(batch, batch_size)` when rehearsal is off).
- Pre-existing test failures, the same in the old tree: 5 in `tests/test_train_model_free.py` (`StateActorCritic` has no `cleanrl_ppo_closer`), and 204 in `tests/test_dreamer.py`. `dreamer4/` and `test_dreamer.py` are not changed.

### 3.2 Pretraining

- Data alignment: 16 collected episodes were replayed in a fresh `HalfCheetah-v4` env with the stored seed and actions. The observations (including the final bootstrap observation) and the rewards are bitwise equal.
- Replay layout: `test_offline_episode_has_the_online_replay_layout` compares an offline episode with an `interact_with_env()` episode field by field.
- Learning: a small world model (model_dim 64, batch 16, window 32) was pretrained on 12 training episodes. The fresh model and the pretrained model were compared on held-out offline episodes and on random-policy rollouts (the first online data of the rl stage).

| Measure | Fresh | 300 steps | 1,000 steps | 3,000 steps |
|---|---|---|---|---|
| Flow loss, held-out offline | 0.30 | 0.030 | | |
| Flow loss, random-policy rollouts | 0.35 | 0.030 | | |
| Reward MAE, expert (mixture-mean predictor: 6.12) | 10.69 | 6.17 | 0.54 | 0.44 |
| Reward MAE, expert + noise (mixture mean: 2.97) | 7.62 | 3.36 | 1.28 | 0.96 |
| Reward MAE, random (mixture mean: 4.84) | 0.57 | 3.75 | 0.62 | 0.56 |

- The dynamics transfer early. The reward head first predicts the mixture mean (at 300 steps it is no better than a constant). It becomes state-dependent by 1,000 steps. Consequence: do not start RL from a step checkpoint much below 1,000 steps. The in-loop 256-step pretraining would be too short for this data mix.
- The held-out flow loss rose slightly from 300 to 3,000 steps (0.024 to 0.026 on expert data). This small test has 12 training episodes, so this is probably overfitting. Watch `offline_pretrain/val/world_model/flow_loss` in the 950-episode run.

## 4. Measurements on this node (CPU)

- The default expert file is 871,773 bytes with sha256 `0000ee72...5488`. It is a `(actor, qf1, qf2)` tuple with `action_scale` 1 and `action_bias` 0, as the plan says.
- The expert validation mean return over 5 deterministic episodes is 10,831 (range 10,697 to 10,889). The model card gives 10,122 ± 120.
- In a 4-episode smoke dataset, the expert + noise episodes returned about 5,900, and the random episode returned -228.
- Disk use: the raw 1 M dataset is about 100 MB. The latent replay is about 0.6 GB for each pretrain name (`latents` are 128 floats per row).

## 5. Not tested on a GPU

The CPU tests run with `compile = false`. The pretrain stage uses the compiled `world_model_loss` and the static batch shape of `main()`. The rehearsal batch keeps the total batch size, so the compiled shapes do not change. A GPU smoke run of each stage is still necessary. Measure the time per world-model step there, and then set `offline_world_model_steps` for one 1:30 job (plan section 13).
