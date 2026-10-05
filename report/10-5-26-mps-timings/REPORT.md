# MPS, Process Memory, and Loop Timings (2026-10-05)

## 1. Summary

- **MPS gives only 1.10-1.20× the throughput of two lanes without MPS.** Two lanes with MPS give 1.32-1.38× the throughput of one run alone. Two lanes without MPS give 1.11-1.17×.
- **Cause: each run waits on its single CPU thread, not on the GPU.** The model is small (dim 128, depth 3), so each kernel runs for microseconds. The main thread spends most of its time on kernel launches, eager Python, and waits for short GPU work. MPS lets the kernels of one run fill the gaps of the other run. It cannot make one thread launch faster.
- **No compiled function uses CUDA graphs.** `train/compile_runtime.py` sets `compile_cudagraphs = False` for the interact forward, the world-model loss, the cached generate forward, and the PPO loss. Thus `compile_mode: reduce-overhead` falls back to normal Inductor. Each Triton kernel gets a Python launch. Closer PPO also turns off the compile of the PPO loss (`train/train_gym.py`, `compile_learn = False if cleanrl_ppo_closer`).
- **Memory:** each env worker loaded torch and the CUDA libraries, about 720 MB for each worker (16 workers for two runs). A new module `train/env_worker.py` holds the worker. Each worker now uses 62 MB, and the env outputs are bitwise the same. New runs use it; the runs in progress keep their workers.
- **The large gains need CUDA graphs for the phases that the CPU limits.** Section 5 gives a ranked list.

## 2. Processes and Memory

Each run has these processes (snapshot 12:30, runs 161 and 138):

| Process | Count for each run | RSS | Swap | Can it stop? |
| --- | --- | --- | --- | --- |
| Main training process (spawn child of `train/ablations.py`) | 1 | 5.7-6.5 GB | 0.7-2.3 GB | No. It is the run. |
| Env workers (`GroupedAsyncVectorEnv`, 8 envs each) | 8 | 330-390 MB | about 440 MB | No. The rollout needs them at each step. |
| Inductor compile worker | 1 | 320-680 MB | 220-360 MB | No. A recompile of a new shape uses it. If it stops, the run can crash. |
| `ablations.py` parent, resource tracker | 2 | 10 MB | 0 | No. |

- The 16 env workers held about 12 GB of RAM plus swap. The cause: a spawn process imports the module of its target function. The target was `train.train_gym._grouped_env_worker`, so each worker imported torch, CUDA, cuDNN, NCCL, and the model code.
- Fix: `train/env_worker.py` imports only gymnasium. `train/train_gym.py` imports `grouped_env_worker` from it.
  - Check (`scripts/env_worker_check.py`): 8 HalfCheetah envs in 4 workers, seed 7, 1,100 fixed random actions (past the 1,000-step time limit and its autoreset). Observations, rewards, terminations, and truncations are bitwise equal. RSS of each worker: 709-724 MB before, 62-63 MB after.
  - Test `test_grouped_env_worker_module_does_not_import_torch` stops a later import of torch in that module.
- Other memory users are not from the runs: the TensorBoard server of another session (0.5 GB RSS, 1 GB swap), open-webui, the desktop, and the browser. I did not stop them.

## 3. MPS against No MPS

Source: the `timings/` scalars in TensorBoard. A loop starts at the start of one rollout and ends at the start of the next. The table compares the same loop numbers, because the loop cost changes during a run. Runs 160 and 137 have the same configurations as the MPS runs 161 and 138 (only the seed differs), and they also started 3-5 minutes apart.

| Lane, loops | MPS: 161, 138 (12:18-12:34) | No MPS, same configs: 160, 137 (08:08-08:25) | No MPS, other runs (10:13-11:36) |
| --- | --- | --- | --- |
| b, loops 13-25 | 16.6 s | 18.2 s | 19.0, 19.2 s |
| b, loops 26-70 | 16.3 s | 18.4 s | 19.6, 19.7 s |
| a, loops 1-15 | 15.6 s | 17.6 s | 19.7, 19.0 s |
| a, loops 16-60 | 15.5 s | 17.9 s | 19.7, 19.8 s |
| One run alone (157, loops 13-60) | 10.8 s | | |

- Throughput of two lanes against one run alone: 1.32-1.38× with MPS, 1.11-1.17× without MPS.
- GPU power (queue trace, mean): 174 W alone, 191-196 W for two lanes without MPS, 206 W with MPS. The cap is 270 W.

### 3.1 Each phase alone, shared, and with MPS (loops 13-60, median for each loop)

| Phase | Alone (157) | No MPS (158) | No MPS (160) | MPS (161) | MPS (138, block2) |
| --- | --- | --- | --- | --- | --- |
| Rollout: model and orchestration | 3.03 s | 5.54 s | 5.52 s | 5.27 s | 4.64 s |
| Rollout: env step | 0.92 s | 0.95 s | 0.95 s | 0.97 s | 0.99 s |
| World-model step (GPU time = wall time) | 3.39 s | 6.34 s | 4.44 s | 4.33 s | 3.46 s |
| Generate | 1.74 s | 3.41 s | 2.26 s | 2.48 s | 2.31 s |
| PPO: forward + backward + clip | 1.09 s | 1.89 s | 2.84 s | 1.33 s | 2.10 s |
| Loop | 10.83 s | 19.62 s | 18.50 s | 16.43 s | 15.71 s |

- With MPS, the world-model step of one run is only 0-28 % slower than alone. The GPU has room while the other run is in a phase that the CPU limits.
- The rollout loses the most: 3.0 s alone, 4.6-5.3 s with MPS. Each rollout step is a chain of small kernels and ends with a sync (`.cpu()` of the actions). Under MPS, these kernels wait in the hardware queue behind the kernels of the other run.
- PPO without MPS suffers most from time slicing: it makes about 20 syncs for each minibatch (6,400 for each loop), and each sync can wait for a time slice of the other process. MPS removes most of this wait.

## 4. Why the GPU Stays Under-Used

### 4.1 Where the main thread spends its time

py-spy, 90 s, `--nonblocking`, 40 samples per second (`scripts/pyspy_phase_leaves.py`). Each row gives the share of the samples of that phase.

| Phase | Share of all samples | MPS run 161 | No-MPS run 158 |
| --- | --- | --- | --- |
| Rollout | 27 % | wait for the GPU at `.cpu()` 46 %, Python and eager torch 25 %, env pipes 15 %, Inductor calls and launches 14 % | wait 34 %, Python 26 %, env 20 %, Inductor 19 % |
| Generate | 26 % | Triton launch 37 %, Inductor call 34 %, Python 25 %, dynamo guards 4 % | Python 39 %, launch 29 %, Inductor 26 %, guards 7 % |
| World model | 19 % | wait for the GPU 34 %, Triton launch 32 %, Inductor call 21 %, Python 12 % | Python 27 %, wait 25 %, launch 24 %, Inductor 17 %, guards 6 % |
| PPO (closer) | 13 % | eager Python and torch dispatch 91 %, autograd 9 % | 87 %, 13 % |

- **Generation is launch-bound.** About 70 % of its CPU time is the Python launch of Triton kernels. The cached generate forward is compiled with dynamic shapes (the KV cache grows), so it cannot use CUDA graphs.
- **The world-model step is near balance.** The forward launches in 1.33 s and runs in 1.72 s on the GPU (alone). The Inductor wrapper of the loss has about 16,000 lines, so there are thousands of kernels for each step. There are 4 micro-batches of 32 windows for each step, so the kernel count is 4× that of one batch of 128.
- **PPO runs as eager PyTorch.** 320 minibatches for each loop, each with a second head forward for metrics (`agent_approx_kl_metrics`) and about 20 GPU-to-CPU syncs. Closer mode logs only the last minibatch.
- **The rollout is a latency chain.** 1,000 sequential steps, each with a compiled forward, eager value and policy heads, sampling, 9 `safe_cat` calls on arrays that grow, and one sync.

### 4.2 Consequence for MPS

A run alone keeps the GPU busy for about 55-65 % of its loop, but with short kernels and gaps. Two runs under MPS fill some of the gaps. Each run keeps its own CPU limit, and its latency chains become slower because of the other run's kernels. The measured result is 1.35× of one run, not 2×.

## 5. Ranked Changes

The gains are estimates from Sections 3 and 4. A GPU benchmark must confirm each one. Flags are necessary where the numerics can change (the default-path rule).

| # | Change | Phase | Expected gain (alone) | Numerics | Effort |
| --- | --- | --- | --- | --- | --- |
| 1 | CUDA graph for the rollout step: static time KV cache of 1,001 frames with a length mask, static input buffers, heads and sampling in the graph | rollout | 3.0 s → about 1 s; under MPS also less queue latency (one graph launch for each step) | flag (masked attention can change the sum order) | large |
| 2 | CUDA graph for the generate step: static KV cache of prompt + horizon (8 + 32 frames) | generate | 1.7 s → about 0.6 s | flag | medium |
| 3 | `torch._inductor.config.cpp_wrapper = True` (C++ kernel launches in place of the Python launcher) | WM, generate, rollout forward | 20-40 % of the launch time; the Triton launcher is 20 % of all samples | same kernels; verify bitwise; flag until verified | small |
| 4 | CUDA graphs for the world-model loss (`compile_cudagraphs = True` for it; cudagraph trees record forward and backward) | WM | 10-25 % of the WM step | same kernels; verify bitwise | small to medium (micro-batch accumulation must work with cudagraph trees) |
| 5 | Closer PPO: compile the minibatch loss; compute `agent_approx_kl_metrics` and the `.item()` metrics only for the last minibatch; keep the clip fraction on the GPU | PPO | 1.1 s → about 0.4 s alone; more with sharing | metrics only: bitwise; compile: flag | small |
| 6 | One generate batch of 2,048 dreams in place of 4 chunks of 512 | generate | 4× fewer launches | flag (RNG order) | small |
| 7 | Preallocated rollout buffers in place of 9 `safe_cat` calls for each step | rollout | 0.3-0.6 s | bitwise | small |
| 8 | High-priority CUDA stream for the rollout and generation (latency phases) | rollout, generate under MPS | unknown; can cut the queue wait behind the other run | scheduling only; verify | small, experimental |
| 9 | A third lane under MPS | all | +20-30 % throughput if the GPU memory allows it | none | config; needs peak-memory data |

- Items 1-5 together: an estimate of 5-6 s for each loop alone, against 10.8 s. Less CPU time per run also makes MPS more useful, because the runs then put more work on the GPU.
- Item 9: each run now holds 5.5 GB of GPU memory, so three runs (16.5 GB) do not fit in 16.3 GB. New runs log `timings/memory/cuda_max_allocated/mebibytes` and `.../cuda_max_reserved/mebibytes` for each loop (added today, logging only). This data gives the memory fraction that a third lane can use. Host memory is no longer a limit after the env-worker fix.
  - First data (13:12, runs 141 depth5 and 147): peak allocated 3,524 MB and 3,073 MB, loop 0 included. Reserved: 7.6 GB in loop 0, then 5.2-6.3 GB (allocator cache). With `cuda_memory_fraction` 0.31 (5.05 GB for each run), three runs fit in 16.3 GB with about 1.5 GB of margin above the peak allocation of each run. Test it with one queue before the use for all queues.
  - Host memory after the fix (13:12): swap 15.0 GB, against 25-26 GB before.
- `activation_memory_budget: 0.825` recomputes activations in the world-model backward. With 7.8 GB for each run (`cuda_memory_fraction: 0.48`), a budget of 1.0 can fit and removes the recompute. This is a config change of the queued experiments, so it needs a decision.

## 6. Changes Made Today

| File | Change | Effect on gradients |
| --- | --- | --- |
| `train/env_worker.py` (new) | the env worker, without a torch import | none (bitwise check, Section 2) |
| `train/train_gym.py` | imports the worker from `train/env_worker.py`; removes the unused `traceback` import; logs the peak GPU memory of each loop (CUDA only) and resets the peak | none |
| `tests/test_train_gym.py` | `test_grouped_env_worker_module_does_not_import_torch` | none |
| `scripts/mps.sh` (new) | MPS daemon start, status, stop | none |

Tests: `tests/test_profiling.py` and `tests/test_train_gym.py` give 41 passed and 1 failed. The failure (`test_main_writes_timings_profiles_nine_steps_and_closes_resources`: `FakeWorldModel` has no `max_steps`, read at `train/train_gym.py` in the world-model call) exists without these changes.

## 7. Method

- `scripts/loop_breakdown.py LO HI 0 RUN...`: per-loop sums of each timing tag, median over loops LO-HI. A CUDA time uses the wall time of the CPU event with the same index, because CUDA times are written at the flush.
- `scripts/loop_phases_range.py`: the phase boundaries from the wall times of the timing events.
- `scripts/pyspy_phase_leaves.py`: py-spy samples by phase and leaf type. The raw samples are in `scripts/pyspy_*.txt`.
- `train/profiling.py`: `cpu_milliseconds` is the wall time of a block without a sync. `cuda_milliseconds` is the time between two CUDA events on the stream. Under sharing, it includes the time that the stream waits for the other process.
- dcgmi values change much within a loop (the phases differ), so this report does not use them.
