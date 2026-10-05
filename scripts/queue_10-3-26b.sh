#!/usr/bin/env bash
# Queue of 10-3-26b (train/ablations.py entries 109-117): learning-rate arms and a deeper no-AttnPool arm of the grouped-normalization base
# (entry 73, a2_grouped_loss_normalization), seeds 4, 42 and 1.
#   109-111 a2_grouped_flat_lr_s4 / s42 / s1   cleanrl_updated_ppo_constant_learning_rate: true (no linear decay)
#   112-114 a2_grouped_flat_lr340_s4 / s42 / s1  constant learning rate and agent_learning_rate: 3.4e-4 (13 % higher)
#   115-117 a2_grouped_no_attn_pool_depth5_s4 / s42 / s1  use_attn_pool: false, depth: 5 (2 time layers; base 3 and 1)
# Lane a runs 109 111 113 115 117, lane b runs 110 112 114 116.
# The script checks the setup at launch, then waits until START_AT (default now) and runs the two lanes.
# The runs save a checkpoint at loops 50 and 100. The replay snapshot is a sparse copy (train_gym.sparse_copy).
# Two runs use the GPU at a time (one for each lane), and two runs never start within START_GAP_SECONDS of each other
# (200 s: a run compiles for several minutes at its start). Each run sets cuda_memory_fraction 0.48
# (train/ablations.py SHARED_GPU_MEMORY). All processes run on CPUs 0-15.
#
# Out of GPU memory: a run that fails with a CUDA out-of-memory error goes back to the queue. The script renames the
# folders of the failed attempt (suffix .oom<attempt>), and runs the entry again, alone, after both lanes end.
# An entry runs at most 1 + MAX_OOM_RETRIES times. Other failures do not go back to the queue.
#
# Usage:   setsid nohup scripts/queue_10-3-26b.sh > runs/queues/queue_10-3-26b_launch.log 2>&1 &
# Options: PYTHON=<python>          (default /data/venvs/dreamer4lucid-x86/bin/python)
#          START_AT=<date string>   (default "now")
#          MAX_OOM_RETRIES=<n>      (default 2)
#          DRY_RUN=1                do the checks and write the record, then stop before the wait
#
# Each start writes runs/queues/10-3-26b_<time>/ with the code state (HEAD + patch), the resolved run settings,
# the package versions, the machine data, one log for each run attempt, status.tsv, a GPU trace, and a memory trace.

set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"

PYTHON=${PYTHON:-/data/venvs/dreamer4lucid-x86/bin/python}
STORAGE=/data/logs/dreamer4lucid
QUEUE_DIR=runs/queues/10-3-26b_$(date +%Y-%m-%d_%H%M%S)
START_AT=${START_AT:-now}
MAX_OOM_RETRIES=${MAX_OOM_RETRIES:-2}
LAUNCH_DELAY_SECONDS=${LAUNCH_DELAY_SECONDS:-60}
# 90 s was too short: a run compiles for several minutes and holds most of the GPU memory (entry 50 OOM).
START_GAP_SECONDS=${START_GAP_SECONDS:-200}
CPUS=0-15
OOM_PATTERN='OutOfMemoryError|CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED|cudaErrorMemoryAllocation|CUDA error: out of memory'

LANE_A=(109 111 113 115 117)
LANE_B=(110 112 114 116)
INDICES=(109 110 111 112 113 114 115 116 117)
declare -A RUN_NAMES=(
    [109]=a2_grouped_flat_lr_s4
    [110]=a2_grouped_flat_lr_s42
    [111]=a2_grouped_flat_lr_s1
    [112]=a2_grouped_flat_lr340_s4
    [113]=a2_grouped_flat_lr340_s42
    [114]=a2_grouped_flat_lr340_s1
    [115]=a2_grouped_no_attn_pool_depth5_s4
    [116]=a2_grouped_no_attn_pool_depth5_s42
    [117]=a2_grouped_no_attn_pool_depth5_s1
)

export PYTHONUNBUFFERED=1
# The /tmp tmpfs has a quota. The inductor cache grew to 12 GB there and stopped a run.
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/data/cache/dreamer4lucid/torchinductor}

fail() { echo "queue: $*" >&2; exit 1; }

# preflight

[[ -x "$PYTHON" ]] || fail "no python at $PYTHON"
[[ -d "$STORAGE" ]] || fail "$STORAGE does not exist; the runs would write to runs/ on /home"
START_EPOCH=$(date -d "$START_AT" +%s) || fail "START_AT=$START_AT is not a date"

# The entry numbers must still give the run names of this queue.
pairs=()
for index in "${INDICES[@]}"; do pairs+=("$index=${RUN_NAMES[$index]}"); done
"$PYTHON" - "${pairs[@]}" <<'EOF' || fail "the entries of train/ablations.py changed; update INDICES and RUN_NAMES"
import sys
from train.ablations import build_ablation_specs
expected = {int(pair.split("=")[0]): pair.split("=")[1] for pair in sys.argv[1:]}
actual = {spec.index: spec.train_kwargs["run_name"] for spec in build_ablation_specs(ablation_indices = list(expected))}
if actual != expected:
    print(f"expected {expected}\nactual   {actual}", file = sys.stderr)
    sys.exit(1)
EOF

check_free_folders() {
    local existing=() index path
    for index in "${INDICES[@]}"; do
        path=runs/gym_imagination/${RUN_NAMES[$index]}
        [[ -e $path || -L $path ]] && existing+=("$path")
    done
    ((${#existing[@]} == 0)) || fail "these folders exist; remove them or rename the runs: ${existing[*]}"
}
check_free_folders

free_gib=$(df --output=avail -BG "$STORAGE" | tail -1 | tr -dc 0-9)
# Each run writes its replay (3.9 GB), one sparse replay snapshot (3.9 GB) and 2 checkpoints (0.1 GB each): about 8 GB,
# 75 GB for the 9 runs. Before each start, a lane waits until MIN_FREE_GIB are free (run_entry).
MIN_FREE_GIB=${MIN_FREE_GIB:-30}
((free_gib >= 60)) || fail "$STORAGE has ${free_gib} GB free; the queue needs at least 60 GB at launch"
((free_gib >= 80)) || echo "queue: $STORAGE has ${free_gib} GB free, less than the 50 GB of all runs and a margin; the last runs can wait for free space"

# reproduction record

mkdir -p "$QUEUE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
{
    echo "launch: $(date --iso-8601=seconds)"
    echo "start_at: $(date -d @"$START_EPOCH" --iso-8601=seconds)"
    echo "host: $(hostname)"
    echo "repo: $REPO"
    echo "branch: $(git rev-parse --abbrev-ref HEAD)"
    echo "head: $(git rev-parse HEAD)"
    echo "python: $PYTHON ($("$PYTHON" -V 2>&1))"
    echo "command: $0 $*"
    echo "lanes: a=${LANE_A[*]} b=${LANE_B[*]}; max_oom_retries: $MAX_OOM_RETRIES"
    echo "env: PYTHONUNBUFFERED=$PYTHONUNBUFFERED TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} OMP_NUM_THREADS=${OMP_NUM_THREADS:-} cpus=$CPUS"
} > "$QUEUE_DIR/queue.txt"

# To rebuild the code: git checkout <head>, then git apply code.patch.
git status --porcelain=v1 > "$QUEUE_DIR/git_status.txt"
git diff --binary HEAD > "$QUEUE_DIR/code.patch"
cp train/config.yaml train/ablations.py train/train_gym.py train/compile_runtime.py "${BASH_SOURCE[0]}" "$QUEUE_DIR/"

"$PYTHON" - "$QUEUE_DIR/run_settings.json" "${INDICES[@]}" <<'EOF'
import json, sys
from train.ablations import build_ablation_specs
specs = build_ablation_specs(ablation_indices = [int(index) for index in sys.argv[2:]])
record = {spec.index: dict(label = spec.label, overrides = spec.overrides, train_kwargs = spec.train_kwargs) for spec in specs}
with open(sys.argv[1], "w") as file:
    json.dump(record, file, indent = 2, sort_keys = True, default = str)
EOF

{
    NO_COLOR=1 uv pip freeze --color never --python "$PYTHON" 2>/dev/null
} > "$QUEUE_DIR/pip_freeze.txt"
{
    "$PYTHON" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version())" 2>/dev/null
    echo; uname -a
    echo; lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core"
    echo; free -g
    echo; df -h /home /data /tmp
    echo; nvidia-smi
} > "$QUEUE_DIR/machine.txt" 2>&1

printf "index\trun_name\tlane\tattempt\tstart\tend\tseconds\texit\tstorage\n" > "$QUEUE_DIR/status.tsv"
: > "$QUEUE_DIR/retry.txt"

echo "queue: record in $QUEUE_DIR"
if [[ ${DRY_RUN:-0} == 1 ]]; then
    echo "queue: dry run, stop before the wait"
    exit 0
fi

# wait for the start time

trap 'echo "queue: interrupted, stopping all runs" >&2; trap - INT TERM EXIT; kill 0' INT TERM

now=$(date +%s)
if ((START_EPOCH > now)); then
    echo "$(date +%T) queue: wait until $(date -d @"$START_EPOCH" '+%F %T')"
    while (( $(date +%s) < START_EPOCH )); do sleep 30; done
fi

# Check again at the start: nothing may have written these folders during the wait.
check_free_folders
while true; do
    gpu_used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    ((gpu_used_mib <= 1024)) && break
    echo "$(date +%T) queue: other processes use ${gpu_used_mib} MiB of GPU memory; check again in 5 minutes"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    sleep 300
done
echo "start: $(date --iso-8601=seconds)" >> "$QUEUE_DIR/queue.txt"

# run

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 10 > "$QUEUE_DIR/gpu_trace.csv" &
GPU_TRACE_PID=$!
# Host memory in MiB (queue 10-1-26 lost two runs to a kill without a Python error; this trace shows a memory cause).
(
    echo "timestamp,mem_used,mem_available,swap_used"
    while true; do
        echo "$(date +%T),$(free -m | awk '/^Mem:/ {printf "%s,%s", $3, $7} /^Swap:/ {printf ",%s", $3}')"
        sleep 10
    done
) > "$QUEUE_DIR/memory_trace.csv" &
MEMORY_TRACE_PID=$!
trap 'kill $GPU_TRACE_PID $MEMORY_TRACE_PID 2>/dev/null' EXIT

record() {  # index lane attempt start_epoch exit
    local index=$1 name=${RUN_NAMES[$1]} end
    end=$(date +%s)
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$index" "$name" "$2" "$3" \
        "$(date -d @"$4" --iso-8601=seconds)" "$(date -d @"$end" --iso-8601=seconds)" "$((end - $4))" "$5" \
        "$(readlink -f "runs/gym_imagination/$name" 2>/dev/null)" >> "$QUEUE_DIR/status.tsv"
}

wait_for_start_slot() {  # Keep START_GAP_SECONDS between the starts of the two lanes.
    local last
    exec 9> "$QUEUE_DIR/start.lock"
    flock 9
    last=$(cat "$QUEUE_DIR/last_start" 2>/dev/null || echo 0)
    (( $(date +%s) - last < START_GAP_SECONDS )) && sleep $(( START_GAP_SECONDS - ($(date +%s) - last) ))
    date +%s > "$QUEUE_DIR/last_start"
    flock -u 9
}

attempts_of() { grep -c "^$1$" "$QUEUE_DIR/attempts.txt" 2>/dev/null; }

add_to_retry() {  # index reason
    exec 8> "$QUEUE_DIR/retry.lock"
    flock 8
    echo "$1" >> "$QUEUE_DIR/retry.txt"
    flock -u 8
    echo "$(date +%T) queue: $1 ${RUN_NAMES[$1]} goes back to the end of the queue ($2)"
}

rename_failed() {  # path attempt: move the folder (or the storage folder of a symlink) aside, on the same disk.
    local path=$1 attempt=$2 target
    [[ -e $path || -L $path ]] || return 0
    if [[ -L $path ]]; then
        target=$(readlink -f "$path")
        [[ -e $target ]] && mv "$target" "$target.oom$attempt"
        rm "$path"
        echo "$path -> $target.oom$attempt" >> "$QUEUE_DIR/failed_attempts.txt"
    else
        mv "$path" "$path.oom$attempt"
        echo "$path -> $path.oom$attempt" >> "$QUEUE_DIR/failed_attempts.txt"
    fi
}

wait_for_free_disk() {  # Do not start a run that can fill /data; wait until space is free (nothing is deleted).
    local lane=$1 index=$2 free
    while true; do
        free=$(df --output=avail -BG "$STORAGE" | tail -1 | tr -dc 0-9)
        ((free >= MIN_FREE_GIB)) && return 0
        echo "$(date +%T) [$lane] $index waits: $STORAGE has ${free} GB free, less than ${MIN_FREE_GIB} GB; check again in 10 minutes"
        sleep 600
    done
}

run_entry() {  # index lane; returns 0, or 2 for an OOM that goes back to the queue
    local index=$1 lane=$2 start status attempt log
    wait_for_free_disk "$lane" "$index"
    wait_for_start_slot
    echo "$index" >> "$QUEUE_DIR/attempts.txt"
    attempt=$(attempts_of "$index")
    log=$QUEUE_DIR/${RUN_NAMES[$index]}.attempt$attempt.log
    start=$(date +%s)
    echo "$(date +%T) [$lane] start $index ${RUN_NAMES[$index]} (attempt $attempt)"
    taskset -c "$CPUS" "$PYTHON" train/ablations.py "$index" > "$log" 2>&1
    status=$?
    record "$index" "$lane" "$attempt" "$start" "$status"
    echo "$(date +%T) [$lane] end   $index ${RUN_NAMES[$index]} exit=$status"

    if ((status != 0)) && grep -q -E "$OOM_PATTERN" "$log"; then
        rename_failed "runs/gym_imagination/${RUN_NAMES[$index]}" "$attempt"
        if ((attempt <= MAX_OOM_RETRIES)); then
            add_to_retry "$index" "out of GPU memory, attempt $attempt"
        else
            echo "$(date +%T) queue: $index ${RUN_NAMES[$index]} ran out of GPU memory $attempt times; no more attempts"
        fi
        return 2
    fi
    return $status
}

lane() {  # name entries...
    local lane=$1 index
    for index in "${@:2}"; do
        run_entry "$index" "$lane"
    done
}

# 1. The two lanes.
lane a "${LANE_A[@]}" 2>&1 | tee -a "$QUEUE_DIR/lane_a.txt" &
LANE_A_PID=$!
# Stagger the starts so the two runs do not compile and pretrain the tokenizer at the same moment.
sleep "$LAUNCH_DELAY_SECONDS"
lane b "${LANE_B[@]}" 2>&1 | tee -a "$QUEUE_DIR/lane_b.txt" &
LANE_B_PID=$!
wait "$LANE_A_PID" "$LANE_B_PID"

# 2. The entries that went back to the queue, one at a time (alone on the GPU).
while [[ -s $QUEUE_DIR/retry.txt ]]; do
    mapfile -t pending < "$QUEUE_DIR/retry.txt"
    : > "$QUEUE_DIR/retry.txt"
    echo "$(date +%T) queue: run again, alone: ${pending[*]}"
    for index in "${pending[@]}"; do
        lane retry "$index" 2>&1 | tee -a "$QUEUE_DIR/lane_retry.txt"
    done
done

echo "end: $(date --iso-8601=seconds)" >> "$QUEUE_DIR/queue.txt"
echo "queue: done"
column -t -s $'\t' "$QUEUE_DIR/status.tsv"
