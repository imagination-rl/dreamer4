#!/usr/bin/env bash
# Recovery and follow-up queue of 9-30-26 (train/ablations.py entries 34, 36, 38, and 40-43), next to the running
# scripts/queue_9-30-26.sh (its lane s4: entries 35, 37, 39).
#
# Lane A (seed 42) starts when entry 35 of the running queue ends (that pretrain holds about 13 GB of GPU memory):
#   34  offline_pretrain_a2_past_runs_scaled_expert_s42            rerun; the first attempt ran out of GPU memory
#   36  offline_pretrained_a2_past_runs_scaled_expert_rehearsal_s42      RL, 0.5 rehearsal, from 34
#   38  offline_pretrained_a2_past_runs_scaled_expert_no_rehearsal_s42   RL, no rehearsal, from 34
#   40  offline_pretrain_a2_past_runs_scaled_expert_reward_grad_s42      pretrain, reward gradient into the transformer x 0.01
#   42  offline_pretrained_a2_past_runs_scaled_expert_reward_grad_no_rehearsal_s42   RL, no rehearsal, from 40
# Lane B (seed 4) starts when the running queue ends:
#   41  offline_pretrain_a2_past_runs_scaled_expert_reward_grad_s4
#   43  offline_pretrained_a2_past_runs_scaled_expert_reward_grad_no_rehearsal_s4
# Entries 40-43 differ from 34-35 and 38-39 only by scale_reward_gradient_into_transformer. The pretrain signature
# includes that flag, so their RL runs need their own pretrain.
# An RL entry runs only if the step 12,000 checkpoint of its pretrain exists. At most two runs use the GPU at a time,
# and two runs never start within START_GAP_SECONDS of each other. All processes run on CPUs 0-15.
#
# Usage:   MAIN_QUEUE_PID=<pid of scripts/queue_9-30-26.sh> MAIN_QUEUE_DIR=<its record folder> scripts/queue_9-30-26b.sh
# Options: PYTHON=<python>  (default /data/venvs/dreamer4lucid-x86/bin/python)
#          DRY_RUN=1        do the checks and write the record, then stop before the runs
#
# Each start writes runs/queues/9-30-26b_<time>/ with the code state (HEAD + patch), the resolved run settings,
# the package versions, the machine data, one log for each run, status.tsv, and a GPU trace.

set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"

PYTHON=${PYTHON:-/data/venvs/dreamer4lucid-x86/bin/python}
MAIN_QUEUE_PID=${MAIN_QUEUE_PID:?set MAIN_QUEUE_PID to the pid of scripts/queue_9-30-26.sh}
MAIN_QUEUE_DIR=${MAIN_QUEUE_DIR:?set MAIN_QUEUE_DIR to the record folder of scripts/queue_9-30-26.sh}
STORAGE=/data/logs/dreamer4lucid
QUEUE_DIR=runs/queues/9-30-26b_$(date +%Y-%m-%d_%H%M%S)
# 90 s was too short: a run compiles for several minutes and holds most of the GPU memory (entry 50 OOM).
START_GAP_SECONDS=200
CPUS=0-15
DATASET=runs/offline_data/HalfCheetah-v4/halfcheetah_v4_mix_past_runs_scaled_expert_5p5m

LANE_A=(34 36 38 40 42)
LANE_B=(41 43)
INDICES=(34 36 38 40 41 42 43)
declare -A RUN_NAMES=(
    [34]=offline_pretrain_a2_past_runs_scaled_expert_s42
    [36]=offline_pretrained_a2_past_runs_scaled_expert_rehearsal_s42
    [38]=offline_pretrained_a2_past_runs_scaled_expert_no_rehearsal_s42
    [40]=offline_pretrain_a2_past_runs_scaled_expert_reward_grad_s42
    [41]=offline_pretrain_a2_past_runs_scaled_expert_reward_grad_s4
    [42]=offline_pretrained_a2_past_runs_scaled_expert_reward_grad_no_rehearsal_s42
    [43]=offline_pretrained_a2_past_runs_scaled_expert_reward_grad_no_rehearsal_s4
)
# The pretrain (offline_pretrain_name) that each entry writes or loads.
declare -A PRETRAIN_OF=(
    [34]=a2_past_runs_scaled_expert_s42
    [36]=a2_past_runs_scaled_expert_s42
    [38]=a2_past_runs_scaled_expert_s42
    [40]=a2_past_runs_scaled_expert_reward_grad_s42
    [41]=a2_past_runs_scaled_expert_reward_grad_s4
    [42]=a2_past_runs_scaled_expert_reward_grad_s42
    [43]=a2_past_runs_scaled_expert_reward_grad_s4
)
PRETRAIN_ENTRIES=" 34 40 41 "

export PYTHONUNBUFFERED=1
# The /tmp tmpfs has a quota. The inductor cache grew to 12 GB there and stopped a run.
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/data/cache/dreamer4lucid/torchinductor}

fail() { echo "queue: $*" >&2; exit 1; }

# preflight

[[ -x "$PYTHON" ]] || fail "no python at $PYTHON"
[[ -d "$STORAGE" ]] || fail "$STORAGE does not exist; the runs would write to runs/ on /home"
[[ -f $DATASET/manifest.json ]] || fail "the offline dataset $DATASET is missing; run train/ablations.py 33"

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

existing=()
for index in "${INDICES[@]}"; do
    path=runs/gym_imagination/${RUN_NAMES[$index]}
    [[ -e $path || -L $path ]] && existing+=("$path")
done
for index in $PRETRAIN_ENTRIES; do
    path=runs/offline_pretrain/${PRETRAIN_OF[$index]}
    [[ -e $path || -L $path ]] && existing+=("$path")
done
((${#existing[@]} == 0)) || fail "these folders exist; remove them or rename the runs: ${existing[*]}"

[[ $(ps -o args= -p "$MAIN_QUEUE_PID" 2>/dev/null) == *queue_9-30-26.sh* ]] || fail "process $MAIN_QUEUE_PID is not scripts/queue_9-30-26.sh"
[[ -f $MAIN_QUEUE_DIR/status.tsv ]] || fail "$MAIN_QUEUE_DIR/status.tsv does not exist"

free_gib=$(df --output=avail -BG "$STORAGE" | tail -1 | tr -dc 0-9)
# Entry 36 writes two 9.9 GB replay checkpoints; the replays and three pretrain folders need about 25 GB.
((free_gib >= 70)) || fail "$STORAGE has ${free_gib} GB free; the queue needs about 70 GB"

# reproduction record

mkdir -p "$QUEUE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
{
    echo "start: $(date --iso-8601=seconds)"
    echo "host: $(hostname)"
    echo "repo: $REPO"
    echo "branch: $(git rev-parse --abbrev-ref HEAD)"
    echo "head: $(git rev-parse HEAD)"
    echo "python: $PYTHON ($("$PYTHON" -V 2>&1))"
    echo "command: $0 $*"
    echo "lanes: A=${LANE_A[*]} (after entry 35 of $MAIN_QUEUE_DIR) B=${LANE_B[*]} (after process $MAIN_QUEUE_PID)"
    echo "dataset: $DATASET"
    echo "env: PYTHONUNBUFFERED=$PYTHONUNBUFFERED TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} OMP_NUM_THREADS=${OMP_NUM_THREADS:-} cpus=$CPUS"
} > "$QUEUE_DIR/queue.txt"

# To rebuild the code: git checkout <head>, then git apply code.patch.
git status --porcelain=v1 > "$QUEUE_DIR/git_status.txt"
git diff --binary HEAD > "$QUEUE_DIR/code.patch"
cp train/config.yaml train/ablations.py "${BASH_SOURCE[0]}" "$DATASET/manifest.json" "$QUEUE_DIR/"

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

printf "index\trun_name\tlane\tstart\tend\tseconds\texit\tstorage\n" > "$QUEUE_DIR/status.tsv"

if [[ ${DRY_RUN:-0} == 1 ]]; then
    echo "queue: dry run, record in $QUEUE_DIR"
    exit 0
fi

# run

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 10 > "$QUEUE_DIR/gpu_trace.csv" &
GPU_TRACE_PID=$!

trap 'echo "queue: interrupted, stopping all runs" >&2; trap - INT TERM EXIT; kill 0' INT TERM
trap 'kill $GPU_TRACE_PID 2>/dev/null' EXIT

record() {  # index lane start_epoch exit
    local index=$1 name=${RUN_NAMES[$1]} end
    end=$(date +%s)
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$index" "$name" "$2" \
        "$(date -d @"$3" --iso-8601=seconds)" "$(date -d @"$end" --iso-8601=seconds)" "$((end - $3))" "$4" \
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

run_entry() {  # index lane
    local index=$1 lane=$2 start status
    wait_for_start_slot
    start=$(date +%s)
    echo "$(date +%T) [$lane] start $index ${RUN_NAMES[$index]}"
    taskset -c "$CPUS" "$PYTHON" train/ablations.py "$index" > "$QUEUE_DIR/${RUN_NAMES[$index]}.log" 2>&1
    status=$?
    record "$index" "$lane" "$start" "$status"
    echo "$(date +%T) [$lane] end   $index ${RUN_NAMES[$index]} exit=$status"
    return $status
}

wait_for_entry_35() {
    until grep -q $'^35\t' "$MAIN_QUEUE_DIR/status.tsv"; do sleep 60; done
    sleep 60  # The GPU memory of the finished run is free a moment later.
}

wait_for_main_queue() {
    while kill -0 "$MAIN_QUEUE_PID" 2>/dev/null; do sleep 60; done
    sleep 60
}

lane() {  # name wait_function entries...
    local lane=$1 wait=$2 index checkpoint
    echo "$(date +%T) [$lane] waiting ($wait)"
    "$wait"
    for index in "${@:3}"; do
        checkpoint=runs/offline_pretrain/${PRETRAIN_OF[$index]}/pretrained_step_12000.pt
        if [[ $PRETRAIN_ENTRIES == *" $index "* || -f $checkpoint ]]; then
            run_entry "$index" "$lane"
        else
            echo "$(date +%T) [$lane] skip  $index ${RUN_NAMES[$index]}: $checkpoint is missing"
            record "$index" "$lane" "$(date +%s)" skipped
        fi
    done
}

echo "queue: record in $QUEUE_DIR"
lane A wait_for_entry_35 "${LANE_A[@]}" 2>&1 | tee -a "$QUEUE_DIR/lane_a.txt" &
LANE_A_PID=$!
lane B wait_for_main_queue "${LANE_B[@]}" 2>&1 | tee -a "$QUEUE_DIR/lane_b.txt" &
LANE_B_PID=$!

wait "$LANE_A_PID" "$LANE_B_PID"
echo "end: $(date --iso-8601=seconds)" >> "$QUEUE_DIR/queue.txt"

echo "queue: done"
column -t -s $'\t' "$QUEUE_DIR/status.tsv"
