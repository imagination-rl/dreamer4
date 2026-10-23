#!/usr/bin/env bash
# Queue of 9-30-26 (train/ablations.py entries 34-39), seeds 42 and 4, on the offline data of entry 33
# (halfcheetah_v4_mix_past_runs_scaled_expert_5p5m: expert, expert + noise, random, past runs, scaled expert):
#   offline_pretrain_a2_past_runs_scaled_expert_s{42,4}                   offline pretrain stage (34, 35)
#   offline_pretrained_a2_past_runs_scaled_expert_rehearsal_s{42,4}       RL from the step 12,000 checkpoint, 0.5 rehearsal (36, 37)
#   offline_pretrained_a2_past_runs_scaled_expert_no_rehearsal_s{42,4}    RL from the step 12,000 checkpoint, no rehearsal (38, 39)
# Each entry differs from entries 23-28 of the 9-29-26 queue only by the offline data.
#
# Two lanes run at the same time, one lane for each seed. Each lane runs its pretrain, then the two RL runs that load
# that pretrain. The RL runs start only after the pretrain of the same lane succeeds. Two runs never start within
# START_GAP_SECONDS of each other (entry 30 of the 9-29-26 queue ran out of GPU memory when two runs started in the
# same second). All processes run on CPUs 0-15.
#
# Usage:   scripts/queue_9-30-26.sh
# Options: PYTHON=<python>  (default /data/venvs/dreamer4lucid-x86/bin/python)
#          FORCE=1          start even if another process uses GPU memory
#          DRY_RUN=1        do the checks and write the record, then stop before the runs
#
# Each start writes runs/queues/9-30-26_<time>/ with the code state (HEAD + patch), the resolved run settings,
# the package versions, the machine data, one log for each run, status.tsv, and a GPU trace.
# The run directories go to /data/logs/dreamer4lucid, with symlinks in runs/ (log_storage_dir in train/config.yaml).

set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"

PYTHON=${PYTHON:-/data/venvs/dreamer4lucid-x86/bin/python}
STORAGE=/data/logs/dreamer4lucid
QUEUE_DIR=runs/queues/9-30-26_$(date +%Y-%m-%d_%H%M%S)
LAUNCH_DELAY_SECONDS=60
START_GAP_SECONDS=90
CPUS=0-15
DATASET=runs/offline_data/HalfCheetah-v4/halfcheetah_v4_mix_past_runs_scaled_expert_5p5m
PRETRAIN_NAME=a2_past_runs_scaled_expert_s

# lane: seed pretrain_index rl_indices...
LANE_S42=(42 34 36 38)
LANE_S4=(4 35 37 39)
INDICES=(34 35 36 37 38 39)
declare -A RUN_NAMES=(
    [34]=offline_pretrain_a2_past_runs_scaled_expert_s42
    [35]=offline_pretrain_a2_past_runs_scaled_expert_s4
    [36]=offline_pretrained_a2_past_runs_scaled_expert_rehearsal_s42
    [37]=offline_pretrained_a2_past_runs_scaled_expert_rehearsal_s4
    [38]=offline_pretrained_a2_past_runs_scaled_expert_no_rehearsal_s42
    [39]=offline_pretrained_a2_past_runs_scaled_expert_no_rehearsal_s4
)

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
for seed in 42 4; do
    path=runs/offline_pretrain/$PRETRAIN_NAME$seed
    [[ -e $path || -L $path ]] && existing+=("$path")
done
((${#existing[@]} == 0)) || fail "these folders exist; remove them or rename the runs: ${existing[*]}"

gpu_used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if ((gpu_used_mib > 1024)) && [[ ${FORCE:-0} != 1 ]]; then
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2
    fail "other processes use ${gpu_used_mib} MiB of GPU memory; stop them or set FORCE=1"
fi

free_gib=$(df --output=avail -BG "$STORAGE" | tail -1 | tr -dc 0-9)
# Two RL runs write two 9.9 GB replay checkpoints each; the replays and pretrain folders need about 30 GB.
((free_gib >= 80)) || fail "$STORAGE has ${free_gib} GB free; the queue needs about 80 GB"

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
    echo "lanes: s42=${LANE_S42[*]} s4=${LANE_S4[*]} (seed pretrain rl...)"
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

lane() {  # seed pretrain rl...
    local seed=$1 pretrain=$2 lane=s$1 rl
    local rl_runs=("${@:3}")
    local checkpoint=runs/offline_pretrain/$PRETRAIN_NAME$seed/pretrained_step_12000.pt

    if run_entry "$pretrain" "$lane" && [[ -f $checkpoint ]]; then
        for rl in "${rl_runs[@]}"; do
            run_entry "$rl" "$lane"
        done
    else
        for rl in "${rl_runs[@]}"; do
            echo "$(date +%T) [$lane] skip  $rl ${RUN_NAMES[$rl]}: the pretrain run failed or $checkpoint is missing"
            record "$rl" "$lane" "$(date +%s)" skipped
        done
    fi
}

echo "queue: record in $QUEUE_DIR"
lane "${LANE_S42[@]}" 2>&1 | tee -a "$QUEUE_DIR/lane_s42.txt" &
LANE_S42_PID=$!
# Stagger the starts so the two runs do not compile and pretrain the tokenizer at the same moment.
sleep "$LAUNCH_DELAY_SECONDS"
lane "${LANE_S4[@]}" 2>&1 | tee -a "$QUEUE_DIR/lane_s4.txt" &
LANE_S4_PID=$!

wait "$LANE_S42_PID" "$LANE_S4_PID"
echo "end: $(date --iso-8601=seconds)" >> "$QUEUE_DIR/queue.txt"

echo "queue: done"
column -t -s $'\t' "$QUEUE_DIR/status.tsv"
