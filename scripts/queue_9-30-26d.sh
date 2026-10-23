#!/usr/bin/env bash
# Queue of 9-30-26d (train/ablations.py entries 49, 50): RL without rehearsal from the scaled40 pretrains of
# entries 45 and 46 (queue 9-30-26c stopped before these ran). One lane for each seed; the two runs share the GPU and
# start START_GAP_SECONDS apart. All processes run on CPUs 0-15.
#   offline_pretrained_a2_scaled40_no_rehearsal_s{42,4}   (49, 50)
#
# Usage:   scripts/queue_9-30-26d.sh
# Options: PYTHON=<python>, FORCE=1, DRY_RUN=1 (as scripts/queue_9-30-26c.sh)

set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"

PYTHON=${PYTHON:-/data/venvs/dreamer4lucid-x86/bin/python}
STORAGE=/data/logs/dreamer4lucid
QUEUE_DIR=runs/queues/9-30-26d_$(date +%Y-%m-%d_%H%M%S)
LAUNCH_DELAY_SECONDS=60
START_GAP_SECONDS=90
CPUS=0-15
DATASET=runs/offline_data/HalfCheetah-v4/halfcheetah_v4_mix_scaled40_5p5m

LANE_S42=(49)
LANE_S4=(50)
INDICES=(49 50)
declare -A RUN_NAMES=(
    [49]=offline_pretrained_a2_scaled40_no_rehearsal_s42
    [50]=offline_pretrained_a2_scaled40_no_rehearsal_s4
)
# The pretrain that each entry loads.
declare -A PRETRAIN_OF=([49]=a2_scaled40_s42 [50]=a2_scaled40_s4)
PRETRAIN_ENTRIES=" "

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

gpu_used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if ((gpu_used_mib > 1024)) && [[ ${FORCE:-0} != 1 ]]; then
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2
    fail "other processes use ${gpu_used_mib} MiB of GPU memory; stop them or set FORCE=1"
fi

free_gib=$(df --output=avail -BG "$STORAGE" | tail -1 | tr -dc 0-9)
# Two RL runs without checkpoints: two replays of about 4 GB.
((free_gib >= 20)) || fail "$STORAGE has ${free_gib} GB free; the queue needs about 20 GB"

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
    echo "lanes: s42=${LANE_S42[*]} s4=${LANE_S4[*]}"
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

lane() {  # name entries...
    local lane=$1 index checkpoint
    for index in "${@:2}"; do
        checkpoint=runs/offline_pretrain/${PRETRAIN_OF[$index]}/pretrained.pt
        if [[ $PRETRAIN_ENTRIES == *" $index "* || -f $checkpoint ]]; then
            run_entry "$index" "$lane"
        else
            echo "$(date +%T) [$lane] skip  $index ${RUN_NAMES[$index]}: $checkpoint is missing"
            record "$index" "$lane" "$(date +%s)" skipped
        fi
    done
}

echo "queue: record in $QUEUE_DIR"
lane s42 "${LANE_S42[@]}" 2>&1 | tee -a "$QUEUE_DIR/lane_s42.txt" &
LANE_S42_PID=$!
# Stagger the starts so the two runs do not compile and pretrain the tokenizer at the same moment.
sleep "$LAUNCH_DELAY_SECONDS"
lane s4 "${LANE_S4[@]}" 2>&1 | tee -a "$QUEUE_DIR/lane_s4.txt" &
LANE_S4_PID=$!

wait "$LANE_S42_PID" "$LANE_S4_PID"
echo "end: $(date --iso-8601=seconds)" >> "$QUEUE_DIR/queue.txt"

echo "queue: done"
column -t -s $'\t' "$QUEUE_DIR/status.tsv"
