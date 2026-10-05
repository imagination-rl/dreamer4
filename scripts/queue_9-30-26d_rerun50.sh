#!/usr/bin/env bash
# Rerun entry 50 (offline_pretrained_a2_scaled40_no_rehearsal_s4) alone, after scripts/queue_9-30-26d.sh ends.
# The first attempt ran out of GPU memory at startup while entry 49 compiled (14.7 GB).
# Options: START_DELAY_SECONDS (default 200): the wait after that process ends.
# Usage: WAIT_FOR_PID=<pid of scripts/queue_9-30-26d.sh> QUEUE_DIR=<its record folder> scripts/queue_9-30-26d_rerun50.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON=${PYTHON:-/data/venvs/dreamer4lucid-x86/bin/python}
export PYTHONUNBUFFERED=1 TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/data/cache/dreamer4lucid/torchinductor}
NAME=offline_pretrained_a2_scaled40_no_rehearsal_s4
echo "$(date +%T) waiting for process $WAIT_FOR_PID"
while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do sleep 30; done
# Wait for the GPU memory of the earlier run to be free.
sleep "${START_DELAY_SECONDS:-200}"
[[ -e runs/gym_imagination/$NAME || -L runs/gym_imagination/$NAME ]] && { echo "runs/gym_imagination/$NAME exists"; exit 1; }
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
((used < 1024)) || { echo "GPU memory in use: $used MiB"; exit 1; }
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 10 > "$QUEUE_DIR/gpu_trace_rerun50.csv" &
TRACE=$!
start=$(date +%s)
echo "$(date +%T) start 50 $NAME (rerun, alone)"
taskset -c 0-15 "$PYTHON" train/ablations.py 50 > "$QUEUE_DIR/$NAME.rerun.log" 2>&1
status=$?
end=$(date +%s)
kill $TRACE 2>/dev/null
printf "50\t%s\ts4\t%s\t%s\t%s\t%s\t%s\n" "$NAME" "$(date -d @$start --iso-8601=seconds)" "$(date -d @$end --iso-8601=seconds)" \
    "$((end - start))" "$status" "$(readlink -f runs/gym_imagination/$NAME)" >> "$QUEUE_DIR/status.tsv"
echo "$(date +%T) end   50 $NAME exit=$status"
