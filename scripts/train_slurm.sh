#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <sbatch|sbatchc|srun> <time> [python args...]" >&2
  exit 2
fi

MODE="$1"
TIME="$2"
shift 2

if [[ "$MODE" != "sbatch" && "$MODE" != "sbatchc" && "$MODE" != "srun" ]]; then
  echo "usage: $0 <sbatch|sbatchc|srun> <time> [python args...]" >&2
  exit 2
fi

slurm_time_to_seconds() {
  local time="$1"
  local days=0
  local hours=0
  local minutes=0
  local seconds=0

  if [[ "$time" =~ ^([0-9]+)-([0-9]+):([0-9]{2}):([0-9]{2})$ ]]; then
    days="${BASH_REMATCH[1]}"
    hours="${BASH_REMATCH[2]}"
    minutes="${BASH_REMATCH[3]}"
    seconds="${BASH_REMATCH[4]}"
  elif [[ "$time" =~ ^([0-9]+)-([0-9]+):([0-9]{2})$ ]]; then
    days="${BASH_REMATCH[1]}"
    hours="${BASH_REMATCH[2]}"
    minutes="${BASH_REMATCH[3]}"
  elif [[ "$time" =~ ^([0-9]+)-([0-9]+)$ ]]; then
    days="${BASH_REMATCH[1]}"
    hours="${BASH_REMATCH[2]}"
  elif [[ "$time" =~ ^([0-9]+):([0-9]{2}):([0-9]{2})$ ]]; then
    hours="${BASH_REMATCH[1]}"
    minutes="${BASH_REMATCH[2]}"
    seconds="${BASH_REMATCH[3]}"
  elif [[ "$time" =~ ^([0-9]+):([0-9]{2})$ ]]; then
    minutes="${BASH_REMATCH[1]}"
    seconds="${BASH_REMATCH[2]}"
  elif [[ "$time" =~ ^[0-9]+$ ]]; then
    minutes="$time"
  else
    echo "invalid Slurm time: $time" >&2
    return 2
  fi

  printf '%d\n' "$((10#$days * 86400 + 10#$hours * 3600 + 10#$minutes * 60 + 10#$seconds))"
}

if [[ "$MODE" == "sbatchc" ]]; then
  SLEEP_SECONDS="$(slurm_time_to_seconds "$TIME")"

  exec sbatch \
    -J dreamer4-sleep \
    -A bguz-dtai-gh \
    -p ghx4 \
    -t "$TIME" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=71 \
    --mem=110g \
    --gpus-per-node=1 \
    --wrap "sleep $(printf '%q' "$SLEEP_SECONDS")"
fi

if [[ "$MODE" == "sbatch" ]]; then
  cmd="source $(printf '%q' "$HOME/lucid.sh") && python $(printf '%q' "$ROOT_DIR/train/ablations.py")"
  for arg in "$@"; do
    cmd+=" $(printf '%q' "$arg")"
  done

  exec sbatch \
    -J dreamer4-train \
    -A bguz-dtai-gh \
    -p ghx4 \
    -t "$TIME" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=71 \
    --mem=110g \
    --gpus-per-node=1 \
    --wrap "bash -lc $(printf '%q' "$cmd")"
fi

exec srun \
  --pty \
  -J dreamer4-train \
  -A bguz-dtai-gh \
  -p ghx4-interactive \
  -t "$TIME" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=71 \
  --mem=110g \
  --gpus-per-node=1 \
  --chdir "$ROOT_DIR" \
  bash -l
