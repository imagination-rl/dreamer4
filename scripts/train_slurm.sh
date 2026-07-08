#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <sbatch|srun> <time> [python args...]" >&2
  exit 2
fi

MODE="$1"
TIME="$2"
shift 2

if [[ "$MODE" != "sbatch" && "$MODE" != "srun" ]]; then
  echo "usage: $0 <sbatch|srun> <time> [python args...]" >&2
  exit 2
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
