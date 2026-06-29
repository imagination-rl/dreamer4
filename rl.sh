#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <sbatch|srun> <time> --run_name <name> --run_details <details> [python args...]" >&2
  exit 2
fi

MODE="$1"
TIME="$2"
shift 2

if [[ "$MODE" != "sbatch" && "$MODE" != "srun" ]]; then
  echo "usage: $0 <sbatch|srun> <time> --run_name <name> --run_details <details> [python args...]" >&2
  exit 2
fi

PYTHON_ARGS=()
RUN_NAME=""
RUN_DETAILS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run_name)
      if [[ $# -lt 2 ]]; then
        echo "error: --run_name requires a value" >&2
        exit 2
      fi

      RUN_NAME="$2"
      PYTHON_ARGS+=("$1" "$2")
      shift 2
      ;;
    --run_name=*)
      RUN_NAME="${1#--run_name=}"
      PYTHON_ARGS+=("$1")
      shift
      ;;
    run_name=*)
      RUN_NAME="${1#run_name=}"
      PYTHON_ARGS+=("$1")
      shift
      ;;
    --run_details)
      if [[ $# -lt 2 ]]; then
        echo "error: --run_details requires a value" >&2
        exit 2
      fi

      RUN_DETAILS="$2"
      PYTHON_ARGS+=("$1" "$2")
      shift 2
      ;;
    --run_details=*)
      RUN_DETAILS="${1#--run_details=}"
      PYTHON_ARGS+=("$1")
      shift
      ;;
    run_details=*)
      RUN_DETAILS="${1#run_details=}"
      PYTHON_ARGS+=("$1")
      shift
      ;;
    *)
      PYTHON_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$RUN_NAME" ]]; then
  echo "error: run_name is required" >&2
  exit 2
fi

if [[ -z "${RUN_NAME//[[:space:]]/}" ]]; then
  echo "error: run_name must not be empty" >&2
  exit 2
fi

if [[ "$RUN_NAME" == */* || "$RUN_NAME" == "." || "$RUN_NAME" == ".." ]]; then
  echo "error: run_name must be a single folder name" >&2
  exit 2
fi

if [[ -z "$RUN_DETAILS" ]]; then
  echo "error: run_details is required" >&2
  exit 2
fi

if [[ -z "${RUN_DETAILS//[[:space:]]/}" ]]; then
  echo "error: run_details must not be empty" >&2
  exit 2
fi

if [[ "$MODE" == "sbatch" ]]; then
  cmd="source $(printf '%q' "$HOME/lucid.sh") && cd $(printf '%q' "$ROOT_DIR") && python $(printf '%q' "$ROOT_DIR/train_halfcheetah_imagination_rl.py")"
  for arg in "${PYTHON_ARGS[@]}"; do
    cmd+=" $(printf '%q' "$arg")"
  done

  exec sbatch \
    -J dreamer4-halfcheetah-rl \
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
  -J dreamer4-halfcheetah-rl \
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
