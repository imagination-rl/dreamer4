#!/usr/bin/env bash
set -euo pipefail

readonly FIELD_IDS="1002,1003,1004,1005"

for command_name in nvidia-smi dcgmi; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "required command not found: $command_name" >&2
    exit 127
  fi
done

if ! nvidia_output="$(nvidia-smi \
  --query-gpu=index,uuid,pci.bus_id,name \
  --format=csv)"; then
  echo "nvidia-smi query failed" >&2
  exit 1
fi

if ! discovery_output="$(dcgmi discovery -l)"; then
  echo "dcgmi discovery failed" >&2
  exit 1
fi

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

declare -A nvidia_index_by_uuid=()
while IFS=, read -r raw_index raw_uuid _pci_bus_id _name; do
  index="$(trim "$raw_index")"
  uuid="$(trim "$raw_uuid")"

  [[ "$index" == "index" ]] && continue
  [[ "$index" =~ ^[0-9]+$ ]] || continue
  [[ "$uuid" =~ ^GPU-[[:alnum:]-]+$ ]] || continue

  uuid="${uuid,,}"
  if [[ -n "${nvidia_index_by_uuid[$uuid]+present}" ]]; then
    echo "duplicate GPU UUID in nvidia-smi output: $uuid" >&2
    exit 1
  fi
  nvidia_index_by_uuid["$uuid"]="$index"
done <<< "$nvidia_output"

declare -A dcgmi_id_by_uuid=()
current_dcgmi_id=""
while IFS= read -r line; do
  if [[ "$line" =~ \|[[:space:]]*([0-9]+)[[:space:]]*\| ]]; then
    current_dcgmi_id="${BASH_REMATCH[1]}"
  fi

  if [[ "$line" =~ UUID:[[:space:]]*(GPU-[[:alnum:]-]+) ]]; then
    if [[ -z "$current_dcgmi_id" ]]; then
      echo "DCGM discovery output has a UUID without a GPU ID" >&2
      exit 1
    fi

    uuid="${BASH_REMATCH[1],,}"
    if [[ -n "${dcgmi_id_by_uuid[$uuid]+present}" && "${dcgmi_id_by_uuid[$uuid]}" != "$current_dcgmi_id" ]]; then
      echo "duplicate GPU UUID with different DCGM IDs: $uuid" >&2
      exit 1
    fi
    dcgmi_id_by_uuid["$uuid"]="$current_dcgmi_id"
  fi
done <<< "$discovery_output"

common_uuid=""
common_count=0
for uuid in "${!nvidia_index_by_uuid[@]}"; do
  if [[ -n "${dcgmi_id_by_uuid[$uuid]+present}" ]]; then
    common_uuid="$uuid"
    ((common_count += 1))
  fi
done

if (( common_count != 1 )); then
  echo "expected exactly one GPU UUID in both command outputs; found $common_count" >&2
  exit 1
fi

gpu_id="${nvidia_index_by_uuid[$common_uuid]}"
echo "monitoring GPU UUID $common_uuid with GPU index $gpu_id" >&2
exec dcgmi dmon -i "$gpu_id" -e "$FIELD_IDS"
