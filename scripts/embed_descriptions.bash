#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/slib/monitor_cmd.bash"

# === Shared configs ===
export CUDA_VISIBLE_DEVICES=0 # One GPU

# Debug mode using `--debug` flag
DEBUG_MODE=${DEBUG_MODE:-0}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--debug)
      DEBUG_MODE=1
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--debug|-d]"
      exit 1
      ;;
  esac
done

DEBUG_ARGS=()
if [[ "${DEBUG_MODE}" -eq 1 ]]; then
  DEBUG_ARGS+=(--debug_mode)
fi



monitor_cmd "descript_embed" "out/datasets" python src/descript_embed.py \
--encoder "siglip" \
"${DEBUG_ARGS[@]}"
