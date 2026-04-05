#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_LOGGING_LEVEL=ERROR
# export VLLM_LOGGING_CONFIG_PATH="$(dirname "$0")/logging_descript.json"
export TRANSFORMERS_VERBOSITY=error

VENV="$(dirname "$0")/../desc-venv"
if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "Error: desc-venv not found. Run scripts/setup_descript_env.bash first."
  exit 1
fi
source "$VENV/bin/activate"
source "$(dirname "$0")/slib/monitor_cmd.bash"
export CUDA_VISIBLE_DEVICES=0 # One GPU

# === Shared configs ===
splits=("train" "dev" "test")

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
  DEBUG_ARGS+=(--debug-mode)
fi

for split in "${splits[@]}"; do
    monitor_cmd "generate_descript" "tmp" \
    python src/generate_descript.py \
    --split $split \
    --chunk-size=10 \
    --extract-hidden-states \
    "${DEBUG_ARGS[@]}"
    if [[ "${DEBUG_MODE}" -eq 1 ]]; then
        break
    fi
done
