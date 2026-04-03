#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_LOGGING_LEVEL=ERROR
export TRANSFORMERS_VERBOSITY=error

VENV="$(dirname "$0")/../desc-venv"
if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "Error: desc-venv not found. Run scripts/setup_descript_env.bash first."
  exit 1
fi
# shellcheck source=/dev/null
source "$VENV/bin/activate"

source "$(dirname "$0")/lib/monitor_cmd.bash"

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
  DEBUG_ARGS+=(--debug-mode)
fi

monitor_cmd "generate_descript" "tmp" python src/generate_descript.py \
"${DEBUG_ARGS[@]}"

