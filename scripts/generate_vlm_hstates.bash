#!/usr/bin/env bash
set -euo pipefail
# Prepend the venv's bundled nvidia CUDA libs so they shadow the system CUDA
# 12.8 libs. The venv has cu129 wheels; the system only has CUDA 12.8, whose
# libnvJitLink.so.12 is missing __nvJitLinkGetErrorLogSize_12_9.
_nvidia_libs=$(find "$(dirname "$0")/../.venv/lib" -type d -name "lib" -path "*/nvidia/*" 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${_nvidia_libs%:}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_LOGGING_LEVEL=ERROR
# export VLLM_LOGGING_CONFIG_PATH="$(dirname "$0")/logging_descript.json"
export TRANSFORMERS_VERBOSITY=error

VENV="$(dirname "$0")/../.venv"
if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "Error: .venv not found. Run setup.fish first."
  exit 1
fi
source "$VENV/bin/activate"
source "$(dirname "$0")/slib/monitor_cmd.bash"
export CUDA_VISIBLE_DEVICES=0 # One GPU

# === Shared configs ===
splits=("train" "dev" "test")

# Debug mode using `--debug` flag
DEBUG_MODE=${DEBUG_MODE:-0}
DEBUG_MODE=1
MODEL_FAMILY="gemma4"
MODEL_ID="unsloth/gemma-4-26B-A4B-it-GGUF" # cyankiwi/gemma-4-31B-it-AWQ-4bit | google/gemma-4-E2B-it | unsloth/gemma-4-E4B-it-GGUF | unsloth/gemma-4-26B-A4B-it-GGUF
# HF model for hidden-state extraction; must be a standard (non-GGUF) repo
# loadable by transformers.  A 4B model needs only ~2 GB VRAM at 4-bit vs
# ~50 GB RAM just to mmap the 26B safetensors shards.
HF_MODEL_ID="google/gemma-4-E2B-it"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--debug)
      DEBUG_MODE=1
      shift
      ;;
    --model_family|--model-family)
      MODEL_FAMILY="$2"
      shift 2
      ;;
    --model_id|--model-id)
      MODEL_ID="$2"
      shift 2
      ;;
    --hf-model-id|--hf_model_id)
      HF_MODEL_ID="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--debug|-d] [--model-family llava|gemma4] [--model-id <hf-repo>] [--hf-model-id <hf-repo>]"
      exit 1
      ;;
  esac
done

DEBUG_ARGS=()
if [[ "${DEBUG_MODE}" -eq 1 ]]; then
  DEBUG_ARGS+=(--debug-mode)
fi

MODEL_ARGS=(--model_family "${MODEL_FAMILY}")
if [[ -n "${MODEL_ID}" ]]; then
  MODEL_ARGS+=(--model_id "${MODEL_ID}")
fi
if [[ -n "${HF_MODEL_ID}" ]]; then
  MODEL_ARGS+=(--hf-model-id "${HF_MODEL_ID}")
fi

for split in "${splits[@]}"; do
    monitor_cmd "generate_descript" "tmp" \
    python src/generate_descript.py \
    --split $split \
    --chunk-size=29 \
    --extract-hidden-states \
    "${MODEL_ARGS[@]}" \
    "${DEBUG_ARGS[@]}"
    if [[ "${DEBUG_MODE}" -eq 1 ]]; then
        break
    fi
done
