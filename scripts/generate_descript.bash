#!/usr/bin/env bash
set -euo pipefail

DEBUG_MODE=0
for arg in "$@"; do
    if [[ "$arg" == "--debug" ]]; then
        DEBUG_MODE=1
        break
    fi
done

# Hide annoying prints
export VLLM_LOGGING_LEVEL=ERROR
export TRANSFORMERS_VERBOSITY=error

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
source "$(dirname "$0")/slib/monitor_cmd.bash"


# === Shared configs ===
splits=("train" "dev" "test")
save_text=1
extract_hidden_states=1
chunk_size=500  # frames per checkpoint
video_bs=32  # inference batch size
MODEL_FAMILY="gemma4"
MODEL_ID="unsloth/gemma-4-26B-A4B-it-GGUF"  # cyankiwi/gemma-4-31B-it-AWQ-4bit | google/gemma-4-E2B-it | unsloth/gemma-4-E4B-it-GGUF | unsloth/gemma-4-26B-A4B-it-GGUF
HF_MODEL_ID="google/gemma-4-E2B-it"  # HF model for hidden-state extraction; must be a standard (non-GGUF) repo
# ======================


for split in "${splits[@]}"; do
    extra_args=()
    if [[ "$extract_hidden_states" -eq 1 ]]; then
        extra_args+=(--extract-hidden-states)
    fi
    if [[ "$save_text" -eq 1 ]]; then
        extra_args+=(--save-text)
    fi

    monitor_cmd "generate_descript" "tmp" \
    python src/generate_descript.py \
    --split "$split" \
    --model_family "$MODEL_FAMILY" \
    --model_id "$MODEL_ID" \
    --hf-model-id "$HF_MODEL_ID" \
    --chunk-size="$chunk_size" \
    --video_bs="$video_bs" \
    "${extra_args[@]}" \
    "$@"

    if [[ "${DEBUG_MODE}" -eq 1 ]]; then
        break
    fi
done
