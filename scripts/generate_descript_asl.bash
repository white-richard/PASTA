#!/usr/bin/env bash
set -euo pipefail

DEBUG_MODE=0
for arg in "$@"; do
    if [[ "$arg" == "--debug" ]]; then
        DEBUG_MODE=1
        break
    fi
done

# Hide noisy logs
export VLLM_LOGGING_LEVEL=ERROR
export TRANSFORMERS_VERBOSITY=error

# CUDA / memory settings
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

source "$(dirname "$0")/slib/monitor_cmd.bash"


# === Dataset config ===
DATASET="asl"

# Change this path to accept the ASL frames
# Expected structure:
#   ${IMG_PATH}/train/<video_name>/*.jpg
#   ${IMG_PATH}/dev/<video_name>/*.jpg
#   ${IMG_PATH}/test/<video_name>/*.jpg
IMG_PATH="${ASL_IMG_PATH:-/path/to/asl/frames/}"

splits=("train" "dev" "test")


# === Model / generation config ===
save_text=1
extract_hidden_states=1
chunk_size=500
video_bs=32

MODEL_FAMILY="gemma4"
MODEL_ID="unsloth/gemma-4-26B-A4B-it-GGUF"
HF_MODEL_ID="google/gemma-4-E2B-it"
# ================================


for split in "${splits[@]}"; do
    extra_args=()

    if [[ "$extract_hidden_states" -eq 1 ]]; then
        extra_args+=(--extract-hidden-states)
    fi

    if [[ "$save_text" -eq 1 ]]; then
        extra_args+=(--save-text)
    fi

    monitor_cmd "generate_descript_${DATASET}_${split}" "tmp" \
    python src/generate_descript_asl.py \
        --dataset "$DATASET" \
        --img_path "$IMG_PATH" \
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