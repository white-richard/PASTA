#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS=1 # Set to 1 or 2

if [[ "${NUM_GPUS}" -ge 2 ]]; then
    export CUDA_VISIBLE_DEVICES=0,1
else
    export CUDA_VISIBLE_DEVICES=0
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1 # flush Python output immediately so errors survive process-group kills
ulimit -n 65536           # prevent "Too many open files" from DataLoader tensor fd sharing
source "$(dirname "$0")/slib/monitor_cmd.bash"

LANGUAGE_DECODER="gemma4" # mbart | gemma4
GEMMA4_MODEL_ID="google/gemma-4-E2B-it"

VISION_BACKBONE=""
CONFIG="src/configs/config_mmslt_phoenix.yaml"
OUTPUT_DIR="out/pasta"
SKIP_VAL="false"
EVAL_EVERY=5           # run dev evaluation every N epochs
EVAL_MAX_NEW_TOKENS=80 # Phoenix avg translation ~10 words; 80 is generous
EVAL_NUM_BEAMS=4       # ignored for gemma4 (greedy), used for mbart
# Set to true to pass --eval-metrics (extra BLEU-1/2/3 and ROUGE during evaluation).
EVAL_METRICS="true"

# Optional: path to a PASTA checkpoint (.pth). When set, run eval-only.
TEST_CHECKPOINT=""

# Set to "true" to freeze all language decoder (mbart/gemma4) parameters.
FREEZE_LLM="true"
# Optional: text prompt prepended to the LLM decoder during training and generation.
# Leave empty to disable.
DECODER_PROMPT="Übersetze die Gebärden in deutschen Text:"

# Optional: path to a pretrained PPASTA checkpoint (from train_ppasta.bash).
# When set, the SigLIP2 ViT + Perceiver from that checkpoint replaces
# VISION_BACKBONE. Leave empty to use the standard backbone.
PPASTA_CHECKPOINT="out/ppasta/best_checkpoint.pth"
# Architecture must match the checkpoint produced by train_ppasta.bash.
PPASTA_MODEL_ID="google/gemma-4-E2B-it"
PPASTA_MODEL_FAMILY="gemma4"
PPASTA_NUM_LATENTS=64
PPASTA_NUM_MEDIA_EMBEDS=512
PPASTA_VISION_CHUNK_SIZE=16
PPASTA_LORA_R=8
PPASTA_LORA_ALPHA=16

DECODER_ARGS=()
if [[ "${LANGUAGE_DECODER}" == "gemma4" ]]; then
    DECODER_ARGS+=(--language_decoder gemma4 --gemma4_model_id "${GEMMA4_MODEL_ID}")
else
    DECODER_ARGS+=(--language_decoder mbart)
fi

PPASTA_ARGS=()
if [[ -n "${PPASTA_CHECKPOINT}" ]]; then
    PPASTA_ARGS+=(
        --ppasta_checkpoint "${PPASTA_CHECKPOINT}"
        --ppasta_model_id "${PPASTA_MODEL_ID}"
        --ppasta_model_family "${PPASTA_MODEL_FAMILY}"
        --ppasta_num_latents "${PPASTA_NUM_LATENTS}"
        --ppasta_num_media_embeds "${PPASTA_NUM_MEDIA_EMBEDS}"
        --ppasta_vision_chunk_size "${PPASTA_VISION_CHUNK_SIZE}"
        --ppasta_lora_r "${PPASTA_LORA_R}"
        --ppasta_lora_alpha "${PPASTA_LORA_ALPHA}"
    )
fi

FILTERED_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
    --test-checkpoint)
        TEST_CHECKPOINT="$2"
        shift 2
        ;;
    *)
        FILTERED_ARGS+=("$1")
        shift
        ;;
    esac
done

EXTRA_ARGS=()
if [[ "${SKIP_VAL}" == "true" ]]; then
    EXTRA_ARGS+=(--skip_val)
fi
if [[ "${EVAL_METRICS}" == "true" ]]; then
    EXTRA_ARGS+=(--eval-metrics)
fi
if [[ -n "${TEST_CHECKPOINT}" ]]; then
    EXTRA_ARGS+=(--eval --resume "${TEST_CHECKPOINT}")
fi
if [[ "${FREEZE_LLM}" == "true" ]]; then
    EXTRA_ARGS+=(--freeze-llm)
fi
if [[ -n "${DECODER_PROMPT}" ]]; then
    EXTRA_ARGS+=(--decoder-prompt "${DECODER_PROMPT}")
fi

if [[ "${NUM_GPUS}" -ge 2 ]]; then
    LAUNCH=(accelerate launch --num_processes "${NUM_GPUS}")
else
    LAUNCH=(python)
fi

monitor_cmd "train_pasta" "${OUTPUT_DIR}" "${LAUNCH[@]}" src/train_pasta.py \
    --batch-size 2 \
    --accum-steps 4 \
    --epochs 50 \
    --opt adamw \
    --lr 1e-4 \
    --lr-llm 1e-4 \
    --min-lr 1e-5 \
    --clip-grad 1.0 \
    --weight-decay 0.01 \
    --warmup-epochs 2 \
    --config "${CONFIG}" \
    --vision_backbone "${VISION_BACKBONE}" \
    --ppasta_feat_cache "datasets/phoenix-vision_feats/A4B_features" \
    --ppasta_n_tokens 0 \
    --output_dir "${OUTPUT_DIR}" \
    --num_workers 8 \
    --eval_num_workers 4 \
    --eval-every "${EVAL_EVERY}" \
    --eval-max-new-tokens "${EVAL_MAX_NEW_TOKENS}" \
    --eval-num-beams "${EVAL_NUM_BEAMS}" \
    "${DECODER_ARGS[@]}" \
    "${PPASTA_ARGS[@]+"${PPASTA_ARGS[@]}"}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
    "${FILTERED_ARGS[@]+"${FILTERED_ARGS[@]}"}" \
    "$@"
