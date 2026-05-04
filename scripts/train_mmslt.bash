#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS=2 # Set to 1 or 2

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

VISION_BACKBONE="resnet18"
CONFIG="src/configs/config_mmslt_phoenix.yaml"
OUTPUT_DIR="out/mmslt"
SKIP_VAL="false"
EVAL_EVERY=5           # run dev evaluation every N epochs
EVAL_MAX_NEW_TOKENS=80 # Phoenix avg translation ~10 words; 80 is generous
EVAL_NUM_BEAMS=4       # ignored for gemma4 (greedy), used for mbart
# Set to true to pass --eval-metrics (extra BLEU-1/2/3 and ROUGE during evaluation).
EVAL_METRICS="true"

# Optional: path to an MMSLT checkpoint (.pth). When set, run eval-only.
TEST_CHECKPOINT=""

# Optional: path to a pretrained GMMLP checkpoint (from train_gmmlp.bash).
# When set, the SigLIP2 ViT + Perceiver from that checkpoint replaces
# VISION_BACKBONE. Leave empty to use the standard backbone.
GMMLP_CHECKPOINT="checkpoint_epoch_99_devloss_3p7550.pth"
# Architecture must match the checkpoint produced by train_gmmlp.bash.
GMMLP_MODEL_ID="google/gemma-4-E2B-it"
GMMLP_MODEL_FAMILY="gemma4"
GMMLP_NUM_LATENTS=64
GMMLP_NUM_MEDIA_EMBEDS=512
GMMLP_VISION_CHUNK_SIZE=8
GMMLP_LORA_R=16
GMMLP_LORA_ALPHA=32

DECODER_ARGS=()
if [[ "${LANGUAGE_DECODER}" == "gemma4" ]]; then
    DECODER_ARGS+=(--language_decoder gemma4 --gemma4_model_id "${GEMMA4_MODEL_ID}")
else
    DECODER_ARGS+=(--language_decoder mbart)
fi

GMMLP_ARGS=()
if [[ -n "${GMMLP_CHECKPOINT}" ]]; then
    GMMLP_ARGS+=(
        --gmmlp_checkpoint "${GMMLP_CHECKPOINT}"
        --gmmlp_model_id "${GMMLP_MODEL_ID}"
        --gmmlp_model_family "${GMMLP_MODEL_FAMILY}"
        --gmmlp_num_latents "${GMMLP_NUM_LATENTS}"
        --gmmlp_num_media_embeds "${GMMLP_NUM_MEDIA_EMBEDS}"
        --gmmlp_vision_chunk_size "${GMMLP_VISION_CHUNK_SIZE}"
        --gmmlp_lora_r "${GMMLP_LORA_R}"
        --gmmlp_lora_alpha "${GMMLP_LORA_ALPHA}"
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

if [[ "${NUM_GPUS}" -ge 2 ]]; then
    LAUNCH=(accelerate launch --num_processes "${NUM_GPUS}")
else
    LAUNCH=(python)
fi

monitor_cmd "train_mmslt" "${OUTPUT_DIR}" "${LAUNCH[@]}" src/train_mmslt.py \
    --batch-size 2 \
    --accum-steps 4 \
    --gradient-checkpointing \
    --epochs 65 \
    --opt adamw \
    --lr 5e-4 \
    --lr-llm 1e-4 \
    --clip-grad 1.0 \
    --weight-decay 0.05 \
    --warmup-epochs 0 \
    --config "${CONFIG}" \
    --vision_backbone "${VISION_BACKBONE}" \
    --gmmlp_feat_cache "datasets/phoenix-vision_feats/A4B_features" \
    --gmmlp_n_tokens 0 \
    --output_dir "${OUTPUT_DIR}" \
    --num_workers 8 \
    --eval_num_workers 4 \
    --eval-every "${EVAL_EVERY}" \
    --eval-max-new-tokens "${EVAL_MAX_NEW_TOKENS}" \
    --eval-num-beams "${EVAL_NUM_BEAMS}" \
    "${DECODER_ARGS[@]}" \
    "${GMMLP_ARGS[@]+"${GMMLP_ARGS[@]}"}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
    "${FILTERED_ARGS[@]+"${FILTERED_ARGS[@]}"}" \
    "$@"
