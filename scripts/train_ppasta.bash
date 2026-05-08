#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1  # flush Python output immediately so errors survive process-group kills
source "$(dirname "$0")/slib/monitor_cmd.bash"

# Pre-extracted ViT features (set to "" to use live ViT + LoRA)
PREEXTRACTED_DIR="datasets/phoenix-vision_feats_pooled/A4B_features"
# PREEXTRACTED_DIR=""

# Translation token features for FILIP text side
TRANSLATION_DIR="datasets/phoenix-translations"
SIGLIP_TRAIN="${TRANSLATION_DIR}/phoenix_translations_siglip2_train.pt"
SIGLIP_DEV="${TRANSLATION_DIR}/phoenix_translations_siglip2_dev.pt"
SIGLIP_TEST="${TRANSLATION_DIR}/phoenix_translations_siglip2_test.pt"

# Set to any non-empty value to skip validation
SKIP_VALIDATION=""

# Set to any non-empty value to use the global per-frame representation from
# PREEXTRACTED_DIR (key: vis_global).  The Perceiver then attends across all
# video frames at once instead of per-frame with temporal pooling.
USE_GLOBAL_REPR="1"

# Set to any non-empty value to apply GradCache at the frame level inside the
# image encoder (live ViT mode only).  Reduces peak VRAM at ~1.5x ViT compute.
FRAME_GRAD_CACHE=""


monitor_cmd "trainppastap" "out/ppasta" python src/train_ppasta.py \
    --batch-size 64 \
    --epochs 25 \
    --opt adamw \
    --lr 0.0005 \
    --min-lr 3e-5 \
    --weight-decay 0.05 \
    --warmup-epochs 7 \
    --finetune "" \
    --output_dir out/ppasta \
    --model_id "google/gemma-4-E2B-it" \
    --model_family gemma4 \
    --lora_r 8 \
    --lora_alpha 16 \
    --num_latents 64 \
    --num_media_embeds 512 \
    --vision_chunk_size 16 \
    --grad_chunk_size 32 \
    --temperature 0.07 \
    --siglip_feat_train "${SIGLIP_TRAIN}" \
    --siglip_feat_dev "${SIGLIP_DEV}" \
    --siglip_feat_test "${SIGLIP_TEST}" \
    --num_workers 4 \
    --eval_num_workers 4 \
    ${SKIP_VALIDATION:+--skip-validation} \
    ${PREEXTRACTED_DIR:+--preextracted_feat_dir "${PREEXTRACTED_DIR}"} \
    ${USE_GLOBAL_REPR:+--use-global-repr} \
    ${FRAME_GRAD_CACHE:+--frame-grad-cache} \
    "$@"
    # --max-frames 32 \
