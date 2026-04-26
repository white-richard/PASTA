#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$(dirname "$0")/slib/monitor_cmd.bash"

LANGUAGE_DECODER="mbart"  # mbart | gemma4
GEMMA4_MODEL_ID="google/gemma-4-E2B-it"

VISION_BACKBONE="resnet18"
CONFIG="src/configs/config_mmslt_phoenix.yaml"
OUTPUT_DIR="out/mmslt"

DECODER_ARGS=()
if [[ "${LANGUAGE_DECODER}" == "gemma4" ]]; then
  DECODER_ARGS+=(--language_decoder gemma4 --gemma4_model_id "${GEMMA4_MODEL_ID}")
else
  DECODER_ARGS+=(--language_decoder mbart)
fi

monitor_cmd "train_mmslt" "${OUTPUT_DIR}" python src/train_mmslt.py \
  --batch-size 8 \
  --epochs 10 \
  --opt adamw \
  --lr 1e-4 \
  --weight-decay 0.001 \
  --warmup-epochs 1 \
  --config "${CONFIG}" \
  --vision_backbone "${VISION_BACKBONE}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_workers 4 \
  --eval_num_workers 1 \
  "${DECODER_ARGS[@]}" \
  "$@"
