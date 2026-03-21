#!/usr/bin/env bash
set -euo pipefail

# pkill -9 -f "train_mmlp.py"

# One GPU
export CUDA_VISIBLE_DEVICES=0

# Set to 1 to enable a script, 0 to disable it.
RUN_TRAIN_MMLP=1
RUN_TRAIN_MMSLT=1

# Shared configs
VISION_BACKBONE="resnet18"
# -----

# Debug mode default with `--debug`
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

if [[ "${RUN_TRAIN_MMLP}" -eq 1 ]]; then
  python src/train_mmlp.py \
    --batch-size 16 \
    --epochs 80 \
    --opt adamw \
    --lr 1e-4 \
    --output_dir pretrain_models/mmlp \
    --vision_backbone "${VISION_BACKBONE}" \
    --grad_chunk_size 4 \
    "${DEBUG_ARGS[@]}"
else
  echo "[SKIP] train_mmlp.py (RUN_TRAIN_MMLP=0)"
fi

if [[ "${RUN_TRAIN_MMSLT}" -eq 1 ]]; then
  python src/train_mmslt.py \
    --batch-size 2 \
    --epochs 200 \
    --opt adamw \
    --lr 1e-4 \
    --finetune pretrain_models/mmlp/best_checkpoint.pth \
    --output_dir out/mmslt \
    --vision_backbone "${VISION_BACKBONE}" \
    "${DEBUG_ARGS[@]}"
else
  echo "[SKIP] train_mmslt.py (RUN_TRAIN_MMSLT=0)"
fi
