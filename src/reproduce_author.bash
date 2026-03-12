#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
MASTER_PORT=1234

# Set to 1 to enable a script, 0 to disable it.
RUN_TRAIN_MMLP=${RUN_TRAIN_MMLP:-1}
RUN_TRAIN_MMSLT=${RUN_TRAIN_MMSLT:-1}

run_dist () {
  local script="$1"; shift
  torchrun \
    --nproc_per_node=1 \
    --master_port="${MASTER_PORT}" \
    "${script}" "$@"
}

if [[ "${RUN_TRAIN_MMLP}" -eq 1 ]]; then
  run_dist train_mmlp.py \
    --batch-size 4 \
    --epochs 80 \
    --opt adamw \
    --lr 1e-4 \
    --output_dir pretrain_models/mmlp
else
  echo "[SKIP] train_mmlp.py (RUN_TRAIN_MMLP=0)"
fi

if [[ "${RUN_TRAIN_MMSLT}" -eq 1 ]]; then
  run_dist train_mmslt.py \
    --batch-size 2 \
    --epochs 200 \
    --opt adamw \
    --lr 1e-4 \
    --finetune pretrain_models/mmlp/best_checkpoint.pth \
    --output_dir out/mmslt
else
  echo "[SKIP] train_mmslt.py (RUN_TRAIN_MMSLT=0)"
fi
