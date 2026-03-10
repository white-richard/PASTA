#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
MASTER_PORT=1234

run_dist () {
  local script="$1"; shift
  torchrun \
    --nproc_per_node=1 \
    --master_port="${MASTER_PORT}" \
    "${script}" "$@"
}

run_dist train_mmlp.py \
  --batch-size 4 \
  --epochs 80 \
  --opt adamw \
  --lr 1e-4 \
  --output_dir pretrain_models/mmlp

run_dist train_mmslt.py \
  --batch-size 2 \
  --epochs 200 \
  --opt adamw \
  --lr 1e-4 \
  --finetune pretrain_models/mmlp/best_checkpoint.pth \
  --output_dir out/mmslt
