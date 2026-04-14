#!/usr/bin/env bash
# GMMLP Stage 1 — Vision Grounding Pretraining
#
# Prerequisites:
#   bash scripts/generate_vlm_hstates.bash  # grounding features
#   bash scripts/embed_descriptions.bash  # SigLIP text features
#
# Usage:
#   bash scripts/train_gmmlp.bash [--debug|-d]
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$(dirname "$0")/slib/monitor_cmd.bash"

export CUDA_VISIBLE_DEVICES=0

# --- Feature paths ---
GROUNDING_DIR="out/descript/features/"
GROUNDING_LAYER=20
SIGLIP_DIR="out/datasets/text_phoenix-descript"
SIGLIP_TRAIN="${SIGLIP_DIR}/phoenix_SLdescriptions_siglip_train.pt"
SIGLIP_DEV="${SIGLIP_DIR}/phoenix_SLdescriptions_siglip_dev.pt"
SIGLIP_TEST="${SIGLIP_DIR}/phoenix_SLdescriptions_siglip_test.pt"

# --- Debug mode ---
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

monitor_cmd "train_gmmlp" "out/gmmlp" python src/train_gmmlp.py \
  --batch-size 4 \
  --epochs 40 \
  --opt adamw \
  --lr 1e-4 \
  --weight-decay 0.05 \
  --warmup-epochs 2 \
  --output_dir out/gmmlp \
  --model_id "llava-hf/llava-onevision-qwen2-7b-ov-hf" \
  --lora_r 16 \
  --lora_alpha 32 \
  --num_latents 64 \
  --num_media_embeds 512 \
  --vision_chunk_size 8 \
  --grad_chunk_size 1 \
  --temperature 0.07 \
  --lambda_ground 0.1 \
  --ground_loss_type mse \
  --grounding_feat_dir "${GROUNDING_DIR}" \
  --grounding_hidden_layer "${GROUNDING_LAYER}" \
  --siglip_feat_train "${SIGLIP_TRAIN}" \
  --siglip_feat_dev   "${SIGLIP_DEV}" \
  --siglip_feat_test  "${SIGLIP_TEST}" \
  --num_workers 4 \
  --eval_num_workers 2 \
  "${DEBUG_ARGS[@]}"
