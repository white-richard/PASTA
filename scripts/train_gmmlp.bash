#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$(dirname "$0")/slib/monitor_cmd.bash"

# Pre-extracted ViT features (set to "" to use live ViT + LoRA + grounding instead)
# PREEXTRACTED_DIR="datasets/phoenix-descript/gmmlp_features"
PREEXTRACTED_DIR=""
SIGLIP_DIR="datasets/phoenix-descript"
# Grounding features — only used when PREEXTRACTED_DIR is empty
GROUNDING_DIR="datasets/phoenix-descript/hidden_states_gemma_4_26B_A4B_it_GGUF"
GROUNDING_LAYER=20
SIGLIP_TRAIN="${SIGLIP_DIR}/phoenix_SLdescriptions_siglip2_train.pt"
SIGLIP_DEV="${SIGLIP_DIR}/phoenix_SLdescriptions_siglip2_dev.pt"
SIGLIP_TEST="${SIGLIP_DIR}/phoenix_SLdescriptions_siglip2_test.pt"

GROUNDING_ARGS=()
if [[ -z "${PREEXTRACTED_DIR}" ]]; then
  GROUNDING_ARGS+=(--grounding_feat_dir "${GROUNDING_DIR}" --grounding_hidden_layer "${GROUNDING_LAYER}")
fi

monitor_cmd "train_gmmlp" "out/gmmlp" python src/train_gmmlp.py \
  --batch-size 64 \
  --epochs 20 \
  --opt adamw \
  --lr 1e-4 \
  --weight-decay 0.05 \
  --warmup-epochs 2 \
  --output_dir out/gmmlp \
  --model_id "google/gemma-4-E2B-it" \
  --model_family gemma4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --num_latents 64 \
  --num_media_embeds 512 \
  --vision_chunk_size 8 \
  --grad_chunk_size 32 \
  --temperature 0.07 \
  --lambda_ground 0.0 \
  --ground_loss_type mse \
  --siglip_feat_train "${SIGLIP_TRAIN}" \
  --siglip_feat_dev   "${SIGLIP_DEV}" \
  --siglip_feat_test  "${SIGLIP_TEST}" \
  --num_workers 4 \
  --eval_num_workers 4 \
  ${PREEXTRACTED_DIR:+--preextracted_feat_dir "${PREEXTRACTED_DIR}"} \
  "${GROUNDING_ARGS[@]+"${GROUNDING_ARGS[@]}"}" \
  "$@"