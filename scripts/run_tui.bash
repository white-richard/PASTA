#!/usr/bin/env bash
# Launch the MMSLT evaluation TUI.
# Mirrors the architecture settings from train_mmslt.bash — keep them in sync.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n 65536

LANGUAGE_DECODER="gemma4"   # mbart | gemma4
GEMMA4_MODEL_ID="google/gemma-4-E2B-it"
VISION_BACKBONE="resnet18"
CONFIG="src/configs/config_mmslt_phoenix.yaml"

# ── Checkpoint ──────────────────────────────────────────────────────────────
# Defaults to the best checkpoint written by train_mmslt.bash.
CHECKPOINT="${1:-out/mmslt/best_checkpoint.pth}"

# ── GMMLP ───────────────────────────────────────────────────────────────────
GMMLP_CHECKPOINT="out/gmmlp/checkpoint.pth"
GMMLP_MODEL_ID="google/gemma-4-E2B-it"
GMMLP_MODEL_FAMILY="gemma4"
GMMLP_NUM_LATENTS=64
GMMLP_NUM_MEDIA_EMBEDS=512
GMMLP_VISION_CHUNK_SIZE=8
GMMLP_LORA_R=16
GMMLP_LORA_ALPHA=32
GMMLP_FEAT_CACHE="out/gmmlp_vit_feats"

# Inference params
EVAL_MAX_NEW_TOKENS=80
EVAL_NUM_BEAMS=4

GMMLP_ARGS=()
if [[ -n "${GMMLP_CHECKPOINT}" && -f "${GMMLP_CHECKPOINT}" ]]; then
  GMMLP_ARGS+=(
    --gmmlp_checkpoint "${GMMLP_CHECKPOINT}"
    --gmmlp_model_id "${GMMLP_MODEL_ID}"
    --gmmlp_model_family "${GMMLP_MODEL_FAMILY}"
    --gmmlp_num_latents "${GMMLP_NUM_LATENTS}"
    --gmmlp_num_media_embeds "${GMMLP_NUM_MEDIA_EMBEDS}"
    --gmmlp_vision_chunk_size "${GMMLP_VISION_CHUNK_SIZE}"
    --gmmlp_lora_r "${GMMLP_LORA_R}"
    --gmmlp_lora_alpha "${GMMLP_LORA_ALPHA}"
    --gmmlp_feat_cache "${GMMLP_FEAT_CACHE}"
  )
fi

uv run python -m tui.app \
  --checkpoint "${CHECKPOINT}" \
  --config "${CONFIG}" \
  --language_decoder "${LANGUAGE_DECODER}" \
  --gemma4_model_id "${GEMMA4_MODEL_ID}" \
  --vision_backbone "${VISION_BACKBONE}" \
  --eval_max_new_tokens "${EVAL_MAX_NEW_TOKENS}" \
  --eval_num_beams "${EVAL_NUM_BEAMS}" \
  "${GMMLP_ARGS[@]+"${GMMLP_ARGS[@]}"}" \
  "${@:2}"
