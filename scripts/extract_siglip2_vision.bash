#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$(dirname "$0")/slib/monitor_cmd.bash"

# === Config ===
HF_MODEL_ID="google/gemma-4-26B-A4B-it"
IMG_PATH="datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/"
SAVE_PATH="datasets/phoenix-descript/gmmlp_features"
BATCH_SIZE=64
NUM_WORKERS=8
SPLITS=(train dev test)
# ==============

# Determine GPUs / shard count
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
else
  mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
NUM_SHARDS=${#GPU_IDS[@]}
if [[ "${NUM_SHARDS}" -eq 0 ]]; then
  echo "No GPUs detected." >&2
  exit 1
fi
echo "Using ${NUM_SHARDS} GPU(s): ${GPU_IDS[*]}"

mkdir -p "${SAVE_PATH}"

# Run per-split: extract shards in parallel, then merge
for SPLIT in "${SPLITS[@]}"; do
  echo "=== [${SPLIT}] extract ==="
  PIDS=()
  for ((k=0; k<NUM_SHARDS; k++)); do
    GPU="${GPU_IDS[$k]}"
    LABEL="extract_siglip2_${SPLIT}_shard${k}"
    (
      export CUDA_VISIBLE_DEVICES="${GPU}"
      monitor_cmd "${LABEL}" "out/gmmlp_features" python src/extract_siglip_gap.py \
        --img_path "${IMG_PATH}" \
        --split "${SPLIT}" \
        --hf-model-id "${HF_MODEL_ID}" \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        --save_path "${SAVE_PATH}" \
        --num-shards "${NUM_SHARDS}" \
        --shard-id "${k}" \
        "$@"
    ) &
    PIDS+=($!)
  done
  for pid in "${PIDS[@]}"; do
    wait "${pid}"
  done

  echo "=== [${SPLIT}] merge ==="
  python src/extract_siglip_gap.py \
    --split "${SPLIT}" \
    --save_path "${SAVE_PATH}" \
    --num-shards "${NUM_SHARDS}" \
    --merge \
    "$@"
done

echo "Done. Features written to ${SAVE_PATH}/features_{train,dev,test}.pt"
