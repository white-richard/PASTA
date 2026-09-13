#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$(dirname "$0")/slib/monitor_cmd.bash"

# === Config ===
HF_MODEL_ID="google/gemma-4-26B-A4B-it"
IMG_PATH="datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/"
SAVE_PATH="datasets/phoenix-vision_feats_pooled/A4B_features"
BATCH_SIZE=128
NUM_WORKERS=8
FEATURE_MODE="gap"
MAX_SOFT_TOKENS=70  # 70->63 ViT tokens (7×9), 140->~126, 280→~252 per frame
DEBUG=true  # set to true for a quick smoke-test (4×BATCH_SIZE frames per shard)
SPLITS=(train dev test)
# ==============

# Determine GPUs / shard count
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPU_IDS <<<"${CUDA_VISIBLE_DEVICES}"
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
    for ((k = 0; k < NUM_SHARDS; k++)); do
        GPU="${GPU_IDS[$k]}"
        LABEL="extract_vision_feats_${SPLIT}_shard${k}"
        (
            export CUDA_VISIBLE_DEVICES="${GPU}"
            monitor_cmd "${LABEL}" ${SAVE_PATH} uv run python -m pasta.extract_vision_feats \
                --img_path "${IMG_PATH}" \
                --split "${SPLIT}" \
                --hf-model-id "${HF_MODEL_ID}" \
                --batch-size "${BATCH_SIZE}" \
                --num-workers "${NUM_WORKERS}" \
                --save_path "${SAVE_PATH}" \
                --num-shards "${NUM_SHARDS}" \
                --shard-id "${k}" \
                --feature-mode "${FEATURE_MODE}" \
                --max-soft-tokens "${MAX_SOFT_TOKENS}" \
                $("${DEBUG}" && echo "--debug") \
                "$@"
        ) &
        PIDS+=($!)
    done
    for pid in "${PIDS[@]}"; do
        wait "${pid}"
    done

    echo "=== [${SPLIT}] merge ==="
    uv run python -m pasta.extract_vision_feats \
        --split "${SPLIT}" \
        --save_path "${SAVE_PATH}" \
        --num-shards "${NUM_SHARDS}" \
        --merge \
        --feature-mode $FEATURE_MODE \
        "$@"
done

echo "Done. Features written to ${SAVE_PATH}/features_{train,dev,test}/"
