#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
source "$(dirname "$0")/slib/monitor_cmd.bash"

DEBUG_MODE=0
for arg in "$@"; do
    if [[ "$arg" == "--debug" ]]; then
        DEBUG_MODE=1
        break
    fi
done

# === Shared configs ===
encoder_name="siglip2"
# ======================

monitor_cmd "descript_embed" "out/datasets" python src/descript_embed.py \
--encoder "$encoder_name" \
"$@"
