#!/usr/bin/env bash
# Sets up desc-venv for generate_descript.py (LLaVA-OV / vllm inference).
# Run once before using generate_descript_author.bash.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/desc-venv"

echo "==> Installing vllm (auto-selects torch for installed CUDA version)"
uv pip install --python "$VENV/bin/python" vllm --torch-backend=auto

# vllm 0.19.0 bug: ImageChunk moved from messages to chunk in mistral_common 1.10.0
echo "==> Patching vllm pixtral.py (mistral_common 1.10+ import path)"
sed -i 's/from mistral_common.protocol.instruct.messages import ImageChunk/from mistral_common.protocol.instruct.chunk import ImageChunk/' \
  "$VENV/lib/python3.10/site-packages/vllm/model_executor/models/pixtral.py"

echo "==> Installing remaining requirements"
uv pip install --python "$VENV/bin/python" \
  -r "$ROOT/desc_requirements.txt" \
  --torch-backend=auto

echo "==> Done. Activate with: source desc-venv/bin/activate"
