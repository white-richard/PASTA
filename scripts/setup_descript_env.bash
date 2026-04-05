#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/desc-venv"

uv pip install --python "$VENV/bin/python" turboquant-vllm[vllm] --torch-backend=auto --index-strategy unsafe-best-match

echo "==> Installing remaining requirements"
uv pip install --python "$VENV/bin/python" \
  -r "$ROOT/desc_requirements.txt" \
  --torch-backend=auto
