#!/usr/bin/env fish

# === Config ===
set python_version 3.12
set DVC_FILES \
    datasets/PHOENIX-2014-T-release-v3.dvc \
    datasets/phoenix-descript.dvc

set ROOT (realpath (dirname (status --current-filename)))

echo "=== Setting up virtual environment ==="
if not test -e .venv
    uv venv --python $python_version
end
source .venv/bin/activate.fish
set VENV $VIRTUAL_ENV

# Append the nvidia LD_LIBRARY_PATH fix into the venv's activate script so any
# future `source .venv/bin/activate.fish` also gets it without a wrapper script.
if not grep -q "nvidia.*LD_LIBRARY_PATH" .venv/bin/activate.fish
    echo '
# Prepend venv nvidia CUDA libs to shadow system CUDA (cu129 vs system 12.8).
set _nvidia_libs (find $VIRTUAL_ENV/lib -type d -name "lib" -path "*/nvidia/*" 2>/dev/null | string join ":")
if test -n "$_nvidia_libs"
    set -gx LD_LIBRARY_PATH "$_nvidia_libs:$LD_LIBRARY_PATH"
end' >> .venv/bin/activate.fish
end

echo "=== Setting up symlinks ==="
if test -z "$HOST_HOME"
    set -x HOST_HOME $HOME
end

set data_repo "$HOST_HOME/.code/latent-space"
if test -d "$data_repo"
    ln -sf "$data_repo/datasets" datasets
end

echo "=== Installing dependencies ==="

uv pip install --python "$VENV/bin/python" turboquant-vllm[vllm] --torch-backend=auto --index-strategy unsafe-best-match

uv pip install -e repos/gradcache

uv pip install --python "$VENV/bin/python" \
  -r "$ROOT/desc_requirements.txt" \
  --torch-backend=auto

# Install transformers from source for Gemma4 support (gemma4 model type not
# yet in any PyPI release as of the last check).
uv pip install --python "$VENV/bin/python" \
  "git+https://github.com/huggingface/transformers.git"

if not git submodule update --init --recursive
    echo "WARNING: git submodule update failed; continuing."
end

# Prepend venv's bundled nvidia CUDA libs so they shadow the system CUDA libs.
# turboquant-vllm installs cu129 wheels; the system has CUDA 12.8, so without
# this the system's libnvJitLink.so.12 (12.8) wins the ld search and is missing
# the 12.9 symbols that libcusparse needs.
set _nvidia_libs (find $VENV/lib -type d -name "lib" -path "*/nvidia/*" 2>/dev/null | string join ":")
if test -n "$_nvidia_libs"
    set -x LD_LIBRARY_PATH "$_nvidia_libs:$LD_LIBRARY_PATH"
end

.venv/bin/python3 -c "import torch; print('all OK')"

echo "=== Syncing DVC ==="
if not command -sq dvc
    echo "WARNING: dvc is not installed; skipping DVC sync."
else
    set STATUS (cd $data_repo; and dvc status $DVC_FILES 2>&1)
    if echo $STATUS | grep -q "up to date"
        echo "All DVC files are up to date. Skipping pull."
    else
        echo "Changes detected. Running dvc pull..."
        echo $STATUS
        cd $data_repo; and dvc pull $DVC_FILES
    end
end
