#!/usr/bin/env fish

# === Config ===
set python_version 3.10
set DVC_FILES \
    datasets/PHOENIX-2014-T-release-v3.dvc \
    datasets/phoenix-descript.dvc

echo "=== Setting up virtual environment ==="
if not test -e .venv
    uv venv --python $python_version
end
source .venv/bin/activate.fish

echo "=== Setting up symlinks ==="
if test -z "$HOST_HOME"
    set -x HOST_HOME $HOME
end
set data_repo "$HOST_HOME/.code/latent-space"
ln -sf "$data_repo/datasets" datasets

echo "=== Installing dependencies ==="

uv sync
uv pip install -e repos/gradcache
uv pip install -r vlm2vev_requirements.txt

if not git submodule update --init --recursive
    echo "WARNING: git submodule update failed; continuing."
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
