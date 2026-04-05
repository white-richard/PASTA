#!/bin/bash
set -euo pipefail
set -x

# === Config ===
RUN_SCRIPT="scripts/generate_descript_author.bash"
OUTPUT_DIRS="out pretrain_models tmp"
DVC_LOCAL_PUSH=0

cd /workspace
export GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=yes"

mkdir -p /root/.ssh
cp /workspace/.deploy_key /root/.ssh/id_ed25519
chmod 600 /root/.ssh/id_ed25519
cp /workspace/.known_hosts /root/.ssh/known_hosts
chmod 644 /root/.ssh/known_hosts
cat > /root/.ssh/config << 'EOF'
Host *
    IdentityFile /root/.ssh/id_ed25519
    StrictHostKeyChecking yes
EOF
chmod 600 /root/.ssh/config

echo "=== Setting up environment ==="
fish setup.fish

echo "=== Starting training ==="
source .venv/bin/activate
chmod +x "$RUN_SCRIPT"
"$RUN_SCRIPT"

echo "=== Creating experiment branch ==="
RUN_ID="${SLURM_JOB_ID}-$(date +%Y%m%dT%H%M%S)"
EXPERIMENT_BRANCH="expr/${GITHUB_BRANCH}/${RUN_ID}"
git checkout -b "$EXPERIMENT_BRANCH"

echo "=== Pushing DVC ==="
if [ "$DVC_LOCAL_PUSH" -eq 1 ]; then
    REPO_NAME=$(basename "$GITHUB_URL" .git | tr '[:upper:]' '[:lower:]')
    cat > /workspace/.dvc/config.local << EOF
[core]
    remote = local-cache
['remote "local-cache"']
    url = $HOST_HOME/.code/$REPO_NAME/.dvc/cache
EOF
fi
mkdir -p sout/${RUN_ID}
mv $OUTPUT_DIRS sout/${RUN_ID}/
dvc add sout/${RUN_ID}
git add sout/${RUN_ID}.dvc sout/.gitignore
git commit -m "Add outputs for job ${EXPERIMENT_BRANCH}"
dvc push
git push origin "$EXPERIMENT_BRANCH"

echo "=== Opening PR ==="
gh pr create \
    --base  "$GITHUB_BRANCH" \
    --head  "$EXPERIMENT_BRANCH" \
    --title "expr: job ${SLURM_JOB_ID} (${GITHUB_BRANCH})" \
    --body  "$(printf 'Automated results from `%s`.\n\n- Base branch: `%s`\n- DVC outputs: `sout/%s/`' \
               "$SLURM_JOB_ID" "$GITHUB_BRANCH" "$RUN_ID")"