#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
PROJECT_NAME="$(basename "$PROJECT_ROOT")"
SECRET_FILE="$HOME/codes/.secrets/proxy.credentials"

if [ ! -f "$SECRET_FILE" ]; then
    echo "ERROR: missing $SECRET_FILE" >&2
    exit 1
fi

NNI="$(sed -n '1p' "$SECRET_FILE" | tr -d '[:space:]')"
nni="${NNI,,}"
if [[ ! "$nni" =~ ^[a-z][a-z0-9_-]*$ ]]; then
    echo "ERROR: the first line of $SECRET_FILE must contain the NNI" >&2
    exit 1
fi

SOURCE_ROOT="$PROJECT_ROOT/"
DESTINATION="$nni@selena.hpc.edf.fr:~/codes/$PROJECT_NAME/"

if [ ! -f "$SOURCE_ROOT/README.md" ] || [ ! -f "$SOURCE_ROOT/sync_code_to_selena.sh" ]; then
    echo "ERROR: project root not found at $SOURCE_ROOT" >&2
    exit 1
fi

echo "Synchronizing $PROJECT_NAME code from DGX to Selena..."
rsync -rlptz --delete --partial --info=progress2 \
    --exclude='.git/' \
    --exclude='.venv/' \
    --exclude='.secrets/' \
    --exclude='pyproject.toml' \
    --exclude='uv.lock' \
    --exclude='datasets/' \
    --exclude='weights/' \
    --exclude='outputs/' \
    --exclude='logs/' \
    --include='outputs_selena/' \
    --include='outputs_selena/.gitkeep' \
    --exclude='outputs_selena/***' \
    --include='logs_selena/' \
    --include='logs_selena/.gitkeep' \
    --exclude='logs_selena/***' \
    "$SOURCE_ROOT" \
    "$DESTINATION"

echo "SUCCESS: Selena's $PROJECT_NAME code matches DGX."
echo "Preserved on Selena: .venv, .secrets, pyproject.toml, uv.lock, datasets, weights, outputs, logs, outputs_selena, and logs_selena."
