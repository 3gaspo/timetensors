#!/bin/bash

set -euo pipefail

usage() {
    printf 'usage: bash sync_results_to_dgx.sh [--size lightweight|detailed|full] [--job-id JOB_ID]\n' >&2
}

SYNC_SIZE="lightweight"
JOB_ID=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --size) SYNC_SIZE="$2"; shift 2 ;;
        --job-id) JOB_ID="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) usage; printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

case "$SYNC_SIZE" in
    lightweight|detailed|full) ;;
    *) usage; printf 'sync size must be lightweight, detailed, or full\n' >&2; exit 2 ;;
esac
if [ -n "$JOB_ID" ] && ! [[ "$JOB_ID" =~ ^[0-9]+$ ]]; then
    usage
    printf 'JOB_ID must be numeric\n' >&2
    exit 2
fi

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

SOURCE_ROOT="$nni@selena.hpc.edf.fr:~/codes/$PROJECT_NAME"
DESTINATION_ROOT="$PROJECT_ROOT"

mkdir -p "$DESTINATION_ROOT/outputs_selena" "$DESTINATION_ROOT/logs_selena"

OUTPUT_FILTERS=()
if [ "$SYNC_SIZE" = lightweight ]; then
    OUTPUT_FILTERS=(
        '--exclude=/reports/**/plots/***'
        '--exclude=/reports/**/averaged_inputs/***'
        '--include=/reports/'
        '--include=/reports/***'
        '--exclude=*'
    )
elif [ "$SYNC_SIZE" = detailed ]; then
    OUTPUT_FILTERS=(
        '--include=*/'
        '--include=*.json'
        '--include=*.csv'
        '--include=*.tsv'
        '--include=*.tex'
        '--include=*.md'
        '--include=*.txt'
        '--include=*.log'
        '--include=*.yaml'
        '--include=*.yml'
        '--include=*.png'
        '--include=*.svg'
        '--include=*.pdf'
        '--include=*.html'
        '--include=*.complete'
        '--include=.gitkeep'
        '--exclude=*'
    )
fi

echo "Pulling $PROJECT_NAME Selena outputs to DGX ($SYNC_SIZE)..."
rsync -rlptz --partial --prune-empty-dirs --info=progress2 \
    "${OUTPUT_FILTERS[@]}" \
    "$SOURCE_ROOT/outputs_selena/" \
    "$DESTINATION_ROOT/outputs_selena/"

if [ -n "$JOB_ID" ]; then
    echo "Pulling $PROJECT_NAME Selena logs for job $JOB_ID to DGX..."
    rsync -rlptz --partial --prune-empty-dirs --info=progress2 \
        '--include=*/' "--include=*_${JOB_ID}.out" "--include=*_${JOB_ID}.err" '--exclude=*' \
        "$SOURCE_ROOT/logs_selena/" \
        "$DESTINATION_ROOT/logs_selena/"
else
    echo "Pulling $PROJECT_NAME Selena logs to DGX..."
    rsync -rlptz --partial --info=progress2 \
        "$SOURCE_ROOT/logs_selena/" \
        "$DESTINATION_ROOT/logs_selena/"
fi

echo "SUCCESS: $SYNC_SIZE outputs_selena artifacts and requested logs_selena were pulled from Selena to DGX."
