#!/usr/bin/env bash
set -euo pipefail

: "${MSWEA_COST_TRACKING:=ignore_errors}"
: "${SUBSET:=verified}"
: "${SPLIT:=test}"
: "${MODEL:=nebius/moonshotai/Kimi-K2.6}"
: "${TASK_SLICE:=0:3}"
: "${WORKERS:=1}"
: "${COST_LIMIT:=0}"
: "${OUTPUT_DIR:=trajectories}"

# Locate the swebench benchmark config bundled with mini-swe-agent.

CONFIG_PATH="${CONFIG_PATH:-}"
if [ -z "$CONFIG_PATH" ]; then
    CONFIG_PATH=$(find .venv/lib -name "swebench.yaml" \
        -path "*/minisweagent/config/benchmarks/*" 2>/dev/null | head -1 || true)
fi
if [ -z "$CONFIG_PATH" ]; then
    echo "ERROR: could not find minisweagent swebench.yaml config. Set CONFIG_PATH explicitly." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Build the command as an array so arguments with spaces are handled safely.
CMD=(
    uv run mini-extra swebench
    --subset "$SUBSET"
    --split  "$SPLIT"
    --model  "$MODEL"
    --slice  "$TASK_SLICE"
    --config "$CONFIG_PATH"
    --workers "$WORKERS"
    -o "$OUTPUT_DIR"
)

# Pass --cost-limit only when it is set to a non-zero value.
if [ "$COST_LIMIT" -gt 0 ] 2>/dev/null; then
    CMD+=(--cost-limit "$COST_LIMIT")
fi

MSWEA_COST_TRACKING="$MSWEA_COST_TRACKING" "${CMD[@]}"
