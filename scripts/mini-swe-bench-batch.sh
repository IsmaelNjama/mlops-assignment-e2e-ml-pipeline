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
: "${CONFIG_PATH:=mini-swe-agent/src/minisweagent/config/benchmarks/swebench.yaml}"

mkdir -p "$OUTPUT_DIR"

MSWEA_COST_TRACKING="$MSWEA_COST_TRACKING" \
uv run mini-extra swebench \
    --subset "$SUBSET" \
    --split "$SPLIT" \
    --model "$MODEL" \
    --slice "$TASK_SLICE" \
    --config "$CONFIG_PATH" \
    --workers "$WORKERS" \
    --cost-limit "$COST_LIMIT" \
    -o "$OUTPUT_DIR"
