#!/usr/bin/env bash
set -euo pipefail

: "${DATASET_NAME:=princeton-nlp/SWE-bench_Verified}"
: "${PREDICTIONS_PATH:=trajectories/preds.json}"
: "${MAX_WORKERS:=1}"
: "${RUN_ID:=test}"
: "${OUTPUT_DIR:=run-eval}"
# Allow caller to override which Python interpreter to use.
# Defaults to the project venv if it exists, then falls back to `python`.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEFAULT_PYTHON="$PROJECT_ROOT/.venv/bin/python"
: "${PYTHON_BIN:=$( [ -x "$DEFAULT_PYTHON" ] && echo "$DEFAULT_PYTHON" || echo python )}"

mkdir -p "$OUTPUT_DIR"
(
    cd "$OUTPUT_DIR"
    "$PYTHON_BIN" -m swebench.harness.run_evaluation \
        --dataset_name "$DATASET_NAME" \
        --predictions_path "$PREDICTIONS_PATH" \
        --max_workers "$MAX_WORKERS" \
        --run_id "$RUN_ID"
) 2>&1 | tee "$OUTPUT_DIR/eval.log"
