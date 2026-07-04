#!/usr/bin/env bash
set -euo pipefail

: "${DATASET_NAME:=princeton-nlp/SWE-bench_Verified}"
: "${PREDICTIONS_PATH:=trajectories/preds.json}"
: "${MAX_WORKERS:=1}"
: "${RUN_ID:=test}"
: "${OUTPUT_DIR:=run-eval}"

mkdir -p "$OUTPUT_DIR"
(
    cd "$OUTPUT_DIR"
    python -m swebench.harness.run_evaluation \
        --dataset_name "$DATASET_NAME" \
        --predictions_path "$PREDICTIONS_PATH" \
        --max_workers "$MAX_WORKERS" \
        --run_id "$RUN_ID"
) 2>&1 | tee "$OUTPUT_DIR/eval.log"
