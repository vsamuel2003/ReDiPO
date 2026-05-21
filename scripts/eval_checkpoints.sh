#!/bin/bash
# Evaluate all training checkpoints found under CHECKPOINT_DIR.
#
# Usage:
#   CHECKPOINT_DIR=/path/to/checkpoints bash scripts/eval_checkpoints.sh
#
# Set CHECKPOINT_DIR to the directory containing checkpoint-* subdirectories.

set -euo pipefail

source env.sh

export HARMBENCH_TC="./evals/harmbench/data/test_cases_directrequest_test.json"
HARMBENCH_TC="./evals/harmbench/data/test_cases_directrequest_test.json"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/path/to/your/checkpoints}"

mapfile -t CHECKPOINTS < <(ls -d "$CHECKPOINT_DIR"/checkpoint-* 2>/dev/null | sort -V)

if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
    echo "No checkpoints found in $CHECKPOINT_DIR"
    exit 1
fi

for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    CKPT_NAME="$(basename "$CHECKPOINT")"
    MODEL_ID="model-recapo-$CKPT_NAME"
    echo "Evaluating checkpoint: $CHECKPOINT"

    bash "./evals/run_all.sh" Qwen/Qwen3-4B-Instruct-2507 \
        --model-id "$MODEL_ID" \
        --model-type lora \
        --lora "$CHECKPOINT" \
        --output-dir "./evals/results/$MODEL_ID" \
        --test-cases-path "$HARMBENCH_TC"
done
