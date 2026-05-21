#!/bin/bash
# Evaluate specific OLMo training checkpoints found under CHECKPOINT_DIR.
#
# Usage:
#   CHECKPOINT_DIR=/path/to/olmo/checkpoints bash scripts/eval_olmo_checkpoints.sh
#
# Set CHECKPOINT_DIR to the directory containing checkpoint-* subdirectories.
# By default, evaluates checkpoint-1000 and checkpoint-1500.

set -euo pipefail

source env.sh

export HARMBENCH_TC="./evals/harmbench/data/test_cases_directrequest_test.json"
HARMBENCH_TC="./evals/harmbench/data/test_cases_directrequest_test.json"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/path/to/your/olmo/checkpoints}"

ALLOWED=("checkpoint-1000" "checkpoint-1500")
CHECKPOINTS=()
for c in "${ALLOWED[@]}"; do
    dir="$CHECKPOINT_DIR/$c"
    if [[ -d "$dir" ]]; then
        CHECKPOINTS+=("$dir")
    else
        echo "WARNING: $dir not found, skipping"
    fi
done

if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
    echo "No allowed checkpoints found in $CHECKPOINT_DIR"
    exit 1
fi

for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    CKPT_NAME="$(basename "$CHECKPOINT")"
    MODEL_ID="olmo-3-7b-recapo-$CKPT_NAME"
    echo "Evaluating checkpoint: $CHECKPOINT"

    bash "./evals/run_all.sh" allenai/OLMo-3-7B-Instruct \
        --model-id "$MODEL_ID" \
        --model-type lora \
        --lora "$CHECKPOINT" \
        --output-dir "./evals/olmo_results/$MODEL_ID" \
        --test-cases-path "$HARMBENCH_TC"
done
