#!/usr/bin/env bash
# Run HarmBench end-to-end for a single model.
#
# Usage:
#   ./run_harmbench.sh <model_path> [options]
#
# Options:
#   --model-type       <str>  base | instruct | lora  (default: instruct)
#   --lora             <dir>  Path to LoRA adapter directory
#   --test-cases-path  <file> Path to test_cases.json (required)
#   --output-dir       <dir>  Where to write completions and results
#   --behaviors-split  <str>  test | val | all (default: test)
#   --classifier       <str>  HarmBench classifier model (default: cais/HarmBench-Llama-2-13b-cls)
#   --max-new-tokens   <int>  Max tokens per completion (default: 256)
#   --batch-size       <int>  Inference batch size (default: 8)
#   --num-tokens       <int>  Max tokens for classifier truncation (default: 512)
#   --include-advbench       Include AdvBench refusal metric

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${1:?Usage: $0 <model_path> --test-cases-path <path> [options]}"
shift

MODEL_TYPE="instruct"
LORA=""
TEST_CASES_PATH=""
OUTPUT_DIR=""
BEHAVIORS_SPLIT="test"
CLASSIFIER="cais/HarmBench-Llama-2-13b-cls"
MAX_NEW_TOKENS=256
BATCH_SIZE=8
NUM_TOKENS=512
ADVBENCH_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-type)       MODEL_TYPE="$2";       shift 2 ;;
        --lora)             LORA="$2";             shift 2 ;;
        --test-cases-path)  TEST_CASES_PATH="$2";  shift 2 ;;
        --output-dir)       OUTPUT_DIR="$2";       shift 2 ;;
        --behaviors-split)  BEHAVIORS_SPLIT="$2";  shift 2 ;;
        --classifier)       CLASSIFIER="$2";       shift 2 ;;
        --max-new-tokens)   MAX_NEW_TOKENS="$2";   shift 2 ;;
        --batch-size)       BATCH_SIZE="$2";       shift 2 ;;
        --num-tokens)       NUM_TOKENS="$2";       shift 2 ;;
        --include-advbench) ADVBENCH_FLAG="--include-advbench-metric"; shift ;;
        --*)                shift 2 2>/dev/null || shift ;;  # silently skip unknown flags
        *)                  shift ;;
    esac
done

if [[ -z "$TEST_CASES_PATH" ]]; then
    echo "Error: --test-cases-path is required" >&2
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$SCRIPT_DIR/results/$(basename "$MODEL_PATH")"
fi

LORA_FLAG=""
[[ -n "$LORA" ]] && LORA_FLAG="--lora $LORA"

BEHAVIORS_PATH="$SCRIPT_DIR/data/behavior_datasets/harmbench_behaviors_text_${BEHAVIORS_SPLIT}.csv"
COMPLETIONS_PATH="$OUTPUT_DIR/completions.json"
RESULTS_PATH="$OUTPUT_DIR/results.json"

mkdir -p "$OUTPUT_DIR"

echo "=== HarmBench Step 1: Generating completions ==="
python "$SCRIPT_DIR/generate_completions.py" \
    --model-path "$MODEL_PATH" \
    --model-type "$MODEL_TYPE" \
    $LORA_FLAG \
    --test-cases-path "$TEST_CASES_PATH" \
    --save-path "$COMPLETIONS_PATH" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --batch-size "$BATCH_SIZE"

echo "=== HarmBench Step 2: Evaluating completions ==="
python "$SCRIPT_DIR/evaluate_completions.py" \
    --cls-path "$CLASSIFIER" \
    --completions-path "$COMPLETIONS_PATH" \
    --save-path "$RESULTS_PATH" \
    --behaviors-path "$BEHAVIORS_PATH" \
    --num-tokens "$NUM_TOKENS" \
    $ADVBENCH_FLAG

echo "HarmBench complete. Results in $RESULTS_PATH"
