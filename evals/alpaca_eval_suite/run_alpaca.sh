#!/usr/bin/env bash
# Run AlpacaEval end-to-end for a single model.
#
# Usage:
#   ./run_alpaca.sh <model_path> [options]
#
# Options:
#   --model-id       <str>  Short identifier (default: basename of model_path)
#   --model-type     <str>  base | instruct | lora  (default: instruct)
#   --lora           <dir>  Path to LoRA adapter directory
#   --output-dir     <dir>  Where to write outputs and leaderboard
#   --annotator      <str>  Alpaca annotator config (default: weighted_alpaca_eval_gpt4_turbo)
#   --max-new-tokens <int>  Max tokens per generation (default: 2048)
#   --batch-size     <int>  Inference batch size (default: 4)
#   --limit          <int>  Only evaluate first N examples (for debugging)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${1:?Usage: $0 <model_path> [options]}"
shift

MODEL_ID="$(basename "$MODEL_PATH")"
MODEL_TYPE="instruct"
LORA=""
OUTPUT_DIR=""
ANNOTATOR="weighted_alpaca_eval_gpt5_4_mini"
MAX_NEW_TOKENS=2048
BATCH_SIZE=4
LIMIT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-id)       MODEL_ID="$2";       shift 2 ;;
        --model-type)     MODEL_TYPE="$2";     shift 2 ;;
        --lora)           LORA="$2";           shift 2 ;;
        --output-dir)     OUTPUT_DIR="$2";     shift 2 ;;
        --annotator)      ANNOTATOR="$2";      shift 2 ;;
        --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --batch-size)     BATCH_SIZE="$2";     shift 2 ;;
        --limit)          LIMIT="$2";          shift 2 ;;
        --*)              shift 2 2>/dev/null || shift ;;  # silently skip unknown flags
        *)                shift ;;
    esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$SCRIPT_DIR/results/$MODEL_ID"
fi

LORA_FLAG=""
[[ -n "$LORA" ]] && LORA_FLAG="--lora $LORA"

LIMIT_FLAG=""
[[ -n "$LIMIT" ]] && LIMIT_FLAG="--limit $LIMIT"

echo "=== AlpacaEval Step 1: Generating model outputs ==="
python "$SCRIPT_DIR/generate_outputs.py" \
    --model-path "$MODEL_PATH" \
    --model-id "$MODEL_ID" \
    --model-type "$MODEL_TYPE" \
    $LORA_FLAG \
    --output-dir "$OUTPUT_DIR" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --batch-size "$BATCH_SIZE" \
    $LIMIT_FLAG

echo "=== AlpacaEval Step 2: Running annotation ==="
python "$SCRIPT_DIR/run_evaluation.py" \
    --outputs-path "$OUTPUT_DIR/outputs.json" \
    --annotator "$ANNOTATOR" \
    --output-dir "$OUTPUT_DIR"

echo "AlpacaEval complete. Results in $OUTPUT_DIR"
