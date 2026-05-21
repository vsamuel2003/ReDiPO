#!/usr/bin/env bash
# Run IFEval end-to-end for a single model via lm-evaluation-harness.
#
# Usage:
#   ./run_ifeval.sh <model_path> [options]
#
# Options:
#   --model-id       <str>  Short identifier (default: basename of model_path)
#   --model-type     <str>  base | instruct | lora  (default: instruct)
#   --lora           <dir>  Path to LoRA adapter directory (required for lora)
#   --output-dir     <dir>  Where to write results (default: ./results/<model_id>)
#   --max-new-tokens <int>  Max generation tokens (default: 1280)
#   --batch-size     <int>  Inference batch size (default: 8)
#   --limit          <int>  Only evaluate first N examples (for debugging)
#   --rejudge              Delete existing results and re-run from scratch

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${1:?Usage: $0 <model_path> [options]}"
shift

MODEL_ID="$(basename "$MODEL_PATH")"
MODEL_TYPE="instruct"
LORA=""
OUTPUT_DIR=""
MAX_NEW_TOKENS=1280
BATCH_SIZE=8
LIMIT=""
REJUDGE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-id)        MODEL_ID="$2";        shift 2 ;;
        --model-type)      MODEL_TYPE="$2";      shift 2 ;;
        --lora)            LORA="$2";            shift 2 ;;
        --output-dir)      OUTPUT_DIR="$2";      shift 2 ;;
        --max-new-tokens)  MAX_NEW_TOKENS="$2";  shift 2 ;;
        --batch-size)      BATCH_SIZE="$2";      shift 2 ;;
        --limit)           LIMIT="$2";           shift 2 ;;
        --rejudge)         REJUDGE="1";          shift ;;
        --*)               shift 2 2>/dev/null || shift ;;  # silently skip unknown flags
        *)                 shift ;;
    esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$SCRIPT_DIR/results/$MODEL_ID"
fi

mkdir -p "$OUTPUT_DIR"

# Qwen detection — mirrors evals/common/prompt_builder.py:26-27
LOWER="$(echo "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')"
IS_QWEN=0; [[ "$LOWER" == *qwen* ]] && IS_QWEN=1

# Build --model_args and --apply_chat_template flag based on model type
CHAT_FLAG=""
case "$MODEL_TYPE" in
    base)
        MODEL_ARGS="pretrained=$MODEL_PATH,dtype=bfloat16"
        ;;
    instruct)
        MODEL_ARGS="pretrained=$MODEL_PATH,dtype=bfloat16"
        [[ "$IS_QWEN" -eq 1 ]] && MODEL_ARGS="$MODEL_ARGS,enable_thinking=false"
        CHAT_FLAG="--apply_chat_template"
        ;;
    lora)
        if [[ -z "$LORA" ]]; then
            echo "Error: --lora <path> is required for --model-type lora" >&2
            exit 1
        fi
        MODEL_ARGS="pretrained=$MODEL_PATH,peft=$LORA,dtype=bfloat16"
        [[ "$IS_QWEN" -eq 1 ]] && MODEL_ARGS="$MODEL_ARGS,enable_thinking=false"
        CHAT_FLAG="--apply_chat_template"
        ;;
    *)
        echo "Error: --model-type must be base, instruct, or lora (got: $MODEL_TYPE)" >&2
        exit 1
        ;;
esac

LIMIT_FLAG=""
[[ -n "$LIMIT" ]] && LIMIT_FLAG="--limit $LIMIT"

if [[ -z "$REJUDGE" ]] && [[ -e "$OUTPUT_DIR/results.json" ]]; then
    echo "IFEval results already present at $OUTPUT_DIR/results.json — skipping. Pass --rejudge to re-run."
    exit 0
fi

echo "=== IFEval: Starting evaluation ==="
echo "  Model path : $MODEL_PATH"
echo "  Model type : $MODEL_TYPE"
echo "  Model ID   : $MODEL_ID"
echo "  Output dir : $OUTPUT_DIR"
[[ -n "$LORA" ]] && echo "  LoRA path  : $LORA"
[[ "$IS_QWEN" -eq 1 ]] && echo "  Qwen model : enable_thinking=false"

python -m lm_eval \
    --model hf \
    --model_args "$MODEL_ARGS" \
    --tasks ifeval \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT_DIR" \
    --log_samples \
    --gen_kwargs "max_gen_toks=$MAX_NEW_TOKENS" \
    $CHAT_FLAG \
    $LIMIT_FLAG

# lm_eval writes timestamped results under a sanitized-model-name subdir.
# Symlink the latest to a stable path for consolidate_results.py.
LATEST="$(ls -t "$OUTPUT_DIR"/*/results_*.json 2>/dev/null | head -1 || true)"
if [[ -n "$LATEST" ]]; then
    ln -sf "$(realpath --relative-to="$OUTPUT_DIR" "$LATEST")" "$OUTPUT_DIR/results.json"
    echo "IFEval complete. Results in $OUTPUT_DIR/results.json"
else
    echo "Warning: could not find results_*.json under $OUTPUT_DIR" >&2
fi
