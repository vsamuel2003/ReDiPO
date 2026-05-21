#!/usr/bin/env bash
# Run MTBench end-to-end for a single model.
#
# Usage:
#   ./run_mtbench.sh <model_path> [options]
#
# Options:
#   --model-id       <str>  Short identifier for output files (default: basename of model_path)
#   --model-type     <str>  base | instruct | lora  (default: instruct)
#   --lora           <dir>  Path to LoRA adapter directory (required for --model-type lora)
#   --output-dir     <dir>  Where to write answers and judgments (default: ./results/<model_id>)
#   --judge-model    <str>  GPT judge model name (default: gpt-5.4-mini-2026-03-17)
#   --mode           <str>  single | pairwise-baseline | pairwise-all (default: single)
#   --parallel       <int>  Concurrent judge API calls (default: 1)
#   --max-new-tokens <int>  Max tokens per generation (default: 1024)
#   --num-choices    <int>  Samples per question (default: 1)
#   --question-begin   <int>  First question index (optional)
#   --question-end     <int>  Last question index (optional)
#   --ref-answer-model <str>  Key in reference_answer/ for math/coding gold refs (default: gpt-5.4-mini-2026-03-17)
#   --rejudge               Delete existing judgment file and re-run from scratch

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${1:?Usage: $0 <model_path> [options]}"
shift

# Defaults
MODEL_ID="$(basename "$MODEL_PATH")"
MODEL_TYPE="instruct"
LORA=""
OUTPUT_DIR=""
JUDGE_MODEL="gpt-5.4-mini-2026-03-17"
REF_ANSWER_MODEL="gpt-4"
REJUDGE=""
MODE="single"
PARALLEL=1
MAX_NEW_TOKENS=1024
NUM_CHOICES=1
QUESTION_BEGIN=""
QUESTION_END=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-id)        MODEL_ID="$2";        shift 2 ;;
        --model-type)      MODEL_TYPE="$2";      shift 2 ;;
        --lora)            LORA="$2";            shift 2 ;;
        --output-dir)      OUTPUT_DIR="$2";      shift 2 ;;
        --judge-model)        JUDGE_MODEL="$2";       shift 2 ;;
        --ref-answer-model)   REF_ANSWER_MODEL="$2";  shift 2 ;;
        --rejudge)            REJUDGE="--rejudge";     shift ;;
        --mode)               MODE="$2";               shift 2 ;;
        --parallel)        PARALLEL="$2";        shift 2 ;;
        --max-new-tokens)  MAX_NEW_TOKENS="$2";  shift 2 ;;
        --num-choices)     NUM_CHOICES="$2";     shift 2 ;;
        --question-begin)  QUESTION_BEGIN="$2";  shift 2 ;;
        --question-end)    QUESTION_END="$2";    shift 2 ;;
        --*)               shift 2 2>/dev/null || shift ;;  # silently skip unknown flags
        *)                 shift ;;
    esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$SCRIPT_DIR/results/$MODEL_ID"
fi

ANSWER_DIR="$OUTPUT_DIR/model_answer"
JUDGMENT_DIR="$OUTPUT_DIR/model_judgment"

mkdir -p "$ANSWER_DIR" "$JUDGMENT_DIR"

LORA_FLAG=""
[[ -n "$LORA" ]] && LORA_FLAG="--lora $LORA"

QB_FLAG=""
[[ -n "$QUESTION_BEGIN" ]] && QB_FLAG="--question-begin $QUESTION_BEGIN"

QE_FLAG=""
[[ -n "$QUESTION_END" ]] && QE_FLAG="--question-end $QUESTION_END"

# Determine output file suffix for show_result
if [[ "$MODE" == "single" ]]; then
    RESULT_FILE="$JUDGMENT_DIR/${JUDGE_MODEL}_single.jsonl"
else
    RESULT_FILE="$JUDGMENT_DIR/${JUDGE_MODEL}_pair.jsonl"
fi

echo "=== MTBench Step 1: Generating model answers ==="
python "$SCRIPT_DIR/gen_model_answer.py" \
    --model-path "$MODEL_PATH" \
    --model-id "$MODEL_ID" \
    --model-type "$MODEL_TYPE" \
    $LORA_FLAG \
    --output-dir "$ANSWER_DIR" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --num-choices "$NUM_CHOICES" \
    $QB_FLAG $QE_FLAG

echo "=== MTBench Step 2: Running judge ==="
python "$SCRIPT_DIR/gen_judgment.py" \
    --judge-model "$JUDGE_MODEL" \
    --ref-answer-model "$REF_ANSWER_MODEL" \
    --mode "$MODE" \
    --model-list "$MODEL_ID" \
    --parallel "$PARALLEL" \
    --answer-dir "$ANSWER_DIR" \
    --output-dir "$JUDGMENT_DIR" \
    $REJUDGE

echo "=== MTBench Step 3: Showing results ==="
python "$SCRIPT_DIR/show_result.py" \
    --judge-model "$JUDGE_MODEL" \
    --mode "$MODE" \
    --input-file "$RESULT_FILE"

echo "MTBench complete. Results in $OUTPUT_DIR"
