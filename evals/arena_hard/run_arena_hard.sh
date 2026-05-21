#!/usr/bin/env bash
# Run Arena-Hard end-to-end for a single model.
#
# Usage:
#   ./run_arena_hard.sh <model_path> [options]
#
# Options:
#   --model-id           <str>    Short identifier for output files (default: basename of model_path)
#   --model-type         <str>    base | instruct | lora  (default: instruct)
#   --lora               <dir>    Path to LoRA adapter directory (required for --model-type lora)
#   --output-dir         <dir>    Where to write answers, judgments, summary (default: ./results/<model_id>/arena_hard)
#   --judge-model        <str>    GPT judge model (default: gpt-5.4-mini-2026-03-17)
#   --max-tokens         <int>    Max tokens for model generation (default: 1024)
#   --max-model-len      <int>    vLLM max model length (default: 4096)
#   --chunk-size         <int>    Prompts per vLLM batch chunk (default: 200)
#   --gpu-util           <float>  vLLM gpu_memory_utilization (default: 0.92)
#   --parallel           <int>    Concurrent judge API calls (default: 8)
#   --rejudge                     Re-run judging even if output already exists

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
MAX_TOKENS=1024
MAX_MODEL_LEN=4096
CHUNK_SIZE=200
GPU_UTIL=0.92
PARALLEL=8
REJUDGE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-id)       MODEL_ID="$2";      shift 2 ;;
        --model-type)     MODEL_TYPE="$2";    shift 2 ;;
        --lora)           LORA="$2";          shift 2 ;;
        --output-dir)     OUTPUT_DIR="$2";    shift 2 ;;
        --judge-model)    JUDGE_MODEL="$2";   shift 2 ;;
        --max-tokens)     MAX_TOKENS="$2";    shift 2 ;;
        --max-model-len)  MAX_MODEL_LEN="$2"; shift 2 ;;
        --chunk-size)     CHUNK_SIZE="$2";    shift 2 ;;
        --gpu-util)       GPU_UTIL="$2";      shift 2 ;;
        --parallel)       PARALLEL="$2";      shift 2 ;;
        --rejudge)        REJUDGE="--rejudge"; shift ;;
        --*)              shift 2 2>/dev/null || shift ;;
        *)                shift ;;
    esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$SCRIPT_DIR/results/$MODEL_ID/arena_hard"
fi

QUESTION_FILE="$SCRIPT_DIR/data/question.jsonl"
BASELINE_FILE="$SCRIPT_DIR/data/baseline_answer.jsonl"
ANSWER_FILE="$OUTPUT_DIR/model_answer/${MODEL_ID}.jsonl"
JUDGMENT_FILE="$OUTPUT_DIR/model_judgment/${JUDGE_MODEL}.jsonl"
SUMMARY_FILE="$OUTPUT_DIR/summary.json"

mkdir -p "$OUTPUT_DIR/model_answer" "$OUTPUT_DIR/model_judgment"

LORA_FLAG=""
[[ -n "$LORA" ]] && LORA_FLAG="--lora $LORA"

# ── Step 0: Guard — baseline must exist ──────────────────────────────────────
N_BASELINE=0
[[ -f "$BASELINE_FILE" ]] && N_BASELINE=$(wc -l < "$BASELINE_FILE" | tr -d ' ')
if [[ "$N_BASELINE" -lt 500 ]]; then
    echo "ERROR: Baseline file not found or incomplete ($N_BASELINE lines, expected ≥500)."
    echo "  File: $BASELINE_FILE"
    echo "  Run the one-time bootstrap first:"
    echo "    python $SCRIPT_DIR/generate_baseline.py"
    exit 1
fi
echo "Baseline OK: $N_BASELINE lines in $BASELINE_FILE"

# ── Step 1: Generate model answers (vLLM) ────────────────────────────────────
echo ""
echo "=== Arena-Hard Step 1: Generating model answers ==="
python "$SCRIPT_DIR/gen_answers.py" \
    --model-path    "$MODEL_PATH" \
    --model-id      "$MODEL_ID" \
    --model-type    "$MODEL_TYPE" \
    $LORA_FLAG \
    --question-file "$QUESTION_FILE" \
    --output-file   "$ANSWER_FILE" \
    --max-tokens    "$MAX_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --chunk-size    "$CHUNK_SIZE" \
    --gpu-memory-utilization "$GPU_UTIL"

# ── Step 2: Run pairwise judge ───────────────────────────────────────────────
echo ""
echo "=== Arena-Hard Step 2: Running pairwise judge ==="
python "$SCRIPT_DIR/run_judge.py" \
    --question-file "$QUESTION_FILE" \
    --baseline-file "$BASELINE_FILE" \
    --answer-file   "$ANSWER_FILE" \
    --output-file   "$JUDGMENT_FILE" \
    --judge-model   "$JUDGE_MODEL" \
    --max-tokens    16000 \
    --parallel      "$PARALLEL" \
    $REJUDGE

# ── Step 3: Compute and print results ────────────────────────────────────────
echo ""
echo "=== Arena-Hard Step 3: Computing results ==="
python "$SCRIPT_DIR/show_result.py" \
    --judgment-file "$JUDGMENT_FILE" \
    --output-file   "$SUMMARY_FILE"

echo ""
echo "Arena-Hard complete. Results in $OUTPUT_DIR"
