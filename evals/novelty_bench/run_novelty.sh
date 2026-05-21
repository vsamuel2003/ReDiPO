#!/usr/bin/env bash
# Run novelty-bench end-to-end for a single model.
#
# Usage:
#   ./run_novelty.sh <model_path> [options]
#
# Options:
#   --model-type     <str>  base | instruct | lora  (default: instruct)
#   --lora           <dir>  Path to LoRA adapter directory
#   --eval-dir       <dir>  Where to write results (default: ./results/<model_basename>)
#   --data           <str>  curated | wildchat  (default: curated)
#   --num-generations <int> Number of responses per prompt (default: 10)
#   --max-tokens     <int>  Max new tokens per generation (default: 512)
#   --patience       <float> Discount factor for scoring (default: 0.8)
#   --alg            <str>  Partition algorithm: classifier | unigram | bertscore (default: classifier)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${1:?Usage: $0 <model_path> [options]}"
shift

MODEL_TYPE="instruct"
LORA=""
EVAL_DIR=""
DATA="curated"
NUM_GENERATIONS=10
MAX_TOKENS=512
PATIENCE=0.8
ALG="classifier"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-type)      MODEL_TYPE="$2";      shift 2 ;;
        --lora)            LORA="$2";            shift 2 ;;
        --eval-dir)        EVAL_DIR="$2";        shift 2 ;;
        --data)            DATA="$2";            shift 2 ;;
        --num-generations) NUM_GENERATIONS="$2"; shift 2 ;;
        --max-tokens)      MAX_TOKENS="$2";      shift 2 ;;
        --patience)        PATIENCE="$2";        shift 2 ;;
        --alg)             ALG="$2";             shift 2 ;;
        --*)               shift 2 2>/dev/null || shift ;;  # silently skip unknown flags
        *)                 shift ;;
    esac
done

if [[ -z "$EVAL_DIR" ]]; then
    EVAL_DIR="$SCRIPT_DIR/results/$(basename "$MODEL_PATH")"
fi

LORA_FLAG=""
[[ -n "$LORA" ]] && LORA_FLAG="--lora $LORA"

echo "=== Novelty-bench Step 1: Generating responses ==="
python "$SCRIPT_DIR/inference.py" \
    --model-path "$MODEL_PATH" \
    --model-type "$MODEL_TYPE" \
    $LORA_FLAG \
    --eval-dir "$EVAL_DIR" \
    --data "$DATA" \
    --num-generations "$NUM_GENERATIONS" \
    --max-tokens "$MAX_TOKENS"

echo "=== Novelty-bench Step 2: Partitioning responses ==="
python "$SCRIPT_DIR/partition.py" \
    --eval-dir "$EVAL_DIR" \
    --alg "$ALG"

echo "=== Novelty-bench Step 3: Scoring partitions ==="
python "$SCRIPT_DIR/score.py" \
    --eval-dir "$EVAL_DIR" \
    --patience "$PATIENCE"

echo "=== Novelty-bench Step 4: Summarizing results ==="
python "$SCRIPT_DIR/summarize.py" \
    --eval-dir "$EVAL_DIR"

echo "Novelty-bench complete. Summary:"
cat "$EVAL_DIR/summary.json"
echo ""
echo "Results in $EVAL_DIR"
