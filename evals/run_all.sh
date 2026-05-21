#!/usr/bin/env bash
# Run all four evals end-to-end for a single model.
#
# Usage:
#   ./run_all.sh <model_path> [options]
#
# Shared options (forwarded to all four evals):
#   --model-id       <str>   Short identifier (default: basename of model_path)
#   --model-type     <str>   base | instruct | lora  (default: instruct)
#   --lora           <dir>   Path to LoRA adapter directory
#   --output-dir     <dir>   Base directory; per-eval subdirs are created under it
#   --max-new-tokens <int>   Max generation tokens (default varies per eval)
#   --batch-size     <int>   Inference batch size
#
# MTBench-specific:
#   --judge-model      <str>   GPT judge (default: gpt-5.4-mini-2026-03-17)
#   --ref-answer-model <str>   Gold ref key in reference_answer/ (default: gpt-5.4-mini-2026-03-17)
#   --mode             <str>   single | pairwise-baseline | pairwise-all (default: single)
#   --parallel         <int>   Concurrent judge API calls (default: 1)
#   --num-choices      <int>   Samples per question (default: 1)
#   --rejudge                  Re-run judging even if output already exists (MTBench + IFEval)
#
# AlpacaEval-specific:
#   --annotator      <str>   Alpaca annotator (default: weighted_alpaca_eval_gpt4_turbo)
#   --limit          <int>   Only evaluate first N examples
#
# Novelty-bench-specific:
#   --data           <str>   curated | wildchat (default: curated)
#   --num-generations <int>  Responses per prompt (default: 10)
#   --patience       <float> Discount factor (default: 0.8)
#   --alg            <str>   classifier | unigram | bertscore (default: classifier)
#
# HarmBench-specific:
#   --test-cases-path <file>  Path to test_cases.json (required for HarmBench)
#   --behaviors-split <str>   test | val | all (default: test)
#   --classifier      <str>   HarmBench classifier (default: cais/HarmBench-Llama-2-13b-cls)
#   --num-tokens      <int>   Max tokens for classifier (default: 512)
#   --include-advbench        Include AdvBench refusal metric
#
# Arena-Hard-specific:
#   --ah-parallel     <int>   Concurrent judge API calls (default: 8)
#   --ah-max-tokens   <int>   Max tokens for model generation via vLLM (default: 1024)
#   --ah-max-model-len <int>  vLLM max model length (default: 4096)
#   --ah-chunk-size   <int>   Prompts per vLLM chunk (default: 200)
#   --ah-gpu-util     <float> vLLM gpu_memory_utilization (default: 0.92)
#   Note: --rejudge is forwarded to Arena-Hard as well.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${1:?Usage: $0 <model_path> [options]}"
shift

# --- Shared defaults ---
MODEL_ID="$(basename "$MODEL_PATH")"
MODEL_TYPE="instruct"
LORA=""
OUTPUT_DIR=""
MAX_NEW_TOKENS=""
BATCH_SIZE=""

# --- MTBench defaults ---
JUDGE_MODEL="gpt-5.4-mini-2026-03-17"
REF_ANSWER_MODEL="gpt-4"
MT_MODE="single"
MT_PARALLEL=1
NUM_CHOICES=1
REJUDGE=""

# --- AlpacaEval defaults ---
ANNOTATOR="weighted_alpaca_eval_gpt5_4_mini"
LIMIT=""

# --- Novelty-bench defaults ---
NB_DATA="curated"
NUM_GENERATIONS=10
PATIENCE=0.8
ALG="classifier"

# --- HarmBench defaults ---
TEST_CASES_PATH=""
BEHAVIORS_SPLIT="test"
HB_CLASSIFIER="cais/HarmBench-Llama-2-13b-cls"
NUM_TOKENS=512
ADVBENCH_FLAG=""

# --- Arena-Hard defaults ---
AH_PARALLEL=8
AH_MAX_TOKENS=1024
AH_MAX_MODEL_LEN=4096
AH_CHUNK_SIZE=200
AH_GPU_UTIL=0.92

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-id)        MODEL_ID="$2";          shift 2 ;;
        --model-type)      MODEL_TYPE="$2";        shift 2 ;;
        --lora)            LORA="$2";              shift 2 ;;
        --output-dir)      OUTPUT_DIR="$2";        shift 2 ;;
        --max-new-tokens)  MAX_NEW_TOKENS="$2";    shift 2 ;;
        --batch-size)      BATCH_SIZE="$2";        shift 2 ;;
        --judge-model)        JUDGE_MODEL="$2";        shift 2 ;;
        --ref-answer-model)   REF_ANSWER_MODEL="$2";  shift 2 ;;
        --rejudge)            REJUDGE="--rejudge";     shift ;;
        --mode)               MT_MODE="$2";            shift 2 ;;
        --parallel)           MT_PARALLEL="$2";        shift 2 ;;
        --num-choices)        NUM_CHOICES="$2";        shift 2 ;;
        --annotator)       ANNOTATOR="$2";         shift 2 ;;
        --limit)           LIMIT="$2";             shift 2 ;;
        --data)            NB_DATA="$2";           shift 2 ;;
        --num-generations) NUM_GENERATIONS="$2";   shift 2 ;;
        --patience)        PATIENCE="$2";          shift 2 ;;
        --alg)             ALG="$2";               shift 2 ;;
        --test-cases-path) TEST_CASES_PATH="$2";   shift 2 ;;
        --behaviors-split) BEHAVIORS_SPLIT="$2";   shift 2 ;;
        --classifier)      HB_CLASSIFIER="$2";     shift 2 ;;
        --num-tokens)      NUM_TOKENS="$2";        shift 2 ;;
        --include-advbench) ADVBENCH_FLAG="--include-advbench"; shift ;;
        --ah-parallel)      AH_PARALLEL="$2";      shift 2 ;;
        --ah-max-tokens)    AH_MAX_TOKENS="$2";    shift 2 ;;
        --ah-max-model-len) AH_MAX_MODEL_LEN="$2"; shift 2 ;;
        --ah-chunk-size)    AH_CHUNK_SIZE="$2";    shift 2 ;;
        --ah-gpu-util)      AH_GPU_UTIL="$2";      shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$SCRIPT_DIR/results/$MODEL_ID"
fi

# Optional shared flags (only pass when user specified them)
LORA_FLAG="";   [[ -n "$LORA" ]] && LORA_FLAG="--lora $LORA"
MNT_FLAG="";    [[ -n "$MAX_NEW_TOKENS" ]] && MNT_FLAG="--max-new-tokens $MAX_NEW_TOKENS"
BS_FLAG="";     [[ -n "$BATCH_SIZE" ]] && BS_FLAG="--batch-size $BATCH_SIZE"
LIMIT_FLAG="";  [[ -n "$LIMIT" ]] && LIMIT_FLAG="--limit $LIMIT"
TCP_FLAG="";    [[ -n "$TEST_CASES_PATH" ]] && TCP_FLAG="--test-cases-path $TEST_CASES_PATH"

echo "======================================================"
echo " Running all evals for: $MODEL_PATH"
echo " Model ID:   $MODEL_ID"
echo " Model type: $MODEL_TYPE"
echo " Output dir: $OUTPUT_DIR"
echo "======================================================"

echo ""
echo "########## MTBench ##########"
"$SCRIPT_DIR/mtbench/run_mtbench.sh" "$MODEL_PATH" \
    --model-id "$MODEL_ID" \
    --model-type "$MODEL_TYPE" \
    $LORA_FLAG \
    --output-dir "$OUTPUT_DIR/mtbench" \
    --judge-model "$JUDGE_MODEL" \
    --ref-answer-model "$REF_ANSWER_MODEL" \
    --mode "$MT_MODE" \
    --parallel "$MT_PARALLEL" \
    --num-choices "$NUM_CHOICES" \
    $MNT_FLAG $REJUDGE || echo "WARNING: MTBench failed"

# echo ""
# echo "########## AlpacaEval ##########"
# "$SCRIPT_DIR/alpaca_eval_suite/run_alpaca.sh" "$MODEL_PATH" \
#     --model-id "$MODEL_ID" \
#     --model-type "$MODEL_TYPE" \
#     $LORA_FLAG \
#     --output-dir "$OUTPUT_DIR/alpaca_eval" \
#     --annotator "$ANNOTATOR" \
#     $MNT_FLAG $BS_FLAG $LIMIT_FLAG || echo "WARNING: AlpacaEval failed"

echo ""
echo "########## IFEval ##########"
"$SCRIPT_DIR/ifeval/run_ifeval.sh" "$MODEL_PATH" \
    --model-id "$MODEL_ID" \
    --model-type "$MODEL_TYPE" \
    $LORA_FLAG \
    --output-dir "$OUTPUT_DIR/ifeval" \
    $MNT_FLAG $BS_FLAG $LIMIT_FLAG $REJUDGE || echo "WARNING: IFEval failed"

echo ""
echo "########## Arena-Hard ##########"
"$SCRIPT_DIR/arena_hard/run_arena_hard.sh" "$MODEL_PATH" \
    --model-id "$MODEL_ID" \
    --model-type "$MODEL_TYPE" \
    $LORA_FLAG \
    --output-dir "$OUTPUT_DIR/arena_hard" \
    --judge-model "$JUDGE_MODEL" \
    --max-tokens "$AH_MAX_TOKENS" \
    --max-model-len "$AH_MAX_MODEL_LEN" \
    --chunk-size "$AH_CHUNK_SIZE" \
    --gpu-util "$AH_GPU_UTIL" \
    --parallel "$AH_PARALLEL" \
    $REJUDGE || echo "WARNING: Arena-Hard failed"

echo ""
echo "########## Novelty-bench ##########"
"$SCRIPT_DIR/novelty_bench/run_novelty.sh" "$MODEL_PATH" \
    --model-type "$MODEL_TYPE" \
    $LORA_FLAG \
    --eval-dir "$OUTPUT_DIR/novelty_bench" \
    --data "$NB_DATA" \
    --num-generations "$NUM_GENERATIONS" \
    --patience "$PATIENCE" \
    --alg "$ALG" || echo "WARNING: Novelty-bench failed"

echo ""
echo "########## HarmBench ##########"
if [[ -z "$TEST_CASES_PATH" ]]; then
    echo "WARNING: --test-cases-path not provided, skipping HarmBench"
else
    "$SCRIPT_DIR/harmbench/run_harmbench.sh" "$MODEL_PATH" \
        --model-type "$MODEL_TYPE" \
        $LORA_FLAG \
        --test-cases-path "$TEST_CASES_PATH" \
        --output-dir "$OUTPUT_DIR/harmbench" \
        --behaviors-split "$BEHAVIORS_SPLIT" \
        --classifier "$HB_CLASSIFIER" \
        --num-tokens "$NUM_TOKENS" \
        $ADVBENCH_FLAG || echo "WARNING: HarmBench failed"
fi

echo ""
echo "======================================================"
echo " All evals complete. Results in $OUTPUT_DIR"
echo "======================================================"
