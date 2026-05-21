#!/usr/bin/env bash
# Run gen_judgment.py for gpt-5.4-mini and gpt-5.4-nano on both evaluated models.
# Outputs land in each model's model_judgment/ directory alongside the existing gpt-4_single.jsonl.
# Usage: bash run_rejudge.sh [--parallel N]
#
# Requires OPENAI_API_KEY in environment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GEN_JUDGMENT="$REPO_ROOT/evals/mtbench/gen_judgment.py"
RESULTS_DIR="$REPO_ROOT/evals/results"

PARALLEL=8
while [[ $# -gt 0 ]]; do
    case "$1" in
        --parallel) PARALLEL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

JUDGES=(
    "gpt-5.4-mini-2026-03-17"
    "gpt-5.4-nano-2026-03-17"
)
MODELS=(
    "qwen3-4b-base"
    "qwen3-4b-instruct-2507"
)

# gen_judgment.py looks up reference answers by judge model name. The reference
# answers (gold standard math/coding answers) are judge-independent, stored only
# as gpt-4.jsonl. Symlink each new judge name to that file so check_data passes.
REF_DIR="$REPO_ROOT/evals/mtbench/data/reference_answer"
for JUDGE in "${JUDGES[@]}"; do
    [ -f "$REF_DIR/$JUDGE.jsonl" ] || ln -s "gpt-4.jsonl" "$REF_DIR/$JUDGE.jsonl"
done

for JUDGE in "${JUDGES[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        ANSWER_DIR="$RESULTS_DIR/$MODEL/mtbench/model_answer"
        OUTPUT_DIR="$RESULTS_DIR/$MODEL/mtbench/model_judgment"
        echo "=== Judge: $JUDGE  |  Model: $MODEL ==="
        python "$GEN_JUDGMENT" \
            --bench-name mt_bench \
            --judge-model "$JUDGE" \
            --mode single \
            --model-list "$MODEL" \
            --answer-dir "$ANSWER_DIR" \
            --output-dir "$OUTPUT_DIR" \
            --parallel "$PARALLEL"
        echo ""
    done
done

echo "All re-judging runs complete."
