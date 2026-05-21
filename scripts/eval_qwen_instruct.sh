#!/bin/bash
set -euo pipefail

source env.sh

export HARMBENCH_TC="./evals/harmbench/data/test_cases_directrequest_test.json"
HARMBENCH_TC="./evals/harmbench/data/test_cases_directrequest_test.json"

bash "./evals/run_all.sh" Qwen/Qwen3-4B-Instruct-2507 \
    --model-id qwen3-4b-instruct-2507 \
    --model-type instruct \
    --output-dir ./evals/qwen_results/qwen3-4b-instruct-2507 \
    --test-cases-path "$HARMBENCH_TC"
