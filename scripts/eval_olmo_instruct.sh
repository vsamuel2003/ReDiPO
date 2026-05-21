#!/bin/bash
set -euo pipefail

source env.sh

export HARMBENCH_TC="./evals/harmbench/data/test_cases_directrequest_test.json"
HARMBENCH_TC="./evals/harmbench/data/test_cases_directrequest_test.json"

bash "./evals/run_all.sh" allenai/OLMo-3-7B-Instruct \
    --model-id olmo-3-7b-instruct \
    --model-type instruct \
    --output-dir ./evals/olmo_results/olmo-3-7b-instruct \
    --rejudge \
    --test-cases-path "$HARMBENCH_TC"
