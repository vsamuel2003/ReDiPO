#!/bin/bash
set -euo pipefail

source env.sh

export HARMBENCH_TC="./evals/harmbench/data/test_cases_directrequest_test.json"
HARMBENCH_TC="./evals/harmbench/data/test_cases_directrequest_test.json"

bash "./evals/run_all.sh" allenai/OLMo-3-1025-7B \
    --model-id olmo-3-1025-7b-base \
    --model-type base \
    --test-cases-path "$HARMBENCH_TC"
