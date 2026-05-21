#!/bin/bash
set -euo pipefail

source env.sh


echo "==================== START: GENERATION ===================="

python data_processing/generate.py \
    --input_file data/instruct_subset.jsonl \
    --models "meta-llama/Llama-3.1-8B-Instruct,meta-llama/Llama-3.1-8B" \
    --k 16 \
    --output_dir ./generated_data \
    --output_file_name generations_llama.jsonl \
    --temperature 0.9 \
    --top_p 0.95 \
    --max_new_tokens 1024

echo "==================== END: GENERATION ======================"
