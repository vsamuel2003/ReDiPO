#!/bin/bash
set -euo pipefail

source env.sh


python data_processing/filter.py \
    --input_file ./generated_data/generations_cleaned.jsonl \
    --safety judge \
    --instruct judge_hard \
    --scorer armorm \
    --output_file ./filtered_data/pilot_hard_filtered.jsonl

# Example with armorm as both filter and scorer (score reused, no double computation):
# python data_processing/filter.py \
#     --input_file ./generated_data/generations.jsonl \
#     --safety llama_guard \
#     --instruct armorm \
#     --scorer armorm \
#     --threshold 2.5 \
#     --output_file ./filtered_data/filtered.jsonl
