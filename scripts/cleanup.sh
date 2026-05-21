#!/bin/bash
source env.sh
python data_processing/cleanup.py \
    --input_file ./generated_data/generations.jsonl \
    --output_file ./generated_data/generations_cleaned.jsonl \
    --diagnostics_file ./generated_data/cleanup_diagnostics.jsonl \
    --batch_size 20
