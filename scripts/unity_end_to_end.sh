#!/bin/bash
set -euo pipefail

source env.sh

echo "==================== START: GENERATION ===================="

python data_processing/generate.py \
    --input_file data/instruct_subset.jsonl \
    --models "allenai/Olmo-3-7B-Instruct,allenai/Olmo-3-1025-7B" \
    --k 16 \
    --output_dir ./generated_data \
    --output_file_name generations_olmo.jsonl \
    --temperature 0.9 \
    --top_p 0.95 \
    --max_new_tokens 1024

echo "==================== END: GENERATION ======================"


echo "==================== START: SOFT CLEANUP =================="

python data_processing/cleanup.py \
    --input_file ./generated_data/generations_olmo.jsonl \
    --output_file ./generated_data/generations_olmo_cleaned.jsonl \
    --diagnostics_file ./generated_data/cleanup_olmo_diagnostics.jsonl \
    --batch_size 25

echo "==================== END: SOFT CLEANUP ===================="


echo "==================== START: BASE MODEL CLEANUP ============"

python data_processing/clean_base_outputs.py \
    --input_file ./generated_data/generations_olmo_cleaned.jsonl \
    --output_file ./generated_data/generations_olmo_full_cleaned.jsonl \
    --cleaner_model allenai/Olmo-3-7B-Instruct \
    --max_new_tokens 1024

echo "==================== END: BASE MODEL CLEANUP =============="


echo "==================== START: FILTER ========================"

python data_processing/filter.py \
    --input_file ./generated_data/generations_olmo_full_cleaned.jsonl \
    --output_file ./filtered_data/olmo_skywork-filtered.jsonl \
    --safety judge \
    --instruct skywork \
    --scorer skywork \
    --threshold 0.15

echo "==================== END: FILTER =========================="


echo "==================== START: GROUP + EMBED ================="

INPUT="filtered_data/olmo_skywork-filtered.jsonl"
MODEL="openai"

BASENAME=$(basename "$INPUT" .jsonl)
GROUPED="filtered_data/${BASENAME}_grouped.jsonl"
OUTPUT="filtered_data/${BASENAME}_embedded.jsonl"

echo "Input:   $INPUT"
echo "Grouped: $GROUPED"
echo "Output:  $OUTPUT"
echo "Model:   $MODEL"

echo "---- GROUPING ----"
python data_processing/group_responses.py \
    --input_file "$INPUT" \
    --output_file "$GROUPED"

echo "---- EMBEDDING ----"
python data_processing/embed_responses.py \
    --input_file "$GROUPED" \
    --output_file "$OUTPUT" \
    --model "$MODEL"

echo "==================== END: GROUP + EMBED ==================="


echo "==================== START: DIVERSITY SCORING ============="

python data_processing/score_diversity.py \
    --input_file "$OUTPUT" \
    --embedding_method openai \
    --diversity_method maxsim \
    --output_file scored_data/olmo_skywork_openai_maxsim.jsonl

echo "==================== END: DIVERSITY SCORING ==============="

echo "==================== START: PAIR SELECTION ================"
python data_processing/select_pairs.py \
    --input_file scored_data/olmo_skywork_openai_maxsim.jsonl \
    --output_file preference_data/olmo_pairs.jsonl \
    --epsilon 6.0
echo "==================== END: PAIR SELECTION =================="


echo "==================== PIPELINE COMPLETE ===================="
