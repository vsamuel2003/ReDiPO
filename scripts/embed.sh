#!/bin/bash
# Embed pipeline: group responses by prompt, then add embeddings.
#
# Usage:
#   bash scripts/embed.sh [input_file] [model]
#
# Defaults:
#   input_file = filtered_data/pilot_hard_filtered.jsonl
#   model      = both  (options: openai | bge | both)
#
# Examples:
#   bash scripts/embed.sh
#   bash scripts/embed.sh filtered_data/pilot_armro-filtered.jsonl bge
#   bash scripts/embed.sh filtered_data/pilot_hard_filtered.jsonl openai

set -euo pipefail

source env.sh

INPUT=${1:-filtered_data/pilot_hard_filtered.jsonl}
MODEL=${2:-both}

# Derive output paths from the input filename
BASENAME=$(basename "$INPUT" .jsonl)
GROUPED="filtered_data/${BASENAME}_grouped.jsonl"
OUTPUT="filtered_data/${BASENAME}_embedded.jsonl"

echo "Input:   $INPUT"
echo "Grouped: $GROUPED"
echo "Output:  $OUTPUT"
echo "Model:   $MODEL"
echo ""

python data_processing/group_responses.py \
    --input_file "$INPUT" \
    --output_file "$GROUPED"

python data_processing/embed_responses.py \
    --input_file "$GROUPED" \
    --output_file "$OUTPUT" \
    --model "$MODEL"
