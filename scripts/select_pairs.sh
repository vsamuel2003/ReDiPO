#!/bin/bash
# Select preference pairs from scored responses for DPO training.
#
# Usage:
#   bash scripts/select_pairs.sh [input_file] [output_file] [mode] [epsilon] [n_cap] [top_quantile] [alpha] [bin_width] [seed]
#
# Defaults:
#   input_file   = filtered_data/pilot_armro-filtered_embedded.jsonl
#   output_file  = filtered_data/preference_pairs.jsonl
#   mode         = epsilon  (options: epsilon | bin | weighted)
#   epsilon      = 0.2      (only used in epsilon mode)
#   n_cap        = 4        (only used in epsilon mode)
#   top_quantile = 0.25     (only used in epsilon mode)
#   alpha        = 0.5      (only used in weighted mode)
#   bin_width    = 0.5      (only used in bin mode)
#   seed         = 42       (only used in weighted mode)
#
# Examples:
#   bash scripts/select_pairs.sh
#   bash scripts/select_pairs.sh filtered_data/pilot_armro-filtered_embedded.jsonl filtered_data/pairs_epsilon.jsonl epsilon 0.2 4 0.25
#   bash scripts/select_pairs.sh filtered_data/pilot_armro-filtered_embedded.jsonl filtered_data/pairs_bin.jsonl bin
#   bash scripts/select_pairs.sh filtered_data/pilot_armro-filtered_embedded.jsonl filtered_data/pairs_weighted.jsonl weighted

set -euo pipefail

source env.sh

INPUT=${1:-filtered_data/pilot_armro-filtered_embedded.jsonl}
OUTPUT=${2:-filtered_data/preference_pairs.jsonl}
MODE=${3:-epsilon}
EPSILON=${4:-0.2}
N_CAP=${5:-4}
TOP_QUANTILE=${6:-0.25}
ALPHA=${7:-0.5}
BIN_WIDTH=${8:-0.5}
SEED=${9:-42}

echo "Input:        $INPUT"
echo "Output:       $OUTPUT"
echo "Mode:         $MODE"
echo "Epsilon:      $EPSILON  (epsilon mode)"
echo "N cap:        $N_CAP  (epsilon mode)"
echo "Top quantile: $TOP_QUANTILE  (epsilon mode)"
echo "Alpha:        $ALPHA  (weighted mode)"
echo "Bin width:    $BIN_WIDTH  (bin mode)"
echo "Seed:         $SEED  (weighted mode)"
echo ""

python data_processing/select_pairs.py \
    --input_file "$INPUT" \
    --output_file "$OUTPUT" \
    --mode "$MODE" \
    --epsilon "$EPSILON" \
    --n_cap "$N_CAP" \
    --top_quantile "$TOP_QUANTILE" \
    --alpha "$ALPHA" \
    --bin_width "$BIN_WIDTH" \
    --seed "$SEED"
