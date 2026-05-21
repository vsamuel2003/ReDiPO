#!/bin/bash

set -e  # stop if any command fails

# echo "Selecting pairs for ARMRO (bge)..."
# bash scripts/select_pairs.sh scored_data/pilot_armro_bge_centroid.jsonl preference_data/pilot_armro_bge_centroid_bin.jsonl bin
# bash scripts/select_pairs.sh scored_data/pilot_armro_bge_centroid.jsonl preference_data/pilot_armro_bge_centroid_weighted.jsonl weighted

# bash scripts/select_pairs.sh scored_data/pilot_armro_bge_maxsim.jsonl preference_data/pilot_armro_bge_maxsim_bin.jsonl bin
# bash scripts/select_pairs.sh scored_data/pilot_armro_bge_maxsim.jsonl preference_data/pilot_armro_bge_maxsim_weighted.jsonl weighted

echo "Selecting pairs for ARMRO (openai)..."
# bash scripts/select_pairs.sh scored_data/pilot_armro_openai_centroid.jsonl preference_data/pilot_armro_openai_centroid_bin.jsonl bin
# bash scripts/select_pairs.sh scored_data/pilot_armro_openai_centroid.jsonl preference_data/pilot_armro_openai_centroid_weighted.jsonl weighted

# bash scripts/select_pairs.sh scored_data/pilot_armro_openai_maxsim.jsonl preference_data/pilot_armro_openai_maxsim_bin.jsonl bin
# bash scripts/select_pairs.sh scored_data/pilot_armro_openai_maxsim.jsonl preference_data/pilot_armro_openai_maxsim_weighted.jsonl weighted

bash scripts/select_pairs.sh scored_data/pilot_armro_openai_centroid.jsonl preference_data/pilot_armro_openai_centroid_epsilon.jsonl epsilon
# bash scripts/select_pairs.sh scored_data/pilot_armro_openai_maxsim.jsonl preference_data/pilot_armro_openai_maxsim_epsilon.jsonl epsilon

# echo "Selecting pairs for HARD (bge)..."
# bash scripts/select_pairs.sh scored_data/pilot_hard_bge_centroid.jsonl preference_data/pilot_hard_bge_centroid_bin.jsonl bin
# bash scripts/select_pairs.sh scored_data/pilot_hard_bge_centroid.jsonl preference_data/pilot_hard_bge_centroid_weighted.jsonl weighted

# bash scripts/select_pairs.sh scored_data/pilot_hard_bge_maxsim.jsonl preference_data/pilot_hard_bge_maxsim_bin.jsonl bin
# bash scripts/select_pairs.sh scored_data/pilot_hard_bge_maxsim.jsonl preference_data/pilot_hard_bge_maxsim_weighted.jsonl weighted

# echo "Selecting pairs for HARD (openai)..."
# bash scripts/select_pairs.sh scored_data/pilot_hard_openai_centroid.jsonl preference_data/pilot_hard_openai_centroid_bin.jsonl bin
# bash scripts/select_pairs.sh scored_data/pilot_hard_openai_centroid.jsonl preference_data/pilot_hard_openai_centroid_weighted.jsonl weighted

# bash scripts/select_pairs.sh scored_data/pilot_hard_openai_maxsim.jsonl preference_data/pilot_hard_openai_maxsim_bin.jsonl bin
# bash scripts/select_pairs.sh scored_data/pilot_hard_openai_maxsim.jsonl preference_data/pilot_hard_openai_maxsim_weighted.jsonl weighted

echo "All pair selection jobs completed!"