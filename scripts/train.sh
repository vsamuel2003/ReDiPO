#!/bin/bash
set -euo pipefail

source env.sh

export WANDB_PROJECT=YOUR_PROJECT
python train/train.py --config configs/llama_config.yaml --train_file preference_data/llama_pairs.jsonl
