#!/bin/bash
set -euo pipefail

source env.sh
python data_processing/clean_base_outputs.py
