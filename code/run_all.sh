#!/usr/bin/env bash
# Run from the repository root:  bash code/run_all.sh
set -e
python src/train.py --features base
python src/train.py --features geo
python src/ablation.py
python src/compare.py
python src/test.py --features geo
python src/make_tables.py
