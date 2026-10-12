#!/usr/bin/env bash
# Run the full pipeline. Put the Kaggle CSV in data/ first (see README), or run
#   python src/make_sample_data.py
# for a synthetic smoke test.
set -e
cd "$(dirname "$0")"
python src/prepare.py
python src/train_logistic.py
python src/train_gbm.py
python src/scorecard.py
echo; echo "Done. See outputs/ for metrics, tables and plots."
