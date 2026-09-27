#!/usr/bin/env bash
# run_train.sh — train the pipeline. See README.md "How this was actually
# run" for the exact per-stage Kaggle commands and measured runtimes on the
# real ~2M-entity dataset.
#
# Two things this script can mean:
#   1. FULL SCALE (what actually produced this submission). Every stage ran
#      as a Kaggle notebook (CPU for normalise/blocking/features/stage-1;
#      GPU T4 for the judge track, which was tried and dropped — see
#      README.md), because the raw data (~2.4 GB, millions of rows) does not
#      fit an 8 GB laptop's RAM (CLAUDE.md SS5/SS6, the scale guard). This
#      script does NOT reproduce that scale locally; the numbered
#      `kaggle kernels push ...` commands that did are listed in README.md,
#      driven through `tools/kaggle_ops.py` with your own Kaggle credentials
#      (not part of this zip).
#   2. SMALL-SAMPLE DEMO (what this script actually runs). Trains a
#      self-contained blocking pruner + stage-1 model + calibrator from
#      scratch on a small dataset's own labels (the bundled `sample/train/`
#      by default), using the exact same src/ code as the real run
#      (src/predict.py's `train_demo`). Saves the result to `out_dir/artifacts/`
#      for `run_predict.sh --demo` to apply to that dataset's TEST split.
#      Useful to verify the code runs end to end after a change.
#
# Usage: ./run_train.sh [data_dir] [artifacts_out_dir]
#   data_dir           folder with train/{train_source1,2,3,train_ground_truth}.tsv
#                      (default: sample/, the bundled sample this repo ships with)
#   artifacts_out_dir  where the trained pruner/model/calibrator are saved
#                      (default: ../../output_demo/artifacts)
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> code/business_entity_resolution/

DATA_DIR="${1:-../../sample}"
ARTIFACTS_DIR="${2:-../../output_demo/artifacts}"
GT="$DATA_DIR/train/train_ground_truth.tsv"
N_JOBS="$(python3 -c 'import os; print(os.cpu_count() or 4)' 2>/dev/null || echo 4)"

echo "== unit tests (src/tests, run on sample/) =="
python3 -m pytest -q src/tests

echo "== small-sample demo training: blocking pruner + stage-1 model from '$DATA_DIR/train/' =="
python3 -m src.predict --config src/configs/default.yaml --train \
  --data-dir "$DATA_DIR" --ground-truth "$GT" \
  --save-artifacts-dir "$ARTIFACTS_DIR" --n-jobs "$N_JOBS"

echo "Trained demo artifacts saved to $ARTIFACTS_DIR/."
echo "Next: ./run_predict.sh --demo $DATA_DIR $ARTIFACTS_DIR"
echo "For the real, full-scale run see README.md."
