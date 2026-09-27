#!/usr/bin/env bash
# run_predict.sh — produce output/matching_results.tsv + candidate_pairs.tsv.
#
# Default mode (no args) reproduces the ACTUAL SHIPPED SUBMISSION exactly,
# from the saved pair-feature table + trained artifacts
# (src/predict.py's `--from-features` path): stage-1 -> isotonic calibration
# -> hard one-owner rule -> expected-F0.5 decoder, byte-for-byte the same as
# the submitted files (same seed, no retraining, no combiner — the combiner
# was evaluated and gated out, see README.md/memory.md). This is what NB11
# checks against the actual submission. It needs:
#   FEATS_TEST     feats_test.parquet, or the feats_test/ folder of
#                  part-*.parquet files (NB06's output) — a Kaggle-notebook
#                  output, too large to ship in this zip (memory.md SS11);
#                  point this at your own copy (`kaggle kernels output`, see
#                  README.md)
#   TEST_S1        test_source1.tsv (for the required row order)
#   ARTIFACTS_DIR  the trained stage-1 model + calibrator + decoder config;
#                  defaults to this zip's own `artifacts/` (the real ones
#                  NB09a saved, ~12 MB — small enough to ship, unlike the
#                  feature table above)
#
# `--demo [data_dir] [artifacts_dir]` runs the full pipeline (blocking ->
# features -> stage-1 -> decode) on a small dataset's TEST split instead,
# using a model already trained (on that same dataset's TRAIN split, by
# run_train.sh) into artifacts_dir — no external artifact needed, but you
# must run run_train.sh first (predicting with a model trained on the very
# rows being predicted would defeat the point of the test split).
#
# Usage:
#   FEATS_TEST=... TEST_S1=... ARTIFACTS_DIR=... ./run_predict.sh
#   ./run_train.sh [data_dir] [artifacts_dir]   # first, to produce a demo model
#   ./run_predict.sh --demo [data_dir] [artifacts_dir]
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> code/business_entity_resolution/

OUT_DIR="${OUT_DIR:-../../output}"
N_JOBS="$(python3 -c 'import os; print(os.cpu_count() or 4)' 2>/dev/null || echo 4)"

if [ "${1:-}" = "--demo" ]; then
  DATA_DIR="${2:-../../sample}"
  ARTIFACTS_DIR="${3:-../../output_demo/artifacts}"
  echo "== small-sample demo: full pipeline, applying the model run_train.sh trained on '$DATA_DIR/train/' =="
  python3 -m src.predict --config src/configs/default.yaml --load-artifacts-dir "$ARTIFACTS_DIR" \
    --data-dir "$DATA_DIR" --split test --out-dir "$OUT_DIR" --n-jobs "$N_JOBS"
  TEST_DIR="$DATA_DIR/test"
else
  : "${FEATS_TEST:?set FEATS_TEST to feats_test.parquet or the feats_test/ folder (NB06 output)}"
  : "${TEST_S1:?set TEST_S1 to test_source1.tsv}"
  ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts}"   # this zip's own saved artifacts/, by default
  echo "== exact reproduction of the shipped submission (--from-features) =="
  python3 -m src.predict --from-features "$FEATS_TEST" --test-s1 "$TEST_S1" \
    --artifacts-dir "$ARTIFACTS_DIR" --out-dir "$OUT_DIR" --n-jobs "$N_JOBS"
  TEST_DIR="$(dirname "$TEST_S1")"
fi

echo "== pre-flight checks (src/check_outputs.py) =="
python3 -m src.check_outputs --out-dir "$OUT_DIR" --test-dir "$TEST_DIR"
echo "Now run the official validator (utils/validate_submission.py, see README.md) before submitting."
