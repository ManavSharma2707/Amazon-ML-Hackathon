# Business Entity Resolution — code package

Amazon ML Challenge 2026, Business Entity Resolution Challenge. This folder is
the exact content of the submission zip's `code/` directory: all source lives
under `src/` (configs, scripts and notebooks included), per the problem
statement's "put all source under `src/`".

**Shipped result** (`output/matching_results.tsv` in the zip root): stage-1
LightGBM classifier -> isotonic calibration -> hard one-owner rule ->
expected-F0.5 decoder. Half-B (validation) macro F0.5 **0.9648** (US 0.9699 /
India 0.9571), LOCO-mean 0.9635, scrambled-letter drop 0.0009. See
`Documentation_template.md` for the full methodology, ablation table and
error analysis.

## What is, and is not, in the shipped inference path

Two things were built, evaluated on Half B, and **dropped** because they did
not clear the project's own keep/drop gate (LOCO-mean gain must exceed noise,
CLAUDE.md SS4.3) — they stay in this code as tested, documented alternatives,
not as the path `run_predict.sh` exercises by default:

- **The combiner** (`src/combiner.py`, `src/collective.py`, NB09): adds
  competition/sibling features and a second LightGBM stage on top of stage-1.
  It improved Half-B F0.5 (0.9674 vs 0.9648, a real, statistically significant
  gain) but showed **no LOCO-mean gain** (0.9635 vs 0.9635 for the stage-1
  pipeline) — the project's primary, cross-country selection metric. Kept as
  an evaluated ablation.
- **The LLM judge** (Qwen3-4B QLoRA, `src/judge_data.py`, `src/judge_train.py`,
  `src/judge_infer.py`, NB06b/NB07/NB08): trained on a 20k-example
  evidence-augmented band from Half A, then scored on the uncertain band of
  Half B. Its gate (judge AUC on the band must beat stage-1's) **failed**:
  0.774 vs 0.877 for stage-1, with a training budget cut to 70 minutes to fit
  the day's schedule. Dropped.

So the base models actually exercised by the shipped submission are just
**LightGBM** (stage-1) plus classical string/number features (`rapidfuzz`,
stdlib). `Qwen/Qwen3-Embedding-0.6B` (dense blocking) and `Qwen/Qwen3-4B` (the
judge) were downloaded and code was written and run for both, but neither is
on the path that produced `output/matching_results.tsv` — see below for why.

## Base models (downloaded once, internet ON, then run fully offline)

| Model | Licence | Actually used in the shipped result? |
|---|---|---|
| `Qwen/Qwen3-Embedding-0.6B` | Apache-2.0 | No — encodes ~400 rec/s on 2xT4; a full-pool dense pass over ~24M train+test records is infeasible (~16 h) at this challenge's scale. Code (`src/embed.py`, `src/finetune_embedder.py`) and a dry-run notebook exist and were verified but never run at scale (Gate G2 = N/A). |
| `Qwen/Qwen3-4B` | Apache-2.0 | No — trained as the judge (`src/judge_train.py`), gated out above. |
| `Qwen/Qwen3-Reranker-0.6B` | Apache-2.0 | No — the documented fallback judge if Qwen3-4B failed outright; not needed. |
| LightGBM | MIT | **Yes** — stage-1 classifier, the blocking pre-rankers, and the (unused) combiner. |

Downloaded once with internet ON (`src/notebooks/nb00_download_models`,
`snapshot_download`), then every other notebook runs with internet OFF,
loading weights from a local Kaggle dataset (`er-models`).

## No external data, no test-time training

- No external APIs, geocoders, business registries, web lookups or
  third-party re-uploads of the dataset anywhere in this pipeline.
- Nothing is trained on test data; test predictions were viewed only to catch
  bugs (format, encoding, empty-rate sanity), never to tune a threshold.
- Country is treated as an open set: the only country-derived feature is the
  `same_country` equality flag; no country value is ever hard-coded,
  filtered on, or one-hot encoded (needed for the unseen France split).
- Every inference notebook runs with internet **OFF**.

## Environment

- Python 3.12 (the Kaggle base image this pipeline was run on); developed and
  unit-tested locally on Python 3.13. `pip install -r requirements.txt`
  installs the CPU pipeline's dependencies (numpy/pandas/scipy/scikit-learn/
  lightgbm/rapidfuzz/pyyaml); the judge track additionally needs
  torch/transformers/peft/accelerate/bitsandbytes (only imported by
  `src/judge_train.py`/`src/judge_infer.py`, not by the shipped inference path).
- Compute actually used: Kaggle CPU notebooks for normalise/blocking/features/
  stage-1/decode; Kaggle GPU T4 (single, then T4x2) for the judge track.
  fp16 only, SDPA attention (no bf16, no flash-attn-2) — a T4 constraint, only
  relevant to the (unused) judge/embedder code paths.
- **8 GB laptop, not the full dataset.** The raw data is ~2.4 GB across
  millions of rows; every full-data step ran as a Kaggle notebook. The laptop
  was used for code authoring, unit tests on the bundled `sample/`, and git.

## How this was actually run (real notebook chain, real runtimes)

Every notebook is generated from `src/` by `tools/bundle_kernel.py` (not part
of this zip — a private-repo dev tool) into one self-contained script per
`src/notebooks/<name>/`, pushed with the Kaggle CLI, and produces a small
`metrics.json` alongside its real outputs. Runtimes below are measured, not
estimated (from this run's `metrics.json` files):

| # | Notebook | Accelerator | Runtime | Purpose |
|---|---|---|---|---|
| 00 | `nb00_download_models` | CPU, internet ON | ~20-40 min | download Qwen3 weights + offline wheels |
| 01 | `nb01_eda` | CPU | ~8 min | EDA findings (memory.md) |
| 02 | `nb02_normalize` | CPU | 57 min | normalise + corpus stats + A/B split |
| 05 | `nb05_blocking_sparse` | CPU | 67 min | candidate generation (blocking), the version that fed the shipped run |
| 06 | `nb06_features_stage1` | CPU | 4 h 58 min | pair features (groups A-H) + stage-1 LightGBM, OOF on A, predictions on B/test |
| 09a | `nb09a_decode_submit` | CPU | 22 min | calibration, one-owner rule, expected-F0.5 decoder, writes the submission TSVs |
| 06b | `nb06b_judge_data` | CPU | 2 min | judge prompts (not on the shipped path) |
| 07 | `nb07_judge_train` | GPU T4 | 76 min | judge QLoRA training (dropped, gate failed) |
| 09 | `nb09_combine` | CPU | 68 min | combiner + collective features (dropped, no LOCO gain) |

Total wall-clock for the shipped path: 00 -> 01 -> 02 -> 05 -> 06 -> 09a,
about 6.5 hours, almost all of it NB06's feature computation over ~40M
candidate pairs.

**Blocking recall is the main limit on the final score.** NB05's channel set
fed the shipped run at 0.9585 pair recall on Half B: char TF-IDF name/address,
a rare-token key, a house-number+street-token key, and unordered name/address
token-pair keys (`src/blocking.py`'s module docstring). Two channels added
afterwards — name-address cross token pairs, then a reverse pool-to-S1 name
lookup — raised recall to 0.9733 and then 0.9736 in later blocking runs, but
arrived too late in the schedule to re-run the ~5-hour feature/stage-1 step
before the modelling freeze; both are implemented, tested and available for
a future run (`src/blocking.py`'s `cross_pair` channel and `rev_name_k`
config; see `Documentation_template.md`'s blocking table for the full
recall/reduction-ratio comparison across versions).

## Reproducing the exact submission

`artifacts/` in this folder **is** what NB09a actually trained and saved for
the shipped run: `stage1_model.txt` (the LightGBM stage-1 booster,
~11.6 MB — small enough to ship, unlike the multi-GB Qwen weights the "no
large model weights" rule targets), `calibrator.npz` (the isotonic
calibrator's breakpoints) and `final_config.json` (the chosen decoder
config: hard one-owner rule, lambda 1.3, expected-F0.5 decoding). Only the
full pair-feature table (`feats_test/`, one row per scored candidate pair
over the whole ~2M-entity test set) is a Kaggle-notebook output too large to
ship in this zip (memory.md SS11). `src/predict.py --from-features`
reproduces `output/matching_results.tsv` exactly (same seed, no
retraining) given a copy of that feature table:

```bash
# from code/business_entity_resolution/, given your own copy of NB06's
# feats_test output (kaggle kernels output er-nb06-features-stage1 -p <dest>):
FEATS_TEST=.../feats_test TEST_S1=.../test_source1.tsv \
  bash src/scripts/run_predict.sh   # ARTIFACTS_DIR defaults to this zip's own artifacts/
```

## Running (self-contained, no external artifacts needed)

```bash
# from code/business_entity_resolution/
pytest -q src/tests/                        # unit tests (run on the bundled sample/)
python -m src.metrics                       # unit-test the official F0.5 scorer

# small-sample demo: trains its own blocking pruner + stage-1 model from
# sample/'s own ground truth, then predicts on sample/'s test split
bash src/scripts/run_train.sh
bash src/scripts/run_predict.sh --demo

# pre-flight checks
python -m src.check_outputs --out-dir ../../output --test-dir ../../student_resource/dataset/test

# official validator (from student_resource/)
python3 utils/validate_submission.py \
  --matching ../output/matching_results.tsv \
  --candidate ../output/candidate_pairs.tsv \
  --test-dir dataset/test --check-ids
```

The bundled `sample/` (~0.2% of the real pool) is for exercising the exact
same code end to end on a laptop; its blocking-recall/F0.5 numbers are not
representative of the real dataset (see `src/tests/test_blocking.py`'s
comments on this).

## Regenerating `requirements.txt`

Versions are pinned from real installs observed on the Kaggle image this
pipeline ran on (`reports/raw/nb00_r1/*.log`, not part of this zip); the exact
patch versions of `pandas`/`pyarrow` (which ship in the Kaggle base image and
were not re-installed) were not captured this run. To regenerate precisely,
run `pip freeze` inside any of the Kaggle notebooks above, or inside NB11 (a
reproducibility check that runs with internet OFF and prints `pip freeze`).

## Licences

| Component | Licence |
|---|---|
| Qwen3-Embedding-0.6B, Qwen3-4B, Qwen3-Reranker-0.6B | Apache-2.0 (downloaded; not on the shipped inference path — see above) |
| LightGBM | MIT (the model actually used) |
| rapidfuzz | MIT |
| bitsandbytes, peft | MIT (judge track only) |
| pandas, numpy, scipy, scikit-learn | BSD |
| transformers, accelerate, huggingface_hub | Apache-2.0 |
| torch | BSD-3-Clause |
| PyYAML | MIT |

No GPL dependencies (e.g. `unidecode`) are used anywhere in this pipeline.
