# architecture.md — Code layout, notebook plan and data flow

> How the plan in `ER_Hackathon_Master_Plan.md` is turned into code and Kaggle notebooks.
> Update this file whenever a module, notebook or artifact name changes.

---

## 1. Principles

- **Logic lives in `src/`; notebooks are thin runners.** A notebook imports `src` modules and calls them with a config. This keeps the final code package reproducible and makes every step runnable both locally and on Kaggle.
- **Each notebook produces a versioned Kaggle Dataset** that later notebooks attach as input. No notebook depends on another notebook's live session.
- **CPU work runs on CPU notebooks or locally, so GPU quota is spent only on embeddings and the judge.**
- **Two GPU runners in parallel:** runner R1 does embeddings, then the embedder fine-tune; runner R2 does judge training and inference. Which Kaggle profile backs each runner is defined only in `CLAUDE.local.md` (not committed).
- **Model weights are downloaded once** (NB00, internet ON). Every other notebook runs with **internet OFF** and loads from `/kaggle/input/er-models/`.

---

## 2. Data-flow overview

```
er-data (raw TSVs)                                            er-models (Qwen weights, NB00)
    │                                                               │
    ▼                                                               │
NB01 EDA ──────────────► memory.md §6 (findings)                    │
    │                                                               │
    ▼                                                               │
NB02 normalize + stats + split ──► er-norm_vN ──────────────┐       │
                                                            ▼       ▼
                                          NB03 frozen embeddings ──► er-emb_vN
                                                            │
                                          NB04 fine-tune embedder (Half A) ──► er-embedder_vN (+ ft vectors)
                                                            │
NB05 blocking (A: frozen, B/test: fine-tuned) ──► er-cands_vN
    │
NB06 features + stage-1 LightGBM ──► er-feats_vN, er-stage1_vN (A OOF, B, test preds)
    │                     │
    │                     └──► NB07 judge train (Half A band) ──► er-judge_vN
    │                                          │
    │                          NB08 judge inference (B + test band) ──► er-judge-scores_vN
    ▼                                          │
NB09 collective + combiner + calibration + exclusivity + decoder + validation ◄──┘
    │        ──► output/matching_results.tsv, output/candidate_pairs.tsv, reports/
    ▼
NB10 error analysis (loop)          NB11 end-to-end predict (reproducibility check, internet OFF)
```

---

## 3. Repository layout (local, mirrors the final zip)

```
<repo root>/
├── CLAUDE.md                       # how Claude Code works on this project
├── memory.md                       # durable facts, decisions, findings
├── progress.md                     # live task board
├── architecture.md                 # this file
├── ER_Hackathon_Master_Plan.md     # full method
├── student_resource/               # organiser files (dataset/, utils/validate_submission.py)
│   └── dataset/{train,test}/*.tsv
├── output/                         # final TSVs (what gets zipped)
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/business_entity_resolution/   # ← goes into the zip as-is
│   ├── README.md
│   ├── requirements.txt
│   └── src/                         # ALL code lives here (problem statement: "Put all source under src/")
│       ├── __init__.py
│       ├── configs/default.yaml
│       ├── io_utils.py
│       ├── normalize.py
│       ├── corpus_stats.py
│       ├── split.py
│       ├── embed.py
│       ├── finetune_embedder.py
│       ├── blocking.py
│       ├── explain_diff.py
│       ├── features.py
│       ├── stage1.py
│       ├── judge_data.py
│       ├── judge_train.py
│       ├── judge_infer.py
│       ├── collective.py
│       ├── combiner.py
│       ├── exclusivity.py
│       ├── decoder.py
│       ├── metrics.py
│       ├── scramble.py
│       ├── check_outputs.py
│       ├── predict.py
│       ├── scripts/
│       │   ├── run_train.sh
│       │   └── run_predict.sh
│       ├── notebooks/               # NB00–NB11 (thin runners)
│       └── tests/                   # pytest unit tests
├── reports/                        # blocking report, ablations, error-analysis dumps (not zipped)
├── approach_summary.md             # 1–2-page document (official guidelines)
└── Documentation_template.md       # full methodology (zipped at the root)
```

Not zipped: the dataset, `CLAUDE.md`, `memory.md`, `progress.md`, `architecture.md`, `ER_Hackathon_Master_Plan.md`, `reports/`, model weights.

---

## 4. Module responsibilities

| Module | Plan step | Main functions | Inputs → Outputs |
|---|---|---|---|
| `io_utils.py` | 0, 22 | `load_tsv`, `load_ground_truth`, `write_id_list_tsv`, `assert_*` | raw TSV → DataFrames; lists → TSV |
| `normalize.py` | 2 | `normalize_record`, `extract_numbers`, `split_landmark`, `fold` | raw df → `raw_*`, `norm_*`, `fold_*`, number fields, landmark field |
| `corpus_stats.py` | 3 | `compute_idf`, `suffix_likeness`, `street_type_likeness` | normalised df → token-stat tables (per run; train-only flag) |
| `split.py` | 4 | `make_ab_split` | S1 + GT → `split` column (A/B), stratified |
| `embed.py` | 5 | `record_texts`, `Embedder` (last-token pooling, MRL), `encode` (multi-GPU threads), `knn_within_groups` (GPU exact kNN, forward + reverse), `pair_recall_at_k` | texts → L2-normalised fp16 256-d vectors; kNN lists |
| `finetune_embedder.py` | 5 | `build_training_triplets`, `train`, `OneS1PerBatchSampler`, scramble augmentation | Half A → fine-tuned weights |
| `blocking.py` | 6 | `run_sparse_channel` (hashed TF-IDF + chunked sparse top-k; char name/addr, rare-token, number key), `union_channels`, `add_dense_cos`, `train_pruner` (LightGBM on Half A), `prune`, `blocking_report` | records + NB03 kNN lists → pruned candidate pairs (+ channel meta) |
| `explain_diff.py` | 7 | `token_relation`, `align_tokens`, `explain_features`, `number_features` | pair → feature dict |
| `features.py` | 8 | `build_pair_features` (groups A–H), multiprocessing | candidates → feature table |
| `stage1.py` | 9 | `train_oof`, `train_full`, `predict` | A features → A OOF; B/test preds |
| `judge_data.py` | 10 | `select_band`, `build_prompt`, `build_dataset` | A OOF + records + explain output → JSONL prompts |
| `judge_train.py` | 10 | `train_qlora` | JSONL → LoRA adapter |
| `judge_infer.py` | 10 | `score_pairs` (yes/no logit, optional order swap) | band pairs → `judge_p` |
| `collective.py` | 11 | `competition_features`, `sibling_features` | stage-1 preds → features |
| `combiner.py` | 12 | `train_oof`, `fit_calibrator`, `apply` | B features → calibrated p |
| `exclusivity.py` | 13 | `hard_one_owner`, `soft_one_owner` | (e, r, p) → adjusted p |
| `decoder.py` | 14 | `poisson_binomial`, `best_k`, `decode_all` | calibrated p per entity → predicted sets |
| `metrics.py` | 21 | `f05_entity`, `f05_macro`, `loco_eval`, `bootstrap_diff`, `blocking_metrics`, `error_buckets` | predictions + GT → scores/reports |
| `scramble.py` | 21 | `make_scrambler`, `scramble_texts`, `scramble_df` | texts → scrambled texts (a–z permutation) |
| `kaggle_env.py` | — | `find_input` / `find_input_dir` (glob under /kaggle/input), `log`, `write_json`, `WORK_DIR` | shared by every notebook driver |
| `check_outputs.py` | 22 | `main` | output TSVs + test dir → PASS/FAIL |
| `predict.py` | all | `run_inference(config)` | artifacts + test TSVs → `output/*.tsv` |

---

## 5. Artifacts (Kaggle notebook outputs / datasets), naming and contents

> Revised 2026-09-26 22:25 IST: pipeline artifacts are **notebook outputs chained with `kernel_sources` on the same runner** (CLAUDE.md §6.3), not separately created datasets. The version is the notebook's version number (`er-nb05-blocking` v1, v2…). Model weights + offline wheels = NB00 output (`nb00-download-models`), attached the same way.

| Dataset name | Produced by | Contents | Consumed by |
|---|---|---|---|
| `er-data` | user upload | organiser TSVs + `utils/validate_submission.py` | all |
| `er-code` | user upload (from `code/business_entity_resolution/`) | `src/` (incl. `src/configs/`) | all notebooks |
| `er-models` | NB00 | `qwen3-emb-0.6b/`, `qwen3-4b/`, `qwen3-reranker-0.6b/` (HF snapshot folders) | NB03, NB04, NB07, NB08, NB11 |
| NB02 output (`er-nb02-normalize` vN) | NB02 | `records_{train,test}_S{1,2,3}.parquet` (raw + norm/fold/number/landmark fields), `stats_{train,test}.parquet` (long token table), `gt.parquet` (s1_id, match_id), `split_s1.parquet`, `split_pool.parquet`, `metrics.json` | NB03–NB09 via `kernel_sources` |
| NB03 output (`er-nb03-embed` vN) | NB03 | `emb_{train,test}_{S1,pool}.npy` (fp16, 256-d; pool = S2 rows then S3 rows), `knn_{split}_{idx,score}.npy` (S1→pool top-30), `rev_{split}_{idx,score}.npy` (pool→S1 top-5), `metrics.json` | NB04, NB05 |
| NB04 output (`er-nb04-finetune` vN) | NB04 | `ft_model/` (merged fp16), `lora_final/`, `train_log.json`, `acceptance.json` (G2) | NB04b; NB04b output (`er-nb04b-ft-encode`): `ft_emb_*`, `ft_knn_*`, `ft_rev_*` (train rows = Half B only) → NB05 v2 |
| NB05 output (`er-nb05-blocking` vN) | NB05 | `cands_{A,B,test}.parquet` (s1_id, cand_id, bitmask, channel scores/ranks, dense_cos, cheap_score), `pruner.txt`, `blocking_report.json`, `missed_B.tsv` | NB06–NB09 |
| `er-feats_vN` | NB06 | `feats_{A,B,test}.parquet` | NB07, NB09 |
| `er-stage1_vN` | NB06 | `stage1_model.txt`, `p1_A_oof.parquet`, `p1_{B,test}.parquet` | NB07, NB08, NB09 |
| `er-judge_vN` | NB07 | LoRA adapter, `train_log.json`, `prompts_sample.jsonl` | NB08 |
| `er-judge-scores_vN` | NB08 | `judge_{B,test}.parquet` | NB09 |
| `er-final_vN` | NB09 | combiner model, calibrator, decoder params, `output/*.tsv`, reports | NB10, NB11, submission |

Rules:
- Bump `_vN` on any change; record the version used by each submission in `progress.md`.
- Every dataset includes a `manifest.json`: creating notebook, config hash, upstream dataset versions, timestamp.

---

## 6. Notebook plan

| NB | Name | Accelerator | Internet | Inputs | Outputs | Est. time | Owner |
|---|---|---|---|---|---|---|---|
| 00 | `nb00_download_models` | None (CPU) | **ON** | — | `er-models` | 20–40 min | GPU track |
| 01 | `nb01_eda` | CPU | OFF | er-data, er-code | findings → `memory.md` | 60–90 min | CPU track |
| 02 | `nb02_normalize` → `er-nb02-normalize` | CPU | OFF | er-data | NB02 output | ~1 h | CPU track |
| 03 | `nb03_embed` (+ `nb03_embed_dry`) → `er-nb03-embed` | GPU T4×2 (`NvidiaTeslaT4`) | OFF | NB02 + NB00 outputs (kernel_sources) | NB03 output | measured in dry run | GPU track |
| 04 | `nb04_finetune` (+ `_dry`), `nb04b_ft_encode` | GPU T4 | OFF | NB02 + NB03 + NB00 outputs | NB04 / NB04b outputs | ~1.5 h + encode | GPU track |
| 05 | `nb05_blocking` → `er-nb05-blocking` | CPU | OFF | NB02 + NB03 (v2: + NB04b) outputs | NB05 output | 1–2 h (est.) | CPU track |
| 06 | `nb06_features_stage1` | CPU | OFF | er-norm, er-cands, er-emb, er-code | `er-feats_v1`, `er-stage1_v1` | 30–90 min | CPU track |
| 07 | `nb07_judge_train` | GPU T4 (single) | OFF | er-norm, er-feats, er-stage1, er-models, er-code | `er-judge_v1` | ≤ 3 h | GPU runner R2 |
| 08 | `nb08_judge_infer` | GPU T4×2 | OFF | er-judge, er-models, er-norm, er-stage1, er-cands, er-code | `er-judge-scores_v1` | 30–90 min | GPU runner R2 |
| 09 | `nb09_combine_decode_validate` | CPU | OFF | all above | `er-final_v1`, output TSVs, reports | 20–40 min | CPU track |
| 10 | `nb10_error_analysis` | CPU | OFF | er-final, er-norm | error buckets report | 15 min/round | Validation track |
| 11 | `nb11_end_to_end_predict` | GPU T4 | **OFF** | er-code, er-models, trained artifacts, er-data | output TSVs (must equal NB09's) | 1–2 h | Packaging |

### 6.1 Notebooks are self-contained (no `er-code` dependency) — revised 2026-09-26

**Superseded:** the original plan had every notebook do `sys.path.insert(0, "/kaggle/input/er-code")` + `from src import ...`. Dropped after NB01 hit `ModuleNotFoundError: No module named 'src'` even with `er-code` correctly attached as a dataset source — this Kaggle environment mounts datasets at `/kaggle/input/datasets/<owner>/<slug>/`, not the classic `/kaggle/input/<slug>/`, and relying on any one mount convention (or a hard-coded owner username) is fragile.

**Current design:** every pushed kernel is fully self-contained — no `sys.path` tricks, no dependency on any dataset for *code* (only for *data*: `er-data`, `er-models`). Each notebook folder under `src/notebooks/<nbNN_name>/` has:
- `driver.py` — the hand-written entry point (business logic only; references `src/` modules by bare name, e.g. `eda.run_eda(...)`, `io_utils.load_tsv(...)`; never imports `src` or touches `sys.path`).
- `bundle_spec.json` — `{"modules": [...in dependency order...], "driver": "driver.py", "output": "<code_file>.py"}`.
- `kernel-metadata.json` — `id` and any `dataset_sources` owner prefixed with the literal string `PLACEHOLDER/` (rewritten to the pushing runner's username at push time, restored after).
- `<code_file>.py` — generated by `tools/bundle_kernel.py` (inlines the listed `src/*.py` modules as real in-memory `types.ModuleType` objects, wired together, followed by the driver code). Regenerated automatically by `tools/kaggle_ops.py push` before every push. Never edit it directly — it says so in its own header.

`src/` stays the single source of truth: it's what pytest tests locally, and the bundler just inlines its current content. A notebook needing a Kaggle-mounted *path* (e.g. where `er-data` landed) must resolve it dynamically — glob for both mount conventions, never hard-code an owner username or assume one mount layout (see `driver.py` in `nb01_eda` for the pattern: `find_data_dir()`).

NB00 needs no `src/` modules at all (it only downloads models with stdlib + `huggingface_hub`), so it has no `bundle_spec.json` and its `.py` file is hand-written and pushed as-is.

**GPU jobs (NB03 onward): dry-run before the full run.** Push a fast/tiny-data variant first to catch import/path/shape/dtype errors on Kaggle's actual GPU environment before committing GPU-hours to a full run (CLAUDE.md SS5, GPU rules).

### 6.2 NB00: download models (internet ON, run once)

1. `pip install -U "transformers>=4.51.0" huggingface_hub`
2. `snapshot_download` for `Qwen/Qwen3-Embedding-0.6B`, `Qwen/Qwen3-4B`, `Qwen/Qwen3-Reranker-0.6B` into `/kaggle/working/models/<short-name>/`.
3. Record the licence field of each model card in `models/LICENSES.md`.
4. Save the notebook output as Kaggle Dataset **`er-models`**.

### 6.3 NB01: EDA

Cells:
1. Load with `io_utils.load_tsv` and run the sanity asserts.
2. Sizes per source × country (E1).
3. Ground-truth parse: singleton rate (E2), matches distribution (E3), multi-owner check (E4), cross-country matches (E5), unmatched S2/S3 share (E6).
4. Positive-pair similarity census: exact-name/address rates (E7); noise census per source using a quick token diff (E8).
5. Hard negatives: top name-similarity non-matches (E9).
6. Field quality (E10); postcode shapes (E11); test set composition and character set (E12).
7. Shortcut check: correlation of ID numbers / row order with match status (detect only).
8. Print a findings block to paste into `memory.md` §6.

### 6.4 NB02: normalise, stats, split

1. `normalize.normalize_df` on train and test (all sources).
2. `corpus_stats` per run: train stats on train; test stats on test (flag `stats_mode: per_run | train_only`).
3. `split.make_ab_split` (stratified by country × match bucket; seed 42).
4. Save `er-norm_v1` + manifest.

### 6.5 NB03: frozen embeddings + GPU dense kNN (T4×2)

1. Load Qwen3-Embedding-0.6B with transformers (fp16, sdpa, left padding, last-token pooling), one replica per GPU; text = `"name | address"` (raw, NFKC + whitespace; country omitted because blocking is within-country).
2. Encode train S1, train pool (S2+S3), test S1, test pool with length-sorted batches; MRL-truncate to 256-d, L2-normalise, save fp16 `.npy` (scale guard: full-record only; name/address-only vectors skipped — ~3× GPU time and > 20 GB output).
3. Exact within-country kNN on the GPU (chunked matmul + top-k): S1→pool top-30 and pool→S1 top-5 from the same score blocks. CPU kNN at 1.7M × 10M is not feasible.
4. metrics.json: throughput, sanity cosines (true vs random same-country pairs), dense recall@k on Half B per country. The dry run (`er-nb03-embed-dry`) also compares with vs without the instruction prefix on a 3k-query mini retrieval task.

### 6.6 NB04: fine-tune embedder (Half A only)

1. Mine hard negatives from frozen kNN on Half A.
2. Build triplets (plus S2↔S3 sibling positives), country-balanced.
3. Train with `MultipleNegativesRankingLoss` (or the cached variant); custom one-S1-per-batch sampler; scramble augmentation 15–20%; field dropout 10%; lr 2e-5; 1 epoch; fp16; checkpoint every 500 steps.
4. Encode B + test with the fine-tuned model; run the acceptance test (recall@k per country, scrambled) → `acceptance.json` → **Gate G2**.

### 6.7 NB05: blocking — current kernel `er-nb05-blocking-sparse` (driver `nb05_blocking/driver.py` + define `NO_DENSE`)

1. **Dense retrieval is not used** (Qwen3-0.6B ~400 rec/s on 2×T4 → ~16 h for all records; memory §5/§9). The `er-nb05-blocking` kernel (with NB03 dense lists) stays in the code for a future faster embedder.
2. Sparse channels, all within-country by label equality (EDA E5); each = hashed TF-IDF + df-capped index + chunked sparse top-k:
   - `name_char` / `addr_char`: char_wb 3–4-grams (k 20 / 15, df cap 2000);
   - `name_tok`: rare norm + fold name tokens (k 20, df cap 200);
   - `num_key`: house number + first street token (k 20, df cap 200);
   - `name_pair` / `addr_pair`: unordered fold-token pairs (k 30, df cap 2000) — needed because names are built from a shared synthetic vocabulary (memory E13);
   - `cross_pair` (v5): name token × address token pairs (k 30, df cap 2000);
   - look-alike digits folded in token/pair keys (v5).
3. Queries: 250k S1 per half (seeded) against the full train pool; all test S1. Both train halves share one pass over the pool.
4. Union → cheap features (channel scores/ranks, gaps to the S1's best, union size) → LightGBM pre-ranker trained on Half A (`pruner.txt`) → top-50 per S1, in 200k-S1 chunks (memory bound).
5. Blocking report on A and B (per country, per channel found/unique, union vs pruned) → **Gate G1**; `missed_B.tsv` (200 missed true B pairs).
6. **Current version: v4** (B pair recall 0.9585); v5 running. Test-side df statistics come from test's own token counts (`per_run`) — see the Q3 blocker in progress.md.

### 6.8 NB06: features + stage-1

1. `features.build_pair_features` for A, B, test (multiprocessing; cached token relations).
2. Stage-1 LightGBM: 5-fold group-k-fold on A → A OOF; full-A model → B and test predictions. **No fine-tuned-embedding features here.**
3. Quick B score with calibration + decoder → basis for **SAFETY SUBMISSION #1** (write the TSVs, run `check_outputs` + validator).

### 6.9 NB07: judge training (runner R2)

1. Band from A OOF [0.05, 0.95] + 10% confident; balance 1:2; build evidence-augmented prompts (order swap 50%, address drop 5%, scramble 5–10%).
2. Load Qwen3-4B in 4-bit NF4 (fp16 compute), LoRA r16/α32 on all linear layers, gradient checkpointing, SDPA.
3. Measure throughput over 50 steps, then cut the dataset to fit ≤ 3 h; train 1 epoch; save the adapter every 200 steps.
4. Quick AUC on a held-out slice of A.

### 6.10 NB08: judge inference (T4×2)

1. Band on B and test: stage-1 p ∈ [0.10, 0.90], sorted by |p − 0.5|, capped by the time budget.
2. Two processes (GPU 0 and GPU 1), batch 16–32, left padding; p = softmax over the "Yes"/"No" logits at the answer position; optional order-swap averaging.
3. Save `judge_{B,test}.parquet` → **Gate G4** evaluated in NB09.

### 6.11 NB09: combine, decode, validate (CPU)

1. Collective features from stage-1 predictions.
2. Combiner: 5-fold group-k-fold on B → OOF; full-B model → test.
3. Calibration (isotonic; Platt fallback); λ grid.
4. Exclusivity mode (none / soft / hard) chosen on B.
5. Decoder → predicted sets; B F0.5; LOCO-mean; scrambled test; bootstrap CIs; ablation rows.
6. Write `output/matching_results.tsv` and `output/candidate_pairs.tsv` (from the exact scored test candidate table); run `check_outputs` + the official validator.
7. Save `er-final_vN`; log in `progress.md`.

### 6.12 NB10: error analysis (loop)

1. Load B OOF predictions + GT; list entities with F0.5 < 1.
2. Bucket: singleton false match / chain branch / multi-claim / blocking miss / heavy noise / other.
3. Show the top 3 buckets with counts and 5 examples each (raw records + key features + probabilities).
4. Propose a fix; log it in `progress.md` → Error-analysis log.

### 6.13 NB11: end-to-end reproducibility (internet OFF)

1. `python -m src.predict --config src/configs/default.yaml` using only `er-code`, `er-models`, the trained artifacts and `er-data`.
2. Assert that the output TSVs are byte-identical (or set-identical) to NB09's final output.
3. Record the runtime for the README.

---

## 7. Configuration (`src/configs/default.yaml`) — key blocks

```yaml
seed: 42
paths:
  data_dir: /kaggle/input/er-data/student_resource/dataset
  models_dir: /kaggle/input/er-models
  work_dir: /kaggle/working
stats_mode: per_run            # per_run | train_only (fallback, plan §9.4)
split: {a_frac: 0.55, stratify: [country, match_bucket]}
embed:
  model: qwen3-emb-0.6b
  max_len: 96
  batch: 256
  instruction: "Instruct: Represent this business record to find records of the same business\nQuery: "
finetune:
  lr: 2.0e-5
  epochs: 1
  batch: 64
  hard_negs: 3
  scramble_frac: 0.2
  field_dropout: 0.1
blocking:
  dense_k: 30
  reverse_k: 5
  tfidf_name_k: 20
  tfidf_addr_k: 15
  rare_token_cap: 20
  prune_top: 50
  within_country: auto         # decided by EDA E5
scale_guard:                   # turned on if EDA shows millions of records (Q8)
  enabled: auto
  mrl_dim: 256                 # truncate Qwen3 embeddings (Matryoshka) for blocking
  faiss_index: ivf             # flat | ivf ; ivf nlist ≈ 4·sqrt(N)
  prerank_top: 20              # cheap vectorised LightGBM keeps top-20 per S1; this is candidate_pairs.tsv
stage1: {num_leaves: 63, lr: 0.05, min_data_in_leaf: 50, folds: 5, monotone: auto}
judge:
  model: qwen3-4b
  lora_r: 16
  lora_alpha: 32
  max_len: 384
  lr: 1.0e-4
  train_band: [0.05, 0.95]
  infer_band: [0.10, 0.90]
  time_budget_min: 45
  order_swap_tta: true
combiner: {num_leaves: 31, folds: 5, calibration: isotonic}
exclusivity: auto              # none | soft | hard, chosen on B
decoder: {lambda_grid: [0.7, 0.85, 1.0, 1.15], max_n: 20, min_p: 0.01}
```

---

## 8. Parallel execution plan (who runs what, when)

| Time (IST) | GPU runner R1 | GPU runner R2 (judge) | CPU track (local / CPU notebook) |
|---|---|---|---|
| 15:30–17:00 | NB00 → NB03 | — | Repo skeleton, io_utils, metrics, NB01 |
| 17:00–19:00 | NB04 (after NB02's split exists) | — | NB02, normalise/stats code |
| 19:00–21:30 | NB04 acceptance; spare | — | NB05, explain_diff, features |
| 21:30–22:30 | — | Prepare NB07 inputs | NB06 → **safety #1** |
| 22:30–02:00 | — | **NB07 judge training** | NB09 v0: collective, combiner, exclusivity, decoder; NB10 round 1 |
| Morning (27 Sep) | spare / reruns | **NB08 judge inference** | **safety #2**; NB09 v1 (+ judge) |
| Until 19:30 IST (freeze) | — | — | NB10 rounds 2–3; **main submission** (~16:00); NB11; packaging |
| 19:30–21:00 IST | — | — | Only packaging, docs, re-validation; **last upload ≤ 21:00 IST** |

---

## 9. How everything fits together (end-to-end workflow)

```
 ┌──────────── Your laptop (repo, git) ────────────┐        ┌────────────── Kaggle ──────────────┐
 │ Claude Code reads CLAUDE.md + context files      │        │ Notebooks NB00–NB11 (thin runners)  │
 │ → writes src/ modules, tests, notebooks          │ upload │  attach: er-data, er-code,          │
 │ → runs CPU steps / unit tests locally            │ ─────► │          er-models, upstream er-*   │
 │ → updates progress.md / memory.md                │ er-code│  save outputs as versioned datasets │
 └──────────────────────────────────────────────────┘        └──────────────────┬──────────────────┘
            ▲                                                                    │ download
            │ output/*.tsv                                                       ▼
 ┌──────────┴─────────────────────────────────────────────────────────────────────────────┐
 │ Laptop: check_outputs.py + official validator → git commit + tag sub-NN                   │
 │ → upload matching_results.tsv on Unstop (≤ 5/day) → log in progress.md                    │
 └───────────────────────────────────────────────────────────────────────────────────────────┘
            │ at the end
            ▼
 Zip: output/ + code/business_entity_resolution/{src, README.md, requirements.txt} + Documentation_template.md
 + 1–2-page approach document (uploaded where Unstop asks for it)
```

Step by step:
1. **Write code locally** with Claude Code (it reads the context files every session).
2. **Upload** `code/business_entity_resolution/` to Kaggle as a new version of `er-code`.
3. **Run the next notebook** on Kaggle with the accelerator and inputs listed in §6; save its output as the next `er-*_vN` dataset.
4. **Report back** the printed metrics to Claude Code, which records them in `progress.md`.
5. **When a submission is ready,** download `output/*.tsv`, run both checks, commit and tag, and upload `matching_results.tsv` on Unstop (from one machine only; simultaneous logins are not allowed).
6. **At the end,** build the zip and the two documents using `memory.md` §11.