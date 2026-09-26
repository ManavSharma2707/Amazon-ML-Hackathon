# progress.md — Live task board

> Updated at the end of every task (see `CLAUDE.md` §2). Timestamps in IST.
> **Last updated:** 2026-09-27 03:35 IST

---

## Autonomous run state (PROMPT FINAL)

- **Updated:** 2026-09-27 04:10 IST
- **Stage:** S0 -> S1 (no submission yet: sub-01 pending NB06 -> NB09a)
- **Iteration #:** 0
- **Current best config:** none yet (sub-01 = stage-1 + isotonic + decoder/threshold, NB09a)
- **Best B F0.5 / LOCO-mean / scrambled drop:** — / — / —
- **Running notebooks:** er-nb06-features-stage1 v1 (R1, pushed ~03:15 IST); er-nb05-blocking-sparse v5 (R1, since ~02:55 IST)
- **Done:** er-nb07-judge-train-dry v1 (R2) COMPLETE 03:35 IST (dry run clean; 3.2 s/step at 4 ex/step)
- **Next action:** NB06 done -> push NB09a + NB06b (R1); fetch TSVs -> checks -> tag sub-01 -> UPLOAD READY; relay judge inputs -> NB07 full (R2); meanwhile write collective.py / combiner.py / NB09
- **Uploads used today (27 Sep):** 0 (assumed — not reported by the user)

## Current focus

**Prompt 3 in progress (03:35 IST).** Code done + unit-tested + local sample smoke runs: `explain_diff`, `features`, `stage1`, `decoder`, `exclusivity`, streaming `check_outputs`, `judge_data`, `judge_train`; kernels NB06 (`er-nb06-features-stage1`), NB09a (`er-nb09a-decode-submit`), NB06b (`er-nb06b-judge-data`), NB07 (+dry). **Running:** NB06 v1 on R1 (pushed ~03:15 IST, on NB05 v4 or v5 output — whichever Kaggle mounted), NB07 dry on R2 (sample-built `er-judge-inputs` v1). **Next:** NB06 done -> push NB09a + NB06b on R1 -> fetch TSVs -> check_outputs + validator -> commit + tag sub-01; relay real judge inputs -> NB07 full on R2. Local disk was full (freed 2.2 GB of stale upload zips; ~2 GB free).

**Prompt 2 (foundations + blocking) done except the G1 target.** NB02 v1 complete. Blocking = sparse only (dense retrieval infeasible at ~400 rec/s). `er-nb05-blocking-sparse` **v4 is the current usable candidate set** (superseded) → **v5 is the current candidate set** (B pair recall 0.9733: US 0.982 / India 0.959; entity-complete 0.916; all test S1 covered, 50 cands/S1; commit `2801f74`). **Next (Prompt 3): features + stage-1 + SAFETY SUBMISSION #1 — there is still no submission.** Pending user decision: train-only vs per-run test statistics (Q3).

## Blockers

- [x] **Kaggle write-auth (resolved 2026-09-26 17:45 IST).** Root cause: the `KGAT_...` values are Kaggle's newer API-token format, not classic kaggle.json keys — they must go through `KAGGLE_API_TOKEN`/`access_token`, not `kaggle.json`'s `key` field (see `memory.md` pitfalls). Fixed by upgrading to `kaggle>=2.0`, switching both runners' credential files to `access_token`, and working around a Windows path-handling bug in the CLI's upload commands (`tools/kaggle_ops.py` now runs uploads with `cwd` set to the target folder + `-p .`). Verified end-to-end on both runners with a throwaway create+delete before touching real data.
- [x] **Notebook self-sufficiency + mount-path issues (resolved 2026-09-26 ~21:00 IST).** NB01 initially broke on `sys.path.insert` + `from src import` (this Kaggle environment mounts datasets at `/kaggle/input/datasets/<owner>/<slug>/`, not the classic path). Fixed by making every notebook fully self-contained via `tools/bundle_kernel.py` (see `architecture.md` §6.1). Two follow-on bugs also fixed: missing `rapidfuzz` on an internet-OFF kernel (swapped to stdlib `difflib`), and a major perf bug in E9 (`isin()` against a 7.6M-item set called once per loop iteration instead of once — cut an ~80 min projected runtime to ~8 minutes actual).
- [x] Dataset present locally (`student_resource/`, 2.4 GB, confirmed 2026-09-26); **local RAM only ~750-822 MB free of ~8 GB** — all CPU-heavy work (EDA done, future steps too) runs as a Kaggle CPU kernel, never locally. Local machine is code-authoring + unit tests + git only.
- [x] Deadline time known: **27 Sep 23:59 IST**; freeze 19:30 IST; final upload by 21:00 IST
- [ ] Google Form questions to organisers: Q3 (test-time statistics), Q5 (which submission counts for private), Q6 (doc length). **Q3 now matters:** NB02 `stats_test` and NB05 TF-IDF df/caps on test use test's own (unlabelled) token counts; a `train_only` path is not yet implemented in NB05 — user asked to decide (2026-09-27 01:15 IST)
- [ ] **No submission exists yet** (plan had safety #1 at 26 Sep 22:30). Top priority for Prompt 3
- [x] v5 completed OK, so its output (current) is the one `kernel_sources` attaches. Rule stays: record the last good config's commit (v5 = `2801f74`)

## Submission budget (max 5 per day, resets daily; unused ones are lost)

| Day | Used | Planned uploads |
|---|---|---|
| 25 Sep | ? (check Unstop) | — |
| 26 Sep | 0 / 5 | Safety #1 (~22:30); 1 spare for a format/bug fix |
| 27 Sep | 0 / 5 | Safety #2 (morning); main (~16:00); up to 3 spares for fixes; **last upload ≤ 21:00 IST** |

---

## Phase checklist

### Phase 0: Setup — Prompt 1 (target: 15:45–16:30)
- [x] Master plan written (`ER_Hackathon_Master_Plan.md`)
- [x] Context files created (`CLAUDE.md`, `memory.md`, `progress.md`, `architecture.md`)
- [x] Compatibility check against the problem statement + official guidelines PDF → `memory.md` §11
- [ ] Send Google Form questions Q3, Q5, Q6 (`memory.md` §10)
- [x] Repo skeleton created as a **git repo** (`code/business_entity_resolution/` per `architecture.md` §3; all code under `src/`) — commit `28628ed`
- [x] `src/io_utils.py` (safe TSV loaders/writers + assertions) — commit `2b19ebf`
- [x] `src/metrics.py` (official F0.5 scorer + unit test = 0.714 example; bootstrap_diff; LOCO/blocking/error-bucket helpers stubbed, pending Prompt 2+) — commit `2b19ebf`
- [x] `src/check_outputs.py` (pre-flight checks) — commit `2b19ebf`
- [x] `configs/default.yaml` (at `src/configs/default.yaml`) — commit `2b19ebf`
- [x] Upload code as `er-code` to both runners — done 2026-09-26 17:5x IST (R1 also holds `er-data`, R2 added as collaborator ~20:05 IST)
- [x] Upload the dataset to Kaggle as `er-data` — done on R1 (single pre-zipped upload, 2404 MiB → 1010 MiB, ~46 min); R2 has collaborator access rather than its own copy (avoids a duplicate ~1 GB upload)
- [x] **NB00**: download Qwen3-Embedding-0.6B, Qwen3-4B, Qwen3-Reranker-0.6B → completed successfully on **both** runners 2026-09-26 ~19:45 IST; all three confirmed Apache-2.0 via the actual model card `license:` field (`LICENSES.md` in each run's output). No HF token needed (rate-limit warning only, no errors)
- [x] `src/eda.py` (E1-E12 + shortcut check, per-stage logging) + `tools/bundle_kernel.py` (self-contained kernel bundling) + `code/.../src/tests/test_eda.py` — 27/27 tests passing; commits `2c905eb`, `90abada`

### Phase 1: Data understanding — Prompt 1 (target: 16:30–17:30)
- [x] **NB01 EDA**: E1–E12 + shortcut check → real findings written into `memory.md` §6 (completed 2026-09-26 ~21:37 IST on R1, full dataset, ~8.2 min runtime after fixing an E9 perf bug)
- [x] Measure dataset scale (Q8) → decide whether the scale guard is on (`memory.md` §5) — **ON**, ~2.4 GB / millions of rows per file
- [x] Decide the one-owner mode (E4) and within-country blocking (E5) → `memory.md` §5/§6 — **E4: HARD** (0/7,638,365 shared), **E5: within-country blocking YES** (0/7,638,365 cross-country); both written into `default.yaml`
- [x] Measure the all-empty floor on train — **0.0558** (memory.md §2, §6 E2)

### Phase 2: Foundations — Prompt 2 (target: 17:30–19:00)
- [x] `src/normalize.py` (Step 2) + unit tests (accents, ligatures, initials, numbers incl. bis/ter, postcodes by shape+position, landmarks, fold, **stdlib Brahmic-script romanisation**, domains, `null` tokens) — commit `02851b6`
- [x] `src/corpus_stats.py` (per-country + global df, IDF, suffix-likeness, street-type-likeness; incremental `TokenCounter`; `stats_mode` flag in config) — commit `02851b6`
- [x] `src/split.py` (A/B stratified by country × match-count bucket; pool records follow their S1; unmatched = shared) — commit `02851b6`
- [x] **NB02** run → `er-nb02-normalize` v1 (R1, CPU, 57 min, 2026-09-26 23:05 IST): row counts = EDA E1; A 1,213,752 / B 993,069 S1, singleton rate 0.05585 in both; name_empty 0%; romanised India S2 23.5% / S3 13%; house number 90–100%, postcode < 1.3%. Report: `reports/raw/nb02/metrics.json`
- [ ] **Fix before Prompt 3:** suffix-likeness top list is dominated by typo variants (`limitet`, `drve`) — the frequency sigmoid is too weak; add a min-frequency floor before using it as a feature
- [x] **NB03** dry runs (`er-nb03-embed-dry` v1–v3): code clean on 2×T4, embeddings good (pos cos 0.85 vs rand 0.39), but **~400 rec/s** → full run (24M records, ~16 h) **not run**. Dense retrieval dropped from blocking (memory §5)

### Phase 3: Blocking — Prompt 2 (target: 19:00–21:00)
- [x] `src/blocking.py`: dense, reverse dense, char TF-IDF name/address, rare-token key (norm+fold), house-number+street key; LightGBM pre-ranker (trained on A) pruning to top-50 — commit `ad14726`
- [x] Blocking report (pair/entity-complete recall union vs pruned, RR, cands/S1 mean+p95, per-channel found/unique, per country; test counts) + `missed_B.tsv`. Scrambled: sparse char/pair channels are permutation-invariant by construction; no dense channel to test
- [~] **Gate G1**: v5 B pair recall **0.9733** — passes hard min 0.95, misses 0.99 target (next levers: larger union re-ranked by stage-1; empty-address and native-script misses)
- [~] **NB04** code + kernels written (`nb04_finetune`, `_dry`, `nb04b_ft_encode`), unit-tested, **not pushed**: moot while dense retrieval is infeasible (re-encoding test alone ~8 h)
- [~] **Gate G2**: N/A for now (no dense channel in blocking). Could return as a pair *feature* on the ≤ 20 pre-ranked candidates if GPU time allows

### Phase 4: Features + Stage-1 + safety submission #1 — Prompt 3 (target: 21:00–23:30)
- [ ] `src/explain_diff.py` (Step 7) + unit tests on the relation tests
- [ ] `src/features.py` (groups A–H)
- [ ] `src/stage1.py` (A OOF + predictions on B/test)
- [ ] Quick calibration + `src/decoder.py` (expected-F0.5)
- [ ] **SAFETY SUBMISSION #1** (stage-1 + calibration + decoder)
- [ ] **Gate G3**: validator PASS and B F0.5 > floor (target ≥ 0.985)

### Phase 5: Collective + combiner — Prompt 4 (target: 23:30–02:00, safety #2 by 09:00)
- [ ] `src/collective.py` (competition + sibling features)
- [ ] `src/combiner.py` (B OOF, isotonic/Platt, λ)
- [ ] `src/exclusivity.py` (none / soft / hard; chosen on B)
- [ ] LOCO + scrambled-letter test + bootstrap CI
- [ ] Error-analysis loop, round 1 (top 3 buckets + fixes)

### Phase 6: LLM judge (stretch) — training started in Prompt 3 on runner R2, inference in Prompt 5
- [ ] `src/judge_data.py` (evidence-augmented prompts; band from A OOF)
- [ ] **NB07** QLoRA train Qwen3-4B (≤ 3 h, checkpoints) → `er-judge_v1`
- [ ] **NB08** judge inference on B + test (2× T4) → `er-judge-scores_v1`
- [ ] **Gate G4**: judge AUC on band > stage-1 AUC on band; combiner LOCO gain > noise

### Phase 7: Final model — Prompt 5 (09:00–17:00; main upload by ~17:00)
- [ ] **SAFETY SUBMISSION #2** (combiner without judge)
- [ ] Combiner v1 (+ judge if G4 passes)
- [ ] Error-analysis loop, rounds 2–3
- [ ] **Gate G5**: final beats safety #2 on LOCO-mean
- [ ] **MAIN SUBMISSION**

### Phase 8: Packaging — Prompt 6 (17:00–19:30 freeze; final upload by 21:00 IST; window closes 23:59 IST)
- [ ] `src/scripts/run_train.sh`, `src/scripts/run_predict.sh`; clean-run test
- [ ] `requirements.txt` pinned (from the Kaggle run)
- [ ] `README.md` (commands, runtimes, licences, no-external-data statement, offline inference)
- [ ] Docstrings/comments on every function (official requirement)
- [ ] **1–2-page approach document** (guidelines) — ML approach, models, experiments, conclusion
- [ ] `Documentation_template.md` filled (full version: ablation table, blocking table, methodology)
- [ ] `check_outputs.py` + official validator PASS
- [ ] Walk through `memory.md` §11 checklist
- [ ] Zip `<team_name>_submission.zip` with the correct structure (no dataset, no context files, no large weights)

---

## Gates status

| Gate | Condition | Status | Result |
|---|---|---|---|
| G1 | Blocking pair recall ≥ 0.99 (min 0.95) on B | **partial** | v2 0.724 → v3 0.952 → v4 0.9585 → **v5 0.9733** (US 0.982 / India 0.959); ≥ hard min, < 0.99 target |
| G2 | Fine-tuned embedder beats frozen on recall@20, both countries | N/A | dense retrieval infeasible at ~400 rec/s on 2×T4; NB04 not run |
| G3 | Safety #1 PASS and B F0.5 > floor (target ≥ 0.985) | pending | |
| G4 | Judge adds a LOCO gain > noise | pending | |
| G5 | Final beats safety #2 on LOCO-mean | pending | |

---

## Experiments log

| # | Time (IST) | Config / change | B F0.5 | LOCO-mean | Scrambled drop | Kept? | Note |
|---|---|---|---|---|---|---|---|
| B1 | 2026-09-27 00:25 | Blocking sparse v2 (name/addr char, rare-token, num key; LGBM pruner top-50), `er-nb05-blocking-sparse` v2 | B pair recall **0.724** (US 0.764 / India 0.664), union 0.726, entity-complete 0.498, 29.3 cands/S1 | — | — | no | G1 FAIL; misses were easy pairs → synthetic shared vocabulary (memory E13) |
| B2 | 2026-09-27 01:10 | + name_pair / addr_pair token-pair channels (k 20, df cap 500), `er-nb05-blocking-sparse` v3 | B pair recall **0.952** (union 0.953), entity-complete 0.867, 48.0 cands/S1 | — | — | yes | addr_pair unique 79.9k, name_pair 42.8k true B pairs; test side OOM-killed on the 48M-pair India union → v4 chunks it |
| B3 | 2026-09-27 02:45 | chunked test union; pair k 30 / df cap 2000, `er-nb05-blocking-sparse` v4 (67 min) | B pair recall **0.9585** (US 0.970 / India 0.942), union 0.964, entity-complete 0.881, RR 0.999995, 49.8 cands/S1 (p95 50) | — | — | superseded by v5 | test: France 49.9 / India 49.9 / US 49.8 cands/S1, 0% S1 without candidates. Missed-B buckets: 65 both-fields-shared (generic/leetspeak), 53 name broken, 49 empty address, 26 pruned out |
| B4 | 2026-09-27 03:20 | + cross name x address pairs, look-alike digit folding, `er-nb05-blocking-sparse` v5 (96 min, commit `2801f74`) | B pair recall **0.9733** (US 0.982 / India 0.959), union 0.978, entity-complete 0.916, RR 0.999995, 50.0 cands/S1 | — | — | **current** | cross_pair unique 11.6k; test 50 cands/S1 in all 3 countries, 0% without candidates |
| — | 2026-09-26 21:37 | all-empty floor (train, full dataset) | 0.0558 | — | — | baseline | Singleton rate; nearly identical US (0.05583) vs India (0.05588) |

---

## Submissions log

| # | Time (IST) | Description | Git tag / commit | B F0.5 | LOCO-mean | Public LB | Validator | File version |
|---|---|---|---|---|---|---|---|---|
| — | | | | | | | | |

---

## Error-analysis log

| Round | Time (IST) | Top buckets (count) | Fix applied | Gain |
|---|---|---|---|---|
| — | | | | |