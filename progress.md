# progress.md — Live task board

> Updated at the end of every task (see `CLAUDE.md` §2). Timestamps in IST.
> **Last updated:** 2026-09-26 15:45 IST

---

## Current focus

**Phase 0: Setup.** Next: **Prompt 1** (repo + Kaggle automation + core utils + EDA + model download). The user does the manual pre-flight first (dataset, Kaggle credentials, `CLAUDE.local.md`).

## Blockers

- [ ] Dataset not yet available (train/test TSVs + `utils/validate_submission.py`) → needed for everything
- [x] Deadline time known: **27 Sep 23:59 IST**; freeze 19:30 IST; final upload by 21:00 IST
- [ ] Google Form questions to organisers: Q3 (test-time statistics), Q5 (which submission counts for private), Q6 (doc length). Not blocking; fallbacks exist

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
- [ ] Repo skeleton created as a **git repo** (`code/business_entity_resolution/` per `architecture.md` §3; all code under `src/`)
- [ ] `src/io_utils.py` (safe TSV loaders/writers + assertions)
- [ ] `src/metrics.py` (official F0.5 scorer + unit test = 0.714 example; LOCO harness; bootstrap)
- [ ] `src/check_outputs.py` (pre-flight checks)
- [ ] `configs/default.yaml` (at `src/configs/default.yaml`)
- [ ] Upload the dataset to Kaggle as `er-data`; upload code as `er-code`
- [ ] **NB00**: download Qwen3-Embedding-0.6B, Qwen3-4B (and Reranker-0.6B fallback) → Kaggle dataset `er-models`

### Phase 1: Data understanding — Prompt 1 (target: 16:30–17:30)
- [ ] **NB01 EDA**: E1–E12 + shortcut check → write the findings into `memory.md` §6
- [ ] Measure dataset scale (Q8) → decide whether the scale guard is on (`memory.md` §5)
- [ ] Decide the one-owner mode (E4) and within-country blocking (E5) → `memory.md` §5
- [ ] Measure the all-empty floor on train

### Phase 2: Foundations — Prompt 2 (target: 17:30–19:00)
- [ ] `src/normalize.py` (Step 2) + unit tests (accents, ligatures, initials, numbers, landmarks, fold)
- [ ] `src/corpus_stats.py` (IDF global/local, suffix-likeness, street-type-likeness; train-only fallback flag)
- [ ] `src/split.py` (A/B stratified by country × match-count bucket)
- [ ] **NB02** run → `er-norm_v1`
- [ ] **NB03** frozen embeddings (GPU) → `er-emb_v1`

### Phase 3: Blocking — Prompt 2 (target: 19:00–21:00)
- [ ] `src/blocking.py`: dense, reverse dense, TF-IDF name/address, number/postcode key, rare-token key; cheap-score pruning to top-50
- [ ] Blocking report: pair recall, entity-complete recall, RR, candidates/S1, per-channel unique contribution, per country, scrambled
- [ ] **Gate G1**: pair recall ≥ 0.99 on B (hard minimum 0.95)
- [ ] **NB04** fine-tune embedder on A (GPU) → `er-embedder_v1`
- [ ] **Gate G2**: fine-tuned beats frozen on recall@20 in both countries

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
| G1 | Blocking pair recall ≥ 0.99 (min 0.95) on B | pending | |
| G2 | Fine-tuned embedder beats frozen on recall@20, both countries | pending | |
| G3 | Safety #1 PASS and B F0.5 > floor (target ≥ 0.985) | pending | |
| G4 | Judge adds a LOCO gain > noise | pending | |
| G5 | Final beats safety #2 on LOCO-mean | pending | |

---

## Experiments log

| # | Time (IST) | Config / change | B F0.5 | LOCO-mean | Scrambled drop | Kept? | Note |
|---|---|---|---|---|---|---|---|
| — | | all-empty floor | | | | | |

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