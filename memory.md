# memory.md — Durable project memory

> Long-lived facts, rules, decisions and findings. Update only with information that stays true across sessions. Live task status lives in `progress.md`.
> Timestamps in IST (Asia/Kolkata).

---

## 1. Project facts

| Item | Value |
|---|---|
| Competition | Business Entity Resolution Challenge |
| Official name | **Amazon ML Challenge 2026** (Unstop), Round 1 = 72-hour hackathon |
| Challenge window | **25 Sep 2026 00:00 IST → 27 Sep 2026 23:59 IST** (official guidelines PDF; Unstop shows the same end as 18:29 UTC) |
| Freeze time | **27 Sep 19:30 IST.** Final upload target **≤ 21:00 IST** (Unstop gets slow near the deadline) |
| Submission limit | **Max 5 leaderboard submissions per day**, over 3 days; the submit button is disabled after that. Unused submissions do not carry over |
| Leaderboards | Public (subset of test) + Private (rest). **Shortlisting uses performance across both**, so the public score matters too |
| Tie-break | Unstop ranks on the best submission score, then **earlier submission time** wins ties |
| Artefacts for the best solution | (a) **1–2-page document** (ML approach, models, experiments, conclusion); (b) source code for experiments, training and inference **with comments/docstrings describing the functions**. Keep **version history of all submissions** |
| Top-100 follow-up | Top 100 teams submit: methodology, candidate generation/blocking strategy, model architecture and feature engineering, other info (= `Documentation_template.md`, no page limit per the problem statement), plus the final zip package |
| Grand finale | Top 10 teams present to Amazon scientists (virtual, 7 Oct 2026) |
| Integrity | Cheating, plagiarism or multiple IDs = instant disqualification. **Simultaneous logins are not allowed:** one laptop/desktop per participant on Unstop |
| Team size / roles | 3–4 members (competition rule). Planned roles: P1 CPU track, P2 GPU track, P3 Validation and packaging (+ P4 helps P1) |
| Compute | Kaggle: P100 or 2× T4 (16 GB each), ~30 GPU h/week, 9–12 h sessions, fp16 only (no bf16), no flash-attn 2. Runner details: `CLAUDE.local.md` (not committed) |
| Leaderboard context (2026-09-26) | Top public scores ≈ **0.99**; the competition is decided in the last 1%. Private board includes France and **both boards count** for shortlisting |

## 2. Problem summary

- Three sources. **S1 = deduplicated reference.** For each S1 entity, output all matching S2/S3 records (0..many).
- Fields: `entity_id`, `business_name`, `business_address`, `country`. The source is given by the ID prefix.
- Train: US + India, with ground truth. Test: US + India + **France (unseen)**.
- **Metric: macro F0.5 per S1 entity.**
  - Truly empty + predicted empty = 1.0.
  - Truly empty + predicted anything = 0.0.
  - Has matches + predicted empty = 0.0.
  - Count form: `1.25·TP / (1.25·TP + 0.25·FN + FP)`, so an FP costs 4× an FN.
- **Floor:** all-empty submission score = singleton rate (fill in below once measured).

## 3. Rules (hard constraints)

1. No external APIs, databases, geocoders, registries, internet augmentation. **No libpostal.**
2. Only the provided training data may be used for training. **Nothing is trained on test data.**
3. Models: MIT/Apache-2.0, ≤ 8B params each. They run **locally** (weights from local folders or Kaggle datasets; inference must work with internet OFF).
4. Country is an open set: no hard-coding, filtering or one-hot. Only `same_country` (equality).
5. Output: one row per test S1; S2-/S3- IDs only, existing in test; no duplicates; matches ⊆ candidates; `candidate_pairs.tsv` = exactly the scored set.
6. No shortcut features (entity_id numbers, row order).
7. Avoid GPL libraries (e.g., `unidecode`).

## 4. Model and licence registry

| Model / library | Licence | Params | Use | Status |
|---|---|---|---|---|
| Qwen/Qwen3-Embedding-0.6B | Apache-2.0 | 0.6B | Frozen + fine-tuned embeddings | planned |
| Qwen/Qwen3-4B | Apache-2.0 (verify on model card) | 4B | LLM judge (QLoRA) — stretch goal | planned |
| Qwen/Qwen3-Reranker-0.6B | Apache-2.0 | 0.6B | Fallback judge (cross-encoder) | fallback |
| intfloat/multilingual-e5-base | verify on model card | ~0.3B | Fallback embedder | fallback |
| LightGBM | MIT | — | Stage-1, combiner | planned |
| rapidfuzz, faiss | MIT | — | Features, kNN | planned |
| sentence-transformers, transformers, peft, accelerate | Apache-2.0 | — | Training / inference | planned |
| bitsandbytes | MIT | — | 4-bit QLoRA | planned |

## 5. Key decisions (with reasons)

| When (IST) | Decision | Reason |
|---|---|---|
| 2026-09-26 15:10 | Country-agnostic design: word-free features for the tree models; no country feature except `same_country` | France is unseen; the private board decides |
| 2026-09-26 15:10 | "Explain-the-difference" features: token alignment + relation types (typo / prefix / skeleton abbreviation / initialism / split-join); rare **unexplained** tokens are the main negative signal | Mirrors the noise process; language-free |
| 2026-09-26 15:10 | Calibrated probabilities + **expected-F0.5 top-k decoder** per entity (includes the "empty" option) | Optimises the exact metric; handles singletons |
| 2026-09-26 15:10 | **One-owner rule** (each S2/S3 → at most one S1), if EDA E4 confirms | S1 is deduplicated; kills chain/branch false merges |
| 2026-09-26 15:10 | Data split: Half A (55%) trains embedder, stage-1 and judge; Half B (45%) trains combiner, calibration and decoder | Prevents leakage and over-confidence |
| 2026-09-26 15:10 | Blocking for A uses frozen embeddings; B and test use fine-tuned (if it passes gate G2) | A in-sample recall would be unrealistic |
| 2026-09-26 15:10 | Model selection by **LOCO-mean** + **scrambled-letter test**, never by the public leaderboard | Proxy for an unseen country |
| 2026-09-26 15:21 | **Error-analysis loop is the main improvement engine**; the LLM judge is a **stretch goal** gated by G4 | Leaderboard ≈ 0.99: gains come from the last 1% of errors |
| 2026-09-26 15:21 | Raised targets: B OOF F0.5 ≥ 0.985 at safety #1; blocking pair recall ≥ 0.99 | Leaderboard context |
| 2026-09-26 15:21 IST | Qwen models downloaded once and loaded from local Kaggle dataset `er-models`; inference with internet OFF | Compliance with the no-external-API rule |
| 2026-09-26 15:40 IST | **Submission policy:** only upload files we would accept as the final answer (no risky public-LB probes) | Unstop ranks on the best submission, and it is unclear which submission's private score counts (Q5) |
| 2026-09-26 15:40 IST | **Submission budget:** today ≤ 2 (safety #1 + 1 spare); tomorrow ≤ 5 (safety #2, main, 3 spares for fixes). Log each one with its git commit | 5/day cap; version history is required |
| 2026-09-26 15:40 IST | Write **two** documents: a 1–2-page approach summary (guidelines) and the full `Documentation_template.md` (problem statement, no page limit) | The two official sources ask for different lengths |
| 2026-09-26 15:40 IST | All code is written by us; **never copy code from public repos** of this challenge (several exist on GitHub/Kaggle) | Plagiarism = disqualification |
| 2026-09-26 15:40 IST | In the zip, **all code lives under `src/`** (configs, scripts and notebooks too); only `README.md` and `requirements.txt` sit next to it | Problem statement: "Put all source under src/" |
| 2026-09-26 15:40 IST | Scale guard: if records are in the millions, use MRL-truncated 256-d embeddings + FAISS IVF (not flat), and a cheap vectorised pre-ranker before the Python-heavy explain features. `candidate_pairs.tsv` = the set **after** the pre-ranker (the set the final model scores) | Keeps runtime inside Kaggle limits; matches the "last filtering stage" rule |
| 2026-09-26 17:15 IST | **Scale guard: ON** (Q8 resolved). Train: S1 2,206,821 rows, S2 5,034,616, S3 5,285,603. Test: S1 1,732,544, S2 4,887,273, S3 5,082,316. Raw dataset ~2.4 GB | File line counts measured directly; matches the validator script's own comment about a "~1.7M-entity test set" |
| 2026-09-26 17:15 IST | **All CPU-heavy work (EDA included) runs as a Kaggle CPU kernel, never locally** | Local machine has only ~822 MB free RAM of 7.9 GB total; pandas over millions of rows would not fit. Local machine is for code authoring, unit tests on small samples, and git only |

## 6. EDA findings (fill in during Step 1)

| ID | Question | Finding |
|---|---|---|
| E1 | Sizes S1/S2/S3 per split and country | Row counts (all rows, pre country split — see SS5 decision above): train S1 2,206,821 / S2 5,034,616 / S3 5,285,603; test S1 1,732,544 / S2 4,887,273 / S3 5,082,316. Per-country breakdown TBD (needs a Kaggle CPU kernel) |
| E2 | Singleton rate overall / per country (= all-empty floor) | TBD |
| E3 | #matches per S1 (S2 vs S3) | TBD |
| E4 | Any S2/S3 ID under two S1s? | TBD → decides the one-owner mode |
| E5 | Matches crossing countries? | TBD → decides within-country blocking |
| E6 | Share of S2/S3 matching nothing | TBD |
| E7 | Exact-name / exact-address rates among positives | TBD |
| E8 | Noise census per source (typo / abbrev / drop / reorder / landmark / missing postcode) | TBD |
| E9 | Hard negatives (high name similarity, non-match) | TBD |
| E10 | Empty / short fields; non-Latin scripts | TBD |
| E11 | Postcode formats | TBD |
| E12 | Test: country labels, sizes, character set | TBD |
| — | Shortcut check: do ID numbers or row order correlate with matches? (detect only, never use) | TBD |

## 7. Metrics definitions

- **B F0.5:** macro F0.5 over Half-B S1 entities using OOF predictions.
- **LOCO-mean:** combiner/calibration/decoder trained on B-countryX → scored on B-countryY, averaged over both directions. **Primary selection metric.**
- **Scrambled drop:** F0.5(B) − F0.5(B with a consistent random a–z permutation applied to all records). Smaller is better (structure over vocabulary).
- **Blocking:** pair recall, entity-complete recall, reduction ratio, mean / p95 candidates per S1, unique contribution per channel.
- **Gate noise:** bootstrap over S1 entities (1,000 resamples).

## 8. Conventions

- Timestamps: IST.
- Artifact naming: see `architecture.md` §5; versions `_v1`, `_v2`…, never overwrite.
- TSV read/write rules: see `CLAUDE.md` §5.
- Random seed: `42` (config `seed`).

## 9. Known pitfalls (keep adding)

- Reading TSV without `keep_default_na=False` turns "NA" (a business name, or Namibia's code) into NaN.
- Reading without `quoting=csv.QUOTE_NONE` lets a stray `"` swallow rows.
- Qwen3 needs `transformers >= 4.51` (older raises `KeyError: 'qwen3'`).
- T4: no bf16 → use fp16; watch for NaN loss (lower lr, grad clip).
- In-batch negatives: two positives of the same S1 in one batch = false negatives → sampler must enforce one S1 per batch.
- `skeleton_abbrev` can be too permissive for 2-letter tokens (e.g., "st"); keep the length-ratio constraint and a separate `n_short_abbrev` counter.
- Qwen3 chat template: use `enable_thinking=False` for the judge.
- `kaggle datasets list --mine` and `kaggle datasets download` do **not** validate credentials — both succeed even with a completely fake username/key. Only a write call (`datasets create`/`version`, `kernels push`) is a real auth test. Don't trust a `list`/`download` success as proof credentials work.
- **Kaggle's newer `KGAT_...` API token is not a classic kaggle.json key.** It's shown on kaggle.com/settings/api as "API TOKEN" and must be used via `KAGGLE_API_TOKEN` env var or a `~/.kaggle/access_token` file (plain text token, no JSON) — putting it in kaggle.json's `key` field 401s on every write call regardless of username. Needs `kaggle>=2.0` (2.2.4 confirmed working); 1.6.17 predates token auth. Resolved 2026-09-26 17:45 IST; see `CLAUDE.local.md` for the full fix.
- **kaggle==2.2.4 on Windows: `datasets create`/`version`/`kernels push` fail with `[Errno 2] No such file or directory`** if the upload folder's path has more than one segment (it string-concatenates the path with the filename to build a cache name, and any `/`/`\` in that path breaks it). Workaround: run with `cwd` set to the folder itself and pass `-p .`. `tools/kaggle_ops.py` does this everywhere.
- On this Windows machine, bare `python`/`python3` on PATH resolve to the Microsoft Store alias and fail with "Python was not found". Use the real interpreter path recorded in `CLAUDE.local.md`.

## 10. Open questions

Ask the organisers through the official **Google Form** linked in the guidelines (not by email; Unstop support won't answer decision questions).

| # | Question | Owner | Status |
|---|---|---|---|
| Q1 | Exact submission deadline time | — | **Resolved:** 27 Sep 23:59 IST |
| Q2 | Team members and git author identity | User | open |
| Q3 | Are rarity/suffix statistics computed on test files at inference (no labels, no training) OK? Fallback: train-only stats (plan §9.4) | User → Google Form | open |
| Q4 | Dataset available locally and as Kaggle dataset `er-data`? | User | open |
| Q5 | Which submission's score is used for the private leaderboard: best public, latest, or a selected one? | User → Google Form | open |
| Q6 | Documentation length: guidelines say 1–2 pages, problem statement says no page limit. Confirm that both a short summary and the full template are acceptable | User → Google Form | open |
| Q7 | Did the team receive AWS credits (SageMaker)? A g5 instance (A10G 24 GB, bf16) would be much faster than a Kaggle T4. Unverified; one participant repo mentions per-participant credits | User | open |
| Q8 | Actual dataset size. Decides whether the scale guard is needed | EDA E1 | **Resolved:** ~2.4 GB, millions of rows per file (see SS5 2026-09-26 17:15 IST) — scale guard ON |

## 11. Submission compatibility checklist (verified against the problem statement and official guidelines, 2026-09-26)

**Leaderboard file (`matching_results.tsv`, uploaded on Unstop):**
- [ ] Tab-separated; header exactly `source1_entity_id<TAB>matched_entity_ids`
- [ ] Exactly one row per test S1 entity, **France included**; no duplicate rows
- [ ] ID lists comma-separated, no spaces, no quotes, no duplicates; only `S2-`/`S3-` IDs that exist in the test set
- [ ] Empty field for predicted singletons
- [ ] Official validator prints PASS
- [ ] Counts toward the 5/day cap → logged in `progress.md` with the git commit hash

**Candidate file (`candidate_pairs.tsv`, in the zip):**
- [ ] Header exactly `source1_entity_id<TAB>candidate_entity_ids`; same row rules
- [ ] Equals the **exact set the final model scores** (the last filtering stage)
- [ ] Every matched ID appears in that entity's candidate list

**Zip `<team_name>_submission.zip`:**
- [ ] `output/matching_results.tsv` + `output/candidate_pairs.tsv`
- [ ] `code/business_entity_resolution/src/` (all code, incl. configs, scripts, notebooks), `README.md`, `requirements.txt` (pinned)
- [ ] `Documentation_template.md` at the zip root (filled; .md or .pdf)
- [ ] README: exact commands data → blocking → matching → output, hardware, runtimes, base-model IDs and licences, "no external data" and "offline inference" statements
- [ ] Code has docstrings/comments on every function (guidelines requirement)
- [ ] Does **not** include: the dataset, our context files (`CLAUDE.md`, `memory.md`, `progress.md`, `architecture.md`), secrets, large model weights

**Rules:**
- [ ] Models MIT/Apache-2.0, ≤ 8B (Qwen3-Embedding-0.6B, Qwen3-4B, LightGBM)
- [ ] No external lookups; models run locally; inference works with internet OFF
- [ ] Nothing trained on test data
- [ ] No hard-coded country values
- [ ] 1–2-page approach document ready