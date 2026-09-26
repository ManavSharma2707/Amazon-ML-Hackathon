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
- **Floor:** all-empty submission score = singleton rate = **0.0558** (measured on train, see SS6 E2; overall and per-country nearly identical).

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
| 2026-09-26 21:40 IST | **E4 confirmed → one-owner mode = HARD** (0/7,638,365 matched IDs shared between two S1s) | `exclusivity.py` should default to `hard`, not `auto`/`soft`, though still verify on Half B per master plan SS13 |
| 2026-09-27 00:35 IST | **Blocking v1 = sparse only (no dense retrieval); token-pair channels added** | Qwen3-0.6B encodes ~400 rec/s on 2×T4 → full-pool dense retrieval infeasible (pitfalls). Sparse v2 (char/rare-token/number key) gave B recall 0.72; pair keys lifted a scaled local proxy from 0.896 to 0.989 union recall |
| 2026-09-27 03:00 IST | **Gate G2 set to N/A; NB04 (fine-tuned embedder) not run** | Its only use was dense blocking for B/test, which needs re-encoding ~22M records (~15 h at ~400 rec/s). Revisit only as a pair-level feature on the pre-ranked ≤ 20 candidates |
| 2026-09-27 03:00 IST | **Candidate generation = `er-nb05-blocking-sparse`**: char TF-IDF name/addr, rare token, number key, name/addr token pairs, cross name x addr pairs (dense/reverse slots unused); LightGBM pre-ranker on Half A → top-50 | v5 B pair recall 0.9733 (US 0.982 / India 0.959), entity-complete 0.916; pair + cross-pair channels supply most unique recall |
| 2026-09-26 21:40 IST | **E5 confirmed → within-country blocking = YES** (0/7,638,365 pairs cross countries) | `blocking.py`'s `within_country` config flag should be set `true` (equality-based, no hard-coded values), not left `auto` |
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
| 2026-09-26 19:50 IST | **Every pushed notebook is fully self-contained** (src/ modules inlined via `tools/bundle_kernel.py` + `bundle_spec.json` + `driver.py`; no `sys.path`/dataset-mounted code, ever) — see architecture.md SS6.1 | NB01 broke on `sys.path.insert(0, "/kaggle/input/er-code")` + `from src import` even with `er-code` correctly attached, because this Kaggle environment mounts datasets at `/kaggle/input/datasets/<owner>/<slug>/`, not `/kaggle/input/<slug>/`. Self-contained kernels remove the dependency entirely instead of chasing mount conventions |
| 2026-09-26 19:50 IST | **GPU notebooks (NB03+) always get a fast/dry-run push before the full run** | GPU quota is scarce (~30 h/week); a crash found only after a multi-hour run is expensive, a dry run costs minutes. User instruction, recorded in CLAUDE.md SS5 |
| 2026-09-26 21:00 IST | **Every Kaggle kernel that can run >1-2 min gets per-stage, flushed progress logging** (start/end of each stage, chunked progress inside big loops) | NB01 ran 45+ minutes with zero visible output, making a stuck run indistinguishable from a slow-but-fine one. User instruction, recorded in CLAUDE.md SS5 (Logging) |
| 2026-09-26 21:51 IST | **Every kernel's output gets fetched into `reports/raw/<name>/` the moment it finishes, before doing anything else with it** | Read eda.json into a summary and deleted the temp download without saving the raw artifact to the repo first — user caught this. `reports/raw/` is git-ignored by design, so this costs nothing. Recorded in CLAUDE.md SS6 |
| 2026-09-27 03:10 IST | **Explain-the-difference alignment = greedy by relation strength** (exact first, then split_join/initialism spans, then greedy 1-1), not Hungarian; 2-letter abbreviations are the weak `short_abbrev` relation; numbers never match as typos | Tokens per field are few, greedy gives the same links and is ~10x faster in pure Python; "st"~"south" must stay weak |
| 2026-09-27 03:10 IST | **Group-A string features: Levenshtein replaces Damerau-Levenshtein; WRatio/partial/token_sort only on names** | Timed per pair: DL 8-27 us, WRatio on addresses 22 us vs ratio/JW/Lev ~1 us; features stay word-free |
| 2026-09-27 03:10 IST | **Stage-1 has no embedding features (group D) and `same_country` is dropped (constant)** | No full-data vectors (NB03/NB04 infeasible); blocking is within-country |
| 2026-09-27 03:10 IST | **NB06 scale guard: if projected full-feature time > 90 min, a cheap pre-ranker (meta + 8 fast rapidfuzz scores, LightGBM on Half A) keeps top-20 per S1; that set is the scored set = candidate_pairs.tsv** | ~250 us/pair Python features x ~111M pairs at top-50 = hours |
| 2026-09-27 03:10 IST | **Stage-1 trains on 150k Half-A S1s (a_s1_cap), B uses all 250k NB05 B queries** | LightGBM time on 4 CPU cores |
| 2026-09-27 03:40 IST | **Judge prompt omits the country field** (deviation from plan SS16.3) | All pairs are same-country; rule 5 allows only the equality flag |
| 2026-09-27 04:05 IST | **Combiner uses within-entity + sibling collective features but NOT cross-entity competition counts/margins** (plan SS17.1) | Half B queries only ~11-18% of train S1s while test has all S1s competing: claimant counts would shift train->test and LOCO cannot see it; cross-entity conflicts go to the one-owner rule |
| 2026-09-27 04:05 IST | **Combiner re-scores only pairs with stage-1 p >= 0.001 (p_floor); others keep p1, identically on test** | LightGBM time on ~5M B rows; true pairs below the floor are reported in NB09 metrics |
| 2026-09-27 04:10 IST | **Judge inference (NB08) scores B and test in one queue, most uncertain first, and cuts both at the same depth abs(p1 - 0.5)** | `judge_scored` must mean the same thing in the combiner's training data (B) and on test |

## 6. EDA findings (fill in during Step 1)

Run 2026-09-26 ~21:37 IST on Kaggle CPU kernel `nb01-eda` (R1), full train+test dataset (no sampling except where noted). Raw output: `reports/eda.json` (not committed; fetch via `kaggle kernels output` if needed again).

| ID | Question | Finding |
|---|---|---|
| E1 | Sizes S1/S2/S3 per split and country | **Train:** S1 2,206,821 (US 1,323,633 / India 883,188); S2 5,034,616 (US 3,016,817 / India 2,017,799); S3 5,285,603 (US 3,170,056 / India 2,115,547). **Test:** S1 1,732,544 (India 809,986 / US 663,106 / France 259,452); S2 4,887,273; S3 5,082,316 |
| E2 | Singleton rate overall / per country (= all-empty floor) | **Overall 0.0558** (India 0.05588, US 0.05583 — nearly identical across countries). **All-empty floor = 0.0558** |
| E3 | #matches per S1 (S2 vs S3) | Mean 3.46; percentiles p50=3, p75=5, p90=6, p95=6, p99=8, max=11. S2 supplies 48.4% of matches, S3 51.6% — roughly even |
| E4 | Any S2/S3 ID under two S1s? | **0 / 7,638,365 matched IDs (0.00000)** → **one-owner mode: HARD** (S1 is truly deduplicated in this data; no soft/none fallback needed) |
| E5 | Matches crossing countries? | **0 / 7,638,365 pairs (0.00000)** → **within-country blocking: YES** (equality-based, never hard-coded values) |
| E6 | Share of S2/S3 matching nothing | S2 26.6% unmatched, S3 25.4% unmatched — about a quarter of each source is pure noise/singletons from the matching perspective |
| E7 | Exact-name / exact-address rates among positives | (300k-pair sample) exact-name 15.8%, exact-address 7.4%, both-exact only 0.83%, name-only-noise 6.6%, address-only-noise 15.0% → **most positives have noise somewhere**; address noise slightly more common than name noise |
| E8 | Noise census per source (typo / abbrev / drop / reorder / landmark / missing postcode) | (300k-pair sample, name field, cheap heuristic — not full explain_diff.py) exact 15.8%, typo 28.3%, abbrev/drop 25.8%, reorder 5.9%, other/unclassified 24.2%. Typo and abbreviation/drop are the two biggest noise categories |
| E9 | Hard negatives (high name similarity, non-match) | 9/500 sampled S1 entities (1.8%) had a same-country non-match with name similarity ≥85 (difflib ratio). Examples are exactly the generic-chain-name pattern master plan risk R20 predicted (India "X Private Limited" / "X Enterprises Private Limited" collisions) — confirms number/unexplained-rare-word features and exclusivity matter |
| E10 | Empty / short fields; non-Latin scripts | S1: 0% empty name/address, 0.16% short names (<5 chars), **0% non-Latin**. **S2: 16.6% non-Latin, 3.4% empty address. S3: 13.0% non-Latin, 3.3% empty address.** S1 is clean; S2/S3 carry real non-Latin-script content (likely native-script India variants) — normalization/embeddings must handle this (risk R21) even though S1 alone looks Latin-only |
| E11 | Postcode formats | Naive shape-only regex (last 4-6 digit run) found US mostly 4-digit (614,255) then 5-digit (141,060); India mostly 4-digit (68,623) then 5-digit (2,441). **This is very likely picking up house/unit numbers, not real postal codes**, for a meaningful share of rows (US ZIPs are 5-digit, India PIN codes are 6-digit) — the naive last-digit-run heuristic is not reliable on its own; Step 8's number/postcode extractor will need position/context refinement, not just "last digit run" |
| E12 | Test: country labels, sizes, character set | France 259,452 (15.0%), India 809,986 (46.8%), US 663,106 (38.3%) of 1,732,544 test S1. **0% non-Latin in test S1** across all three countries (consistent with S1's E10 finding — S1 itself stays Latin-only even in the unseen France split) |
| E13 | Name/address vocabulary (found in blocking, 2026-09-27 00:30 IST) | **Names are combinations of a synthetic, shared vocabulary** (e.g. "Glypheus" starts many unrelated S1 names: Glypheus Capital / Municipals LLC / Platforms PLLC …). Single tokens and char n-grams are therefore common (pruned by any df cap), while **token pairs** ("glypheus|platforms", "238|houston") are rare. Consequences: rare-token / char-TF-IDF blocking alone fails (B pair recall 0.72); pair-key channels are needed; for features, single-token IDF is a weak identity signal — compare token *combinations* |
| — | Shortcut check: do ID numbers or row order correlate with matches? (detect only, never use) | Row-index vs. match-count correlation ≈ **0.00028 (effectively zero)** — no shortcut. ID-number correlation returned NaN (extraction/dtype edge case in the diagnostic itself, not a data signal); row-index result alone is sufficient to confirm no shortcut, but the NaN should be fixed if this check is reused later |

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
- **kaggle CLI on Windows silently writes 0-byte output files** (`'charmap' codec can't encode characters...`) when its own stdout contains non-ASCII bytes and `PYTHONUTF8`/`PYTHONIOENCODING` aren't set to UTF-8 — no error is raised, the file is just empty. `tools/kaggle_ops.py`'s `_run_kaggle` sets both env vars on every call; set them manually for any ad hoc CLI invocation outside that tool.
- **NB00 downloads succeed without an HF token**, just a "sending unauthenticated requests" warning (checked the actual kernel log — no 401/403/rate-limit errors). An HF token is available (see `CLAUDE.local.md`) if a future run ever hits real rate-limiting; not needed so far.
- **`Series.isin(a_huge_python_set)` re-converts the set into a hashable structure on every call** — calling it inside a loop against a set that doesn't change (e.g. `matched_ids` with 7.6M entries) turns an O(1) fixed cost into O(loop_iterations). Measured on NB01's E9: 100 loop iterations took 923s (~9.2s/iteration) with the `.isin()` call inside the loop; moving it to a single call before the loop was the fix. Always hoist a large-set membership filter out of any loop it doesn't need to be inside.
- **`kaggle kernels output` returns nothing while a kernel is still `RUNNING`** — it only serves files once the run finishes (COMPLETE or ERROR). Per-stage logging (added to `eda.py`, see the SS5 decision above) is therefore only visible after the run ends, via the log file — not streamable through the CLI/API mid-run. Live progress during a run is only visible on the Kaggle website itself (the kernel's page shows a live console).
- **A kernel with `enable_internet: false` cannot `pip install` anything** ("No matching distribution found"). Prefer stdlib over a third-party package for any CPU kernel that doesn't need internet for real work (e.g. `eda.py` uses `difflib` instead of `rapidfuzz` for its approximate name-similarity checks — good enough for a census/illustrative sample, and keeps the kernel both internet-OFF and dependency-free). Only reach for a real dependency (and flip internet ON, or pre-install via wheels the way NB00 does) when the approximation genuinely isn't good enough.
- **Suffix/street-type-likeness as specified (end-ratio × sigmoid of z-scored log freq) ranks typo variants (`limitet`, `drve`) above real suffixes** on this data: most tokens have tiny df, so the z-score is poorly scaled. Measured in NB02 v1 (2026-09-26 23:05 IST). Needs a minimum-frequency floor before use.
- **Qwen3-Embedding-0.6B encodes only ~400 records/s on Kaggle 2×T4 (fp16 confirmed, GPU-bound: ~12k tokens/s per T4, forward ≈ 98% of time; batch 128 vs 512 no difference; instruction prefix costs ~10% speed, gains ~1 pt recall@k).** At 24M records (train+test) that is ~16 h, test alone ~8 h → full-pool dense retrieval with this model is infeasible here. Measured in NB03 dry run v3, 2026-09-27 00:05 IST (`reports/raw/nb03_dry_v3/metrics.json`). Also: Kaggle's image ships transformers 5.0.0, where `torch_dtype=` is renamed `dtype=` (`embed.py` passes both ways and casts explicitly).
- **Look-alike digits inside words** (`k01kata`, `h0spital`, `capita1`, `8ombay`) are a real noise type (NB05 v4 missed pairs); `blocking.deleet` maps them back for blocking keys but skips number-like tokens (`3rd`, `12b`, `12bis`). Reuse it in pair features (Prompt 3).
- **Scoring a 48M-pair union with ~25 float features at once OOM-kills a Kaggle CPU kernel (30 GB).** NB05 computes channel lists per country but unions/features/prunes in 200k-S1 chunks.
- **`np.add.reduceat` fails when trailing segments are empty** (index == len(data)); use `np.bincount(row_ids, weights=...)` for per-row sums of sparse data.
- **A notebook consumed via `kernel_sources` exposes only its latest version's output** — a failed newer version can hide a good older output. Keep the last good config's commit hash in progress.md.
- `kaggle kernels output --file-pattern '<regex>'` downloads only matching files — use it to fetch `metrics.json`/logs without pulling multi-GB parquet/npy outputs.
- On this Windows machine, bare `python`/`python3` on PATH resolve to the Microsoft Store alias and fail with "Python was not found". Use the real interpreter path recorded in `CLAUDE.local.md`.
- **Kaggle dataset mount path is not always `/kaggle/input/<slug>/`.** Observed on this environment (2026-09-26): datasets mount at `/kaggle/input/datasets/<owner>/<slug>/` instead. Never hard-code either convention or an owner username in a notebook — resolve the path dynamically (glob both patterns). This is also why notebooks must never depend on a code dataset being mounted at all (see the SS5 decision below and architecture.md SS6.1) — `sys.path.insert(0, "/kaggle/input/er-code")` broke for the same reason.
- **Local C: drive filled up completely (5.6 MB free of 191 GB) on 2026-09-27 03:25 IST.** Two stale 1.1 GB `dataset.zip` temp files from the 26 Sep er-data upload were in `%TEMP%` (deleted). Check `df -h /c` before fetching output TSVs (candidate_pairs.tsv ~0.5 GB); pagefile.sys was 11.5 GB from RAM pressure.
- **Bundled kernels run module code before the driver installs offline wheels**: never bind an optional import (rapidfuzz) at module import time — resolve it lazily (explain_diff._distances) or the slow fallback is used silently. Also: the bundler strips only top-level `from . import X`; an indented one breaks the bundle.
- **86M-row candidate tables as pandas object strings need ~11 GB**: read Parquet with Arrow-backed strings (`io_utils.read_parquet_compact`) and stream test candidates in whole-S1 batches.
- **Original check_outputs loaded all test S2/S3 IDs + per-S1 Python sets (GBs)**: rewritten to stream with int64-encoded IDs (fine on the 8 GB laptop).
- **Qwen3-4B QLoRA on one T4 (dry run): 3.2 s per 4-example step; prompts ~291 tokens (4.7% hit 384); "Yes"/"No" are single tokens** (reports/raw/nb07_dry_v1).

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