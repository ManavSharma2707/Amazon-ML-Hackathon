# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Brocode
**Team Members:** Manav Sharma, Atharva Rathi
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We treat entity resolution as candidate generation (sparse lexical blocking)
followed by a country-agnostic, word-free "explain-the-difference" classifier
(LightGBM) that scores every candidate pair, an isotonic calibration and hard
one-owner step, and an expected-F0.5 decoder that picks each S1 entity's
predicted set (including "no match") to maximise the official metric in
expectation rather than by a single global threshold. The core innovation is
scoring pairs by *typed token relations* (typo / abbreviation / split-join /
initialism / number agreement) instead of raw text or country-specific
vocabulary, so the model generalises to the unseen France test split without
ever seeing a French training example. Shipped result: Half-B macro F0.5
**0.9648** (US 0.9699 / India 0.9571), LOCO-mean 0.9635, scrambled-letter drop
0.0009 — see Section 5 for what limits it short of the ~0.99 target and why.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA (full train + test, `src/eda.py`, `memory.md` SS6) drove every later
design choice:

- **Scale.** Train: S1 2,206,821 / S2 5,034,616 / S3 5,285,603. Test: S1
  1,732,544 (India 810k, US 663k, **France 259k, unseen**). Millions of rows
  per file rule out loading the full dataset on an 8 GB laptop; every
  full-data step runs as a Kaggle notebook (CPU for normalise/blocking/
  features/stage-1, GPU T4 for the judge track).
- **Floor.** All-empty submission scores 0.0558 (the singleton rate), nearly
  identical across countries — any real model must clear this by a wide
  margin to be worth shipping.
- **Structural facts that shape the design, both confirmed on the full
  ground truth:** (a) an S2/S3 record never matches two different S1
  entities (0 / 7,638,365 matched IDs shared) → a **hard one-owner rule** is
  safe and, per E4/E9, actively helps (chain/near-duplicate businesses
  collide on name similarity, so a greedy global assignment resolves the
  conflict rather than double-counting it); (b) no match ever crosses a
  country label (0 / 7,638,365 pairs) → blocking and features can restrict
  to **within-country candidates**, with the only country signal ever used
  being the `same_country` equality flag (never a raw country value, needed
  because France is unseen at training time).
- **Noise.** Positives show typos (28.3%), abbreviations/drops (25.8%),
  reordering (5.9%) and exact matches (15.8%) on the name field; addresses
  add native-script (non-Latin) content in S2/S3 (13-24% depending on
  source/country) even though S1 itself stays Latin-only in every split,
  including France. This is why features are built on *token relations*
  (typo distance, abbreviation subsequence, initialism, split/join) rather
  than raw string identity or a language-specific dictionary.
- **A synthetic, combinatorial name vocabulary.** Names are built from a
  shared pool of words recombined across unrelated businesses (e.g.
  "Glypheus" starts many unrelated S1 names — "Glypheus Capital",
  "... Municipals LLC", "... Platforms PLLC"). Single tokens and even
  char n-grams are therefore weak identity signals; **token *pairs*** are
  rare and far more discriminating — the single biggest lever in blocking
  (Section 3).
- **Number agreement is the main false-merge trap.** A conflicting house
  number on an otherwise near-identical name/address pair is the signature
  of a different branch of the same chain — handled as its own feature
  group (Section 4), separate from string similarity.

### 2.2 Solution Strategy

**Approach type:** Blocking + classifier + metric-aware decoder (not
end-to-end, not graph-based). **Core innovation:** every classifier feature
is either (a) a typed, symmetric token-relation count (explain-the-difference,
Section 4) that is language- and country-free, or (b) a number-agreement
code — never a raw token, a country value, or an ID/row-order shortcut
(checked directly in EDA: row-index-vs-match-count correlation ≈ 0.00028).
Decisions are made by an **expected-F0.5 decoder** (Section 20 of the
internal master plan) that tries every top-k set per entity, including the
empty one, and picks the one maximising E[F0.5] under the fitted
calibration — this is what lets one model handle both "one confident match"
and "several moderate matches" correctly, which a single global threshold
cannot.

---

## 3. Candidate Generation (Blocking)

**Blocking keys used** (`src/blocking.py`; all restricted to equal country
labels, EDA-confirmed, never by a hard-coded country value):

- **Char TF-IDF** (3-4 char n-grams) on the normalised name and address.
- **Rare-token key**: normalised + phonetically-folded name tokens.
- **House-number + street-token key** (stands in for a postcode key —
  postcodes are present in only ~1-2% of records; house numbers in ~90%+).
- **Unordered token-pair keys** on folded name and address tokens (the main
  lever against the synthetic combinatorial vocabulary above — e.g.
  `"glypheus|platforms"`), plus a name-token x address-token cross-pair key.

The union of channels is pruned to the top 50 candidates per S1 by a small
LightGBM pre-ranker (trained on Half A) using channel scores/ranks and
per-S1 gaps — this pruned set *is* `candidate_pairs.tsv`.

**Candidate pairs generated (shipped run):** ~5.0M for Half B (250k S1),
~34.65M for the full test set (1,732,544 S1, 20 per S1 after the scale-guard
pre-ranker described in Section 4).

**Recall by blocking version** (Half B; the shipped run used v4):

| Blocking version | B pair recall | B entity-complete recall | Reduction ratio | Candidates/S1 (mean) |
|---|---|---|---|---|
| v4 (**shipped**: fed the final stage-1/decoder run) | 0.9585 | 0.8810 | 0.999995 | 49.8 |
| v5 (+ cross name x address pairs, look-alike digit folding) | 0.9732 | 0.9164 | 0.999995 | 50.0 |
| v6 (+ reverse pool->S1 name lookup) | 0.9736 | 0.9175 | 0.999995 | 50.0 |

Per-channel contribution, shipped v4 (Half B; `found` = a channel's top-k
list included the true pair; `unique` = found by no other channel):

| Channel | Found | Unique |
|---|---|---|
| name_char | 159,430 | 609 |
| addr_char | 309,987 | 2,232 |
| name_tok | 140,528 | 176 |
| num_key | 431,304 | 1,956 |
| name_pair | 560,467 | 43,194 |
| addr_pair | 754,464 | 76,177 |

**How true matches were not lost:** the token-pair channels (found via the
EDA vocabulary finding above) supply the large majority of *unique* true
pairs — 43,194 + 76,177 of them are found by no other channel, i.e. blocking
would lose ~48% of true pairs without them. v5/v6's extra channels
(cross-field pairs, a reverse pool-to-S1 lookup) raise recall further
(0.9585 -> 0.9736) but arrived after the ~5-hour feature/stage-1 step had
already run on v4 candidates, too late to re-run before the modelling
freeze — see Section 5's honest assessment of what this costs the final
score.

---

## 4. Matching Model

**Features used** (`src/features.py`, `src/explain_diff.py`; all word-free
and country-free, groups A-G of the internal plan; ~120 features total):

- **Classic string similarities** (`rapidfuzz`) on normalised and folded name
  and address: ratio, partial/token-sort/token-set ratio, WRatio,
  Jaro-Winkler, Levenshtein, plus char-3-gram / token / IDF-weighted-token
  Jaccard.
- **Explain-the-difference** (name and address, symmetric and cached): every
  token pair is tested for `exact` / `fold_exact` (look-alike-digit +
  phonetic fold) / `numeric_equal` (12 vs 12bis, never a typo) / `typo`
  (edit-distance-gated) / `prefix_abbrev` / `skeleton_abbrev` (ordered
  subsequence, length-ratio gated — "st"~"south" is deliberately only the
  *weak* 2-letter case, never a strong relation) / `split_join`
  (Walmart~Wal Mart) / `initialism` (SBI~State Bank of India, short
  connectors like "of" may be skipped). A maximum-weight greedy alignment
  produces per-pair counts of each relation type, the IDF-weighted share of
  explained tokens, and the rarest *unexplained* token on each side — the
  single strongest negative signal.
- **Number agreement:** house-number relation (equal / same-base-different-
  suffix, e.g. 12 vs 12bis / conflict / missing), postcode relation
  (equal / first-3-digit match / conflict / missing), unit/other-number
  agreement, shared-number Jaccard.
- **Structure/missingness:** token counts, length ratios, empty-field flags,
  postcode/landmark presence, landmark similarity, romanised-script flag.
- **Blocking meta:** channel-hit bitmask, per-channel score/rank, gap to the
  S1's best candidate, candidates-per-S1, and `is_S3` (a source flag, not a
  country value).
- Group H (`same_country`) is dropped: blocking is already within-country, so
  it is constant and carries no information.

**Scale guard.** A 10k-pair timing check projected > 90 minutes for the full
feature set over all candidates, so a cheap pre-ranker (the 8 fastest
`rapidfuzz` scores + blocking-meta features, LightGBM, cross-fitted on Half
A) prunes each S1's 50 blocking candidates to the top 20 before the full
feature set runs on those — this top-20 set is what is actually classified
and scored.

**Model type:** LightGBM (`objective=binary`, `num_leaves=63`,
`learning_rate=0.05`, 5-fold group-k-fold by S1 id, no class re-weighting so
probabilities stay calibratable). Held-out AUC 0.99961 (B), 0.99959 (A OOF)
— the classifier separates true from false candidates almost perfectly at
the pair level; the remaining ~1% of the metric is lost to blocking misses
and calibration/decoding, not pair-level discrimination (Section 5).

**Threshold selection method:** not a single global threshold. Probabilities
are isotonic-calibrated on Half B (cross-fitted to avoid leakage), then a
hard one-owner rule resolves any record claimed by more than one S1, then an
**expected-F0.5 decoder** evaluates every top-k prediction per entity
(k = 0..20, k = 0 being the empty prediction) under a Poisson-binomial model
of true/false positives plus an estimated blocking-miss prior, and picks the
k maximising E[F0.5]. A global-threshold baseline was also fit on Half B and
came in a hair below the decoder (0.964613 vs 0.964763); the decoder was
kept.

---

## 5. Results & Error Analysis

- **F0.5 Score (macro), Half-B validation, shipped pipeline:** **0.9648**
  overall (floor 0.0547). LOCO-mean (train on one country, score the other,
  both directions) **0.9635**. Scrambled-letter drop (a random, consistent
  a-z permutation applied to every record, as a proxy for the unseen France
  split) **0.0009** — a very small drop, meaning the model is reasoning
  about structure (typo distance, abbreviation shape, number agreement)
  rather than memorising US/India vocabulary.

| Country | n S1 | F0.5 | All-empty floor | Singleton accuracy | Non-singleton F0.5 |
|---|---|---|---|---|---|
| ALL | 250,000 | 0.9648 | 0.0547 | 0.9447 | 0.9659 |
| India | 100,041 | 0.9571 | 0.0550 | 0.9395 | 0.9581 |
| US | 149,959 | 0.9699 | 0.0546 | 0.9482 | 0.9711 |

**Error buckets on Half B** (56,376 of 250,000 S1 entities score below 1.0;
CLAUDE.md's error-analysis convention: singleton false match / blocking miss
/ multi-claim conflict / chain branch / heavy noise / other, in priority
order):

| Bucket | Count | Share of wrong entities |
|---|---|---|
| blocking_miss | 28,702 | 50.9% |
| other | 21,336 | 37.8% |
| heavy_noise | 4,975 | 8.8% |
| chain_branch | 627 | 1.1% |
| false_match_on_singleton | 520 | 0.9% |
| multi_claim_conflict | 216 | 0.4% |

**Common false negatives (missed matches):** dominated by blocking misses
(50.9% of wrong entities) — a true match never reached the scored candidate
list at all. Manual inspection of 200 sampled misses (v4/v5 blocking) shows
these are mostly native-script names transliterated so differently that no
shared token or token-pair survives (~30%), one side's address field empty
or near-empty so the address channels contribute nothing (~35%), and
website/handle-style names (`gurgaonprojects.com`) with almost no lexical
overlap with the real name (~20%). These are largely a *blocking* problem,
not a classifier problem — v5/v6's extra channels (cross-field pairs,
reverse lookup) measurably help (recall 0.9585 -> 0.9736) but arrived too
late to re-run the ~5-hour feature/stage-1 step before the freeze.

**Common false positives (wrong merges):** rare by design (the metric
weights a false positive 4x a false negative) — only 520 singleton false
matches and 627 "chain branch" cases (same name, conflicting house number)
out of 250,000 entities, thanks to the number-agreement features and the
hard one-owner rule.

**What was tried and dropped** (both evaluated honestly, kept as tested code
paths, not shipped):

| Configuration | B F0.5 | LOCO-mean | Scrambled drop |
|---|---|---|---|
| All-empty floor | 0.0547 | | |
| Stage-1 + global threshold | 0.9642 | | |
| + calibration + expected-F decoder | 0.9641 | | |
| + one-owner rule (**shipped: sub-01**) | 0.9648 | 0.9635 | 0.0009 |
| + combiner with collective features (evaluated, dropped: no LOCO gain) | 0.9674 | 0.9635 | 0.0010 |

*The combiner* (competition/sibling features + a second LightGBM stage on
Half B) improved the Half-B score by a statistically real margin
(bootstrap 90% CI [0.0025, 0.0028], excludes 0) but showed **no LOCO-mean
gain** — the project's primary cross-country generalisation metric — so it
was not shipped (a Half-B-only gain that does not transfer is exactly the
overfitting risk LOCO-mean is meant to catch).

*The LLM judge* (Qwen3-4B, QLoRA, evidence-augmented prompts on the
uncertain stage-1 band) was trained and evaluated against its own gate
(judge AUC on the band must beat stage-1's):

| | AUC (held-out) | Log-loss (held-out) |
|---|---|---|
| Judge | 0.8003 | 0.4884 |
| Stage-1 | 0.8959 | 0.3721 |

On the uncertain band only (929 pairs):

| | AUC |
|---|---|
| Judge | 0.7739 |
| Stage-1 | 0.8767 |

**Gate FAILED** (0.774 < 0.877) — dropped. Training was cut to a 70-minute
wall-clock budget (5,184 of 18,969 available examples seen) to fit the
day's schedule; a longer run may have closed the gap, but was not
justified given a stage-1 classifier already at 0.996 pair-level AUC.

---

## 6. Conclusion

Word-free, country-free "explain-the-difference" features plus a
metric-aware decoder reach 0.9648 Half-B F0.5 (LOCO-mean 0.9635, scrambled
drop 0.0009) without any language- or country-specific logic, which matters
directly for the unseen France test split. The main remaining gap is
**blocking recall**, not classifier quality (pair-level AUC 0.996): about
half of all wrong Half-B entities are blocking misses, and two blocking
improvements that would have helped (cross-field token pairs, a reverse
name lookup, recall 0.9585 -> 0.9736) were built and validated but arrived
too late in the schedule to feed a full re-run before the modelling freeze.
The combiner and LLM-judge tracks were both honestly evaluated against
pre-registered gates and dropped rather than shipped on a Half-B-only gain
— the key lesson being that LOCO-mean (not the in-country validation score)
is what actually predicts generalisation to an unseen country here.

---

## Appendix

### A. Code Artefacts

The complete, runnable code ships under `code/business_entity_resolution/`
(`src/`, `README.md`, `requirements.txt`). Entry points:

- `src/predict.py --from-features` reproduces the shipped
  `output/matching_results.tsv` / `candidate_pairs.tsv` exactly from the
  bundled `artifacts/` (the real trained stage-1 model, calibrator and
  decoder config) plus a copy of NB06's feature table (too large to ship,
  see README.md).
- `src/scripts/run_train.sh` + `run_predict.sh --demo` run the identical
  code end to end, self-contained, on the bundled `sample/` (trains its own
  small blocking pruner + stage-1 model from `sample/`'s own labels, then
  predicts on `sample/`'s test split) — no external artifact needed.
- `pytest -q src/tests/` — unit tests for every module (explain-the-difference
  relations, feature building, the expected-F0.5 decoder checked against the
  official metric's reference implementation, the blocking pipeline, etc.).
- The real, full-scale run is the numbered Kaggle notebooks in
  `src/notebooks/`; README.md lists the exact commands, accelerators and
  measured runtimes for each.

### B. Additional Results

See Section 5's tables above for the full ablation, per-country, blocking-
version and judge-vs-stage-1 comparisons; `reports/` (not part of this zip)
holds the underlying `metrics.json` files these tables were generated from,
via `src/scripts/make_doc_tables.py`.

---

**Note:** Teams can modify sections according to their approach while
maintaining clarity and technical depth.
