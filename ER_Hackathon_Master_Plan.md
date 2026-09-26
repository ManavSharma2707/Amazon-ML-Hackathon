# Business Entity Resolution Challenge — Master Plan

> **Status:** Final approach, approved for build · **Deadline:** 27 Sep 2026 23:59 IST (freeze 19:30 IST, last upload by 21:00 IST)
> **Compute:** Kaggle (P100 or 2× T4, 16 GB each) · **Team roles:** CPU track, GPU track, Validation/Packaging track

> **Updates (2026-09-26 15:45 IST), checked against the official guidelines PDF:**
> - Max **5 leaderboard submissions per day**; ties are broken by earlier submission time; shortlisting uses **both** public and private boards.
> - Two documents are needed: a **1–2-page approach summary** (guidelines) and the full `Documentation_template.md` (problem statement).
> - Submitted code needs **docstrings/comments on every function**, and a version history of all submissions (git tags).
> - In the zip, **all code lives under `src/`**. `architecture.md` §3 supersedes §26 of this plan.
> - A **scale guard** applies if the data has millions of records (architecture.md §7). The submission checklist is in `memory.md` §11.

---

## Table of contents

0. [TL;DR](#0-tldr)
1. [What the problem really is](#1-what-the-problem-really-is)
2. [Rules, compliance and licences](#2-rules-compliance-and-licences)
3. [Design principles](#3-design-principles)
4. [Architecture at a glance](#4-architecture-at-a-glance)
5. [Environment and compute setup](#5-environment-and-compute-setup)
6. [Step 0 — Data loading and sanity checks](#6-step-0--data-loading-and-sanity-checks)
7. [Step 1 — Exploratory data study (decisions it drives)](#7-step-1--exploratory-data-study-decisions-it-drives)
8. [Step 2 — Universal normalisation](#8-step-2--universal-normalisation)
9. [Step 3 — Corpus statistics: rarity and suffix-likeness](#9-step-3--corpus-statistics-rarity-and-suffix-likeness)
10. [Step 4 — Data split and leakage control](#10-step-4--data-split-and-leakage-control)
11. [Step 5 — Embeddings: frozen and fine-tuned](#11-step-5--embeddings-frozen-and-fine-tuned)
12. [Step 6 — Candidate generation (blocking)](#12-step-6--candidate-generation-blocking)
13. [Step 7 — Explain-the-difference features](#13-step-7--explain-the-difference-features)
14. [Step 8 — Full feature set](#14-step-8--full-feature-set)
15. [Step 9 — Stage-1 LightGBM](#15-step-9--stage-1-lightgbm)
16. [Step 10 — LLM judge (Qwen3-4B, QLoRA)](#16-step-10--llm-judge-qwen3-4b-qlora)
17. [Step 11 — Competition and sibling features (collective signals)](#17-step-11--competition-and-sibling-features-collective-signals)
18. [Step 12 — Combiner and calibration](#18-step-12--combiner-and-calibration)
19. [Step 13 — One-owner rule (exclusivity)](#19-step-13--one-owner-rule-exclusivity)
20. [Step 14 — Expected-F0.5 decoder](#20-step-14--expected-f05-decoder)
21. [Validation protocol](#21-validation-protocol)
22. [Output writing and submission checks](#22-output-writing-and-submission-checks)
23. [Unseen-country (France) safeguards](#23-unseen-country-france-safeguards)
24. [Sprint timeline, gates and fallbacks](#24-sprint-timeline-gates-and-fallbacks)
25. [Master risk register](#25-master-risk-register)
26. [Code package structure and reproducibility](#26-code-package-structure-and-reproducibility)
27. [Mapping to the methodology document](#27-mapping-to-the-methodology-document)
28. [Appendix — reference code](#28-appendix--reference-code)
29. [Sources](#29-sources)

---

## 0. TL;DR

We build a **country-agnostic, metric-aware** entity-resolution pipeline:

1. **Universal normalisation.** No country-specific dictionaries do the heavy lifting.
2. **Hybrid blocking.** Fine-tuned multilingual embeddings, character TF-IDF, number/postcode keys and rare-token keys.
3. **Explain-the-difference features.** Every difference between two records is classified as a known noise operation (typo, abbreviation, initials, dropped word, reorder, legal suffix). Rare words that **cannot be explained** are the main evidence against a match.
4. **Stage-1 LightGBM** on word-free features.
5. **LLM judge.** Qwen3-4B with QLoRA, applied only to uncertain pairs, using an **evidence-augmented prompt**.
6. **Combiner LightGBM**, then calibration.
7. **One-owner rule.** A Source 2/3 record belongs to at most one Source 1 entity.
8. **Expected-F0.5 decoder.** For each entity, pick the prediction set (including "empty") with the highest expected score.

**What makes us different from most teams:**
- a decoder built directly for the metric;
- the structural exclusivity constraint;
- word-free features designed to transfer to France;
- a *scrambled-letter* test that proves the model doesn't rely on language-specific vocabulary.

---

## 1. What the problem really is

### 1.1 The task

Source 1 (S1) is a deduplicated reference list. For each S1 entity, output every Source 2 (S2) and Source 3 (S3) record describing the same real-world business. The answer may be zero, one or many records.

### 1.2 The metric, decoded

The metric is F-beta with β = 0.5, **macro-averaged over S1 entities**:

```
F0.5 = 1.25·P·R / (0.25·P + R)
     = 1.25·TP / (1.25·TP + 0.25·FN + FP)       (equivalent count form)
```

| Situation for one S1 entity | Score |
|---|---|
| Truly no matches, we predict empty | **1.0** |
| Truly no matches, we predict anything | **0.0** |
| Has matches, we predict empty | **0.0** |
| Has matches, we predict a set | F0.5 of that set |

Consequences:
- **A false positive costs 4× more than a false negative** in the count form (FP has weight 1, FN has weight 0.25).
- **Every entity weighs the same.** An entity with 1 true match counts as much as one with 10. Getting the *small* entities exactly right matters most.
- **The "empty vs. non-empty" decision is binary and brutal:** 1.0 or 0.0. For many entities this is the single most important decision.
- **Baseline to beat:** predicting empty for everyone scores exactly the singleton rate. Measure this on day one; it is our floor.

### 1.3 Structural facts we exploit (to be verified in Step 1)

| Fact | Source | How we exploit it |
|---|---|---|
| S1 is deduplicated | Problem statement | Each S2/S3 record should belong to at most one S1 entity → exclusivity (Step 13) and competition features (Step 11) |
| One S1 can match many S2/S3 records | Problem statement | S2/S3 contain internal duplicates → sibling evidence (Step 11) |
| Test contains an unseen country (France) | Problem statement | Word-free features, no country feature, scrambled-letter validation (Section 23) |
| Noise follows listed patterns | Problem statement | Explain-the-difference features mirror the corruption process (Step 7) |

---

## 2. Rules, compliance and licences

### 2.1 Hard rules and how we comply

| Rule | Our compliance |
|---|---|
| Use only the provided training data | Nothing is trained on test records: no pseudo-labels, no test-derived synthetic pairs, no fine-tuning on test data |
| No external lookups (APIs, registries, geocoders, internet augmentation) | None used. No libpostal (its parser is trained on OpenStreetMap/OpenAddresses address data, which could be judged external data). No geocoding. |
| Final model MIT/Apache 2.0, ≤ 8B parameters | Qwen3-Embedding-0.6B (Apache 2.0) + Qwen3-4B (Apache 2.0) + LightGBM (MIT). Total learned parameters ≈ 4.6B, so we are under 8B **even if the limit is read as a pipeline total**. |
| Country is an open set | No hard-coded country values, no one-hot country, no country filters. Only an equality flag (`same_country`), and only if Step 1 justifies it. |
| Every test S1 entity has exactly one row | Enforced by construction and asserted before writing |
| `candidate_pairs.tsv` = exactly the set scored by the model | We write the candidate file from the same in-memory table the models score. Final matches are filtered from it, so they are a subset by construction. |

### 2.2 Grey areas and our position

| Grey area | Position | Fallback |
|---|---|---|
| Computing token rarity / suffix frequency on the test files at inference | No labels, no gradient updates. Identical in nature to fitting TF-IDF on the corpus being processed, which every team does. **Send a one-line confirmation request to the organisers.** | Compute statistics on train only, plus a character-shape prior for unseen tokens (Section 9.4) |
| Pretrained models (Qwen3) carry world knowledge | Explicitly allowed by the licence clause; they are not lookups | None needed |
| Manually looking at test predictions | Allowed **only for bug detection** (encoding errors, crashes). Never to tune thresholds or rules on test. | None needed |
| A small universal lexicon (connector words like and/et/und/y; common legal forms) | General linguistic knowledge, not entity data. Kept small and multilingual, and **not** relied upon (statistical detection is primary). | Remove the lexicon entirely; the model still works through statistics |

### 2.3 Library licences (for the audit)

| Library | Licence | Note |
|---|---|---|
| pandas, numpy, scipy, scikit-learn | BSD | |
| LightGBM | MIT | |
| rapidfuzz | MIT | Fast string metrics |
| faiss-cpu / faiss-gpu | MIT | |
| sentence-transformers | Apache 2.0 | |
| transformers, peft, accelerate | Apache 2.0 | |
| bitsandbytes | MIT | |
| **Avoid:** `unidecode` | **GPL** | Use stdlib `unicodedata` NFKD, or `anyascii` (ISC), instead |
| **Avoid:** libpostal | MIT code, but OSM-trained | External-data risk (see 2.1) |

Before packaging, open each model card and confirm the licence field says Apache-2.0 or MIT. Put the licence list in the README.

---

## 3. Design principles

1. **The model never sees words.** The final classifiers see only *measurements of difference*: similarities, counts, agreement flags, rarity-weighted scores. Words are processed by functions that behave the same way in any Latin-script language. The pretrained LLM and embedder see text, but are guarded by validation (Section 21).
2. **Optimise the metric directly.** Calibrated probabilities plus a decision-theoretic decoder, instead of a hand-picked threshold.
3. **Use the structure.** Exclusivity plus sibling evidence.
4. **Every component must earn its place.** It stays only if it improves leave-one-country-out (LOCO) F0.5 beyond noise (Section 21.5).
5. **Always keep a valid submission in hand.** Build in layers; each layer is independently shippable.
6. **Symmetric processing.** Every transformation is applied identically to both sides of a pair. A normalisation that is "wrong" but symmetric rarely hurts matching.

---

## 4. Architecture at a glance

```
             ┌──────────────── TRAIN (S1/S2/S3 + GT) ─────────────────┐     ┌──────── TEST (S1/S2/S3) ────────┐
             │                                                        │     │                                 │
 Step 0-1    │   Load (dtype=str, no NA coercion) → sanity → EDA      │     │  Load → sanity                  │
 Step 2      │   Universal normalisation (both sides, same code)      │     │  Same normalisation             │
 Step 3      │   Corpus stats (rarity, suffix-likeness)               │     │  Stats recomputed on this run   │
 Step 4      │   Split S1 entities → Half A | Half B (country-strat.) │     │                                 │
             │                                                        │     │                                 │
 Step 5      │   Frozen embeddings (all) ── fine-tune embedder on A   │     │  Frozen + fine-tuned embeddings │
 Step 6      │   Blocking: A (frozen-based) · B (fine-tuned-based)    │     │  Blocking (fine-tuned-based)    │
             │           → candidate pairs                            │     │  → candidate_pairs.tsv          │
 Step 7-8    │   Explain-the-difference + all pair features           │     │  Same features                  │
 Step 9      │   Stage-1 LightGBM trained on A (OOF within A)         │     │  Stage-1 predicts               │
 Step 10     │   LLM judge QLoRA trained on A's hard pairs            │     │  Judge scores uncertain band    │
 Step 11     │   Competition + sibling features (from stage-1 probs)  │     │  Same                           │
 Step 12     │   Combiner LightGBM + calibration trained on B (OOF)   │     │  Combiner → calibrated p        │
 Step 13     │   One-owner rule (tuned on B)                          │     │  Same                           │
 Step 14     │   Expected-F0.5 decoder (tuned on B)                   │     │  → matching_results.tsv         │
             └────────────────────────────────────────────────────────┘     └─────────────────────────────────┘
```

---

## 5. Environment and compute setup

### 5.1 Kaggle facts to plan around

- **Accelerators:** P100, or 2× T4, each with 16 GB. Weekly GPU quota is typically about 30 hours.
- **Session limits:** sources report 9–12 hours per session. **Plan every GPU job to finish in ≤ 8 hours, with checkpoints.**
- **Working directory:** about 20 GB of saved output.
- **Precision:** T4 has **no bf16 support**, so use fp16 compute. Flash-Attention 2 is not supported on T4; use SDPA/eager attention.
- **Access:** phone verification is required for GPU. Enable **Internet** in notebook settings for `pip` and model downloads.
- **Background runs:** "Save & Run All (Commit)" keeps a run going after you close the tab. Outputs persist as notebook output, so turn them into **Kaggle Datasets** for reuse.
- **Parallelism:** run the embedder and the judge as separate GPU sessions at the same time.

### 5.2 Environment pinning

At the start of each GPU notebook:

```bash
pip install -q "transformers>=4.51.0" "sentence-transformers>=3.0" "peft>=0.11" \
               "bitsandbytes>=0.43" "accelerate>=0.30" rapidfuzz lightgbm faiss-cpu
python -c "import transformers, peft, bitsandbytes, torch; print(transformers.__version__, peft.__version__, bitsandbytes.__version__, torch.__version__, torch.cuda.get_device_name(0))"
```

Freeze the exact versions printed into `requirements.txt` at packaging time. Qwen3 architectures need `transformers >= 4.51`; older versions fail with a `KeyError: 'qwen3'`.

### 5.3 Artifact flow between notebooks

| Artifact | Produced by | Stored as |
|---|---|---|
| `normalized_{train,test}.parquet` | CPU track | Kaggle Dataset `er-norm` |
| `emb_frozen_{train,test}.npy` | GPU track | Kaggle Dataset `er-emb` |
| `embedder_ft/` (weights) | GPU track | Kaggle Dataset `er-embedder` |
| `candidates_{A,B,test}.parquet` | CPU track | `er-cands` |
| `features_{A,B,test}.parquet` | CPU track | `er-feats` |
| `judge_lora/` (adapter only, about 60 MB) | GPU track | `er-judge` |
| `judge_scores_{B,test}.parquet` | GPU track | `er-judge-scores` |

Every artifact filename carries a version suffix (`_v1`, `_v2`) so teammates never overwrite each other.

### 5.4 Seeds and determinism

Set seeds for Python, NumPy, PyTorch and LightGBM (`seed`, `bagging_seed`, `feature_fraction_seed`). GPU training is not bit-exact across runs; that is acceptable, but log the metrics of the run that produced the submission.

---

## 6. Step 0 — Data loading and sanity checks

### 6.1 Loading, done safely

```python
import csv, pandas as pd
def load_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       na_values=[], quoting=csv.QUOTE_NONE, encoding="utf-8")
```

Why each argument matters:
- `dtype=str`: otherwise IDs or postcodes like `01234` lose their leading zeros.
- `keep_default_na=False`: otherwise a business named "NA", or the country code "NA" (Namibia), silently becomes NaN. The test set has an open country set.
- `quoting=csv.QUOTE_NONE`: a stray `"` inside a business name (`"Joe's "Best" Diner`) would otherwise swallow following lines.

### 6.2 Assertions (fail loudly)

- Row count equals the file's line count minus 1 (catches quote-swallowing).
- Exactly four columns: `entity_id, business_name, business_address, country`.
- IDs are unique within each file; prefixes match the file (`S1-`, `S2-`, `S3-`).
- The ground-truth file parses: an empty `matched_entity_ids` becomes an empty list, and every listed ID exists in S2/S3 train.
- Every train S1 ID appears exactly once in the ground truth (or the missing ones are treated as singletons; log it).
- Strip a UTF-8 BOM from the first header if present.
- Trim whitespace, and collapse internal whitespace runs **only in the normalised copy**. Keep the raw text for the LLM prompt.

---

## 7. Step 1 — Exploratory data study (decisions it drives)

Time box: **about 90 minutes.** Every question maps to a design decision.

| # | Question | Decision it drives |
|---|---|---|
| E1 | Sizes of S1/S2/S3 per split and country | Compute budget, blocking k, judge band size |
| E2 | **Singleton rate** overall and per country | Floor score; decoder prior; sanity check on test predictions |
| E3 | Distribution of #matches per S1 (and S2 vs S3 split) | Cap on predicted set size; decoder behaviour |
| E4 | **Does any S2/S3 ID appear under two S1 entities?** | If ≈ 0% → hard one-owner rule. If some → soft rule. |
| E5 | Do matches ever cross countries? | If never → block within country (equality only, never hard-coded values). If sometimes → don't. |
| E6 | What fraction of S2/S3 records match *nothing*? | Negative-heavy candidate space; how strict to be |
| E7 | Among positives: exact-name rate, exact-address rate, name-only-noise vs. address-only-noise | Feature priorities; whether address or name is more reliable |
| E8 | Noise census on positives (token-aligned diff): typo rate, abbreviation rate, dropped-token rate, reorder rate, landmark insertion rate, missing postcode rate, **per source (S2 vs S3)** | Tells us which Step 7 operations matter; whether S2 and S3 need a `source` feature |
| E9 | Hardest negatives: pairs with high name similarity but not matched | How much chain/franchise confusion exists; number features' importance |
| E10 | Empty or very short names/addresses; non-Latin scripts | Missing-value handling; normalisation robustness |
| E11 | Postcode formats per country (digit-run lengths) | Number extractor generality (5-digit, 6-digit, alphanumeric) |
| E12 | Test set: country label values, sizes per country, script/character distribution | France share; confirm our normaliser handles French characters (é, è, ç, œ, ') |

Write every number into `eda_findings.md`; it feeds the methodology document.

**What can go wrong:**
- *Findings contradict our assumptions*, e.g. S2 IDs shared across S1s. **Fix:** the design already has soft variants: soft exclusivity, and the decoder working without the one-owner rule.
- *We spend too long.* **Fix:** hard 90-minute timebox. Skip E9–E12 depth if behind; E2, E4, E5 and E8 are mandatory.

---

## 8. Step 2 — Universal normalisation

Applied **identically** to every record from every source and country. We keep both `raw_*` (for the LLM prompt) and `norm_*` fields.

### 8.1 Pipeline

1. **Unicode:** NFKC, then NFKD with combining marks removed (é→e, ç→c). Store `norm` without accents, and keep a lowercased-with-accents copy for the embedder.
2. **Case:** `casefold()` (better than `lower()` for ß and similar).
3. **Ligatures and specials:** œ→oe, æ→ae, ß→ss, `&`→` and `. Typographic apostrophes and quotes become `'`.
4. **Connector unification (small universal set):** {and, et, und, y, e, en, &} → `&`, applied only as a **standalone token**. This is linguistic, not country data.
5. **Punctuation:** `.` `,` `-` `/` `(` `)` `#` `'` become spaces. Exceptions:
   - dots inside initials: `s.b.i.` → `sbi` (join single letters separated by dots);
   - hyphens inside digit runs: `12-14` stays as a number range token.
6. **Number extraction** (into separate fields, *removed* from the text used by word-level features):
   - `numbers_all`: every digit run, plus alphanumeric house numbers (`12b`, `12bis`, `14ter`).
   - `postcode_candidates`: digit runs of length 4–6 at the end or near the end of the address, and alphanumeric patterns like `sw1a 1aa`. Found **by shape, not by country**.
   - `house_number`: the first short digit run (1–5 digits) adjacent to a street-type-like token or at the start of the address.
   - `unit`: numbers after unit-like tokens detected statistically (suite, ste, unit, apt, floor, bureau, etage…; see 8.2).
7. **Landmark phrases:** a small multilingual trigger set {near, nr, opp, opposite, behind, beside, next to, in front of, adjacent, besides, près de, pres de, en face de, derrière, derriere, à côté de, a cote de}. The trigger plus the next **up to 4 tokens** are moved to `landmark_text`. They are compared separately with a low-weight feature, so they don't pollute address similarity.
8. **Phonetic/transliteration folding** (applied to a *separate* `fold` field, symmetric on both sides):
   - vowel-run reduction (aa→a, ee→i, oo→u, ii→i);
   - digraphs ph→f, sh→s, th→t, kh→k, gh→g, ck→k, qu→k, w→v, z→s;
   - duplicate-consonant collapse (ll→l).

   Word-level features are computed on **both** `norm` and `fold`, and the model decides how much to trust each.
9. **Whitespace collapse;** tokens are split on whitespace.

### 8.2 Legal suffix handling

Legal suffixes are **not stripped by a hard list**. Instead:
- Tokens get a **suffix-likeness** score from corpus statistics (Section 9).
- Name tokens are split into `core_tokens` (low suffix-likeness) and `suffix_tokens` (high suffix-likeness, found in the last 1–3 positions).
- Features compare the cores and the suffixes separately. A suffix mismatch ("Ltd" vs "Pvt Ltd" vs nothing) is weak evidence; a core mismatch is strong evidence.
- A universal backup list (ltd, limited, llc, inc, corp, corporation, co, company, pvt, private, plc, gmbh, ag, sa, sarl, sas, sasu, eurl, sci, snc, bv, nv, srl, spa, llp, lp, pty) is merged in **only as a prior**. If the statistics disagree, the statistics win.

### 8.3 What can go wrong in normalisation

| Failure | Example | Fix |
|---|---|---|
| Over-normalisation merges different things | `st` = Street vs. Saint; `dr` = Drive vs. Doctor | We do **not** expand abbreviations globally; they are handled pairwise in Step 7, where context (the other record) disambiguates |
| Numbers inside names | "7-Eleven", "24x7 Pharmacy", "3M" | Name numbers go into a `name_numbers` field and are compared separately; they are not treated as house numbers |
| Initials broken | `S. B. I.` vs `SBI` vs `State Bank of India` | Dot-joined single letters handled in 8.1(5); initialism matching in Step 7 |
| Accent stripping merges distinct words | French `côte` vs `cote` | Harmless for matching because it is symmetric; the embedder still sees accents |
| Non-Latin scripts (Devanagari, Arabic) | Empty result after ASCII folding | We never delete characters we can't fold; keep the Unicode letters. Embeddings and character n-grams still work. |
| Empty field after normalisation | Name was only punctuation | Keep an `is_empty` flag; similarity features return NaN (LightGBM handles missing values natively) |
| Landmark trigger eats the real address | "Near Road" as an actual street name | Only extract when the trigger is **not** the first token after a house number; the 4-token cap limits damage; the raw address is still used by the embedder and judge |
| Phonetic folding creates collisions | `vine` and `wine` both become `vine` | That's why `fold` features sit alongside `norm` features, never replacing them |

---

## 9. Step 3 — Corpus statistics: rarity and suffix-likeness

### 9.1 Rarity (IDF)

For each token t in names (and separately in addresses):

```
df(t)  = number of records containing t   (computed over S1 ∪ S2 ∪ S3 of the current run)
idf(t) = log( (N + 1) / (df(t) + 1) ) + 1
```

- Computed **per run** over the records being processed. For test, that is the test files; for validation, the training files. No labels are involved.
- Optionally computed **per country group** as well (`idf_local`), since "Sharma" is common in India and rare in the US. The model gets both global and local rarity.
- Also computed on the `fold` form.

### 9.2 Suffix-likeness

For each token t in names:

```
end_ratio(t)   = count(t in last 2 positions) / count(t anywhere)
freq(t)        = count(t) / number of names
suffix_score(t)= end_ratio(t) · sigmoid( (log freq(t) − μ) / σ )
```

A token that is frequent **and** almost always at the end (ltd, llc, sarl, sas…) scores high. It works for any language with end-position legal forms. It will also flag frequent end words like "store" or "restaurant". That is fine: those are **generic** words that carry little identity, and treating them as low-weight is correct. They also get low IDF.

### 9.3 Street-type-likeness (for addresses)

The same idea: tokens frequent in addresses that sit immediately after a house number or before a name-like token (road, rd, street, st, rue, avenue, av, bd, marg, nagar…). This is used only to flag "type" tokens, so their abbreviation variants are treated as weak evidence.

### 9.4 Fallback if test-time statistics are ruled out

If the organisers disallow computing statistics on the test files:
- Compute IDF on train only.
- For unseen tokens, use a **shape prior**: expected IDF as a function of token length and character-class pattern, fitted on train tokens (short tokens tend to be generic; long tokens tend to be rare).
- Replace suffix-likeness with the universal backup list plus the position-in-name feature.

### 9.5 What can go wrong

| Failure | Fix |
|---|---|
| Tiny corpora (small country subset) → noisy statistics | Additive smoothing, and blend local with global IDF: `idf = w·local + (1−w)·global`, where w = n_local/(n_local + 500) |
| The same generic word dominates (e.g., "enterprises") | That is exactly what IDF down-weights; verify with a top-50 list printout |
| Rare typos get very high IDF ("Corporaton") | Typos are explained in Step 7 before rarity is used for *unexplained* tokens, so a typo token aligned to its correct form doesn't count as unexplained |

---

## 10. Step 4 — Data split and leakage control

### 10.1 Why we split

Three models learn from labels: the embedder, stage-1 LightGBM and the judge. Then a combiner learns from their outputs. If the combiner trains on pairs those models already saw, it learns to over-trust them. On the test set that trust is misplaced, which means over-confident false positives, exactly what F0.5 punishes.

### 10.2 The split

- Split **S1 entities** (with all their positives) into **Half A (55%)** and **Half B (45%)**.
- **Stratify** by country × number of matches bucket (0, 1, 2–3, 4+), so both halves have the same singleton rate and country mix.
- S2/S3 records belong to the half of the S1 they match. Unmatched S2/S3 records are shared (they are only ever negatives, so no leak).

### 10.3 Who trains where

| Component | Trained on | Applied to |
|---|---|---|
| Fine-tuned embedder | A (positives + hard negatives) | Blocking and similarity for B and test |
| Stage-1 LightGBM | A (features exclude fine-tuned-embedding scores) | OOF predictions on A (5-fold group-k-fold, for judge band selection); predictions on B and test |
| LLM judge | A's hard pairs (selected by A's OOF stage-1) | B and test uncertain band |
| Combiner LightGBM | B (5-fold group-k-fold by S1 → OOF) | Test |
| Calibration, one-owner mode, decoder parameters | B OOF | Test |
| LOCO and scrambled-letter validation | B (plus A where noted) | Reporting |

Key detail: **blocking for A uses frozen embeddings**, since fine-tuned ones would give A an unrealistically easy recall. B and test use fine-tuned embeddings. Stage-1 is trained without fine-tuned-embedding features, so it never learns from a feature that is in-sample on A.

### 10.4 Size-dependent adjustments

- **If train is small** (< 3,000 S1 entities): halving hurts. Use **2-fold cross-fitting**: train the embedder and judge on A, apply them to B; then train them on B, apply them to A. The combiner then gets OOF on all data. This roughly doubles GPU time, so only do it if the GPU track is ahead.
- **If train is large** (> 50,000 S1): subsample training pairs for the judge (Section 16.4) and cap the embedder's training steps.

### 10.5 What can go wrong

| Failure | Fix |
|---|---|
| The same real business appears in A and B through S2/S3 duplicates | Impossible by construction: S2/S3 records follow their S1's half, and S1 is deduplicated |
| Group-k-fold not grouped by S1 | Always pass `groups = s1_entity_id` |
| Combiner still over-trusts the judge | The judge is out-of-sample on B by design; check the combiner's feature importance and the calibration curve |

---

## 11. Step 5 — Embeddings: frozen and fine-tuned

### 11.1 Model choice

We use **Qwen3-Embedding-0.6B**:
- Apache 2.0, multilingual (100+ languages, including French);
- about 1.2 GB in fp16, fast on a T4;
- supports instructions and Matryoshka (MRL) dimension truncation.

**Fallback** if it misbehaves (fp16 NaNs, version errors): `intfloat/multilingual-e5-base`. Confirm the licence on its model card before use.

### 11.2 Text templates

- **Full record:** `"{name} | {address} | {country}"`. Including country is fine: it's text, not a one-hot. It helps disambiguate similar names in different countries, and it's the same label on both sides of a same-country pair.
- **Name only:** `"{name}"`. **Address only:** `"{address}"`.
- **Instruction** (identical on both sides, since matching is symmetric): `"Instruct: Represent this business record to find records of the same business\nQuery: "`.
- Use **raw text lightly cleaned** (whitespace, Unicode NFKC, accents kept). The embedder benefits from natural text; heavy normalisation is for the string features.

### 11.3 Frozen embeddings (GPU, about 30–60 minutes)

- Encode all records (train + test) in fp16, batch size 128–256, `max_length = 96` tokens (business records are short).
- L2-normalise and store as `float16` `.npy` together with an ID-order file.
- Use **exact search** (`faiss.IndexFlatIP`). Record counts are small enough that approximate indexes add risk with no benefit.

### 11.4 Fine-tuning (GPU, about 1–1.5 hours)

**Training pairs (from Half A only):**
- Anchor = S1 record text; positive = each matched S2/S3 record.
- Extra positives: S2↔S3 pairs that share the same S1. These teach duplicate-within-source structure.
- **Hard negatives:** for each anchor, the top-ranked non-matches from the *frozen* kNN, ranks 1–30 excluding true matches. Take 3 per anchor, preferring:
  1. same name, different number;
  2. high name similarity, different city token;
  3. high address similarity, different name.

**Loss:** `MultipleNegativesRankingLoss` (in-batch negatives) with explicit hard negatives, as `(anchor, positive, negative)` triplets. Use `CachedMultipleNegativesRankingLoss` if memory allows a larger effective batch.

**Hyperparameters:** lr 2e-5, 1 epoch, warmup 5%, batch 64 (fp16), max_len 96, gradient clipping 1.0, weight decay 0.01. Save a checkpoint every 500 steps.

**Country-agnostic regularisation:**
- **Scrambled-letter augmentation on 15–20% of batches:** pick a random permutation of a–z and apply it *identically* to anchor, positive and negatives. Digits and spaces stay unchanged. Abbreviation, typo and reorder relations are preserved while vocabulary is destroyed, so the model has to learn *structure*.
- **Country-balanced sampling:** equal US/India anchors per batch, so neither country dominates.
- **Field dropout (10%):** randomly drop the address or the name from one side, mimicking missing components.

**Batch hygiene (important):** with in-batch negatives, two positives of the **same S1 entity** in one batch become false negatives for each other. Build batches with a custom sampler so **each S1 entity appears at most once per batch**.

### 11.5 Acceptance test for the fine-tuned embedder

On Half B, compare frozen vs fine-tuned:
- blocking recall@k for k ∈ {5, 10, 20, 50};
- MRR of true matches;
- the same metrics **per country**, and under the scrambled-letter transform.

**Accept** if recall@20 improves on both countries, and the scrambled-letter drop is not bigger than the frozen model's. Otherwise use frozen embeddings for blocking, and keep the fine-tuned cosine only as a combiner feature (the combiner will ignore it if it's useless).

### 11.6 What can go wrong

| Failure | Detection | Fix |
|---|---|---|
| fp16 overflow → NaN loss | Loss prints `nan` | Lower lr to 1e-5; keep grad clipping; load the model in fp32 with AMP autocast; fall back to e5-base |
| OOM | CUDA OOM error | Batch 32 plus CachedMNRL mini-batch 16; max_len 64 |
| Catastrophic forgetting of multilingual ability | Scrambled-letter and French-character sanity drop | 1 epoch only, low lr, keep the frozen feature |
| Overfits US/India vocabulary | LOCO gap widens vs frozen | Increase scrambled share to 30%; fewer steps |
| Wrong pooling / padding side | Similarities near-random | Use sentence-transformers loading (it applies last-token pooling and left padding for Qwen3-Embedding); verify on 5 known positive pairs |
| Session dies mid-training | | Checkpoints every 500 steps; resume |

---

## 12. Step 6 — Candidate generation (blocking)

### 12.1 Channels (union)

| Channel | What it catches | Per-S1 k |
|---|---|---|
| **Dense kNN** (full-record embedding) S1 → S2∪S3 | Paraphrase, reorder, abbreviations, French | 30 |
| **Reverse dense kNN** S2/S3 → S1 (keep if S1 is in the record's top-5) | Records whose S1 has many near neighbours | — |
| **Char 3–4-gram TF-IDF, name** (cosine top-k) | Typos, partial names | 20 |
| **Char 3–4-gram TF-IDF, address** | Same address, renamed/DBA business | 15 |
| **Number/postcode key**: same postcode AND ≥1 shared name core token (or name TF-IDF > 0.3) | Heavy name noise with an intact postcode | all |
| **Rare-token key**: share a name core token with IDF in the top 20% | Distinctive-word matches the others missed | cap 20 by IDF sum |

If Step 1 shows **matches never cross countries**, restrict every channel to same-country pairs using an equality check on the label. There is still no hard-coded list, so France simply works.

### 12.2 Pruning to the final candidate set

The union can be large. Compute a **cheap score** for every candidate: max of the channel scores, plus the number of channels that found it. Keep the **top 50 per S1**. This pruned set is what the model scores and what we write to `candidate_pairs.tsv`, so the file is always honest.

### 12.3 Metrics to report (train, Half B)

- **Pair recall** = true pairs in candidates / all true pairs. **Target ≥ 0.98.**
- **Entity-complete recall** = share of S1 entities whose *all* true matches are in candidates.
- **Reduction ratio** = 1 − |candidates| / (|S1| · |S2 ∪ S3|).
- **Mean and 95th-percentile candidates per S1.**
- **Per channel:** unique contribution (true pairs found *only* by this channel). Use this for the methodology document and to drop useless channels.
- All of the above **per country**, and under the scrambled-letter transform.

### 12.4 Implementation notes

- TF-IDF: `TfidfVectorizer(analyzer="char_wb", ngram_range=(3,4), min_df=2, sublinear_tf=True)`. Fit on S1 ∪ S2 ∪ S3 of the current run. Top-k via chunked sparse matrix multiplication: 2,000 S1 rows at a time × transposed S2∪S3 matrix, then `argpartition`.
- Dense: FAISS `IndexFlatIP` on the GPU if available, CPU otherwise.
- Deduplicate the union; keep `channels_hit` as a bitmask feature.

### 12.5 What can go wrong

| Failure | Fix |
|---|---|
| Recall < 0.95 | Look at the missed pairs (by noise type from E8); raise k on the weakest channel; add a channel (e.g., `fold`-form TF-IDF for transliteration misses) |
| Candidate explosion for generic names ("City Pharmacy") | Cap per S1 by cheap score; rare-token channel already excludes common tokens |
| Very common postcodes (big cities) | The postcode channel requires a shared name core token as well |
| Memory blow-up on sparse multiplication | Smaller chunks; `float32`; only keep the top-k per chunk |
| Train/test blocking mismatch | Same code and parameters; report candidate-count distributions for train vs test per country; similar counts are expected |
| France candidates much fewer or more than others | Signals a normalisation or embedding problem; inspect 10 French records' candidate lists (bug detection only) |

---

## 13. Step 7 — Explain-the-difference features

This is the core idea of the solution. The same procedure runs on **name core tokens** and on **address tokens** separately.

### 13.1 Token-pair relation tests (ordered from strongest to weakest)

For tokens a (record 1) and b (record 2):

| Relation | Test | Strength |
|---|---|---|
| `exact` | a == b | 1.00 |
| `fold_exact` | fold(a) == fold(b) | 0.95 |
| `numeric_equal` | same number after stripping bis/ter/letter suffix | 0.95 |
| `typo` | Damerau-Levenshtein ≤ 1 (len ≤ 5) or ≤ 2 (len > 5), or Jaro-Winkler ≥ 0.92 | 0.85 |
| `prefix_abbrev` | shorter is a prefix of longer, len(shorter) ≥ 2 | 0.80 |
| `skeleton_abbrev` | same first letter AND shorter's letters are an ordered subsequence of longer's (Blvd→boulevard, Pvt→private, Ltd→limited, Bd→boulevard), len(shorter) ≥ 2, and len(shorter) ≤ 0.7·len(longer) | 0.70 |
| `split_join` | a == b1+b2 (e.g., walmart ↔ wal mart) | 0.90 |
| `initialism` | a's letters equal the initials of 2–5 consecutive tokens on the other side (sbi ↔ state bank of india) | 0.80 |
| `none` | — | 0 |

The strengths are **priors for the alignment only**. The model sees counts per relation type and learns their true value.

### 13.2 Alignment

1. Build the relation-strength matrix S between tokens of side 1 and side 2 (multi-token relations like `initialism` and `split_join` are handled as one-to-many links before the 1-to-1 step).
2. Solve a maximum-weight 1-to-1 assignment (`scipy.optimize.linear_sum_assignment` on −S; only links with S > 0 count).
3. Every token on either side is now either **explained** (aligned, with a relation type) or **unexplained**.
4. Unexplained tokens that are suffix-like or connector tokens are relabelled `dropped_generic` (weak evidence). Other unexplained tokens are **unexplained_content** (strong evidence).

### 13.3 Features produced (for each of name and address)

- `expl_frac_idf`: IDF-weighted share of tokens explained (both sides combined).
- `expl_frac_idf_min`: minimum over the two sides (catches "short name fully contained in a long one").
- `unexpl_max_idf_1`, `unexpl_max_idf_2`: the rarest unexplained word on each side. **The key negative signal.**
- `unexpl_sum_idf_1`, `unexpl_sum_idf_2`.
- `n_exact`, `n_typo`, `n_abbrev`, `n_initialism`, `n_splitjoin`, `n_dropped_generic`, `n_unexplained`.
- `order_kendall_tau` between aligned positions (reorder detection).
- `core_equal_after_expl`: 1 if every core token is explained on both sides.
- `first_token_explained`: the first name token usually carries identity.

### 13.4 Number agreement (separate, very important)

| Feature | Values |
|---|---|
| `house_no_rel` | equal / conflict / one-missing / both-missing |
| `postcode_rel` | equal / prefix-equal (first 3 digits) / conflict / missing |
| `unit_rel` | equal / conflict / missing |
| `name_number_rel` | equal / conflict / missing (7-Eleven vs 24x7) |
| `any_number_conflict` | 1 if any extracted number conflicts **and** is not explained by a range (12-14 contains 12) |
| `n_shared_numbers`, `jaccard_numbers` | counts |

A conflicting house number on otherwise identical records is the signature of a **different branch**. That is the main false-merge trap under F0.5.

### 13.5 What can go wrong

| Failure | Fix |
|---|---|
| `skeleton_abbrev` too permissive ("st" ↔ "south") | Length-ratio constraint; the model learns the weight; strongest relation wins in alignment; ambiguous short tokens (≤ 2 chars) get their own counter `n_short_abbrev` so the model can discount them |
| Speed (Python loops over millions of pairs) | Tokens per field are few (typically ≤ 10); use `multiprocessing` over pair chunks; cache token relations in a dict keyed by (a, b); rapidfuzz for distance |
| Landmark text treated as unexplained content | Landmarks are removed from address tokens (Step 2) and compared in their own feature |
| Legal-form difference counted as content | Suffix-likeness gating (Section 9.2) |

---

## 14. Step 8 — Full feature set

All features are **word-free and country-free**. Groups:

**A. Classic string similarities** (rapidfuzz), on name and address, each on `norm` and `fold`:
`ratio`, `partial_ratio`, `token_sort_ratio`, `token_set_ratio`, `WRatio`, Jaro-Winkler, normalised Damerau-Levenshtein, char 3-gram Jaccard, token Jaccard, IDF-weighted token Jaccard, TF-IDF cosine (name, address).

**B. Explain-the-difference** (Section 13.3): name and address.

**C. Numbers** (Section 13.4).

**D. Embedding similarities:** frozen full-record, name and address cosines. Fine-tuned full-record cosine **only in the combiner** (Section 10.3).

**E. Structure and missingness:** token counts per side, length ratio, empty flags, address has postcode (each side), landmark present (each side), landmark similarity.

**F. Blocking meta:** `channels_hit` bitmask, number of channels, rank of the candidate in each channel for this S1, candidate's dense rank among this S1's candidates.

**G. Source:** `is_S3` (S2 vs S3). Allowed, since it's a source, not a country. It lets the model learn per-source noise.

**H. Country:** only `same_country` (equality flag). **No raw country value.** If Step 1 shows every match is same-country and blocking already restricts to it, drop this feature (it would be constant).

**I. Collective** (Section 17): competition and sibling features, **combiner only**.

**Excluded on purpose:** raw tokens, token IDs, country one-hots, anything derived from the ground truth of other pairs (except via proper OOF).

---

## 15. Step 9 — Stage-1 LightGBM

### 15.1 Training

- Data: Half A candidate pairs, label = 1 if the pair is in the ground truth.
- Objective `binary`; `learning_rate 0.05`; `num_leaves 63`; `min_data_in_leaf 50`; `feature_fraction 0.8`; `bagging_fraction 0.8`, `bagging_freq 1`; `lambda_l2 1.0`; early stopping on group-k-fold validation (groups = S1).
- **No class re-weighting.** We need calibrated probabilities later. The imbalance is fine for trees.
- 5-fold group-k-fold on A → **OOF probabilities for A** (used to select the judge's training band), plus one model on all of A → predictions for B and test.

### 15.2 Optional robustness: monotone constraints

For an unseen country, it helps if the model *can't* learn nonsensical directions. Set monotone constraints:
- `+1` on `expl_frac_idf`, the name/address similarities and the embedding cosines;
- `−1` on `unexpl_max_idf_*` and `any_number_conflict`.

Validate on LOCO: keep them if LOCO F0.5 is equal or better (typically slightly lower in-country, better cross-country).

### 15.3 What can go wrong

| Failure | Fix |
|---|---|
| Model relies on one blocking-meta feature (e.g., dense rank) that behaves differently on test | Compare feature distributions train vs test (PSI); drop features with PSI > 0.25 on the non-France part of test |
| Overfitting A | Early stopping; LOCO check |
| Country leakage through proxies (e.g., postcode length = India) | Acceptable only if it doesn't hurt LOCO. Postcode *length* is not a feature, only agreement; check importance for proxies |

---

## 16. Step 10 — LLM judge (Qwen3-4B, QLoRA)

### 16.1 Why a judge, and why only on the uncertain band

The trees are fast and strong on clear cases. The judge adds *reading comprehension* where it matters: DBA/trade names, landmark-heavy addresses, and unusual formats (French). Scoring only the uncertain band keeps inference cheap and puts the judge's impact exactly where decisions flip.

### 16.2 Model and setup

- **Qwen3-4B** (Apache 2.0). Qwen3-8B is possible but roughly doubles T4 time for a small gain, so it is not worth it before the deadline.
- 4-bit NF4 quantisation (bitsandbytes), double quantisation on, `bnb_4bit_compute_dtype=torch.float16`.
- LoRA: r = 16, alpha = 32, dropout 0.05, targets = all linear projections (q, k, v, o, gate, up, down).
- Gradient checkpointing on; `attn_implementation="sdpa"`; `max_length = 384`.
- Optimiser `paged_adamw_8bit`, lr 1e-4, cosine schedule, warmup 3%, 1 epoch, effective batch 32 (micro-batch 4 × grad-accum 8), grad clip 0.3.
- Train on **one GPU** (DDP with 4-bit is fiddly and not worth it now). Use the second T4 for inference or embedding work.
- Qwen3 has a thinking mode: build prompts with the chat template and `enable_thinking=False`, and make the assistant turn start directly with the answer.

### 16.3 The evidence-augmented prompt (our distinctive twist)

```
<system>
You decide whether two business records describe the same real-world business
(same legal entity at the same location). Differences in abbreviations, legal
suffixes, word order, typos, transliteration and missing address parts are normal.
Different branches of the same chain at different addresses are NOT the same business.
Answer only "Yes" or "No".
</system>
<user>
Record A: name = "{raw_name_1}" | address = "{raw_addr_1}" | country = "{c1}"
Record B: name = "{raw_name_2}" | address = "{raw_addr_2}" | country = "{c2}"

Evidence (automatically computed):
- Name alignment: {e.g. 'corp'~'corporation' (abbreviation); 'intl'~'international' (abbreviation)}
- Unexplained name words: A: {..} | B: {..}
- Address alignment: {e.g. 'rd'~'road' (abbreviation); 'mg'='mg'}
- Unexplained address words: A: {..} | B: {..}
- Numbers: house {12 vs 12: equal}; postcode {411001 vs missing}; unit {none}
- Landmarks: A: {near sbi atm} | B: {none}
- Similarity: name {0.82}, address {0.64}, embedding {0.91}
Same business?
</user>
<assistant>Yes|No
```

The judge learns to weigh *structured evidence*, which transfers across languages, instead of memorising US/India vocabulary. The raw text is still present, so it can use its multilingual knowledge for trade names and French forms.

### 16.4 Training data (from Half A)

- **Band selection:** A-pairs with OOF stage-1 probability in [0.05, 0.95], plus a random 10% of confident pairs so the judge doesn't learn "everything is borderline".
- **Balance:** about 1 positive : 2 negatives, keeping all positives in the band.
- **Size:** **12k–20k examples.** On a T4 with 4-bit 4B at 384 tokens, expect roughly 1.5–3 hours per epoch. **Measure throughput in the first 50 steps and cut the dataset to fit a 3-hour budget.**
- **Augmentations:**
  - swap A/B order on 50% (removes position bias);
  - drop the address on one side for 5%;
  - scrambled letters on 5–10% (applied to raw text and evidence tokens consistently). Keep this low, since the judge's value partly *is* its language knowledge.
- **Loss** only on the answer token (completion-only).

### 16.5 Inference

- One forward pass per pair: `p = softmax(logit["Yes"], logit["No"])[Yes]`, read at the answer position. Check that "Yes" and "No" are **single tokens** in the Qwen3 tokenizer. If not, use the first sub-token of each consistently.
- **Test-time symmetry:** score both orders (A,B) and (B,A) and average. This doubles cost; drop it if time is short.
- Batch 16–32 with left padding; run two processes on the two T4s (`CUDA_VISIBLE_DEVICES=0` and `=1`), each on half the pairs.
- **Band at inference:** stage-1 probability in [0.10, 0.90] on B and test, capped at a budget N (sorted by |p − 0.5|, most uncertain first). Measure pairs/second, then set N to fit about 45 minutes per split.
- Pairs outside the band get `judge_p = NaN` (the combiner handles missing values), plus a flag `judge_scored`.

### 16.6 What can go wrong

| Failure | Detection | Fix |
|---|---|---|
| bitsandbytes / CUDA mismatch on Kaggle | Import error | Pin versions from 5.2; restart the kernel after pip installs |
| fp16 instability (NaN loss) | Loss `nan` | lr 5e-5; grad clip 0.3; `bnb_4bit_compute_dtype=float16` with LoRA weights in fp32 (peft default) |
| Too slow | Throughput < 1 it/s | Reduce max_length to 256 (shorter evidence block); reduce examples; micro-batch 2 with grad-accum 16 |
| Session timeout | | Save the adapter every 200 steps; resume from checkpoint |
| Judge collapses to always "No" | OOF AUC on B ≈ 0.5 | Rebalance positives; check the label-token masking; verify the loss is only on the answer |
| Judge over-trusts the "Similarity" numbers in the evidence | Ablation: judge with/without numbers | Keep the evidence focused on explanations and numbers, and drop the similarity scores if the ablation says so |
| Position bias | p(A,B) ≠ p(B,A) systematically | Order-swap augmentation plus test-time averaging |
| Judge doesn't help | Combiner LOCO gain < noise | Drop it; ship without it (safety submission) |
| Fallback judge if 4B fails entirely | | Fine-tune **Qwen3-Reranker-0.6B** (Apache 2.0) as a cross-encoder on the same pairs (much faster, about 40 min) |

---

## 17. Step 11 — Competition and sibling features (collective signals)

These are computed from **stage-1 probabilities** (out-of-sample on B and test) and feed the combiner.

### 17.1 Competition (S1s competing for one record)

For candidate record r and S1 entity e:
- `n_claimants(r)`: number of S1 entities with r in their candidates and stage-1 p > 0.05.
- `rank_among_claimants(e, r)`: 1 = e is the strongest claimant.
- `margin_to_best_other(e, r)` = p(e, r) − max over e′ ≠ e of p(e′, r).
- `second_best_p(r)`.

### 17.2 Within-entity context (candidates competing inside one S1)

- `rank_in_entity(e, r)`, `gap_to_top(e, r)`, `n_candidates_above_0.5(e)`, `entropy(e)` of normalised probabilities.

### 17.3 Sibling support (S2↔S3 duplicate evidence)

- Let `Conf(e)` be e's candidates with stage-1 p ≥ 0.7, excluding r.
- `sib_max_name_sim(e, r)`: max TF-IDF name cosine between r and any record in Conf(e).
- `sib_max_addr_sim`, `sib_number_agree` (shares house number or postcode with a confident sibling), `sib_count`.
- Needs pairwise S2↔S3 similarity only between candidates of the same S1 entity (cheap: ≤ 50² per entity).

### 17.4 What can go wrong

| Failure | Fix |
|---|---|
| Feedback loop: a wrong confident match drags in its siblings | Only use siblings with p ≥ 0.7; the combiner learns how much to trust sibling features; the exclusivity step later removes conflicts |
| Competition features differ in distribution between train and test (different density) | They are relative (ranks, margins), which transfers better than raw counts; check PSI |

---

## 18. Step 12 — Combiner and calibration

### 18.1 Combiner

- Data: Half B candidate pairs.
- Features: all groups A–H, plus `p_stage1`, `judge_p`, `judge_scored`, fine-tuned embedding cosine, and competition/sibling features.
- LightGBM with the same settings as stage-1 but `num_leaves 31` (smaller data).
- 5-fold group-k-fold by S1 → **OOF probabilities on all of B.** The final model is trained on all of B → applied to test.

### 18.2 Calibration

- Fit **isotonic regression** on B OOF (probability → empirical match rate).
- Check the reliability diagram and **ECE** per country. If isotonic overfits (step artefacts, small B), use **Platt scaling** on the logit.
- **Temperature/shrink parameter λ** for the decoder: `p' = sigmoid(λ · logit(p))`, tuned on B (Section 20.4). This protects against over-confidence on unseen countries.

### 18.3 What can go wrong

| Failure | Fix |
|---|---|
| Calibration fitted on US+India is wrong for France | We can't measure it on France. We choose **conservative** λ (slightly < 1 if LOCO shows over-confidence out of country), and check the predicted singleton rate on French test entities vs the train singleton rates (Section 23) |
| Leakage through group-k-fold misconfiguration | Groups = S1 id; assert no S1 in both train and validation fold |

---

## 19. Step 13 — One-owner rule (exclusivity)

Applied only if **E4** confirms that S2/S3 records map to ≤ 1 S1 in the ground truth.

### 19.1 Hard version (greedy, global)

1. Collect all (e, r, p) with p ≥ p_min (e.g., 0.05).
2. Sort by p, descending.
3. Walk down the list: assign r to e if r is unassigned; otherwise drop the pair (set p = 0 for that e).

### 19.2 Soft version

For each r with several claimants: `p'(e, r) = p(e, r) · (p(e, r) / Σ_e′ p(e′, r))^γ`, with γ tuned on B (γ = 0 means no rule, large γ approaches the hard rule).

### 19.3 Choosing

Evaluate none / soft / hard on B OOF with the decoder downstream; pick the best. With a clean deduplicated S1, **hard** usually wins.

### 19.4 What can go wrong

| Failure | Fix |
|---|---|
| The winner of a conflict is wrong, and both lose | Only compare when both claimants are candidates; the combiner's competition features already reduce such conflicts; the soft rule is available |
| Ground truth itself has a few multi-owner records | Soft rule, or hard rule with a small exception: keep both if their p difference is < δ and both are > 0.9 (tuned on B) |

---

## 20. Step 14 — Expected-F0.5 decoder

### 20.1 Idea

For each S1 entity we have calibrated probabilities p₁ ≥ p₂ ≥ … ≥ pₙ for its candidates. Try every "top-k" set, k = 0…n, and choose the one with the **highest expected F0.5**. This is the standard, well-studied way to maximise an F-measure in expectation: under independence, the optimal prediction is a top-k by probability, and scanning k finds the best one.

### 20.2 Exact computation

Let Y_i ~ Bernoulli(p_i), independent, plus an extra Bernoulli q for "a true match exists outside the candidate set" (q = estimated blocking miss rate for this entity, from B).

For a top-k prediction:
- TP ~ PoissonBinomial(p₁…p_k), FN ~ PoissonBinomial(p_{k+1}…p_n, q), FP = k − TP.
- Score(k, tp, fn):
  - k = 0: 1 if fn = 0, else 0;
  - k > 0, tp = 0: 0;
  - otherwise 1.25·tp / (1.25·tp + 0.25·fn + (k − tp)).
- E[F](k) = Σ over tp, fn of P(tp)·P(fn)·Score(k, tp, fn).

The Poisson-binomial distributions are computed by repeated convolution. Complexity is O(n³) per entity with n ≤ 50, fast with NumPy; we can also truncate to the top 20 candidates and fold the rest into FN. Reference code is in Appendix 28.3.

### 20.3 Why this beats a threshold

- The "empty" option is valued at P(no true match at all), exactly the singleton payoff.
- An entity with one very strong candidate and several weak ones gets top-1; an entity with three moderate candidates may still get all three. A single threshold can't do both.

### 20.4 Parameters tuned on B OOF

- λ (probability sharpening/softening), q scaling, and optionally a small `min_p` below which candidates are ignored.
- **Tune by a coarse grid only** (e.g., λ ∈ {0.7, 0.85, 1.0, 1.15}), to avoid overfitting.

### 20.5 What can go wrong

| Failure | Fix |
|---|---|
| Independence assumption is wrong (siblings are correlated) | λ tuning absorbs part of it; compare the decoder with a tuned global threshold on B. Keep the better one (the decoder usually wins, but we verify) |
| Miscalibrated probabilities → wrong k | Calibration (Step 12); λ; LOCO check |
| Slow on huge candidate lists | Truncate to top 20; vectorise |

---

## 21. Validation protocol

### 21.1 Scorer

Implement the official metric exactly (Appendix 28.1) and unit-test it on the worked example in the problem statement: predicted {47, 193, 812} vs true {47, 812} → **0.714**.

### 21.2 Main split

B OOF F0.5 (macro over B's S1 entities), reported overall and per country, plus:
- precision, recall and singleton accuracy (share of true singletons predicted empty);
- non-singleton F0.5;
- **the floor**: all-empty score = singleton rate.

### 21.3 Leave-one-country-out (LOCO): our main model-selection metric

- Combiner, calibration and decoder trained on B-US → evaluated on B-India, and the reverse.
- Report both directions and their mean. **All go/no-go decisions use LOCO-mean.**
- The embedder and judge are trained on A (both countries). For a stricter check, if time allows, also train stage-1 on A-US only and evaluate on B-India.

### 21.4 Scrambled-letter test (our France proxy)

- Apply one consistent random letter permutation to **every** record in B, then run the full inference pipeline (normalisation, blocking, features, models, decoder).
- Digits are unchanged, so numbers still work. Relationships between noisy variants survive; vocabulary is destroyed.
- Report F0.5 **drop vs unscrambled**. A small drop means the model reasons about structure. A large drop means it memorises vocabulary; if so, increase augmentation or remove the offending features.
- Also run a **dictionary-off test**: disable the universal suffix/connector lexicon and confirm the drop is small.

### 21.5 Gate for every component

A component is kept only if its **LOCO-mean** gain is above noise. Noise is estimated by bootstrap over S1 entities (1,000 resamples) of the difference in per-entity scores; keep the component if the 90% CI of the gain excludes 0, or the gain ≥ +0.003 with a non-negative scrambled-test effect.

### 21.6 Ablation table (goes into the methodology document)

| Configuration | B F0.5 | LOCO mean | Scrambled drop |
|---|---|---|---|
| All-empty floor | | | |
| Stage-1 + global threshold | | | |
| + calibration + expected-F decoder | | | |
| + one-owner rule | | | |
| + competition/sibling features | | | |
| + fine-tuned embedder | | | |
| + LLM judge | | | |

### 21.7 Public leaderboard usage

The public leaderboard is only a **sanity check** (format, gross bugs). Never pick between close variants by public score: it's a subset, the private set decides, and France is where the difference is.

---

## 22. Output writing and submission checks

### 22.1 Writing (manual, no pandas quoting surprises)

```python
def write_tsv(path, rows, header):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for s1, ids in rows:
            f.write(f"{s1}\t{','.join(ids)}\n")
```

### 22.2 Pre-flight assertions (script `check_outputs.py`)

- One row per test S1 ID, same set as `test_source1.tsv`; no duplicates.
- IDs in lists start with `S2-` or `S3-` and exist in the test S2/S3 files.
- No duplicates inside a list; no spaces; no quotes.
- Every matched ID ∈ that entity's candidate list.
- Empty lists written as an empty field (the line ends right after the tab).
- **France present:** count test S1 rows per country in the output = counts in the input.
- **Distribution sanity:** predicted empty-rate per country vs train singleton rates; mean predicted set size per country.
- Then run the official validator:
  `python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test` and expect **PASS**.

### 22.3 What can go wrong

| Failure | Fix |
|---|---|
| Windows line endings / BOM | `newline="\n"`, `encoding="utf-8"` (not `utf-8-sig`) |
| Float-formatted IDs | IDs read as `str` from the start |
| A matched ID not in candidates | Impossible by construction (final = filtered candidates); asserted anyway |
| Missing France rows because a country filter slipped in | Assertion per country |

---

## 23. Unseen-country (France) safeguards

| Layer | Safeguard |
|---|---|
| Normalisation | Accents, ligatures (œ), apostrophes (l', d'), bis/ter house numbers, 5-digit postcodes, "cedex", arrondissement formats ("75011", "Paris 11e") all handled by generic rules (digits as tokens, ordinal suffix stripping of e/er/eme/ème) |
| Abbreviations | Structure-based (prefix, skeleton, initialism), so Av/Bd/Chem/Imp/Pl are covered without a French list. French street-type abbreviations commonly shorten words not in the official AFNOR list to their first four letters (Chemin→CHEM), which our prefix test catches directly |
| Legal suffixes | Statistical suffix-likeness picks up SARL/SAS/SASU/EURL/SA from the test corpus itself (no labels) |
| Features | No country value, no vocabulary |
| Embedder / judge | Multilingual pretrained; fine-tuning restrained (1 epoch, low lr, scrambled augmentation); frozen cosine kept as a fallback feature |
| Calibration | Conservative λ; LOCO-verified |
| Monitoring without labels | For French test entities: predicted empty rate, mean set size, probability histogram, candidate counts. If they are far from US/India (e.g., 90% empty predicted while train singleton rate is 30%), investigate normalisation/blocking bugs. **Do not tune thresholds on this**, only fix bugs |

---

## 24. Sprint timeline, gates and fallbacks

Assume the deadline is tomorrow at time **D**; the **freeze is at D − 4h**. Clock times below assume starting at 15:30 today.

| Time | CPU track (Person 1) | GPU track (Person 2) | Validation / packaging (Person 3) |
|---|---|---|---|
| 15:30–17:00 | Loading + sanity (Step 0), EDA (Step 1) | Env setup; frozen embeddings for train + test | Scorer + unit test; `check_outputs.py`; repo skeleton |
| 17:00–19:00 | Normalisation, corpus statistics, A/B split | Hard-negative mining (needs split); start embedder fine-tune | Blocking metrics script; scrambled-letter transform |
| 19:00–21:30 | TF-IDF / number / rare-token blocking; recall report; explain-the-difference features | Embedder acceptance test; B + test dense candidates; start building judge prompts once stage-1 OOF exists | LOCO harness; bootstrap CI |
| 21:30–22:30 | Stage-1 LightGBM (A, OOF); **SAFETY SUBMISSION #1** (stage-1 + calibration + decoder on test) | Judge dataset build (band from A OOF) | Validate & upload #1; log score |
| 22:30–02:00 | Competition + sibling features; combiner v0 (without judge); one-owner rule; decoder tuning | **Judge QLoRA training** (≤ 3h, checkpoints) | Ablation table filling; README draft |
| Morning (08:00–10:00) | **SAFETY SUBMISSION #2** (combiner v0) | Judge inference on B + test (2 GPUs) | Methodology doc draft |
| 10:00–D−4h | Combiner v1 (+ judge); final LOCO + scrambled; **MAIN SUBMISSION** | Buffer: rerun if needed; optional order-swap TTA | Packaging, requirements pin, validator PASS |
| D−4h → D | **FREEZE.** Only packaging, docs, re-validation | — | Zip, final checks, upload |

### Gates (go/no-go)

| Gate | Condition | If it fails |
|---|---|---|
| G1 (19:00) | Blocking pair recall ≥ 0.95 on B | Add fold-TF-IDF channel, raise k; don't proceed until ≥ 0.95 |
| G2 (20:30) | Fine-tuned embedder beats frozen on recall@20 in both countries | Use frozen for blocking; fine-tuned cosine only as a feature |
| G3 (22:30) | Safety #1 validator PASS and > all-empty floor on B | Debug before anything else |
| G4 (morning) | Judge OOF AUC on B band > stage-1 AUC on same band | Drop the judge or switch to the Reranker-0.6B fallback |
| G5 (D−4h) | Final model beats safety #2 on LOCO-mean | Submit safety #2 as final |

---

## 25. Master risk register

| # | Risk | Likelihood | Impact | Detection | Mitigation |
|---|---|---|---|---|---|
| R1 | TSV parsing corrupts rows (quotes, NA) | Med | High | Row-count assert | Safe loader (6.1) |
| R2 | Leakage inflates validation → over-confident test | Med | High | LOCO vs B gap; calibration curve | A/B split, OOF, groups = S1 |
| R3 | France performance collapses | Med | **Very high** (private LB) | Scrambled test; French prediction distribution | Word-free features, no country feature, multilingual models, conservative λ |
| R4 | Blocking recall too low | Med | High | Recall report | Multi-channel union; per-channel diagnostics |
| R5 | Candidate explosion / memory | Med | Med | Candidate stats | Top-50 cap; chunked sparse ops |
| R6 | Judge training too slow / fails | Med | Med | Throughput log | Budget-based dataset size; Reranker-0.6B fallback; ship without |
| R7 | Kaggle session timeout / quota | Med | Med | Session timer | Checkpoints; parallel GPU sessions; ≤ 8h jobs |
| R8 | Library/version breakage on Kaggle | Med | Med | Import check cell | Pinned versions; restart after install |
| R9 | fp16 NaNs on T4 | Low-Med | Med | Loss log | Lower lr, clipping, fp32 fallback |
| R10 | Overfitting public LB | High (for others) | High | — | Select by LOCO only |
| R11 | Format rejection | Low | Very high | Validator | `check_outputs.py` + official validator before every upload |
| R12 | `candidate_pairs.tsv` not matching model input (audit fail) | Low | High (disqualification-level) | Assertion | Write candidates from the scored table; matches ⊆ candidates |
| R13 | Rule violation (external data) | Low | Disqualification | Code review | No libpostal/geocoders/APIs; no test-data training; licence list in README |
| R14 | Test-time statistics judged non-compliant | Low | Med | Organiser reply | Train-only statistics fallback (9.4) behind a config flag |
| R15 | Exclusivity assumption false | Low | Med | EDA E4 | Soft rule / none, chosen on B |
| R16 | Independence assumption in decoder | Med | Low-Med | Decoder vs threshold on B | λ tuning; keep whichever is better |
| R17 | Time overrun | High | High | Gates | Layered safety submissions; freeze |
| R18 | Team merge conflicts / overwritten artifacts | Med | Med | — | Versioned artifact names; one owner per module |
| R19 | Non-reproducible final run | Med | High (top-team review) | Clean-run test | `run_all.sh`; seeds; pinned requirements; README with timings |
| R20 | Generic chain names → false merges | High | High | Hard-negative analysis | Number features, unexplained-rare-word features, exclusivity, judge prompt rule on branches |
| R21 | Transliteration misses (India) | Med | Med | Missed-pair analysis | `fold` channel and features; char n-grams; embeddings |
| R22 | Landmark-only addresses | Med | Med | E8 | Separate landmark field/feature; judge sees raw text |
| R23 | Duplicate S1 rows in test | Low | High | Assertion | Dedupe on write; assert uniqueness |
| R24 | Non-Latin scripts in test | Low | Med | EDA E12 | Unicode-preserving normalisation; embeddings |

---

## 26. Code package structure and reproducibility

```
code/business_entity_resolution/
├── README.md                 # exact commands, hardware, expected runtimes, licences
├── requirements.txt          # pinned versions (frozen from the Kaggle run)
├── configs/
│   └── default.yaml          # all hyperparameters, k values, band limits, λ, flags
├── src/
│   ├── io_utils.py           # safe loaders/writers, assertions
│   ├── normalize.py          # Step 2
│   ├── corpus_stats.py       # Step 3 (IDF, suffix-likeness, street-type-likeness)
│   ├── split.py              # Step 4
│   ├── embed.py              # Step 5 frozen encode
│   ├── finetune_embedder.py  # Step 5 fine-tune
│   ├── blocking.py           # Step 6
│   ├── explain_diff.py       # Step 7
│   ├── features.py           # Step 8
│   ├── stage1.py             # Step 9
│   ├── judge_data.py         # Step 10 prompt building
│   ├── judge_train.py        # Step 10 QLoRA
│   ├── judge_infer.py        # Step 10 scoring
│   ├── collective.py         # Step 11
│   ├── combiner.py           # Step 12 + calibration
│   ├── exclusivity.py        # Step 13
│   ├── decoder.py            # Step 14
│   ├── metrics.py            # scorer, LOCO, bootstrap
│   ├── scramble.py           # scrambled-letter transform
│   └── predict.py            # end-to-end inference on test → output/
├── scripts/
│   ├── run_train.sh          # data → trained artifacts
│   └── run_predict.sh        # artifacts + test → output/*.tsv
└── notebooks/                # Kaggle notebooks mirroring the scripts (optional)
```

README must include:
1. environment (Python version, GPU);
2. how to download base models (HF IDs);
3. `run_train.sh` then `run_predict.sh`;
4. expected runtimes;
5. where outputs land;
6. licence table;
7. a statement that no external data or lookups are used.

---

## 27. Mapping to the methodology document

| Template section | Content source in this plan |
|---|---|
| Methodology used | Sections 3, 4, 20 (metric-aware decoding), 13 (explain-the-difference) |
| Candidate generation / blocking | Section 12 + recall/reduction table (12.3) + per-channel contribution |
| Model architecture and feature engineering | Sections 13–18 |
| Other relevant information | Validation (21), France safeguards (23), compliance (2), ablations (21.6), risks handled (25) |

---

## 28. Appendix — reference code

### 28.1 Official-metric scorer

```python
def f05_entity(pred: set, true: set) -> float:
    if not true and not pred:
        return 1.0
    if not true or not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)

def f05_macro(pred_map: dict, true_map: dict) -> float:
    ids = list(true_map)
    return sum(f05_entity(pred_map.get(i, set()), true_map[i]) for i in ids) / len(ids)

# unit test from the problem statement
assert abs(f05_entity({"S2-00047","S2-00193","S3-00812"}, {"S2-00047","S3-00812"}) - 0.714) < 1e-3
```

### 28.2 Scrambled-letter transform

```python
import random, string
def make_scrambler(seed: int):
    rng = random.Random(seed)
    src = string.ascii_lowercase
    dst = list(src); rng.shuffle(dst)
    table = str.maketrans(src + src.upper(), "".join(dst) + "".join(dst).upper())
    return lambda s: s.translate(table)   # digits, spaces, punctuation unchanged

# apply the SAME scrambler to both records of a pair (or to the whole split for the test)
```

Apply it **after** Unicode accent folding, so accented letters are scrambled consistently too.

### 28.3 Expected-F0.5 decoder

```python
import numpy as np

def poisson_binomial(ps):
    pmf = np.array([1.0])
    for p in ps:
        pmf = np.convolve(pmf, [1.0 - p, p])
    return pmf

def best_k(probs, q_outside=0.0, beta2=0.25, max_n=20):
    order = np.argsort(-np.asarray(probs))
    p = np.asarray(probs)[order][:max_n]
    rest = np.asarray(probs)[order][max_n:]
    n = len(p)
    best_val, best_kk = -1.0, 0
    for k in range(0, n + 1):
        tp_pmf = poisson_binomial(p[:k])                       # len k+1
        fn_pmf = poisson_binomial(np.concatenate([p[k:], rest, [q_outside]]))
        tp = np.arange(len(tp_pmf))[:, None]
        fn = np.arange(len(fn_pmf))[None, :]
        if k == 0:
            score = (fn == 0).astype(float) * np.ones_like(tp)
        else:
            fp = k - tp
            denom = (1 + beta2) * tp + beta2 * fn + fp
            score = np.where(tp > 0, (1 + beta2) * tp / np.maximum(denom, 1e-12), 0.0)
        val = float((tp_pmf[:, None] * fn_pmf[None, :] * score).sum())
        if val > best_val:
            best_val, best_kk = val, k
    return order[:best_kk], best_val
```

### 28.4 Abbreviation tests

```python
def is_subsequence(short, long):
    it = iter(long)
    return all(ch in it for ch in short)

def token_relation(a, b):
    if a == b: return "exact"
    s, l = (a, b) if len(a) <= len(b) else (b, a)
    if len(s) >= 2 and l.startswith(s): return "prefix_abbrev"
    if len(s) >= 2 and s[0] == l[0] and len(s) <= 0.7 * len(l) and is_subsequence(s, l):
        return "skeleton_abbrev"
    return None   # typo / fold / split-join / initialism tested elsewhere
```

### 28.5 Safe loader

See Section 6.1.

---

## 29. Sources

- Qwen3 Embedding / Reranker licence and sizes: https://qwenlm.github.io/blog/qwen3-embedding/ · https://huggingface.co/Qwen/Qwen3-Reranker-0.6B · https://arxiv.org/abs/2506.05176
- Kaggle GPU limits: https://www.kaggle.com/docs/efficient-gpu-usage · https://www.kaggle.com/product-feedback/361104 · https://aimultiple.com/free-cloud-gpu
- libpostal training data (why we avoid it): https://github.com/openvenues/libpostal
- Expected F-measure maximisation via top-k scanning: https://arxiv.org/pdf/1904.09235 · https://arxiv.org/pdf/1505.01802 · Top-Personalized-K (expected utility with calibrated probabilities): https://arxiv.org/pdf/2402.16304
- Rarity-weighted token matching for company names: https://tilores.io/content/company-name-normalization-isnt-enough-for-fuzzy-matching/ · https://etinginfrati.com/company-matching/
- French street-type abbreviation convention (AFNOR XP Z10-011): http://ressources.sitilr.fr/document/types-de-voie-norme-afnor-xp-z-10-011/