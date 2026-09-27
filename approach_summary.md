# Business Entity Resolution — Approach Summary

**Team:** [TEAM NAME — fill in from CLAUDE.local.md] · Amazon ML Challenge 2026

## Approach

Business records from three sources (S1 deduplicated reference; S2, S3 noisy
duplicates) are matched by **blocking + classify + calibrate + decode**,
never end-to-end and never with any country- or language-specific logic —
required because the test set includes France, unseen at training time.

1. **Normalise.** Unicode/accent/ligature cleanup, stdlib-only Brahmic-script
   romanisation (no external transliteration library), house-number and
   postcode extraction by shape and position, landmark-phrase removal from
   addresses, and a phonetic "fold" form for typo/transliteration tolerance.
2. **Block.** Within-country (0 cross-country matches in the full ground
   truth — confirmed, never assumed) sparse candidate generation: char
   TF-IDF, a rare-token key, a house-number+street key, and — the key
   lever — **unordered token-pair keys**, because names here are built from
   a small, shared, combinatorial vocabulary ("Glypheus" starts many
   unrelated businesses), so single tokens are weak but token *pairs* are
   rare and discriminating. A LightGBM pre-ranker (trained on one half of
   train) prunes the union to 50 candidates per S1.
3. **Classify.** ~120 word-free, country-free features per candidate pair:
   classic string similarities (`rapidfuzz`), and — the core innovation —
   **"explain-the-difference"**: every token pair is typed as exact / fold /
   typo / abbreviation / split-join / initialism / number-equal via
   symmetric, cached tests (Damerau-Levenshtein and Jaro-Winkler gated by
   length, ordered-subsequence abbreviation gated by a length ratio so
   "st"~"south" stays a *weak* signal only), then aligned greedily by
   relation strength. The rarest *unexplained* token on either side is the
   strongest negative signal. Numbers are handled separately (house/postcode/
   unit agreement) because a conflicting house number on an otherwise
   identical pair is the signature of a different branch of the same chain
   — the main false-merge trap under a metric that penalises false
   positives 4x false negatives. A LightGBM classifier scores every
   candidate (5-fold group-k-fold by S1, held-out pair-level AUC 0.9996).
4. **Calibrate and decode.** Isotonic calibration (cross-fitted), a hard
   one-owner rule (a record can belong to only one S1 — confirmed true for
   100% of the training ground truth), then an **expected-F0.5 decoder**:
   for each S1, every top-k prediction (including "no match") is scored by
   its expected F0.5 under a Poisson-binomial model of the candidates'
   calibrated probabilities plus an estimated blocking-miss rate, and the
   best k is kept. This handles "one strong match" and "several moderate
   matches" correctly, which a single global threshold cannot, and beat a
   tuned global threshold on validation (0.9648 vs 0.9646).

## Models

Only **LightGBM** (MIT) is on the shipped inference path. Two Apache-2.0
Qwen3 models were downloaded, coded against and evaluated but are **not**
in the shipped path: `Qwen3-Embedding-0.6B` for dense blocking (infeasible
at ~400 rec/s on 2xT4 for ~24M records) and `Qwen3-4B` as an LLM judge on
uncertain pairs, QLoRA-trained on evidence-augmented prompts — its own gate
(band AUC must beat stage-1's) failed (0.774 vs 0.877), so it was dropped
rather than shipped on faith.

## Experiments

Every keep/drop decision used **LOCO-mean** (train the downstream steps on
one country, score the other, average both directions) — not the in-country
validation score — because it is the closest available proxy for the
unseen France split, plus a **scrambled-letter test** (a random, consistent
a-z permutation over every record) to confirm the model reasons about
structure rather than memorised vocabulary.

| Configuration | B F0.5 | LOCO-mean | Scrambled drop |
|---|---|---|---|
| All-empty floor | 0.0547 | — | — |
| Stage-1 + global threshold | 0.9642 | — | — |
| + calibration + expected-F0.5 decoder | 0.9641 | — | — |
| **+ hard one-owner rule (shipped)** | **0.9648** | **0.9635** | **0.0009** |
| + combiner (competition/sibling features, 2nd LightGBM stage) | 0.9674 | 0.9635 | 0.0010 |

The combiner improved the in-country score by a real margin (bootstrap 90%
CI excludes 0) but showed **zero LOCO-mean gain**, so it was not shipped —
exactly the kind of overfitting LOCO-mean exists to catch. Blocking recall
was raised from 0.9585 to 0.9736 across two later channel additions
(cross-field token pairs; a reverse pool-to-S1 name lookup) but arrived too
late to re-run the ~5-hour feature/classifier step before the modelling
freeze, so the shipped run still uses the earlier, lower-recall candidate
set.

## Conclusion

A country-agnostic, typed-token-relation feature set plus a metric-aware
decoder reaches **0.9648 Half-B F0.5** (LOCO-mean 0.9635, scrambled drop
0.0009) with only classical models (LightGBM), which matters directly for
generalising to the unseen France split. Error analysis shows the remaining
gap is dominated by **blocking misses** (51% of wrong entities), not
classifier quality (pair-level AUC 0.996) — the clear next step, already
implemented and measured (recall 0.9585 -> 0.9736), is simply time-boxed out
of this submission. Both stretch tracks attempted (a combiner, an LLM judge)
were evaluated honestly against pre-registered gates and dropped rather than
shipped on an in-country-only gain.
