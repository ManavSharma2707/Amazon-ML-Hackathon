# Business Entity Resolution

A solution to the Business Entity Resolution Challenge, built for the Amazon ML Challenge 2026. The task is to match business records describing the same real-world entity across three noisy, differently-sourced tables, at a scale of several million records per file, without using any external data, lookups, or pretrained knowledge beyond the models permitted by the competition rules.

## Problem

Three sources of business records are provided: `S1`, a deduplicated reference table, and `S2` and `S3`, two independently collected tables containing duplicates, typos, transliterations, and other real-world noise. For every record in `S1`, the task is to find every record in `S2` and `S3` that refers to the same business, or to correctly report that none exists. Training data covers the United States and India; the test set adds France, a country never seen during training, which rules out any approach that relies on country-specific rules, vocabulary, or hand-tuned thresholds.

The evaluation metric is macro-averaged F0.5 per `S1` entity, which weights precision roughly four times as heavily as recall. A single false positive costs as much as four missed matches, which shapes nearly every design decision in this pipeline, from blocking through the final decision rule.

## Approach

The pipeline follows a standard four-stage entity-resolution design, kept deliberately simple and free of language- or country-specific logic throughout:

**Normalisation.** Unicode and accent cleanup, ligature folding, and a from-scratch, stdlib-only romanisation of Brahmic scripts (no external transliteration library, since none is permitted under the competition rules). House numbers and postcodes are extracted by shape and position rather than by country-specific format, landmark phrases are separated out of addresses, and a phonetic "fold" form is derived for typo and transliteration tolerance.

**Blocking.** Candidate generation is restricted to matching records within the same country, a property confirmed directly from the training ground truth rather than assumed. Several sparse channels contribute candidates: character-level TF-IDF, a rare-token key, a house-number-and-street key, and, as the strongest single contributor, unordered token-pair keys. Business names in this dataset are built from a small, shared, combinatorial vocabulary, so any single token is a weak signal on its own, but a pair of tokens occurring together is rare and highly discriminating. A LightGBM pre-ranker, trained on a held-out half of the training data, prunes the resulting candidate union down to fifty candidates per `S1` entity.

**Feature extraction and classification.** Each candidate pair is scored using around 120 features that avoid raw vocabulary or country signals. Alongside standard string-similarity measures, the central idea is an "explain-the-difference" feature set: every pair of tokens across two records is classified as an exact match, a phonetic fold, a typo, an abbreviation, a split-or-joined form, an initialism, or a numeric match, using a set of symmetric, cached comparisons. The rarest token left unexplained on either side turns out to be the strongest signal of a genuine mismatch. Numeric fields (house numbers, postcodes) are handled separately, since a conflicting house number on an otherwise near-identical pair is the classic signature of two different branches of the same chain, exactly the kind of false positive the metric punishes most severely. A LightGBM classifier, trained with grouped cross-validation so that no `S1` entity leaks across folds, scores every candidate pair.

**Calibration and decoding.** Raw classifier scores are calibrated with cross-fitted isotonic regression. A hard one-owner rule is then applied, since the training data confirms that no `S2` or `S3` record ever belongs to more than one `S1` entity. Finally, an expected-F0.5 decoder considers every plausible prediction set for each entity, including the empty set, scores each one under a Poisson-binomial model of the calibrated probabilities, and keeps the set with the highest expected score. This handles both "one confident match" and "several moderate matches" correctly in a way that a single global threshold cannot.

## Results

| Configuration | Half-B F0.5 | LOCO-mean | Scrambled-letter drop |
|---|---|---|---|
| All-empty baseline | 0.0547 | — | — |
| Stage-1 classifier, global threshold | 0.9642 | — | — |
| Calibration and expected-F0.5 decoder | 0.9641 | — | — |
| Hard one-owner rule (shipped configuration) | 0.9648 | 0.9635 | 0.0009 |
| Second-stage combiner (evaluated, not shipped) | 0.9674 | 0.9635 | 0.0010 |

Model selection throughout this project was driven by a metric called LOCO-mean: the pipeline is trained on one country and scored on the other, in both directions, and the two scores are averaged. Because France appears only at test time, in-country validation scores are a poor proxy for how a change will generalise, while LOCO-mean approximates exactly that. A scrambled-letter test, which applies a consistent random substitution cipher to every character before scoring, was used alongside it to confirm that the model reasons about structural relationships between tokens rather than memorising specific vocabulary; a small drop under scrambling is the desired outcome.

A second-stage combiner, adding features about how many other candidates compete for the same records, improved the in-country score by a statistically significant margin but produced no LOCO-mean gain, so it was evaluated, documented, and deliberately left out of the shipped pipeline rather than shipped on the strength of an in-country number alone.

## Models used

Only LightGBM (MIT licence) sits on the inference path that produces the submitted output, alongside classical string-similarity functions from `rapidfuzz`. Two additional models permitted under the competition's licence and parameter-count rules, both from the Qwen3 family under an Apache-2.0 licence, were downloaded, integrated, and evaluated but are not part of the shipped pipeline:

- `Qwen3-Embedding-0.6B` was intended for dense semantic blocking. Measured throughput on the available hardware was roughly 400 records per second, which made a full pass over the multi-million-record pool infeasible within the competition's time budget, so this path was dropped after being fully implemented and measured rather than left untested.
- `Qwen3-4B` was fine-tuned with QLoRA as a judge model for the most uncertain candidate pairs. Its own pre-registered gate required its accuracy on that band to exceed the existing classifier's, and it did not clear that bar, so it was dropped rather than shipped on faith.

Both attempts, along with the reasoning behind dropping each one, are documented in full inside `Documentation_template.md` and in the code's own comments, since an honest account of what did not work was treated as being as important as what did.

## Compliance

No external APIs, geocoders, business registries, web lookups, or third-party re-uploads of the dataset are used anywhere in this pipeline. Nothing is trained on the test set; test predictions were inspected only to catch formatting or encoding bugs, never to tune a threshold or feature. Country is treated as an open set throughout: no country value is ever hard-coded, one-hot encoded, or filtered on, and the only country-derived signal used anywhere is a same-country equality flag, which is why the pipeline transfers to France without modification. Every model runs entirely offline once its weights are downloaded once at the start of the project.

## Repository layout

```
code/business_entity_resolution/   all source code for the submitted pipeline
    src/                            normalisation, blocking, features, models, decoding
    src/notebooks/                  thin Kaggle-notebook drivers, one per pipeline stage
    src/tests/                      unit tests, runnable against a small local sample
    src/configs/default.yaml        every tunable parameter in one place
    requirements.txt                pinned dependencies, captured from the actual run
tools/                              helper scripts for driving Kaggle notebooks and sampling
output/                             final submission files (matching results, candidate pairs)
reports/                            diagnostic notes and experiment logs
Documentation_template.md           full methodology, ablations, and error analysis
approach_summary.md                 a short summary of the approach and results
architecture.md                     how the pipeline maps onto notebooks and data artifacts
```

## Reproducing this work

The pipeline is organised as a sequence of notebooks, each one a thin driver around the code in `src/`, meant to run on Kaggle against the competition dataset. `src/notebooks/` contains one folder per stage, from normalisation and blocking through feature extraction, model training, and final decoding, along with a few exploratory variants that were tried and evaluated but not shipped. `code/business_entity_resolution/README.md` documents the exact commands, expected runtimes, and hardware used for each stage. Unit tests in `src/tests/` run against a small local data sample and do not require the full dataset or any Kaggle infrastructure.

## Acknowledgements

Built for the Amazon ML Challenge 2026, Business Entity Resolution Challenge.
