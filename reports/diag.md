# diag.md — Read-only diagnostic (2026-09-27, no Kaggle jobs; built from `reports/raw/*` already on disk)

**Scope note:** run without new Kaggle kernels (packaging is on the critical path to the 19:30 freeze; the
14:20 IST decision to abort further model changes stands — see `progress.md`). Several D-items need data
that was never computed/saved (candidate-level neighbour ranks, full missed-pair sets, per-group feature
timing) and are marked N/A below rather than triggering a new run.

## D1 — Error composition (proxy: NB09 combiner B OOF, 56,376/250,000 wrong; exact sub-01/stage-1-pipeline
buckets were never computed — combiner gains only +0.0026 F0.5 so this is a close proxy, not identical)
| bucket | count | share of wrong |
|---|---|---|
| blocking_miss | 28,702 | 50.9% |
| other | 21,336 | 37.9% |
| heavy_noise | 4,975 | 8.8% |
| chain_branch | 627 | 1.1% |
| false_match_on_singleton | 520 | 0.9% |
| multi_claim_conflict | 216 | 0.4% |
F0.5-lost-per-bucket not separately logged (only counts) — N/A without a rerun.
**Singleton non-empty rate:** stage-1 pipeline (sub-01) `singleton_acc` 0.9447 → **5.53% of true singletons predicted non-empty**; combiner 0.9620 → 3.80%.

## D2 — Blocking funnel (v6, B ALL, `nb05_sparse_v6/blocking_report.json`)
Union recall (pre-prune) **0.97862**; pair recall after top-50 prune **0.97358** → **pruning loss 0.0050**. Entity-complete recall **0.91747**.
Per-channel found/unique (B ALL): addr_pair 754,458/25,823; name_pair 566,362/19,737; cross_pair 730,656/11,517; num_key 431,304/575; addr_char 309,987/804; name_tok 141,842/91; reverse 121,574/301; name_char 159,430/146; dense 0/0 (dropped, infeasible per memory §5).

## D3 — Missed true pairs, rank/quality (approx: v5's 200-row `missed_B.tsv` sample only — v6 saved no
missed-pairs file, and a full k=500 re-rank was not run; difflib ratio used as a name/address similarity proxy for TF-IDF cosine)
- n=200 sampled misses: **22% already in the pre-prune union but cut by the top-50 prune**; 78% not in the union at all (blocking-channel miss, not a pruning loss).
- Same-country rate: **100%** (structural — within-country blocking, E5).
- Exact-name rate: **1%**. Name-sim quartiles (p25/p50/p75): **0.13 / 0.65 / 0.83**. Address-sim quartiles: **0.00 / 0.35 / 0.52**.
- 34.5% of sampled misses have an empty name/address field on one side.
Full k=500 rank histogram: **N/A** — would need a rerun of blocking with an extended k (a new Kaggle job), out of scope for this pass.

## D4 / D5 — Sibling reachability / reverse coverage: **N/A**. Neither the per-S1 neighbour lists nor the
full candidate table are in `reports/raw/` (by design — large parquet/npy is never fetched, per CLAUDE.md §6). Answering these needs a Kaggle job over the saved candidate parquet; not run here.

## D6 — Recall by #true-matches-per-S1: **N/A** in saved artifacts (blocking_report.json reports aggregate recall only, not stratified by match count).

## D7 — Ten missed pairs, raw text (train, v5 sample; bucket = D3 "not in union" unless noted)
1. S1 "Future Technology Pvt Ltd" / Delhi… vs S3 "फ्यूचर टेक्नोलॉजी प्रा. लि." — native-script + address drift.
2. "Frerichs Entertainment LP" vs "FRERICHS ENTERTOIMNNET LP" (heavy typo, empty address on match side).
3. "Gujarat Consultants LLP" vs Bengali-script name, address barely overlapping.
4. "Pediatric Center of Omaha" vs domain-handle name "centeromaha.com".
5. "Future Enterprises Private Limited" vs Devanagari name, reordered address.
6. **[in-union, pruned]** "Global Energy Private Limited" vs Devanagari name, partial address overlap.
7. "J/U Earths LLC" vs "J/U LLC Center" — generic-token collision, empty address.
8. "Ace Digital Agro Private Limited" vs Devanagari, address near-miss.
9. **[in-union, pruned]** "Temple Chapel" vs "Temple Chabpsl" (typo) + city typo "Buccksport".
10. "Supertech & Co" vs "Sueretecih & [Co]" — heavy character-level typo on both fields.
(Full sample: `reports/raw/nb05_sparse_v5/missed_B.tsv`.)

## D8 — Non-blocking errors
Decoder/exclusivity in production (sub-01, `nb09a_v2`): **method=decoder, mode=hard (one-owner), lam=1.3**; combiner variant (not shipped) best at lam=1.0/1.15. ECE (B, isotonic-calibrated, overall): **6e-05** (raw 5.6e-04) — **not split by country in any saved run**.
5 FP-on-singleton examples (raw text + p, from `nb09_v2/reports/errors.md`):
- "Kwality Estates" (Haveri) vs "Smt Kwality Estates Enterprises Corp" — p=0.587
- "Ram Green Construction LLP" vs unrelated "Arcdovahalo" sharing the same address string — p=0.888
- "Priester Webster Corporation" vs "Priester Webster Co" (house-no off by 5) — p=0.915
- same S1 vs "Priester Webster Corp" (different branch cue "Fruitridge Pocket") — p=0.946

## D9 — Feature cost profile
Per-feature-group timing: **N/A** (NB06 logs only full-pipeline vs cheap-prerank timing, not per feature group). Aggregate (v4 candidates, from `nb06_v1/metrics.json`): full features **0.4786 ms/pair**, cheap pre-rank **0.0676 ms/pair**; total pairs A 3.0M / B 5.0M / test 34.65M (v4; **v6 was never run through NB06** — Track 1 aborted before this step, so no v6 pair counts exist). Share of pairs the cheap model alone resolves confidently (p<0.001 or >0.999): **N/A**, not logged.

## D10 — Resources
- **GPU hours:** not tracked as a running total in any artifact. Measured GPU spend so far: NB07 dry (~min) + NB07 full train **69.4 min** (`train_s` 4193.9s) + judge inference dry (T4×2, negligible) on R2; NB03 dry v1–v3 (few min each) on R1/R2. Rough estimate **≤2 h used of the ~30 h/week quota per runner** — not verified against the Kaggle usage page.
- **Max concurrent CPU sessions on Kaggle:** not checked (would need the Kaggle site/API, out of scope for a read-only local pass).
- **Local disk free:** last measured **~2 GB** (`progress.md`, after freeing 2.2 GB of stale upload zips ~03:25 IST) — stale, worth rechecking before fetching more large outputs.
- **sub-01 public LB score:** not yet given by the user (`progress.md`: "pending, user uploads").
- **Uploads used today (27 Sep):** **1 / 5** (sub-01).
