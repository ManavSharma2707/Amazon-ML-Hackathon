"""NB01 logic — exploratory data study (master plan SS7 / architecture.md SS6.3).

Answers E1-E12 plus the shortcut check, and measures the all-empty floor.
Designed to run as a Kaggle CPU kernel: the full dataset (~2.4 GB, millions of
rows) does not fit comfortably in this project's local RAM budget (memory.md
SS5), so this module is meant to be pushed and run there, not locally.

Country is treated as an open set throughout: no country value is ever
hard-coded into a comparison; every country-conditioned statistic groups by
whatever string is present in the data (CLAUDE.md SS3 rule 4).
"""

from __future__ import annotations

import difflib
import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from . import io_utils

_START_TIME = time.time()


def _log(msg: str) -> None:
    """Print a timestamped progress line and flush immediately.

    `run_eda` can take tens of minutes on the full dataset with nothing
    printed until the very end otherwise (CLAUDE.md SS5: log timings per
    stage). `flush=True` matters because a long-running Kaggle kernel's
    stdout is otherwise buffered and invisible in `kaggle kernels output`
    until the process exits.

    Inputs: msg - the message to log. Outputs: None (prints to stdout).
    """
    elapsed = time.time() - _START_TIME
    print(f"[{elapsed:7.1f}s] {msg}", flush=True)

TOKEN_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    """Casefold + tokenize a name/address string into word/number tokens (no punctuation).

    Inputs: text - raw string. Outputs: list of lowercase tokens.
    """
    return TOKEN_RE.findall(text.casefold())


def _is_non_latin(text: str) -> bool:
    """Detect whether a string contains any non-Latin-script letter.

    Inputs: text - raw string. Outputs: True if any alphabetic character's
    Unicode name doesn't mention "LATIN".
    """
    for ch in text:
        if ch.isalpha() and "LATIN" not in unicodedata.name(ch, ""):
            return True
    return False


def load_train(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load and sanity-check every train file (master plan SS6 assertions).

    Inputs: data_dir - path to the dataset root (contains train/ and test/).
    Outputs: dict with keys s1, s2, s3, gt_raw (DataFrames).
    """
    train = data_dir / "train"
    _log("  reading train_source1.tsv...")
    s1 = io_utils.load_tsv(train / "train_source1.tsv")
    _log(f"  train_source1.tsv: {len(s1):,} rows")
    _log("  reading train_source2.tsv...")
    s2 = io_utils.load_tsv(train / "train_source2.tsv")
    _log(f"  train_source2.tsv: {len(s2):,} rows")
    _log("  reading train_source3.tsv...")
    s3 = io_utils.load_tsv(train / "train_source3.tsv")
    _log(f"  train_source3.tsv: {len(s3):,} rows")
    io_utils.assert_source_columns(s1, "train_source1.tsv")
    io_utils.assert_source_columns(s2, "train_source2.tsv")
    io_utils.assert_source_columns(s3, "train_source3.tsv")
    io_utils.assert_unique_ids(s1, "S1-", "train_source1.tsv")
    io_utils.assert_unique_ids(s2, "S2-", "train_source2.tsv")
    io_utils.assert_unique_ids(s3, "S3-", "train_source3.tsv")
    _log("  reading train_ground_truth.tsv...")
    gt_raw = io_utils.load_tsv(train / "train_ground_truth.tsv")
    _log(f"  train_ground_truth.tsv: {len(gt_raw):,} rows")
    return {"s1": s1, "s2": s2, "s3": s3, "gt_raw": gt_raw}


def load_test(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load and sanity-check every test file.

    Inputs: data_dir - path to the dataset root.
    Outputs: dict with keys s1, s2, s3 (DataFrames).
    """
    test = data_dir / "test"
    _log("  reading test_source1.tsv...")
    s1 = io_utils.load_tsv(test / "test_source1.tsv")
    _log(f"  test_source1.tsv: {len(s1):,} rows")
    _log("  reading test_source2.tsv...")
    s2 = io_utils.load_tsv(test / "test_source2.tsv")
    _log(f"  test_source2.tsv: {len(s2):,} rows")
    _log("  reading test_source3.tsv...")
    s3 = io_utils.load_tsv(test / "test_source3.tsv")
    _log(f"  test_source3.tsv: {len(s3):,} rows")
    io_utils.assert_source_columns(s1, "test_source1.tsv")
    io_utils.assert_source_columns(s2, "test_source2.tsv")
    io_utils.assert_source_columns(s3, "test_source3.tsv")
    io_utils.assert_unique_ids(s1, "S1-", "test_source1.tsv")
    io_utils.assert_unique_ids(s2, "S2-", "test_source2.tsv")
    io_utils.assert_unique_ids(s3, "S3-", "test_source3.tsv")
    return {"s1": s1, "s2": s2, "s3": s3}


def e1_sizes(train: dict, test: dict) -> dict:
    """E1: sizes of S1/S2/S3 per split and per country.

    Inputs: train, test - dicts from load_train/load_test.
    Outputs: dict of counts, overall and per country per source.
    """
    out = {"train": {}, "test": {}}
    for split_name, split in (("train", train), ("test", test)):
        for src in ("s1", "s2", "s3"):
            df = split[src]
            out[split_name][src] = {
                "total": int(len(df)),
                "per_country": df["country"].value_counts().to_dict(),
            }
    return out


def explode_ground_truth(gt_raw: pd.DataFrame) -> pd.DataFrame:
    """Turn the wide ground-truth table into one row per (s1_id, matched_id) pair.

    Inputs: gt_raw - raw ground-truth DataFrame (source1_entity_id, matched_entity_ids).
    Outputs: DataFrame with columns s1_id, matched_id (empty-match S1s excluded).
    """
    gt = gt_raw.copy()
    gt["matched_list"] = gt["matched_entity_ids"].apply(lambda s: s.split(",") if s else [])
    exploded = gt[["source1_entity_id", "matched_list"]].explode("matched_list")
    exploded = exploded.dropna(subset=["matched_list"])
    exploded = exploded.rename(columns={"source1_entity_id": "s1_id", "matched_list": "matched_id"})
    return exploded.reset_index(drop=True)


def e2_singleton_rate(gt_raw: pd.DataFrame, s1: pd.DataFrame) -> dict:
    """E2: singleton rate (fraction of S1 entities with zero matches), overall and per country.

    This is also the all-empty-submission floor score.

    Inputs: gt_raw - raw ground-truth DataFrame; s1 - train S1 DataFrame (for country).
    Outputs: dict with overall rate and per-country rates.
    """
    gt = gt_raw.merge(s1[["entity_id", "country"]], left_on="source1_entity_id", right_on="entity_id", how="left")
    gt["is_singleton"] = gt["matched_entity_ids"] == ""
    overall = float(gt["is_singleton"].mean())
    per_country = gt.groupby("country")["is_singleton"].mean().to_dict()
    return {"overall": overall, "per_country": per_country}


def e3_match_count_distribution(gt_raw: pd.DataFrame, exploded: pd.DataFrame) -> dict:
    """E3: distribution of #matches per S1, and the S2-vs-S3 split within matches.

    Inputs: gt_raw - raw ground-truth DataFrame; exploded - output of explode_ground_truth.
    Outputs: dict with percentiles of match count, and S2/S3 share of matched IDs.
    """
    counts = gt_raw["matched_entity_ids"].apply(lambda s: len(s.split(",")) if s else 0)
    percentiles = {f"p{p}": float(np.percentile(counts, p)) for p in (50, 75, 90, 95, 99, 100)}
    is_s2 = exploded["matched_id"].str.startswith("S2-")
    return {
        "mean": float(counts.mean()),
        "percentiles": percentiles,
        "max": int(counts.max()),
        "s2_share_of_matches": float(is_s2.mean()),
        "s3_share_of_matches": float((~is_s2).mean()),
    }


def e4_multi_owner_check(exploded: pd.DataFrame) -> dict:
    """E4: does any S2/S3 ID appear under two different S1 entities?

    Drives the one-owner-mode decision (memory.md SS5).

    Inputs: exploded - output of explode_ground_truth.
    Outputs: dict with the count/fraction of multi-owner IDs and a few examples.
    """
    owners_per_id = exploded.groupby("matched_id")["s1_id"].nunique()
    multi = owners_per_id[owners_per_id > 1]
    return {
        "n_matched_ids": int(len(owners_per_id)),
        "n_multi_owner": int(len(multi)),
        "frac_multi_owner": float(len(multi) / len(owners_per_id)) if len(owners_per_id) else 0.0,
        "examples": multi.index[:5].tolist(),
    }


def e5_cross_country_matches(exploded: pd.DataFrame, s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame) -> dict:
    """E5: do matches ever cross countries? Drives the within-country-blocking decision.

    Inputs: exploded - s1_id/matched_id pairs; s1, s2, s3 - train source DataFrames.
    Outputs: dict with the count/fraction of cross-country matches.
    """
    s1_country = s1.set_index("entity_id")["country"]
    s23_country = pd.concat([s2.set_index("entity_id")["country"], s3.set_index("entity_id")["country"]])
    joined = exploded.copy()
    joined["s1_country"] = joined["s1_id"].map(s1_country)
    joined["matched_country"] = joined["matched_id"].map(s23_country)
    cross = joined["s1_country"] != joined["matched_country"]
    return {
        "n_pairs": int(len(joined)),
        "n_cross_country": int(cross.sum()),
        "frac_cross_country": float(cross.mean()) if len(joined) else 0.0,
    }


def e6_unmatched_share(exploded: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame) -> dict:
    """E6: what fraction of S2/S3 records match nothing?

    Inputs: exploded - s1_id/matched_id pairs; s2, s3 - train source DataFrames.
    Outputs: dict with the unmatched fraction, overall and per source.
    """
    matched_ids = set(exploded["matched_id"])
    s2_unmatched = float((~s2["entity_id"].isin(matched_ids)).mean())
    s3_unmatched = float((~s3["entity_id"].isin(matched_ids)).mean())
    return {"s2_unmatched_frac": s2_unmatched, "s3_unmatched_frac": s3_unmatched}


def e7_exact_rate(joined_sample: pd.DataFrame) -> dict:
    """E7: among positive pairs, exact-name and exact-address rates.

    Inputs: joined_sample - positive pairs with columns name_a, name_b, addr_a, addr_b
            (already casefolded/whitespace-normalized).
    Outputs: dict with exact-name-rate, exact-address-rate, name-only-noise-rate, addr-only-noise-rate.
    """
    exact_name = joined_sample["name_a"] == joined_sample["name_b"]
    exact_addr = joined_sample["addr_a"] == joined_sample["addr_b"]
    return {
        "exact_name_rate": float(exact_name.mean()),
        "exact_addr_rate": float(exact_addr.mean()),
        "name_only_noise_rate": float((~exact_name & exact_addr).mean()),
        "addr_only_noise_rate": float((exact_name & ~exact_addr).mean()),
        "both_exact_rate": float((exact_name & exact_addr).mean()),
        "n_sampled": int(len(joined_sample)),
    }


def _classify_name_relation(name_a: str, name_b: str) -> str:
    """Classify one pair's name relation with cheap, language-free heuristics for E8.

    Not the full explain_diff.py logic (Prompt 2) — a lightweight census used
    only to get approximate noise-type proportions during EDA.

    Inputs: name_a, name_b - casefolded name strings.
    Outputs: one of "exact", "typo", "abbrev_or_drop", "reorder", "other_noise".
    """
    if name_a == name_b:
        return "exact"
    tok_a, tok_b = _tokens(name_a), _tokens(name_b)
    set_a, set_b = set(tok_a), set(tok_b)
    if set_a == set_b and tok_a != tok_b:
        return "reorder"
    if set_a and set_b and (set_a <= set_b or set_b <= set_a) and set_a != set_b:
        return "abbrev_or_drop"
    # difflib.SequenceMatcher.ratio() (stdlib, no dependency) as an approximate
    # stand-in for a Levenshtein-ratio threshold — good enough for a census.
    if difflib.SequenceMatcher(None, name_a, name_b).ratio() > 0.8:
        return "typo"
    return "other_noise"


def e8_noise_census(joined_sample: pd.DataFrame) -> dict:
    """E8: noise census on positives (typo/abbrev-drop/reorder/other rates), on a sample.

    Inputs: joined_sample - positive pairs with name_a/name_b columns (casefolded).
    Outputs: dict of relation -> fraction, plus missing-postcode-style checks skipped
    (postcode census lives in E11).
    """
    chunk_size = 50_000
    chunks = []
    n = len(joined_sample)
    for start in range(0, n, chunk_size):
        chunk = joined_sample.iloc[start : start + chunk_size]
        chunks.append(chunk.apply(lambda r: _classify_name_relation(r["name_a"], r["name_b"]), axis=1))
        _log(f"  E8 progress: {min(start + chunk_size, n):,}/{n:,} pairs classified")
    relations = pd.concat(chunks) if chunks else pd.Series(dtype=object)
    return relations.value_counts(normalize=True).to_dict()


def e9_hard_negatives(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame, matched_ids: set, n_s1_sample: int = 500, n_cand_sample: int = 300, seed: int = 42) -> dict:
    """E9: hardest negatives — high name similarity, non-matched pairs (illustrative sample).

    Not exhaustive (that needs blocking, built in Prompt 2): samples S1 entities
    and same-country candidate records, scores name similarity, and reports the
    top non-matches.

    Inputs: s1, s2, s3 - train source DataFrames; matched_ids - set of every
            matched S2/S3 ID (to exclude true positives); n_s1_sample,
            n_cand_sample - sample sizes; seed - RNG seed.
    Outputs: dict with the count of near-duplicate non-matches found and a few examples.
    """
    rng = np.random.default_rng(seed)
    records23 = pd.concat([s2, s3], ignore_index=True)
    # Precompute country groups once instead of re-filtering the full ~10M-row
    # table inside the loop on every one of the n_s1_sample iterations.
    country_groups = {country: g for country, g in records23.groupby("country")}
    s1_sample = s1.sample(n=min(n_s1_sample, len(s1)), random_state=seed)
    examples = []
    near_dup_count = 0
    total_checked = 0
    for i, (_, row) in enumerate(s1_sample.iterrows()):
        if i > 0 and i % 100 == 0:
            _log(f"  E9 progress: {i}/{len(s1_sample)} S1 entities checked")
        same_country = country_groups.get(row["country"])
        if same_country is None or same_country.empty:
            continue
        cand = same_country.sample(n=min(n_cand_sample, len(same_country)), random_state=int(rng.integers(1_000_000)))
        cand = cand[~cand["entity_id"].isin(matched_ids)]
        if cand.empty:
            continue
        # difflib (stdlib) ratio() on a 0-100 scale, as an approximate stand-in
        # for rapidfuzz.fuzz.ratio() — good enough for an illustrative EDA sample.
        s1_name_cf = row["business_name"].casefold()
        scores = cand["business_name"].apply(
            lambda n: difflib.SequenceMatcher(None, s1_name_cf, n.casefold()).ratio() * 100
        )
        total_checked += len(cand)
        top_idx = scores.idxmax()
        if scores.loc[top_idx] >= 85:
            near_dup_count += 1
            if len(examples) < 5:
                examples.append(
                    {
                        "s1_name": row["business_name"],
                        "candidate_name": cand.loc[top_idx, "business_name"],
                        "similarity": float(scores.loc[top_idx]),
                    }
                )
    return {
        "n_s1_sampled": len(s1_sample),
        "n_candidates_checked": total_checked,
        "n_near_dup_nonmatches": near_dup_count,
        "examples": examples,
    }


def e10_field_quality(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame) -> dict:
    """E10: empty/very short names or addresses, and non-Latin-script share.

    Inputs: s1, s2, s3 - train source DataFrames.
    Outputs: dict of per-source stats.
    """
    out = {}
    for name, df in (("s1", s1), ("s2", s2), ("s3", s3)):
        name_len = df["business_name"].str.len()
        addr_len = df["business_address"].str.len()
        non_latin = (df["business_name"] + " " + df["business_address"]).apply(_is_non_latin).astype(bool)
        out[name] = {
            "empty_name_frac": float((name_len == 0).mean()),
            "short_name_frac_lt5": float((name_len < 5).mean()),
            "empty_addr_frac": float((addr_len == 0).mean()),
            "short_addr_frac_lt10": float((addr_len < 10).mean()),
            "non_latin_frac": float(non_latin.mean()),
        }
    return out


POSTCODE_RE = re.compile(r"\d{4,6}(?:-\d{3,4})?\b")


def e11_postcode_formats(s1: pd.DataFrame) -> dict:
    """E11: postcode digit-run length distribution per country, detected by shape only.

    Inputs: s1 - a source DataFrame with business_address and country columns.
    Outputs: dict {country: {length: count}} for the last digit-run found per address.
    """
    out: dict[str, dict] = {}
    for country, group in s1.groupby("country"):
        lengths = []
        for addr in group["business_address"]:
            matches = POSTCODE_RE.findall(addr)
            if matches:
                lengths.append(len(re.sub(r"\D", "", matches[-1])))
        if lengths:
            out[country] = pd.Series(lengths).value_counts().to_dict()
    return out


def e12_test_composition(test_s1: pd.DataFrame) -> dict:
    """E12: test set country labels, sizes, and non-Latin character share.

    Inputs: test_s1 - test source1 DataFrame.
    Outputs: dict with per-country counts and non-Latin fraction.
    """
    non_latin = (test_s1["business_name"] + " " + test_s1["business_address"]).apply(_is_non_latin).astype(bool)
    return {
        "per_country_counts": test_s1["country"].value_counts().to_dict(),
        "non_latin_frac": float(non_latin.mean()),
        "non_latin_frac_per_country": test_s1.assign(non_latin=non_latin).groupby("country")["non_latin"].mean().to_dict(),
    }


def shortcut_check(gt_raw: pd.DataFrame) -> dict:
    """Detect (never use) any correlation between ID numbers / row order and match status.

    Inputs: gt_raw - raw ground-truth DataFrame, in file order.
    Outputs: dict with correlation coefficients (id-number vs. match-count, row-index vs. match-count).
    """
    id_num = gt_raw["source1_entity_id"].str.extract(r"(\d+)")[0].astype(float)
    match_count = gt_raw["matched_entity_ids"].apply(lambda s: len(s.split(",")) if s else 0)
    row_index = np.arange(len(gt_raw))
    return {
        "corr_id_number_vs_match_count": float(np.corrcoef(id_num, match_count)[0, 1]),
        "corr_row_index_vs_match_count": float(np.corrcoef(row_index, match_count)[0, 1]),
    }


def build_positive_pairs_sample(exploded: pd.DataFrame, s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame, n: int = 300_000, seed: int = 42) -> pd.DataFrame:
    """Sample positive pairs and attach casefolded name/address fields for E7/E8.

    Inputs: exploded - s1_id/matched_id pairs; s1, s2, s3 - train source DataFrames;
            n - sample size; seed - RNG seed.
    Outputs: DataFrame with name_a/name_b/addr_a/addr_b (casefolded, whitespace-collapsed).
    """
    sample = exploded.sample(n=min(n, len(exploded)), random_state=seed).copy()
    s1_lookup = s1.set_index("entity_id")[["business_name", "business_address"]]
    s23 = pd.concat([s2, s3], ignore_index=True).set_index("entity_id")[["business_name", "business_address"]]

    def norm(text: str) -> str:
        return re.sub(r"\s+", " ", text.casefold()).strip()

    sample = sample.join(s1_lookup.rename(columns={"business_name": "name_a", "business_address": "addr_a"}), on="s1_id")
    sample = sample.join(s23.rename(columns={"business_name": "name_b", "business_address": "addr_b"}), on="matched_id")
    sample = sample.dropna(subset=["name_a", "name_b", "addr_a", "addr_b"])
    sample["name_a"] = sample["name_a"].map(norm)
    sample["name_b"] = sample["name_b"].map(norm)
    sample["addr_a"] = sample["addr_a"].map(norm)
    sample["addr_b"] = sample["addr_b"].map(norm)
    return sample


def run_eda(data_dir: Path) -> dict:
    """Run every EDA question end to end and return the full findings dict.

    Inputs: data_dir - path to the dataset root (contains train/ and test/).
    Outputs: dict keyed by question ID (e1..e12, shortcut_check, all_empty_floor).
    """
    _log("loading train (source1/2/3 + ground truth)...")
    train = load_train(data_dir)
    _log(f"train loaded: S1={len(train['s1']):,} S2={len(train['s2']):,} S3={len(train['s3']):,}")

    _log("loading test (source1/2/3)...")
    test = load_test(data_dir)
    _log(f"test loaded: S1={len(test['s1']):,} S2={len(test['s2']):,} S3={len(test['s3']):,}")

    _log("exploding ground truth into (s1_id, matched_id) pairs...")
    exploded = explode_ground_truth(train["gt_raw"])
    matched_ids = set(exploded["matched_id"])
    _log(f"exploded to {len(exploded):,} pairs, {len(matched_ids):,} unique matched IDs")

    _log("building 300k-pair positive sample for E7/E8...")
    positive_sample = build_positive_pairs_sample(exploded, train["s1"], train["s2"], train["s3"])
    _log(f"positive sample built: {len(positive_sample):,} rows")

    findings: dict = {}

    _log("E1: sizes...")
    findings["e1_sizes"] = e1_sizes(train, test)

    _log("E2: singleton rate...")
    e2 = e2_singleton_rate(train["gt_raw"], train["s1"])
    findings["e2_singleton_rate"] = e2
    _log(f"  overall singleton/floor = {e2['overall']:.4f}")

    _log("E3: match count distribution...")
    findings["e3_match_count_distribution"] = e3_match_count_distribution(train["gt_raw"], exploded)

    _log("E4: multi-owner check (groupby nunique over exploded pairs)...")
    findings["e4_multi_owner_check"] = e4_multi_owner_check(exploded)

    _log("E5: cross-country matches...")
    findings["e5_cross_country_matches"] = e5_cross_country_matches(exploded, train["s1"], train["s2"], train["s3"])

    _log("E6: unmatched share (isin over S2/S3)...")
    findings["e6_unmatched_share"] = e6_unmatched_share(exploded, train["s2"], train["s3"])

    _log("E7: exact-name/address rates (on sample)...")
    findings["e7_exact_rate"] = e7_exact_rate(positive_sample)

    _log("E8: noise census (row-wise apply + difflib on 300k-row sample -- can take a while)...")
    findings["e8_noise_census"] = e8_noise_census(positive_sample)

    _log("E9: hard negatives (500 S1 x 300 candidates, difflib -- can take a while)...")
    findings["e9_hard_negatives"] = e9_hard_negatives(train["s1"], train["s2"], train["s3"], matched_ids)

    _log("E10: field quality...")
    findings["e10_field_quality"] = e10_field_quality(train["s1"], train["s2"], train["s3"])

    _log("E11: postcode formats (per-row regex over all of train S1)...")
    findings["e11_postcode_formats"] = e11_postcode_formats(train["s1"])

    _log("E12: test composition...")
    findings["e12_test_composition"] = e12_test_composition(test["s1"])

    _log("shortcut check...")
    findings["shortcut_check"] = shortcut_check(train["gt_raw"])

    findings["all_empty_floor"] = e2["overall"]
    _log("run_eda complete.")
    return findings


def print_summary(findings: dict) -> None:
    """Print a <=20-line summary of the findings, for pasting into memory.md SS6.

    Inputs: findings - output of run_eda(). Outputs: none (prints to stdout).
    """
    e1 = findings["e1_sizes"]
    print(f"train S1={e1['train']['s1']['total']:,} S2={e1['train']['s2']['total']:,} S3={e1['train']['s3']['total']:,}")
    print(f"test  S1={e1['test']['s1']['total']:,} S2={e1['test']['s2']['total']:,} S3={e1['test']['s3']['total']:,}")
    print(f"E2 singleton/floor: overall={findings['e2_singleton_rate']['overall']:.4f} per_country={findings['e2_singleton_rate']['per_country']}")
    print(f"E3 match count: mean={findings['e3_match_count_distribution']['mean']:.2f} p95={findings['e3_match_count_distribution']['percentiles']['p95']}")
    e4 = findings["e4_multi_owner_check"]
    print(f"E4 multi-owner: {e4['n_multi_owner']}/{e4['n_matched_ids']} ({e4['frac_multi_owner']:.5f}) -> one-owner mode: {'HARD' if e4['frac_multi_owner'] < 0.001 else 'SOFT'}")
    e5 = findings["e5_cross_country_matches"]
    print(f"E5 cross-country matches: {e5['frac_cross_country']:.5f} -> within-country blocking: {'YES' if e5['frac_cross_country'] < 0.001 else 'NO'}")
    print(f"E6 unmatched: S2={findings['e6_unmatched_share']['s2_unmatched_frac']:.3f} S3={findings['e6_unmatched_share']['s3_unmatched_frac']:.3f}")
    print(f"E7 exact rates: {findings['e7_exact_rate']}")
    print(f"E8 noise census (sample): {findings['e8_noise_census']}")
    print(f"E9 near-dup non-matches: {findings['e9_hard_negatives']['n_near_dup_nonmatches']}/{findings['e9_hard_negatives']['n_s1_sampled']} S1 sampled")
    print(f"E10 field quality (s1): {findings['e10_field_quality']['s1']}")
    print(f"E11 postcode lengths (train S1, top country): {list(findings['e11_postcode_formats'].items())[:1]}")
    print(f"E12 test composition: {findings['e12_test_composition']['per_country_counts']} non_latin={findings['e12_test_composition']['non_latin_frac']:.4f}")
    print(f"Shortcut check (should be ~0, detect only): {findings['shortcut_check']}")
    print(f"ALL-EMPTY FLOOR: {findings['all_empty_floor']:.4f}")


def main() -> None:
    """CLI entry point: run the full EDA and write reports/eda.json."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Path to the dataset root (contains train/ and test/)")
    parser.add_argument("--out", default="reports/eda.json")
    args = parser.parse_args()

    findings = run_eda(Path(args.data_dir))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(findings, indent=2, default=str), encoding="utf-8")
    print_summary(findings)
    print(f"\nFull findings written to {out_path}")


if __name__ == "__main__":
    main()
