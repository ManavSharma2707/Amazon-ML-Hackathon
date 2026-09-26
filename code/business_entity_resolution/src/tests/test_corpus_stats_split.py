"""Unit tests for src/corpus_stats.py and src/split.py."""

import numpy as np
import pandas as pd

from src import corpus_stats, normalize, split


def _records():
    """Tiny normalised record frame with two country labels."""
    raw = pd.DataFrame(
        {
            "entity_id": [f"S1-{i}" for i in range(6)],
            "business_name": ["Alpha Pvt Ltd", "Beta Pvt Ltd", "Gamma Ltd", "Delta Inc", "Zeta Ltd", "Ltd Omega"],
            "business_address": ["1 Main Road, X", "2 Hill Road, Y", "3 Oak Road, Z", "4 Oak Street, W", "5 Elm Road", "Q"],
            "country": ["C1", "C1", "C1", "C2", "C2", "C2"],
        }
    )
    return normalize.normalize_df(raw, "S1")


def test_counts_df_and_country_groups():
    """df counts records (not occurrences); global = sum over countries."""
    stats = corpus_stats.count_tokens([_records()], min_df=1)
    ltd = stats[(stats.field == "norm_name") & (stats.token == "ltd")].set_index("country")["df"]
    assert ltd[""] == 5 and ltd["C1"] == 3 and ltd["C2"] == 2
    assert int(stats[(stats.field == "norm_name") & (stats.country == "")]["n_docs"].iloc[0]) == 6


def test_incremental_equals_batch():
    """Adding frames one by one gives the same table as one batch count."""
    recs = _records()
    batch = corpus_stats.count_tokens([recs], min_df=1)
    inc = corpus_stats.count_tokens([recs.iloc[:3], recs.iloc[3:]], min_df=1)
    key = ["field", "country", "token"]
    a = batch.sort_values(key).reset_index(drop=True)
    b = inc.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_min_df_and_idf_lookup():
    """Tokens under min_df are dropped; unseen tokens get the df=1 default IDF (the max)."""
    stats = corpus_stats.count_tokens([_records()], min_df=2)
    idf, default = corpus_stats.idf_lookup(stats, "norm_name")
    assert "alpha" not in idf
    assert default > idf["ltd"]


def test_suffix_and_street_type_likeness():
    """'ltd' (frequent, at the end) scores higher than 'alpha'; 'road' is street-type-like."""
    stats = corpus_stats.count_tokens([_records()], min_df=1)
    suf = corpus_stats.suffix_likeness(stats)
    assert suf["ltd"] > suf.get("alpha", 0) and suf["ltd"] > suf["pvt"] * 0.5
    st = corpus_stats.street_type_likeness(stats)
    assert st["road"] > st.get("x", 0)


def _s1_gt(n=400):
    """Synthetic S1 frame + GT with a known match-count mix across 2 countries."""
    s1 = pd.DataFrame({"entity_id": [f"S1-{i}" for i in range(n)], "country": np.where(np.arange(n) % 2, "X", "Y")})
    gt = {f"S1-{i}": {f"S2-{i}-{j}" for j in range(i % 6)} for i in range(n)}
    return s1, gt


def test_split_stratified_and_deterministic():
    """55/45 within every country x bucket stratum; same result for a shuffled input."""
    s1, gt = _s1_gt()
    a = split.make_ab_split(s1, gt, 0.55, 42)
    b = split.make_ab_split(s1.sample(frac=1, random_state=1), gt, 0.55, 42)
    assert a.set_index("entity_id")["split"].equals(b.set_index("entity_id")["split"])
    for _, g in a.groupby(["country", "bucket"]):
        assert abs((g["split"] == "A").mean() - 0.55) < 0.05
    rep = split.split_report(a)
    assert abs(rep["A"]["singleton_rate"] - rep["B"]["singleton_rate"]) < 0.02


def test_pool_split_follows_owner():
    """S2/S3 records get their S1's half; unmatched ones are 'shared'."""
    s1, gt = _s1_gt(40)
    sp = split.make_ab_split(s1, gt)
    pairs = split.gt_long(gt)
    pool_ids = pd.Series(list(pairs["match_id"]) + ["S3-unmatched"])
    ps = split.pool_split(sp, pairs, pool_ids).set_index("entity_id")["split"]
    owner = pairs.merge(sp, left_on="s1_id", right_on="entity_id").set_index("match_id")["split"]
    assert (ps.loc[owner.index] == owner).all()
    assert ps["S3-unmatched"] == "shared"
    assert set(split.match_bucket(pd.Series([0, 1, 2, 3, 4, 9]))) == {"0", "1", "2-3", "4+"}
