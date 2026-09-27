"""Local smoke test for src/features_v7.py (V7's 9-channel feature layout, incl. translit),
run on sample/ before any Kaggle push -- fast feedback for exactly the class of bug
(schema/shape/NaN handling) that a Kaggle dry run would also catch, without needing a
kernel push + queue wait or cross-runner dataset placement."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import corpus_stats, features_v7 as features, io_utils, normalize

SAMPLE = Path(__file__).resolve().parents[4] / "sample"
N_S1 = 300


@pytest.fixture(scope="module")
def setup():
    if not (SAMPLE / "train" / "train_source1.tsv").exists():
        pytest.skip("sample/ not built")
    raw = {s: io_utils.load_tsv(SAMPLE / "train" / f"train_source{s}.tsv") for s in (1, 2, 3)}
    gt = io_utils.load_ground_truth(SAMPLE / "train" / "train_ground_truth.tsv")
    s1 = normalize.normalize_df(raw[1].iloc[:N_S1], "S1")
    pool = pd.concat([normalize.normalize_df(raw[s], f"S{s}") for s in (2, 3)], ignore_index=True)
    stats = corpus_stats.count_tokens([s1, pool], min_df=1)
    lookups = features.token_lookups(stats, min_df_generic=20)
    rng = np.random.default_rng(0)
    rows = []
    pool_by_cty = {c: g["entity_id"].to_numpy() for c, g in pool.groupby("country")}
    pool_ids = set(pool["entity_id"])
    for sid, cty in zip(s1["entity_id"], s1["country"]):
        pos = [m for m in gt.get(sid, set()) if m in pool_ids]
        neg = rng.choice(pool_by_cty[cty], 8, replace=False).tolist()
        for c in dict.fromkeys(pos + neg):
            rows.append((sid, c, c in pos))
    cands = pd.DataFrame(rows, columns=["s1_id", "cand_id", "label"])
    n = len(cands)
    cands["bitmask"] = rng.integers(0, 1 << len(features.SPARSE_CHANNELS), n).astype(np.int32)
    cands["n_channels"] = rng.integers(1, 5, n).astype(np.int8)
    for c in features.SPARSE_CHANNELS:
        cands[f"{c}_score"] = rng.random(n).astype(np.float32)
    cands["cheap_score"] = rng.random(n).astype(np.float32)
    return s1, pool, lookups, cands


def test_v7_feature_table_includes_translit_channels(setup):
    """SPARSE_CHANNELS/FEATURES include translit_name/translit_addr; build_features runs end to end."""
    assert "translit_name" in features.SPARSE_CHANNELS and "translit_addr" in features.SPARSE_CHANNELS
    assert "ch_translit_name" in features.META_FEATURES and "translit_addr_score" in features.META_FEATURES
    s1, pool, lookups, cands = setup
    X = features.build_features(cands, s1, pool, lookups, n_jobs=1, log=lambda m: None)
    assert list(X.columns) == features.FEATURES and len(X) == len(cands)
    assert X.dtypes.eq(np.float32).all()
    assert np.isfinite(X["cheap_rank"]).all() and (X["cheap_rank"] >= 0).all()


def test_v7_handles_missing_translit_columns_like_v6_cands(setup):
    """Candidates from a v6-shaped file (no translit_*_score columns) don't crash -- the
    NB06v7 driver's `fill_missing` NaN-fills these; build_features must tolerate NaN there
    exactly the way it already tolerates NaN dense_cos etc."""
    s1, pool, lookups, cands = setup
    c6 = cands.drop(columns=["translit_name_score", "translit_addr_score"]).copy()
    c6["translit_name_score"] = np.float32(np.nan)
    c6["translit_addr_score"] = np.float32(np.nan)
    c6 = c6[cands.columns]  # restore original column order (fill_missing appends at the end too, order doesn't matter to build_features)
    X = features.build_features(c6, s1, pool, lookups, n_jobs=1, log=lambda m: None)
    assert len(X) == len(c6)
    assert X["translit_name_score"].isna().all() and X["translit_addr_score"].isna().all()
