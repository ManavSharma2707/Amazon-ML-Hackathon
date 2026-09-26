"""Tests for src/features.py and src/stage1.py on the local sample/ (skipped if absent)."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import corpus_stats, features, io_utils, normalize, stage1

SAMPLE = Path(__file__).resolve().parents[4] / "sample"
N_S1 = 600


@pytest.fixture(scope="module")
def setup():
    """Normalised sample records, token lookups and a candidate frame (true matches + same-country negatives)."""
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
    cands["bitmask"] = rng.integers(0, 512, n).astype(np.int16)
    cands["n_channels"] = rng.integers(1, 5, n).astype(np.int8)
    for c in features.SPARSE_CHANNELS:
        cands[f"{c}_score"] = rng.random(n).astype(np.float32)
    cands["cheap_score"] = rng.random(n).astype(np.float32)
    return s1, pool, lookups, cands


def test_feature_table_shape_and_signal(setup):
    """All features present, float32, finite where expected; explained-name fraction separates labels."""
    s1, pool, lookups, cands = setup
    t0 = time.time()
    X = features.build_features(cands, s1, pool, lookups, n_jobs=1, log=lambda m: None)
    dt = (time.time() - t0) / len(cands)
    print(f"\n{len(cands)} pairs, {dt * 1e6:.0f} us/pair (1 process)")
    assert list(X.columns) == features.FEATURES and len(X) == len(cands)
    assert X.dtypes.eq(np.float32).all()
    y = cands["label"].to_numpy()
    assert X.loc[y, "name_expl_frac_idf"].mean() > X.loc[~y, "name_expl_frac_idf"].mean() + 0.3
    assert X.loc[y, "norm_name_token_set_ratio"].mean() > X.loc[~y, "norm_name_token_set_ratio"].mean()
    assert np.isfinite(X["cheap_rank"]).all() and (X["cheap_rank"] >= 0).all()
    # prerank variant = cheap features only
    Xp = features.build_features(cands.iloc[:200], s1, pool, lookups, with_py=False, log=lambda m: None)
    assert list(Xp.columns) == features.PRERANK_FEATURES


def test_stage1_oof_auc(setup):
    """Group-k-fold OOF on sample pairs reaches a high AUC and never leaks S1 groups."""
    s1, pool, lookups, cands = setup
    X = features.build_features(cands, s1, pool, lookups, n_jobs=1, log=lambda m: None)
    y = cands["label"].to_numpy().astype(np.int8)
    params = stage1.lgb_params({"lr": 0.1, "min_data_in_leaf": 20}, 42)
    r = stage1.train_oof(X.to_numpy(np.float32), y, cands["s1_id"].to_numpy(), params, folds=3, max_rounds=200,
                         feature_names=features.FEATURES, log=lambda m: None)
    ev = stage1.eval_by_group(y, r["oof"], np.array(["X"] * len(y)))
    assert ev["ALL"]["auc"] > 0.97
    m = stage1.train_full(X.to_numpy(np.float32), y, params, int(np.mean(r["best_iters"])) + 1, features.FEATURES)
    assert stage1.predict(m, X.to_numpy(np.float32)[:10]).shape == (10,)


def test_group_folds_no_leak():
    """Every group lands in exactly one fold."""
    g = np.repeat(np.arange(50), 3)
    f = stage1.group_folds(g, 5, 0)
    assert pd.DataFrame({"g": g, "f": f}).groupby("g")["f"].nunique().max() == 1
    assert set(f.tolist()) == set(range(5))
