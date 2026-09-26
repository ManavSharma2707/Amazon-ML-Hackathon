"""Unit tests for src/embed.py: text building and the chunked within-group kNN (CPU)."""

import numpy as np
import pandas as pd
import pytest

from src import embed

torch = pytest.importorskip("torch")


def test_record_texts():
    """NFKC + whitespace clean, accents kept, empty address -> name only."""
    assert embed.record_texts(["Café  Rouge", "X"], ["1 Rue A,  Paris", ""]) == ["Café Rouge | 1 Rue A, Paris", "X"]


def _unit(rng, n, d):
    """Random L2-normalised float16 rows."""
    x = rng.normal(size=(n, d)).astype(np.float32)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16)


def test_knn_matches_brute_force_within_groups():
    """Chunked GPU-style kNN == brute force per group; never crosses group labels."""
    rng = np.random.default_rng(0)
    q, p = _unit(rng, 57, 16), _unit(rng, 230, 16)
    qg = np.array(["a", "b", "c"])[rng.integers(0, 3, 57)]
    pg = np.array(["a", "b", "d"])[rng.integers(0, 3, 230)]  # "c" has no pool, "d" no queries
    fi, fs, ri, rs = embed.knn_within_groups(q, qg, p, pg, k=5, rk=2, devices=["cpu"], q_chunk=10, p_chunk=37)
    sims = q.astype(np.float32) @ p.astype(np.float32).T
    for i in range(len(q)):
        cand = np.flatnonzero(pg == qg[i])
        if len(cand) == 0:
            assert (fi[i] == -1).all()
            continue
        exp = set(cand[np.argsort(-sims[i, cand])[:5]])
        got = set(fi[i][fi[i] >= 0])
        assert got == exp
        assert (pg[fi[i][fi[i] >= 0]] == qg[i]).all()
    for j in range(len(p)):
        cand = np.flatnonzero(qg == pg[j])
        if len(cand) == 0:
            assert (ri[j] == -1).all()
            continue
        exp = set(cand[np.argsort(-sims[cand, j])[:2]])
        assert set(ri[j][ri[j] >= 0]) == exp


def test_pair_recall_at_k():
    """Recall@k counts a pair as found when its match sits in the first k slots."""
    fwd = np.array([[1, 2, 0], [0, 2, 1]])
    pairs = pd.DataFrame({"s1_id": ["q0", "q1", "q1"], "match_id": ["p0", "p0", "p1"]})
    r = embed.pair_recall_at_k(fwd, np.array(["q0", "q1"]), np.array(["p0", "p1", "p2"]), pairs, ks=(1, 3))
    assert r["recall@1"] == pytest.approx(1 / 3, abs=1e-4)
    assert r["recall@3"] == 1.0 and r["n_pairs"] == 3
