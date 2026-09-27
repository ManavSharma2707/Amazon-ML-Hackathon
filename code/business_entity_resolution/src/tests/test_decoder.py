"""Unit tests for src/decoder.py and src/exclusivity.py."""

import numpy as np
import pandas as pd

from src import decoder, exclusivity, metrics


def _reference_best_k(probs, q_outside=0.0, beta2=0.25):
    """Plan SS28.3 reference decoder (per entity loop)."""
    def pb(ps):
        pmf = np.array([1.0])
        for p in ps:
            pmf = np.convolve(pmf, [1.0 - p, p])
        return pmf

    p = np.sort(np.asarray(probs))[::-1]
    best_val, best_k = -1.0, 0
    for k in range(len(p) + 1):
        tp_pmf, fn_pmf = pb(p[:k]), pb(np.concatenate([p[k:], [q_outside]]))
        tp, fn = np.arange(len(tp_pmf))[:, None], np.arange(len(fn_pmf))[None, :]
        if k == 0:
            score = (fn == 0).astype(float) * np.ones_like(tp)
        else:
            denom = (1 + beta2) * tp + beta2 * fn + (k - tp)
            score = np.where(tp > 0, (1 + beta2) * tp / np.maximum(denom, 1e-12), 0.0)
        val = float((tp_pmf[:, None] * fn_pmf[None, :] * score).sum())
        if val > best_val + 1e-12:
            best_val, best_k = val, k
    return best_k, best_val


def test_vectorised_matches_reference():
    """Vectorised E[F] argmax equals the reference loop on random entities."""
    rng = np.random.default_rng(1)
    P = -np.sort(-(rng.random((200, 7)) ** 2), axis=1)
    q = rng.random(200) * 0.2
    ef = decoder.expected_f_matrix(P, q)
    for e in range(200):
        k, v = _reference_best_k(P[e], q[e])
        assert abs(ef[e].max() - v) < 1e-9
        assert ef[e, k] >= ef[e].max() - 1e-9


def test_decode_empty_and_strong():
    """Weak candidates -> empty prediction; one strong + weak -> top-1; three moderate-high -> all three."""
    df = pd.DataFrame({
        "s1_id": ["a"] * 3 + ["b"] * 3 + ["c"] * 3,
        "cand_id": [f"S2-{i}" for i in range(9)],
        "p": [0.05, 0.03, 0.01, 0.97, 0.10, 0.05, 0.9, 0.85, 0.8],
    })
    out = decoder.decode(df, "p")
    got = out.groupby("s1_id")["cand_id"].apply(set).to_dict()
    assert "a" not in got and got["b"] == {"S2-3"} and got["c"] == {"S2-6", "S2-7", "S2-8"}


def test_macro_f05_matches_official_scorer():
    """Pair-table macro F0.5 equals metrics.f05_macro on a small example (incl. singletons)."""
    truth = pd.DataFrame({"s1_id": ["a", "a", "b"], "match_id": ["x", "y", "z"]})
    pred = pd.DataFrame({"s1_id": ["a", "c"], "cand_id": ["x", "w"]})
    ids = ["a", "b", "c", "d"]
    got = decoder.macro_f05(pred, truth, ids)
    pm = {"a": {"x"}, "c": {"w"}}
    tm = {"a": {"x", "y"}, "b": {"z"}, "c": set(), "d": set()}
    assert abs(got - metrics.f05_macro(pm, tm)) < 1e-12


def test_hard_and_soft_one_owner():
    """The higher-probability claimant keeps a shared record; soft shrinks the loser more."""
    df = pd.DataFrame({"s1_id": ["a", "b", "b"], "cand_id": ["x", "x", "y"], "p": [0.9, 0.6, 0.7]})
    assert exclusivity.hard_one_owner(df).tolist() == [0.9, 0.0, 0.7]
    soft = exclusivity.soft_one_owner(df, gamma=1.0)
    assert soft[1] < soft[0] and abs(soft[2] - 0.7) < 1e-12


def test_isotonic_and_sharpen():
    """Isotonic output is monotone in p; sharpen(lam=1) is the identity."""
    rng = np.random.default_rng(0)
    p = rng.random(2000)
    y = (rng.random(2000) < p ** 2).astype(int)
    c = decoder.Isotonic().fit(p, y).predict(np.linspace(0, 1, 50))
    assert (np.diff(c) >= -1e-12).all()
    assert np.allclose(decoder.sharpen(p, 1.0), p)


def test_isotonic_save_load_roundtrip(tmp_path):
    """A calibrator saved to disk and reloaded (NumPy only, no sklearn) reproduces the same predictions."""
    rng = np.random.default_rng(1)
    p = rng.random(5000)
    y = (rng.random(5000) < p ** 1.5).astype(int)
    cal = decoder.Isotonic().fit(p, y)
    path = tmp_path / "calibrator.npz"
    cal.save(path)
    loaded = decoder.load_isotonic(path)
    query = np.linspace(0, 1, 200)
    assert np.allclose(cal.predict(query), loaded.predict(query), atol=1e-9)
