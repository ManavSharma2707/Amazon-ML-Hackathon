"""Unit tests for src/finetune_embedder.py and src/scramble.py (no model download)."""

import numpy as np
import pandas as pd
import pytest

from src import finetune_embedder as fe
from src import scramble


def test_one_s1_per_batch_never_repeats_an_s1():
    """Every batch holds distinct S1s, and every row is used at most once."""
    rng = np.random.default_rng(0)
    s1 = np.array([f"s{i}" for i in rng.integers(0, 40, 500)])
    batches = fe.one_s1_per_batch(s1, 16, seed=1)
    used = np.concatenate(batches)
    assert len(used) == len(set(used.tolist()))
    for b in batches:
        assert len(set(s1[b].tolist())) == len(b)


def test_build_triplets_excludes_matches_from_negatives():
    """Hard negatives come from kNN minus the anchor's own matches; one row per positive."""
    anchors = pd.DataFrame({"entity_id": ["S1-a", "S1-b"], "text": ["A co | 1 x st", "B co | 2 y st"]})
    pool = pd.DataFrame({
        "entity_id": ["S2-a1", "S3-a2", "S2-b1", "S2-n1", "S2-n2"],
        "text": ["a co", "a company", "b co", "n1", "n2"],
        "owner": ["S1-a", "S1-a", "S1-b", None, None],
    })
    pairs = pd.DataFrame({"s1_id": ["S1-a", "S1-a", "S1-b"], "match_id": ["S2-a1", "S3-a2", "S2-b1"]})
    knn = np.array([[0, 3, 1, 4], [2, 0, 4, -1]])
    t = fe.build_triplets(anchors, pairs, pool, knn, hard_negs=2, sibling_frac=0.0)
    assert len(t) == 3
    a = t[t.s1_id == "S1-a"]
    assert set(a["n_text"]) <= {"n1", "n2"}
    b = t[t.s1_id == "S1-b"]
    assert b["n_text"].iloc[0] in {"a co", "n2"} and b["p_text"].iloc[0] == "b co"


def test_mnrl_mask_removes_same_owner_candidates():
    """A candidate owned by the anchor's S1 (not its label) must not act as a negative."""
    torch = pytest.importorskip("torch")
    a = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
    # positives: exact; negatives: neg0 identical to anchor 1's vector and owned by anchor 1's S1.
    c = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.7, 0.7]])
    c = torch.nn.functional.normalize(c, dim=-1)
    owners = torch.tensor([0, 1])
    unmasked = fe.mnrl_loss(a, c, owners, torch.tensor([0, 1, -1, -1]))
    masked = fe.mnrl_loss(a, c, owners, torch.tensor([0, 1, 1, -1]))
    assert float(masked) < float(unmasked)


def test_field_dropout_and_recall_mrr():
    """Field dropout keeps one side; recall/MRR count ranks correctly."""
    import random

    out = {fe.field_dropout("N | A", random.Random(s)) for s in range(20)}
    assert out == {"N", "A"}
    r = fe.recall_mrr(np.array([[0, 1], [0, 1]]), np.array(["q0", "q1"]), np.array(["p0", "p1"]),
                      pd.DataFrame({"s1_id": ["q0", "q1"], "match_id": ["p0", "p1"]}), ks=(1, 2))
    assert r["recall@1"] == 0.5 and r["recall@2"] == 1.0 and r["mrr"] == 0.75


def test_scrambler_is_consistent_permutation():
    """Same permutation for both cases; digits/spaces unchanged; bijective on a-z."""
    t = scramble.make_scrambler(7)
    s = "Main St 12".translate(t)
    assert s.endswith(" 12")
    assert len({chr(t[ord(c)]) for c in "abcdefghijklmnopqrstuvwxyz"}) == 26
    assert "Aa".translate(t)[0].lower() == "Aa".translate(t)[1]
