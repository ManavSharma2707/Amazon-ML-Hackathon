"""Unit tests for src/blocking.py: sparse channels, union, pruning, report; plus a
recall smoke test of the sparse channels on sample/ when it exists."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import blocking, io_utils, normalize, split

SAMPLE = Path(__file__).resolve().parents[4] / "sample"


def test_char_channel_finds_typo_and_ranks():
    """A typo'd name is the top char-TF-IDF neighbour; scores sorted, ranks from 0."""
    q = ["alpha trading company", "beta foods"]
    p = ["gamma foods", "alpha tradng company", "beta food", "zeta"]
    qi, pi, sc, rk = blocking.run_sparse_channel(q, p, "char", k=2, max_df=10)
    top = {int(a): int(b) for a, b, r in zip(qi, pi, rk) if r == 0}
    assert top == {0: 1, 1: 2}
    for row in (0, 1):
        s = sc[qi == row]
        assert (np.diff(s) <= 1e-6).all()


def test_empty_rows_anywhere_do_not_crash():
    """Empty texts (e.g. empty addresses) at the start, middle and end of either side are fine."""
    q = ["", "main street 12", ""]
    p = ["", "main st 12", "oak road", ""]
    qi, pi, sc, _ = blocking.run_sparse_channel(q, p, "char", k=2, max_df=10)
    assert set(qi.tolist()) == {1} and pi[0] == 1 and np.isfinite(sc).all()


def test_deleet_and_cross_pairs():
    """Look-alike digits map back to letters only inside words; cross keys pair name x address."""
    assert blocking.deleet("k01kata") == "kolkata" and blocking.deleet("capita1") == "capital"
    assert blocking.deleet("8ombay") == "bombay" and blocking.deleet("12bis") == "12bis"
    assert blocking.deleet("238") == "238" and blocking.deleet("3rd") == "3rd" and blocking.deleet("b2") == "b2"
    keys = blocking._cross_pair_analyzer(blocking.cross_text(pd.Series(["acme 24 & co"]), pd.Series(["12 main"]))[0])
    assert sorted(keys) == ["acme^12", "acme^main", "co^12", "co^main"]
    assert blocking.cross_text(pd.Series(["x"]), pd.Series([""])) == [""]


def test_max_df_prunes_frequent_features():
    """A token present in every pool record is dropped from the index (no match through it)."""
    q = ["common"]
    p = ["common"] * 5
    qi, _, _, _ = blocking.run_sparse_channel(q, p, "word", k=3, max_df=3)
    assert len(qi) == 0
    qi, _, _, _ = blocking.run_sparse_channel(q, p, "word", k=3, max_df=10)
    assert len(qi) == 3


def test_channel_texts():
    """Rare-token text carries fold forms; num key = house number + next word token."""
    t = blocking.name_tok_text(pd.Series(["phillip shah 24 & co"]), pd.Series(["filip sah 24 & co"]))
    assert t == ["phillip shah co ~filip ~sah"]
    k = blocking.num_key_text(pd.Series(["35840 chester rd 3409", "main st", "12 34 oak"]), pd.Series(["35840", "", "12"]))
    assert k == ["35840_chester", "", "12_oak"]


def test_union_prune_and_report():
    """Union merges channels per pair; prune keeps top-n per S1; report counts recall."""
    parts = {
        "dense": (np.array([0, 0, 1]), np.array([5, 6, 7]), np.array([0.9, 0.8, 0.7], np.float32), np.array([0, 1, 0], np.int16)),
        "name_char": (np.array([0, 1]), np.array([6, 8]), np.array([0.5, 0.4], np.float32), np.array([0, 0], np.int16)),
    }
    u = blocking.union_channels(parts, n_pool=10)
    assert len(u) == 4
    row = u[(u.q_row == 0) & (u.p_row == 6)].iloc[0]
    assert row["n_channels"] == 2 and row["dense_rank"] == 1 and row["name_char_rank"] == 0
    assert row["bitmask"] == blocking._BIT["dense"] | blocking._BIT["name_char"]
    pr = blocking.prune(u, u["dense_score"].fillna(0).to_numpy(), top=1)
    assert pr.groupby("q_row").size().max() == 1
    true_keys = np.sort(np.array([0 * 10 + 6, 1 * 10 + 8, 1 * 10 + 9], dtype=np.int64))
    rep = blocking.blocking_report(
        u, pr, true_keys, np.array([0, 1]), np.array(["X", "Y"]), np.array([1, 2]), n_pool=10, n_pool_total=10
    )
    assert rep["ALL"]["pair_recall_union"] == pytest.approx(2 / 3, abs=1e-4)
    assert rep["ALL"]["channels"]["name_char"]["unique"] == 1  # (1, 8) only via name_char
    assert rep["ALL"]["pair_recall"] == 0.0  # the top-1 by dense score drops both true pairs


@pytest.mark.skipif(not (SAMPLE / "train" / "train_ground_truth.tsv").exists(), reason="sample/ not built")
def test_sparse_channels_recall_on_sample():
    """On sample/, the union of the sparse channels alone recovers most true pairs within country."""
    frames = {s: normalize.normalize_df(io_utils.load_tsv(SAMPLE / "train" / f"train_source{s}.tsv"), f"S{s}") for s in "123"}
    s1 = frames["1"]
    pool = pd.concat([frames["2"], frames["3"]], ignore_index=True)
    gt = io_utils.load_ground_truth(SAMPLE / "train" / "train_ground_truth.tsv")
    pairs = split.gt_long(gt)
    parts = {c: [] for c in ("name_char", "addr_char", "name_tok", "num_key")}
    for c in s1["country"].unique():
        qi = np.flatnonzero(s1["country"].to_numpy() == c)
        pi = np.flatnonzero(pool["country"].to_numpy() == c)
        q, p = s1.iloc[qi], pool.iloc[pi]
        texts = {
            "name_char": (q["norm_name"].tolist(), p["norm_name"].tolist(), "char", 20),
            "addr_char": (q["norm_addr"].tolist(), p["norm_addr"].tolist(), "char", 15),
            "name_tok": (blocking.name_tok_text(q["norm_name"], q["fold_name"]), blocking.name_tok_text(p["norm_name"], p["fold_name"]), "word", 20),
            "num_key": (blocking.num_key_text(q["norm_addr"], q["house_number"]), blocking.num_key_text(p["norm_addr"], p["house_number"]), "word", 20),
        }
        for ch, (qt, pt, kind, k) in texts.items():
            a, b, s, r = blocking.run_sparse_channel(qt, pt, kind, k=k, max_df=500)
            parts[ch].append((qi[a], pi[b], s, r))
    parts = {c: tuple(np.concatenate([x[i] for x in v]) for i in range(4)) for c, v in parts.items()}
    u = blocking.union_channels(parts, n_pool=len(pool))
    s1_pos = pd.Series(np.arange(len(s1)), index=s1["entity_id"])
    p_pos = pd.Series(np.arange(len(pool)), index=pool["entity_id"])
    keys = s1_pos.loc[pairs["s1_id"]].to_numpy().astype(np.int64) * len(pool) + p_pos.loc[pairs["match_id"]].to_numpy()
    got = np.isin(keys, u["q_row"].to_numpy().astype(np.int64) * len(pool) + u["p_row"].to_numpy())
    # The sample pool is ~0.2% of the real one, so this is an easy setting: a
    # loose floor that catches a broken channel, not a quality claim.
    assert got.mean() > 0.95


@pytest.mark.skipif(not (SAMPLE / "train" / "train_ground_truth.tsv").exists(), reason="sample/ not built")
def test_build_candidates_pipeline_on_sample():
    """`build_union` + `train_pruner` + `build_candidates` (src/predict.py's blocking step) on sample/."""
    cfg = io_utils.load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")["blocking"]
    frames = {s: normalize.normalize_df(io_utils.load_tsv(SAMPLE / "train" / f"train_source{s}.tsv"), f"S{s}") for s in "123"}
    s1 = frames["1"]
    pool = pd.concat([frames["2"], frames["3"]], ignore_index=True)
    gt = io_utils.load_ground_truth(SAMPLE / "train" / "train_ground_truth.tsv")
    pairs = split.gt_long(gt)

    u = blocking.build_union(s1, pool, cfg)
    s1_pos = pd.Series(np.arange(len(s1)), index=s1["entity_id"])
    p_pos = pd.Series(np.arange(len(pool)), index=pool["entity_id"])
    keys_true = s1_pos.loc[pairs["s1_id"]].to_numpy().astype(np.int64) * len(pool) + p_pos.loc[pairs["match_id"]].to_numpy()
    y = np.isin(u["q_row"].to_numpy().astype(np.int64) * len(pool) + u["p_row"].to_numpy(), keys_true).astype(np.int8)
    pruner = blocking.train_pruner(u, y, seed=0)

    cands = blocking.build_candidates(s1, pool, cfg, pruner=pruner)
    assert set(cands.columns) >= {"s1_id", "cand_id", "bitmask", "n_channels", "cheap_score"}
    assert not cands.duplicated(["s1_id", "cand_id"]).any()
    assert (cands.groupby("s1_id").size() <= cfg["prune_top"]).all()
    got = np.isin(pairs["s1_id"] + "|" + pairs["match_id"], (cands["s1_id"] + "|" + cands["cand_id"]).to_numpy())
    assert got.mean() > 0.90  # same easy small-pool setting as the channel-only test above
