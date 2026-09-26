"""Step 6 — candidate generation (blocking), master plan SS12.

Channels (all restricted to equal country labels — EDA E5 found 0 cross-
country matches — by label equality only, never by value):
- dense       S1 -> pool top-k from NB03 (frozen) or NB04 (fine-tuned) vectors
- reverse     pool -> S1 top-rk from the same run (keeps pairs whose S1 is
              crowded by near neighbours)
- name_char   char_wb 3-4-gram TF-IDF on the normalised name
- addr_char   char_wb 3-4-gram TF-IDF on the normalised address
- name_tok    rare-token key: word tokens of norm + fold name (typo- and
              transliteration-tolerant via the fold form), IDF-weighted
- num_key     house number + first street token ("35840_chester"); stands in
              for the plan's postcode key, since NB02 found postcodes in only
              ~1-2% of records while house numbers are in ~90%

All sparse channels share one mechanism (`sparse_topk`): hashed features ->
TF-IDF (sublinear tf, smoothed idf, L2 rows) -> chunked sparse product ->
top-k per query. Index-side features with df above `max_df` are dropped
before the product (they carry little identity and dominate the cost); the
scores are then partial cosines, fine for ranking.

The union is pruned to `prune_top` per S1 by a small LightGBM pre-ranker
trained on Half A labels (cheap vectorised features: channel scores/ranks,
dense cosine, gaps to the S1's best). The pruned set is what later stages
score, i.e. `candidate_pairs.tsv` (up to the scale-guard cut in NB06).
"""

from __future__ import annotations

import time
import multiprocessing

import numpy as np
import pandas as pd
import scipy.sparse as sp

CHANNELS = ["dense", "reverse", "name_char", "addr_char", "name_tok", "num_key"]
_BIT = {c: 1 << i for i, c in enumerate(CHANNELS)}

# ---------------------------------------------------------------------------
# Per-record channel texts
# ---------------------------------------------------------------------------


def name_tok_text(norm_name: pd.Series, fold_name: pd.Series) -> list[str]:
    """Token text for the rare-token channel: norm tokens + '~'-prefixed fold tokens.

    Digit-only tokens are dropped (numbers are compared elsewhere).
    Inputs: norm_name, fold_name - aligned Series. Outputs: list of strings.
    """
    out = []
    for n, f in zip(norm_name.tolist(), fold_name.tolist()):
        toks = [t for t in n.split() if not t.isdigit() and t != "&"]
        toks += ["~" + t for t in f.split() if not t.isdigit() and t != "&" and t not in toks]
        out.append(" ".join(toks))
    return out


def num_key_text(norm_addr: pd.Series, house_number: pd.Series) -> list[str]:
    """"<house number>_<first non-number token after it>" key per record, or "".

    Inputs: norm_addr, house_number - aligned Series. Outputs: list of keys.
    """
    out = []
    for a, h in zip(norm_addr.tolist(), house_number.tolist()):
        key = ""
        if h:
            toks = a.split()
            try:
                i = toks.index(h)
            except ValueError:
                i = -1
            for t in toks[i + 1 :]:
                if not any(ch.isdigit() for ch in t):
                    key = f"{h}_{t}"
                    break
        out.append(key)
    return out


def _fork_pool(n_jobs: int):
    """A fork-based process pool, or None where fork is unavailable (Windows: run in-process).

    Fork lets workers share the big matrices set in module globals without
    pickling them.
    """
    if n_jobs > 1 and "fork" in multiprocessing.get_all_start_methods():
        return multiprocessing.get_context("fork").Pool(n_jobs)
    return None


# ---------------------------------------------------------------------------
# Hashed TF-IDF matrices
# ---------------------------------------------------------------------------
_N_FEATURES = 2**22


def _vectorizer(kind: str):
    """HashingVectorizer for a channel kind ("char" or "word")."""
    from sklearn.feature_extraction.text import HashingVectorizer

    if kind == "char":
        return HashingVectorizer(
            analyzer="char_wb", ngram_range=(3, 4), n_features=_N_FEATURES,
            alternate_sign=False, norm=None, dtype=np.float32, lowercase=False,
        )
    return HashingVectorizer(
        analyzer=str.split, n_features=_N_FEATURES, alternate_sign=False, norm=None, dtype=np.float32, lowercase=False,
    )


def _hash_chunk(args):
    """Worker: hash one chunk of texts."""
    kind, texts = args
    return _vectorizer(kind).transform(texts)


def hashed_counts(texts: list[str], kind: str, n_jobs: int = 1, chunk: int = 200_000) -> sp.csr_matrix:
    """Term-count matrix of texts with the channel's hashing vectorizer (parallel).

    Inputs: texts; kind - "char"/"word"; n_jobs; chunk - texts per task.
    Outputs: csr_matrix [len(texts), 2**22] float32 counts.
    """
    parts = [(kind, texts[i : i + chunk]) for i in range(0, len(texts), chunk)]
    if not parts:
        return sp.csr_matrix((0, _N_FEATURES), dtype=np.float32)
    pool = _fork_pool(n_jobs) if len(parts) > 1 else None
    if pool is not None:
        with pool:
            mats = pool.map(_hash_chunk, parts)
    else:
        mats = [_hash_chunk(p) for p in parts]
    return sp.vstack(mats).tocsr()


def _l2_rows(m: sp.csr_matrix) -> None:
    """L2-normalise csr rows in place (empty rows stay empty)."""
    # bincount over row ids, not np.add.reduceat: reduceat fails when trailing
    # rows are empty (NB05 sparse v1 crashed on empty addresses at the end).
    counts = np.diff(m.indptr)
    rows = np.repeat(np.arange(m.shape[0]), counts)
    sq = np.bincount(rows, weights=m.data.astype(np.float64) ** 2, minlength=m.shape[0])
    norms = np.where(sq > 0, np.sqrt(sq), 1.0)
    m.data /= norms[rows].astype(np.float32)


def tfidf_pair(q_counts: sp.csr_matrix, p_counts: sp.csr_matrix, max_df: int) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """TF-IDF weight query and pool counts with shared df; prune frequent pool features.

    idf = log((N+1)/(df+1)) + 1 over query + pool rows; tf = 1 + log(count);
    rows L2-normalised; then pool columns with df > max_df are removed.

    Inputs: q_counts, p_counts - hashed count matrices; max_df - index cap.
    Outputs: (Q weighted, P weighted and pruned), both csr float32.
    """
    df = np.bincount(p_counts.indices, minlength=_N_FEATURES) + np.bincount(q_counts.indices, minlength=_N_FEATURES)
    n = q_counts.shape[0] + p_counts.shape[0]
    idf = (np.log((n + 1) / (df + 1)) + 1).astype(np.float32)
    out = []
    for m in (q_counts, p_counts):
        m = m.copy()
        m.data = (1 + np.log(m.data)) * idf[m.indices]
        _l2_rows(m)
        out.append(m)
    q, p = out
    keep = (df <= max_df)[p.indices]
    p.data = p.data * keep
    p.eliminate_zeros()
    return q, p


# ---------------------------------------------------------------------------
# Chunked sparse top-k
# ---------------------------------------------------------------------------
_G: dict = {}  # worker globals (inherited through fork)


def _topk_rows(start: int, end: int):
    """Worker: top-k pool rows for query rows [start, end) of _G['Q'] against _G['PT']."""
    r = (_G["Q"][start:end] @ _G["PT"]).tocsr()
    k = _G["k"]
    qi, pi, sc, rk = [], [], [], []
    for row in range(r.shape[0]):
        a, b = r.indptr[row], r.indptr[row + 1]
        if a == b:
            continue
        d, ix = r.data[a:b], r.indices[a:b]
        if b - a > k:
            sel = np.argpartition(-d, k)[:k]
            d, ix = d[sel], ix[sel]
        order = np.argsort(-d, kind="stable")
        qi.append(np.full(len(order), start + row, dtype=np.int32))
        pi.append(ix[order].astype(np.int32))
        sc.append(d[order].astype(np.float32))
        rk.append(np.arange(len(order), dtype=np.int16))
    if not qi:
        return (np.empty(0, np.int32),) * 2 + (np.empty(0, np.float32), np.empty(0, np.int16))
    return np.concatenate(qi), np.concatenate(pi), np.concatenate(sc), np.concatenate(rk)


def _topk_task(bounds):
    """Pool.map adapter for `_topk_rows`."""
    return _topk_rows(*bounds)


def sparse_topk(q: sp.csr_matrix, p: sp.csr_matrix, k: int, n_jobs: int = 1, chunk: int = 1000):
    """Top-k pool rows per query row by sparse dot product.

    Inputs: q [nq, F], p [np, F] csr (already weighted/normalised); k;
            n_jobs - fork workers (1 = in-process, used by tests/Windows);
            chunk - query rows per task.
    Outputs: (q_row, p_row, score, rank) arrays; rank 0 = best.
    """
    _G.update(Q=q, PT=p.T.tocsr(), k=k)
    bounds = [(s, min(s + chunk, q.shape[0])) for s in range(0, q.shape[0], chunk)]
    pool = _fork_pool(n_jobs) if len(bounds) > 1 else None
    if pool is not None:
        with pool:
            parts = pool.map(_topk_task, bounds, chunksize=4)
    else:
        parts = [_topk_task(b) for b in bounds]
    _G.clear()
    parts = [x for x in parts if len(x[0])]
    if not parts:
        return np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32), np.empty(0, np.int16)
    return tuple(np.concatenate([x[i] for x in parts]) for i in range(4))


def run_sparse_channel(q_texts: list[str], p_texts: list[str], kind: str, k: int, max_df: int, n_jobs: int = 1):
    """Hash + TF-IDF + top-k for one channel within one country group.

    Inputs: query/pool channel texts; kind "char"/"word"; k; max_df; n_jobs.
    Outputs: (q_row, p_row, score, rank) local to the given lists.
    """
    qc = hashed_counts(q_texts, kind, n_jobs)
    pc = hashed_counts(p_texts, kind, n_jobs)
    q, p = tfidf_pair(qc, pc, max_df)
    del qc, pc
    return sparse_topk(q, p, k, n_jobs)


# ---------------------------------------------------------------------------
# Union of channels + cheap features
# ---------------------------------------------------------------------------


def union_channels(parts: dict[str, tuple], n_pool: int) -> pd.DataFrame:
    """Merge per-channel (q_row, p_row, score, rank) lists into one row per pair.

    Inputs: parts - {channel: (q_row, p_row, score, rank)} with global rows;
            n_pool - pool size (for the int64 pair key).
    Outputs: DataFrame sorted by (q_row, p_row) with q_row, p_row, bitmask,
    n_channels, and <channel>_score / <channel>_rank (NaN / 999 when absent).
    """
    keys = {c: v[0].astype(np.int64) * n_pool + v[1].astype(np.int64) for c, v in parts.items()}
    allk = np.unique(np.concatenate(list(keys.values()))) if keys else np.empty(0, np.int64)
    out = pd.DataFrame({"q_row": (allk // n_pool).astype(np.int32), "p_row": (allk % n_pool).astype(np.int32)})
    bitmask = np.zeros(len(allk), dtype=np.int16)
    for c in CHANNELS:
        score = np.full(len(allk), np.nan, dtype=np.float32)
        rank = np.full(len(allk), 999, dtype=np.int16)
        if c in keys and len(keys[c]):
            # A channel can list a pair twice only through a bug; keep the best rank.
            kk, first = np.unique(keys[c], return_index=True)
            pos = np.searchsorted(allk, kk)
            score[pos] = parts[c][2][first]
            rank[pos] = parts[c][3][first]
            bitmask[pos] |= _BIT[c]
        out[f"{c}_score"] = score
        out[f"{c}_rank"] = rank
    out["bitmask"] = bitmask
    out["n_channels"] = np.array([bin(b).count("1") for b in range(1 << len(CHANNELS))], dtype=np.int8)[bitmask]
    return out


def add_dense_cos(u: pd.DataFrame, q_emb: np.ndarray, p_emb: np.ndarray, chunk: int = 1_000_000) -> None:
    """Add `dense_cos` (embedding dot product) for every union pair, in place, chunked."""
    cos = np.empty(len(u), dtype=np.float32)
    qr, pr = u["q_row"].to_numpy(), u["p_row"].to_numpy()
    for s in range(0, len(u), chunk):
        a = q_emb[qr[s : s + chunk]].astype(np.float32)
        b = p_emb[pr[s : s + chunk]].astype(np.float32)
        cos[s : s + chunk] = np.einsum("ij,ij->i", a, b)
    u["dense_cos"] = cos


def add_gap_features(u: pd.DataFrame) -> None:
    """Add per-S1 relative features (score minus the S1's best), in place.

    Requires `u` sorted by q_row (as `union_channels` returns it).
    """
    starts = np.flatnonzero(np.r_[True, u["q_row"].to_numpy()[1:] != u["q_row"].to_numpy()[:-1]])
    sizes = np.diff(np.r_[starts, len(u)])
    for col in ("dense_cos", "name_char_score", "addr_char_score", "name_tok_score"):
        v = u[col].to_numpy()
        best = np.fmax.reduceat(np.nan_to_num(v, nan=-1.0), starts)
        u[f"{col}_gap"] = v - np.repeat(best, sizes)
    u["union_size"] = np.repeat(sizes, sizes).astype(np.int16)


PRUNE_FEATURES = [
    "dense_cos", "dense_cos_gap", "dense_rank", "reverse_rank", "reverse_score",
    "name_char_score", "name_char_rank", "name_char_score_gap",
    "addr_char_score", "addr_char_rank", "addr_char_score_gap",
    "name_tok_score", "name_tok_rank", "name_tok_score_gap",
    "num_key_score", "n_channels", "union_size",
]


def train_pruner(u: pd.DataFrame, y: np.ndarray, seed: int = 42, max_rows: int = 3_000_000):
    """Fit the cheap LightGBM pre-ranker on (Half A) union pairs.

    Inputs: u - union frame with PRUNE_FEATURES; y - 0/1 labels; seed;
            max_rows - subsample cap for speed.
    Outputs: trained lightgbm.Booster.
    """
    import lightgbm as lgb

    idx = np.arange(len(u))
    if len(idx) > max_rows:
        idx = np.random.default_rng(seed).choice(idx, max_rows, replace=False)
        idx.sort()
    params = {
        "objective": "binary", "learning_rate": 0.1, "num_leaves": 31, "min_data_in_leaf": 100,
        "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "seed": seed,
        "bagging_seed": seed, "feature_fraction_seed": seed, "deterministic": True, "verbose": -1,
    }
    ds = lgb.Dataset(u[PRUNE_FEATURES].iloc[idx].to_numpy(np.float32), label=y[idx], feature_name=PRUNE_FEATURES)
    return lgb.train(params, ds, num_boost_round=150)


def prune(u: pd.DataFrame, score: np.ndarray, top: int) -> pd.DataFrame:
    """Keep the `top` highest-scoring pairs per S1 (ties broken by pool row).

    Inputs: u - union frame; score - pre-ranker score per row; top.
    Outputs: pruned frame (sorted by q_row, score desc) with `cheap_score`.
    """
    u = u.assign(cheap_score=score.astype(np.float32))
    order = np.lexsort((u["p_row"].to_numpy(), -u["cheap_score"].to_numpy(), u["q_row"].to_numpy()))
    u = u.iloc[order]
    q = u["q_row"].to_numpy()
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    rank = np.arange(len(u)) - np.repeat(starts, np.diff(np.r_[starts, len(u)]))
    return u[rank < top].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Blocking report (plan SS12.3)
# ---------------------------------------------------------------------------


def blocking_report(
    union: pd.DataFrame, pruned: pd.DataFrame, true_keys: np.ndarray, q_rows: np.ndarray, q_country: np.ndarray,
    n_true_by_q: np.ndarray, n_pool: int, n_pool_total: int,
) -> dict:
    """Pair recall, entity-complete recall, RR, candidates/S1 and per-channel unique contribution.

    Inputs: union/pruned - candidate frames (q_row, p_row, bitmask); true_keys -
            sorted int64 keys q_row*n_pool+p_row of all true pairs of the
            queried S1s; q_rows - queried S1 rows; q_country - their country
            labels; n_true_by_q - true-match count per queried S1 (aligned to
            q_rows); n_pool - key base; n_pool_total - pool size for RR.
    Outputs: nested dict, overall and per country.
    """
    def _is_true(df):
        k = df["q_row"].to_numpy().astype(np.int64) * n_pool + df["p_row"].to_numpy()
        return np.isin(k, true_keys)

    country_of = pd.Series(q_country, index=q_rows)
    ntrue_of = pd.Series(n_true_by_q, index=q_rows)
    u_true = _is_true(union)
    p_true = _is_true(pruned)

    def _one(sel_q: np.ndarray) -> dict:
        qs = set(sel_q.tolist())
        um = union["q_row"].isin(qs).to_numpy()
        pm = pruned["q_row"].isin(qs).to_numpy()
        n_true = int(ntrue_of.loc[sel_q].sum())
        found = pd.Series(p_true[pm]).groupby(pruned["q_row"].to_numpy()[pm]).sum()
        need = ntrue_of.loc[sel_q]
        need = need[need > 0]
        complete = (found.reindex(need.index).fillna(0).to_numpy() == need.to_numpy()).mean() if len(need) else 1.0
        cps = pruned.loc[pm].groupby("q_row").size().reindex(sel_q).fillna(0)
        uni = {}
        bm_true = union.loc[um & u_true, "bitmask"].to_numpy()
        for c, bit in _BIT.items():
            uni[c] = {
                "found": int(((bm_true & bit) > 0).sum()),
                "unique": int((bm_true == bit).sum()),
            }
        return {
            "n_s1": int(len(sel_q)),
            "n_true_pairs": n_true,
            "pair_recall_union": round(float(u_true[um].sum() / max(n_true, 1)), 5),
            "pair_recall": round(float(p_true[pm].sum() / max(n_true, 1)), 5),
            "entity_complete_recall": round(float(complete), 5),
            "reduction_ratio": round(1 - float(pm.sum()) / max(len(sel_q) * n_pool_total, 1), 8),
            "cands_per_s1_mean": round(float(cps.mean()), 2),
            "cands_per_s1_p95": float(np.quantile(cps, 0.95)) if len(cps) else 0.0,
            "union_per_s1_mean": round(float(um.sum() / max(len(sel_q), 1)), 2),
            "channels": uni,
        }

    rep = {"ALL": _one(q_rows)}
    for c in sorted(set(q_country.tolist())):
        rep[c] = _one(q_rows[q_country == c])
    return rep


def main() -> None:
    """Smoke test: one sparse channel on four toy names."""
    q = ["alpha trading co", "beta foods"]
    p = ["alpha tradng company", "gamma foods", "beta food"]
    print(run_sparse_channel(q, p, "char", k=2, max_df=10))


if __name__ == "__main__":
    main()
