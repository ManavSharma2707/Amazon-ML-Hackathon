"""Pair feature table for candidate pairs (master plan SS14, groups A-H).

Groups:
  A  string similarities (rapidfuzz, vectorised `cpdist`) on norm/fold name
     and address, plus token / IDF-weighted token / char-3-gram Jaccard;
  B  explain-the-difference features for name and address (explain_diff.py);
  C  number agreement (explain_diff.number_features);
  D  embedding cosines: not available (dense encoding of all records is
     infeasible at ~400 rec/s, memory.md SS5) -> omitted;
  E  structure and missingness (token counts, length ratio, empty flags,
     postcode / landmark presence, landmark similarity, romanised script);
  F  blocking meta (channel bits, channel scores, pre-ranker score, rank and
     gap to the S1's best, candidates per S1);
  G  source (`is_S3`);
  H  `same_country` is constant 1 (blocking is within-country, EDA E5) -> dropped.
No raw tokens, no country values, no IDs. Fine-tuned embeddings are never used
here (stage-1 is trained on Half A, plan SS10.3).

Per-pair Python work (groups A-Jaccard, B, C, E) runs in a fork-based process
pool; every worker keeps its own token-relation cache (explain_diff._CACHE).
Record tables are shared with the workers through a module global set before
the fork, never pickled.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import blocking, corpus_stats, explain_diff

SPARSE_CHANNELS = ["name_char", "addr_char", "name_tok", "num_key", "name_pair", "addr_pair", "cross_pair"]
REC_COLS = ["entity_id", "country", "norm_name", "fold_name", "norm_addr", "fold_addr", "name_numbers",
            "addr_numbers", "house_number", "postcodes", "landmark", "name_romanized"]

# Group A, vectorised: (feature name, record field, scorer name). Scorers were
# timed per pair (WRatio / Damerau-Levenshtein on addresses cost 20+ us each),
# so addresses get the cheap ones and Levenshtein replaces Damerau-Levenshtein.
_RF_SPECS = [(f"norm_name_{s}", "norm_name", s)
             for s in ("ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio", "WRatio", "jw", "lev")]
_RF_SPECS += [(f"norm_addr_{s}", "norm_addr", s) for s in ("ratio", "token_set_ratio", "jw", "lev")]
_RF_SPECS += [(f"{f}_{s}", f, s) for f in ("fold_name", "fold_addr") for s in ("ratio", "jw")]
# Pre-ranker subset (~8 us/pair): the fastest scorers only.
_PRE_SPECS = [(f"norm_name_{s}", "norm_name", s) for s in ("ratio", "token_set_ratio", "jw", "lev")]
_PRE_SPECS += [(f"norm_addr_{s}", "norm_addr", s) for s in ("ratio", "jw", "lev")] + [("fold_name_ratio", "fold_name", "ratio")]
PRE_RF_FEATURES = [n for n, _, _ in _PRE_SPECS]
RF_FEATURES = [n for n, _, _ in _RF_SPECS]

PY_FEATURES = (
    [f"name_{f}" for f in explain_diff.FIELD_FEATURES]
    + [f"addr_{f}" for f in explain_diff.FIELD_FEATURES]
    + explain_diff.NUMBER_FEATURES
    + ["name_tok_jacc", "name_idf_jacc", "name_c3_jacc", "addr_tok_jacc", "addr_idf_jacc", "addr_c3_jacc",
       "name_len_ratio", "addr_len_ratio", "addr_empty_1", "addr_empty_2", "has_postcode_1", "has_postcode_2",
       "landmark_1", "landmark_2", "landmark_jacc", "romanized_1", "romanized_2"]
)
META_FEATURES = (
    [f"ch_{c}" for c in SPARSE_CHANNELS] + [f"{c}_score" for c in SPARSE_CHANNELS]
    + ["n_channels", "cheap_score", "cheap_rank", "cheap_gap", "n_cands", "is_S3"]
)
# Scale-guard pre-ranker (plan SS12.2 / memory.md SS5): cheap, fully vectorised features only.
PRERANK_FEATURES = META_FEATURES + PRE_RF_FEATURES
FEATURES = META_FEATURES + RF_FEATURES + PY_FEATURES

_CTX: dict = {}  # shared with forked workers (see module docstring)


# ---------------------------------------------------------------------------
# Token statistics -> IDF and generic-token lookups
# ---------------------------------------------------------------------------


def token_lookups(stats: pd.DataFrame, min_df_generic: int = 1000, name_end_ratio: float = 0.7,
                  addr_type_ratio: float = 0.5) -> dict:
    """IDF dicts per country and generic-token sets from the NB02 token statistics.

    Generic tokens (legal forms, street types) are frequent AND positional:
    name tokens mostly in the last two positions, address tokens mostly right
    after the house number. The frequency floor fixes the NB02 finding that
    the unfloored likeness score ranks typo variants first (memory.md SS9).
    No word lists, no country values: the country is only a grouping key.

    Inputs: stats - long token table (field, country, token, df, pos_count, n_docs).
    Outputs: {"idf": {country: (name_idf, name_default, addr_idf, addr_default)},
              "generic_name": set, "generic_addr": set}.
    """
    out: dict = {"idf": {}}
    countries = sorted(c for c in stats["country"].unique().tolist() if c != "")
    for c in countries:
        ni, nd = corpus_stats.idf_lookup(stats, "norm_name", c)
        ai, ad = corpus_stats.idf_lookup(stats, "norm_addr", c)
        out["idf"][c] = (ni, nd, ai, ad)
    for field, key, ratio in (("norm_name", "generic_name", name_end_ratio), ("norm_addr", "generic_addr", addr_type_ratio)):
        g = stats[(stats["field"] == field) & (stats["country"] == "")]
        keep = (g["df"] >= min_df_generic) & (g["pos_count"] / g["df"].clip(lower=1) >= ratio)
        out[key] = set(g.loc[keep, "token"].tolist()) | {"&"}
    return out


# ---------------------------------------------------------------------------
# Context shared with workers
# ---------------------------------------------------------------------------


def set_context(s1: pd.DataFrame, pool: pd.DataFrame, lookups: dict) -> None:
    """Store the record columns (as Python lists) and lookups for the workers.

    Must be called before `pair_features` (and before any fork).
    Inputs: s1, pool - record frames with REC_COLS (row order = row ids used
            by the pair arrays); lookups - output of `token_lookups`.
    """
    _CTX.clear()
    for side, df in (("s", s1), ("p", pool)):
        for c in REC_COLS[1:]:
            col = df[c]
            _CTX[f"{side}_{c}"] = (np.asarray(col, dtype=bool).tolist() if c == "name_romanized"
                                   else [str(x) for x in col.tolist()])
        # fresh Python objects for the (subset) records: forked workers then only touch these pages
        _CTX[f"{side}_arr"] = {c: np.array([str(x) for x in df[c].tolist()], dtype=object)
                               for c in ("norm_name", "fold_name", "norm_addr", "fold_addr")}
        _CTX[f"{side}_len"] = {c: np.fromiter(map(len, a), dtype=np.int32, count=len(a)) for c, a in _CTX[f"{side}_arr"].items()}
    _CTX["lookups"] = lookups
    explain_diff._CACHE.clear()


def _c3(s: str) -> set:
    """Character 3-grams of a string (with boundary spaces)."""
    s = f" {s} "
    return {s[i : i + 3] for i in range(len(s) - 2)}


def _jacc(a: set, b: set) -> float:
    """Jaccard of two sets (NaN if both empty)."""
    if not a and not b:
        return float("nan")
    return len(a & b) / len(a | b)


def _idf_jacc(t1: list[str], w1: list[float], t2: list[str], w2: list[float]) -> float:
    """IDF-weighted token Jaccard."""
    d1, d2 = dict(zip(t1, w1)), dict(zip(t2, w2))
    inter = sum(d1[t] for t in d1.keys() & d2.keys())
    union = sum(d1.values()) + sum(v for t, v in d2.items() if t not in d1)
    return inter / union if union > 0 else float("nan")


def _side(tokens: list[str], idf: dict, default: float, generic: set) -> tuple[list[float], list[bool]]:
    """IDF values and generic flags of a token list."""
    return [idf.get(t, default) for t in tokens], [t in generic for t in tokens]


def _py_block(q_rows: np.ndarray, p_rows: np.ndarray) -> np.ndarray:
    """Per-pair Python features (PY_FEATURES) for aligned arrays of S1 / pool rows."""
    C = _CTX
    lk = C["lookups"]
    gen_n, gen_a = lk["generic_name"], lk["generic_addr"]
    out = np.full((len(q_rows), len(PY_FEATURES)), np.nan, dtype=np.float32)
    last_i, s_cache = -1, None
    for k in range(len(q_rows)):
        i, j = int(q_rows[k]), int(p_rows[k])
        if i != last_i:  # pairs are grouped by S1: prepare the S1 side once
            ni, nd, ai, ad = lk["idf"].get(C["s_country"][i], ({}, 1.0, {}, 1.0))
            n1, a1 = C["s_norm_name"][i].split(), C["s_norm_addr"][i].split()
            nw1, ng1 = _side(n1, ni, nd, gen_n)
            aw1, ag1 = _side(a1, ai, ad, gen_a)
            rec1 = {c: C[f"s_{c}"][i] for c in ("house_number", "postcodes", "addr_numbers", "name_numbers")}
            lm1 = set(C["s_landmark"][i].split())
            s_cache = (n1, a1, nw1, ng1, aw1, ag1, rec1, lm1, _c3(C["s_norm_name"][i]), _c3(C["s_norm_addr"][i]),
                       len(C["s_norm_name"][i]), len(C["s_norm_addr"][i]), float(C["s_name_romanized"][i]))
            last_i = i
        n1, a1, nw1, ng1, aw1, ag1, rec1, lm1, c3n1, c3a1, ln1, la1, rom1 = s_cache
        pn, pa = C["p_norm_name"][j], C["p_norm_addr"][j]
        n2, a2 = pn.split(), pa.split()
        nw2, ng2 = _side(n2, ni, nd, gen_n)
        aw2, ag2 = _side(a2, ai, ad, gen_a)
        rec2 = {c: C[f"p_{c}"][j] for c in ("house_number", "postcodes", "addr_numbers", "name_numbers")}
        lm2 = set(C["p_landmark"][j].split())
        fn, _ = explain_diff.explain_features(n1, n2, nw1, nw2, ng1, ng2)
        fa, _ = explain_diff.explain_features(a1, a2, aw1, aw2, ag1, ag2)
        num = explain_diff.number_features(rec1, rec2)
        ln2, la2 = len(pn), len(pa)
        extra = [
            _jacc(set(n1), set(n2)), _idf_jacc(n1, nw1, n2, nw2), _jacc(c3n1, _c3(pn)),
            _jacc(set(a1), set(a2)), _idf_jacc(a1, aw1, a2, aw2), _jacc(c3a1, _c3(pa)) if la1 and la2 else np.nan,
            min(ln1, ln2) / max(ln1, ln2) if max(ln1, ln2) else np.nan,
            min(la1, la2) / max(la1, la2) if max(la1, la2) else np.nan,
            float(la1 == 0), float(la2 == 0), float(bool(rec1["postcodes"])), float(bool(rec2["postcodes"])),
            float(bool(lm1)), float(bool(lm2)), _jacc(lm1, lm2) if lm1 and lm2 else np.nan,
            rom1, float(C["p_name_romanized"][j]),
        ]
        out[k] = fn + fa + num + extra
    return out


def _rf_block(q_rows: np.ndarray, p_rows: np.ndarray, specs: list = _RF_SPECS, workers: int = 1) -> np.ndarray:
    """Vectorised rapidfuzz similarities (0-100) for aligned S1 / pool rows; NaN where a side is empty."""
    from rapidfuzz import fuzz, process
    from rapidfuzz.distance import JaroWinkler, Levenshtein

    scorers = {"ratio": fuzz.ratio, "partial_ratio": fuzz.partial_ratio, "token_sort_ratio": fuzz.token_sort_ratio,
               "token_set_ratio": fuzz.token_set_ratio, "WRatio": fuzz.WRatio,
               "jw": JaroWinkler.normalized_similarity, "lev": Levenshtein.normalized_similarity}
    out = np.empty((len(q_rows), len(specs)), dtype=np.float32)
    sa, pa, sl, pl = _CTX["s_arr"], _CTX["p_arr"], _CTX["s_len"], _CTX["p_len"]
    texts: dict = {}
    for c, (_, field, s) in enumerate(specs):
        if field not in texts:
            texts[field] = (sa[field][q_rows].tolist(), pa[field][p_rows].tolist())
        a, b = texts[field]
        v = process.cpdist(a, b, scorer=scorers[s], workers=workers, dtype=np.float32)
        if s in ("jw", "lev"):
            v = v * 100.0  # same 0-100 scale as the fuzz scorers
        v[(sl[field][q_rows] == 0) | (pl[field][p_rows] == 0)] = np.nan  # unknown, not "different"
        out[:, c] = v
    return out


def _worker(args):
    """Pool task: (q_rows, p_rows, with_py) -> RF + PY feature block, or the pre-ranker RF subset."""
    q_rows, p_rows, with_py = args
    if not with_py:
        return _rf_block(q_rows, p_rows, _PRE_SPECS)
    rf = _rf_block(q_rows, p_rows)
    return np.hstack([rf, _py_block(q_rows, p_rows)])


def meta_features(cands: pd.DataFrame, pool_is_s3: np.ndarray, p_rows: np.ndarray, q_rows: np.ndarray,
                  score_col: str = "cheap_score") -> np.ndarray:
    """Blocking-meta and source features (META_FEATURES), fully vectorised.

    Ranks/gaps are computed within each S1 over the given candidate rows.
    Inputs: cands - candidate rows (bitmask, n_channels, <channel>_score,
            `score_col`); pool_is_s3 - bool per pool row; p_rows, q_rows - row ids.
    Outputs: float32 array (len(cands), len(META_FEATURES)).
    """
    n = len(cands)
    cols = []
    bm = np.asarray(cands["bitmask"], dtype=np.int64)
    for c in SPARSE_CHANNELS:
        cols.append(((bm & blocking._BIT[c]) > 0).astype(np.float32))
    for c in SPARSE_CHANNELS:
        cols.append(np.asarray(cands[f"{c}_score"], dtype=np.float32))
    cols.append(np.asarray(cands["n_channels"], dtype=np.float32))
    sc = np.asarray(cands[score_col], dtype=np.float32)
    order = np.lexsort((-sc, q_rows))
    qs = q_rows[order]
    starts = np.flatnonzero(np.r_[True, qs[1:] != qs[:-1]])
    sizes = np.diff(np.r_[starts, n])
    rank_sorted = np.arange(n) - np.repeat(starts, sizes)
    best_sorted = np.repeat(sc[order][starts], sizes)
    rank, gap, size = np.empty(n, np.float32), np.empty(n, np.float32), np.empty(n, np.float32)
    rank[order], gap[order], size[order] = rank_sorted, sc[order] - best_sorted, np.repeat(sizes, sizes)
    cols += [sc, rank, gap, size, pool_is_s3[p_rows].astype(np.float32)]
    return np.column_stack(cols).astype(np.float32)


def pair_features(q_rows: np.ndarray, p_rows: np.ndarray, with_py: bool = True, n_jobs: int = 1,
                  block: int = 20_000, log_every: int = 50, log=print) -> np.ndarray:
    """RF (+ PY) features for aligned S1/pool row arrays, in parallel blocks.

    `set_context` must have been called. Pairs should be grouped by S1 row
    (the S1-side preparation is reused across consecutive pairs).
    Inputs: q_rows, p_rows - int arrays; with_py - include PY_FEATURES;
            n_jobs - worker processes (fork; in-process where unavailable);
            block - pairs per task; log_every - progress line every N blocks.
    Outputs: float32 array (n, len(RF_FEATURES) + len(PY_FEATURES)), or
    (n, len(PRE_RF_FEATURES)) if not with_py.
    """
    tasks = [(q_rows[s : s + block], p_rows[s : s + block], with_py) for s in range(0, len(q_rows), block)]
    t0 = time.time()
    parts = []
    pool = blocking._fork_pool(n_jobs)
    try:
        it = pool.imap(_worker, tasks, chunksize=1) if pool else map(_worker, tasks)
        for b, res in enumerate(it):
            parts.append(res)
            if log_every and (b + 1) % log_every == 0:
                done = min((b + 1) * block, len(q_rows))
                log(f"    features {done:,}/{len(q_rows):,} pairs ({time.time() - t0:.0f}s, {done / max(time.time() - t0, 1e-9):,.0f}/s)")
    finally:
        if pool:
            pool.close()
            pool.join()
    width = len(RF_FEATURES) + len(PY_FEATURES) if with_py else len(PRE_RF_FEATURES)
    return np.vstack(parts) if parts else np.empty((0, width), np.float32)


def rows_for(cands: pd.DataFrame, s1: pd.DataFrame, pool: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Map candidate entity IDs to S1 / pool row ids (fails loudly on unknown IDs)."""
    q = pd.Index(s1["entity_id"]).get_indexer(cands["s1_id"])
    p = pd.Index(pool["entity_id"]).get_indexer(cands["cand_id"])
    assert (q >= 0).all() and (p >= 0).all(), "candidate IDs missing from the record tables"
    return q.astype(np.int64), p.astype(np.int64)


def build_features(cands: pd.DataFrame, s1: pd.DataFrame, pool: pd.DataFrame, lookups: dict,
                   n_jobs: int = 1, with_py: bool = True, score_col: str = "cheap_score", log=print) -> pd.DataFrame:
    """Full feature table for a candidate frame (FEATURES, or PRERANK_FEATURES if not with_py).

    Inputs: cands - s1_id, cand_id, bitmask, n_channels, channel scores,
            `score_col`; s1, pool - record frames with REC_COLS; lookups -
            `token_lookups` output; n_jobs; with_py; score_col; log.
    Outputs: DataFrame (same row order as cands) of float32 features.
    """
    cands = cands.reset_index(drop=True)
    q, p = rows_for(cands, s1, pool)
    set_context(s1, pool, lookups)
    is_s3 = np.asarray(pool["entity_id"].str.startswith("S3-"), dtype=bool)
    meta = meta_features(cands, is_s3, p, q, score_col)
    # sort by S1 row for the per-S1 cache in the workers, then restore order
    order = np.argsort(q, kind="stable")
    rest = pair_features(q[order], p[order], with_py=with_py, n_jobs=n_jobs, log=log)
    back = np.empty_like(order)
    back[order] = np.arange(len(order))
    arr = np.hstack([meta, rest[back]])
    names = FEATURES if with_py else PRERANK_FEATURES
    return pd.DataFrame(arr, columns=names)


def time_per_pair(cands: pd.DataFrame, s1: pd.DataFrame, pool: pd.DataFrame, lookups: dict, n_jobs: int,
                  n: int = 10_000, with_py: bool = True) -> float:
    """Wall-clock seconds per pair of `build_features` on the first `n` candidate rows (all workers)."""
    sub = cands.iloc[:n]
    t0 = time.time()
    build_features(sub, s1, pool, lookups, n_jobs=n_jobs, with_py=with_py, log=lambda m: None)
    return (time.time() - t0) / max(len(sub), 1)


def main() -> None:
    """Print the feature list sizes (smoke test)."""
    print(f"{len(META_FEATURES)} meta + {len(RF_FEATURES)} rapidfuzz + {len(PY_FEATURES)} python = {len(FEATURES)}")


if __name__ == "__main__":
    main()
