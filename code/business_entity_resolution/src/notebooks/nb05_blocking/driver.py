"""NB05 driver — blocking: channels -> union -> cheap pre-ranker -> top-k (architecture.md SS6.7).

Bundled with kaggle_env/io_utils/blocking into `nb05_blocking.py`
(CLAUDE.md SS6.3). CPU kernel, internet OFF.

Inputs (globbed under /kaggle/input):
- NB02 output: records_*.parquet, gt.parquet, split_s1.parquet
- NB03 output: emb_{split}_{S1,pool}.npy, knn_/rev_{split}_{idx,score}.npy (frozen)
- optional NB04 output (DENSE_FT = True): ft_emb_*/ft_knn_*/ft_rev_* for B and
  test; Half A always uses the frozen vectors (plan SS10.3).

Outputs in /kaggle/working:
- cands_{A,B,test}.parquet  s1_id, cand_id, bitmask, channel scores/ranks,
  dense_cos, cheap_score (the pruned candidate set)
- pruner.txt                LightGBM pre-ranker (trained on Half A)
- blocking_report.json      SS12.3 metrics for A and B (per country, per channel)
- missed_B.tsv              <= 200 true B pairs missing from the candidates
- metrics.json
"""

import gc
import os
import time

import numpy as np
import pandas as pd

DENSE_FT = globals().get("DENSE_FT", False)
# NO_DENSE (kernel er-nb05-blocking-sparse): sparse channels only, no NB03
# input; dense features are NaN for the pre-ranker (LightGBM handles missing).
NO_DENSE = globals().get("NO_DENSE", False)
WORK = kaggle_env.WORK_DIR
N_JOBS = os.cpu_count() or 1
BC = CONFIG["blocking"]
CHUNK_S1 = int(os.environ.get("ER_CHUNK_S1", 200_000))  # S1 rows per union/feature/prune chunk (memory bound)
COLS = ["entity_id", "country", "norm_name", "fold_name", "norm_addr", "fold_addr", "house_number"]


def load_split(recs_dir, split: str):
    """S1 and pool (S2 then S3) frames with the channel columns."""
    s1 = pd.read_parquet(recs_dir / f"records_{split}_S1.parquet", columns=COLS)
    pool = pd.concat([pd.read_parquet(recs_dir / f"records_{split}_S{s}.parquet", columns=COLS) for s in (2, 3)], ignore_index=True)
    return s1, pool


def dense_parts(emb_dir, prefix: str, split: str, q_rows: np.ndarray, n_s1: int):
    """Forward and reverse dense channel lists for the queried S1 rows.

    Outputs: {"dense": (q_row, p_row, score, rank), "reverse": (...)} with global rows.
    """
    fi = np.load(emb_dir / f"{prefix}knn_{split}_idx.npy", mmap_mode="r")
    fs = np.load(emb_dir / f"{prefix}knn_{split}_score.npy", mmap_mode="r")
    f_idx, f_sc = np.asarray(fi[q_rows]), np.asarray(fs[q_rows]).astype(np.float32)
    k = f_idx.shape[1]
    qr = np.repeat(q_rows, k).astype(np.int32)
    rk = np.tile(np.arange(k, dtype=np.int16), len(q_rows))
    ok = f_idx.ravel() >= 0
    dense = (qr[ok], f_idx.ravel()[ok], f_sc.ravel()[ok], rk[ok])
    ri = np.load(emb_dir / f"{prefix}rev_{split}_idx.npy")
    rs = np.load(emb_dir / f"{prefix}rev_{split}_score.npy").astype(np.float32)
    rk = ri.shape[1]
    p_rows = np.repeat(np.arange(len(ri), dtype=np.int32), rk)
    s1r, sc = ri.ravel(), rs.ravel()
    rank = np.tile(np.arange(rk, dtype=np.int16), len(ri))
    want = np.zeros(n_s1, dtype=bool)
    want[q_rows] = True
    ok = (s1r >= 0) & want[np.clip(s1r, 0, None)]
    reverse = (s1r[ok].astype(np.int32), p_rows[ok], sc[ok], rank[ok])
    return {"dense": dense, "reverse": reverse}


def sparse_parts(s1: pd.DataFrame, pool: pd.DataFrame, q_rows: np.ndarray) -> dict:
    """Run the four sparse channels per country group; return global-row lists."""
    parts = {c: [] for c in ("name_char", "addr_char", "name_tok", "num_key", "name_pair", "addr_pair", "cross_pair")}
    s1_cty, p_cty = s1["country"].to_numpy(), pool["country"].to_numpy()
    for c in sorted(set(s1_cty[q_rows].tolist())):
        qi = q_rows[s1_cty[q_rows] == c]
        pi = np.flatnonzero(p_cty == c)
        if len(pi) == 0:
            continue
        q, p = s1.iloc[qi], pool.iloc[pi]
        specs = {
            "name_char": (lambda d: d["norm_name"].tolist(), "char", BC["tfidf_name_k"], BC["name_char_max_df"]),
            "addr_char": (lambda d: d["norm_addr"].tolist(), "char", BC["tfidf_addr_k"], BC["addr_char_max_df"]),
            "name_tok": (lambda d: blocking.name_tok_text(d["norm_name"], d["fold_name"]), "word", BC["rare_token_cap"], BC["name_tok_max_df"]),
            "num_key": (lambda d: blocking.num_key_text(d["norm_addr"], d["house_number"]), "word", BC["num_key_k"], BC["num_key_max_df"]),
            "name_pair": (lambda d: d["fold_name"].tolist(), "name_pair", BC["pair_k"], BC["pair_max_df"]),
            "addr_pair": (lambda d: d["fold_addr"].tolist(), "addr_pair", BC["pair_k"], BC["pair_max_df"]),
            "cross_pair": (lambda d: blocking.cross_text(d["fold_name"], d["fold_addr"]), "cross_pair", BC["pair_k"], BC["pair_max_df"]),
        }
        for ch, (text_fn, kind, k, max_df) in specs.items():
            t0 = time.time()
            a, b, s, r = blocking.run_sparse_channel(text_fn(q), text_fn(p), kind, k=k, max_df=max_df, n_jobs=N_JOBS)
            parts[ch].append((qi[a].astype(np.int32), pi[b].astype(np.int32), s, r))
            kaggle_env.log(f"    {c!r} {ch}: {len(qi):,} q x {len(pi):,} pool -> {len(a):,} pairs ({time.time() - t0:.0f}s)")
            gc.collect()
    out = {}
    for ch, v in parts.items():
        if v:
            out[ch] = tuple(np.concatenate([x[i] for x in v]) for i in range(4))
    return out


def _features(u, split, emb_src):
    """Add dense cosine (or NaN) and per-S1 gap features to a union frame, in place."""
    if NO_DENSE:
        u["dense_cos"] = np.float32(np.nan)
    else:
        ddir, dprefix = emb_src
        q_emb = np.load(ddir / f"{dprefix}emb_{split}_S1.npy", mmap_mode="r")
        p_emb = np.load(ddir / f"{dprefix}emb_{split}_pool.npy", mmap_mode="r")
        blocking.add_dense_cos(u, q_emb, p_emb)
    blocking.add_gap_features(u)


def build_candidates(split, s1, pool, q_rows, emb_src, model=None, chunk=CHUNK_S1):
    """All channels -> union -> cheap features; pruned by `model` if given.

    Channel lists are computed once for all queried rows (compact arrays);
    union + features + pre-ranking run over chunks of `chunk` S1 rows so peak
    memory stays bounded (NB05 sparse v3 was OOM-killed building a 48M-pair
    union with features for one country at once).

    Outputs: model None -> the full union frame (train: needed to fit the
    pre-ranker); model given -> the concatenated pruned frames.
    """
    t0 = time.time()
    parts = {}
    if not NO_DENSE:
        ddir, dprefix = emb_src
        parts = dense_parts(ddir, dprefix, split, q_rows, len(s1))
        kaggle_env.log(f"  {split}: dense parts {len(parts['dense'][0]):,} + reverse {len(parts['reverse'][0]):,}")
    parts.update(sparse_parts(s1, pool, q_rows))
    if model is None:
        u = blocking.union_channels(parts, len(pool))
        del parts
        gc.collect()
        _features(u, split, emb_src)
        kaggle_env.log(f"  {split}: union {len(u):,} pairs for {len(q_rows):,} S1 ({time.time() - t0:.0f}s)")
        return u
    chunk_of = np.full(len(s1), -1, dtype=np.int32)
    chunk_of[q_rows] = np.arange(len(q_rows)) // chunk
    out, n_union = [], 0
    for ci in range(int(chunk_of[q_rows].max()) + 1 if len(q_rows) else 0):
        sub = {c: tuple(a[chunk_of[v[0]] == ci] for a in v) for c, v in parts.items()}
        u = blocking.union_channels(sub, len(pool))
        del sub
        _features(u, split, emb_src)
        n_union += len(u)
        score = model.predict(u[blocking.PRUNE_FEATURES].to_numpy(np.float32), num_threads=N_JOBS)
        out.append(blocking.prune(u, score, BC["prune_top"]))
        del u, score
        gc.collect()
    kaggle_env.log(f"  {split}: union {n_union:,} pairs -> pruned for {len(q_rows):,} S1 ({time.time() - t0:.0f}s)")
    pr = pd.concat(out, ignore_index=True) if out else None
    pr.attrs["n_union"] = n_union
    return pr


def true_keys_for(s1, pool, pairs, q_rows):
    """Sorted int64 keys (s1_row * n_pool + pool_row) of true pairs of the queried S1 rows."""
    s1_pos = pd.Series(np.arange(len(s1)), index=s1["entity_id"].to_numpy())
    p_pos = pd.Series(np.arange(len(pool)), index=pool["entity_id"].to_numpy())
    qset = set(s1["entity_id"].to_numpy()[q_rows].tolist())
    pr = pairs[pairs["s1_id"].isin(qset)]
    keys = s1_pos.loc[pr["s1_id"]].to_numpy().astype(np.int64) * len(pool) + p_pos.loc[pr["match_id"]].to_numpy()
    n_true = pr.groupby("s1_id").size()
    n_by_q = n_true.reindex(s1["entity_id"].to_numpy()[q_rows]).fillna(0).astype(int).to_numpy()
    return np.sort(keys), n_by_q


def to_cands(pr: pd.DataFrame, s1, pool) -> pd.DataFrame:
    """Pruned union rows -> saved candidate table with entity IDs."""
    keep = ["bitmask", "n_channels", "dense_rank", "reverse_rank", "name_char_score", "addr_char_score",
            "name_tok_score", "num_key_score", "name_pair_score", "addr_pair_score", "cross_pair_score", "dense_cos",
            "cheap_score"]
    out = pr[keep].copy()
    out.insert(0, "cand_id", pool["entity_id"].to_numpy()[pr["p_row"].to_numpy()])
    out.insert(0, "s1_id", s1["entity_id"].to_numpy()[pr["q_row"].to_numpy()])
    return out


def dump_missed(recs_dir, s1, pool, pairs, pr, union, q_rows, n=200):
    """Write <= n true pairs of the queried rows missing from the pruned set (raw text, channel info)."""
    keys_pr = pr["q_row"].to_numpy().astype(np.int64) * len(pool) + pr["p_row"].to_numpy()
    keys_u = union["q_row"].to_numpy().astype(np.int64) * len(pool) + union["p_row"].to_numpy()
    s1_pos = pd.Series(np.arange(len(s1)), index=s1["entity_id"].to_numpy())
    p_pos = pd.Series(np.arange(len(pool)), index=pool["entity_id"].to_numpy())
    qset = set(s1["entity_id"].to_numpy()[q_rows].tolist())
    tp = pairs[pairs["s1_id"].isin(qset)].copy()
    tp["key"] = s1_pos.loc[tp["s1_id"]].to_numpy().astype(np.int64) * len(pool) + p_pos.loc[tp["match_id"]].to_numpy()
    miss = tp[~np.isin(tp["key"].to_numpy(), keys_pr)]
    miss = miss.sample(min(n, len(miss)), random_state=0) if len(miss) else miss
    if miss.empty:
        return 0
    in_union = np.isin(miss["key"].to_numpy(), keys_u)
    raw_cols = ["entity_id", "raw_name", "raw_addr", "country"]
    ids = set(miss["s1_id"]) | set(miss["match_id"])
    raws = pd.concat(
        [pd.read_parquet(recs_dir / f"records_train_S{s}.parquet", columns=raw_cols) for s in (1, 2, 3)], ignore_index=True
    )
    raws = raws[raws["entity_id"].isin(ids)].set_index("entity_id")
    lines = ["s1_id\tmatch_id\tcountry\tin_union\ts1_name\ts1_addr\tm_name\tm_addr"]
    for (s1_id, m_id), iu in zip(miss[["s1_id", "match_id"]].itertuples(index=False), in_union):
        a, b = raws.loc[s1_id], raws.loc[m_id]
        clean = lambda x: str(x).replace("\t", " ").replace("\n", " ")
        lines.append("\t".join([s1_id, m_id, clean(a["country"]), str(bool(iu)), clean(a["raw_name"]), clean(a["raw_addr"]),
                                clean(b["raw_name"]), clean(b["raw_addr"])]))
    (WORK / "missed_B.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(miss)


def main() -> None:
    """Run NB05 end to end."""
    t0 = time.time()
    io_utils.set_seeds(CONFIG["seed"])
    WORK.mkdir(parents=True, exist_ok=True)
    recs_dir = kaggle_env.find_input("records_train_S1.parquet").parent
    frozen_dir = None if NO_DENSE else kaggle_env.find_input("knn_train_idx.npy").parent
    ft_dir = kaggle_env.find_input("ft_knn_test_idx.npy").parent if DENSE_FT else None
    kaggle_env.log(f"records {recs_dir}; frozen {frozen_dir}; ft {ft_dir}; workers {N_JOBS}")
    metrics: dict = {"dense_ft": DENSE_FT, "no_dense": NO_DENSE, "blocking_config": BC}
    report: dict = {"dense_source_B_test": "none" if NO_DENSE else ("fine-tuned" if DENSE_FT else "frozen")}

    # ---------------- train: A and B query samples ----------------
    s1, pool = load_split(recs_dir, "train")
    pairs = pd.read_parquet(recs_dir / "gt.parquet")
    sp1 = pd.read_parquet(recs_dir / "split_s1.parquet").set_index("entity_id")
    half = sp1["split"].reindex(s1["entity_id"]).to_numpy()
    rng = np.random.default_rng(CONFIG["seed"])
    q = {}
    for h in ("A", "B"):
        rows = np.flatnonzero(half == h)
        n = min(BC["train_sample_per_half"], len(rows))
        q[h] = np.sort(rng.choice(rows, n, replace=False)).astype(np.int32)
    s1_cty = s1["country"].to_numpy()
    unions = {}
    if DENSE_FT:
        # A must use frozen vectors and B fine-tuned ones, so build them separately.
        for h in ("A", "B"):
            src = (ft_dir, "ft_") if h == "B" else (frozen_dir, "")
            kaggle_env.log(f"train half {h}: {len(q[h]):,} S1 queries (dense: {src[1] or 'frozen '}vectors)")
            unions[h] = build_candidates("train", s1, pool, q[h], src)
    else:
        # One pass over the pool for both halves (hashing the 10M-row pool is the cost), then split.
        both = np.sort(np.concatenate([q["A"], q["B"]]))
        kaggle_env.log(f"train halves A+B: {len(both):,} S1 queries in one pass")
        u = build_candidates("train", s1, pool, both, (frozen_dir, ""))
        in_a = np.isin(u["q_row"].to_numpy(), q["A"])
        unions["A"] = u[in_a].reset_index(drop=True)
        unions["B"] = u[~in_a].reset_index(drop=True)
        del u, in_a
        gc.collect()

    kaggle_env.log("training the cheap pre-ranker on Half A")
    keys_a, _ = true_keys_for(s1, pool, pairs, q["A"])
    ua = unions["A"]
    y_a = np.isin(ua["q_row"].to_numpy().astype(np.int64) * len(pool) + ua["p_row"].to_numpy(), keys_a).astype(np.int8)
    model = blocking.train_pruner(ua, y_a, seed=CONFIG["seed"])
    model.save_model(str(WORK / "pruner.txt"))
    metrics["pruner_feature_gain"] = dict(zip(blocking.PRUNE_FEATURES, model.feature_importance("gain").round(1).tolist()))

    for h in ("A", "B"):
        u = unions[h]
        pr = blocking.prune(u, model.predict(u[blocking.PRUNE_FEATURES].to_numpy(np.float32), num_threads=N_JOBS), BC["prune_top"])
        keys, n_by_q = true_keys_for(s1, pool, pairs, q[h])
        report[h] = blocking.blocking_report(u, pr, keys, q[h], s1_cty[q[h]], n_by_q, len(pool), len(pool))
        kaggle_env.log(f"  {h}: {report[h]['ALL']}")
        to_cands(pr, s1, pool).to_parquet(WORK / f"cands_{h}.parquet", index=False, compression="zstd")
        if h == "B":
            metrics["n_missed_dumped"] = dump_missed(recs_dir, s1, pool, pairs, pr, u, q[h])
        del pr
        gc.collect()
    kaggle_env.write_json(report, WORK / "blocking_report.json")
    del unions, u, ua, s1, pool, pairs, sp1
    gc.collect()

    # ---------------- test: all S1, pruned per country to bound memory ----------------
    s1, pool = load_split(recs_dir, "test")
    src = (ft_dir, "ft_") if DENSE_FT else (frozen_dir, "")
    s1_cty = s1["country"].to_numpy()
    outs, per_country = [], {}
    for c in sorted(set(s1_cty.tolist())):
        qr = np.flatnonzero(s1_cty == c).astype(np.int32)
        kaggle_env.log(f"test {c!r}: {len(qr):,} S1 queries")
        pr = build_candidates("test", s1, pool, qr, src, model=model)
        cps = pr.groupby("q_row").size().reindex(qr).fillna(0)
        per_country[c] = {"n_s1": int(len(qr)), "union_per_s1_mean": round(pr.attrs["n_union"] / len(qr), 2),
                          "cands_per_s1_mean": round(float(cps.mean()), 2), "cands_per_s1_p95": float(np.quantile(cps, 0.95)),
                          "share_s1_zero_cands": round(float((cps == 0).mean()), 5)}
        outs.append(to_cands(pr, s1, pool))
        del pr
        gc.collect()
    cands = pd.concat(outs, ignore_index=True)
    assert cands["s1_id"].nunique() <= len(s1)
    assert not cands.duplicated(["s1_id", "cand_id"]).any()
    cands.to_parquet(WORK / "cands_test.parquet", index=False, compression="zstd")
    report["test"] = per_country
    kaggle_env.write_json(report, WORK / "blocking_report.json")
    metrics["test"] = per_country
    metrics["runtime_s"] = round(time.time() - t0, 1)
    kaggle_env.write_json(metrics, WORK / "metrics.json")
    kaggle_env.log(f"done in {metrics['runtime_s']}s")


if __name__ == "__main__":
    main()
