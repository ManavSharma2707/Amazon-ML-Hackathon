"""NB05v7 driver — v6 blocking + learned transliteration channels (PROMPT V7 STEP1a/b/d).

Bundled with kaggle_env/io_utils/normalize/translit/blocking into
`nb05v7_translit.py` (CLAUDE.md SS6.3). CPU kernel, internet OFF, no
`er-code`/`sys.path` dependency (architecture.md SS6.1) -- only `er-data`
mounted, this notebook's own upstream (`er-nb02-normalize`) attached via
`kernel_sources`.

Builds on the shipped sparse-only NB05 (`er-nb05-blocking-sparse` v6):
same 9 channels (memory.md SS5/SS9), plus two new ones learned from a
train-only transliteration dictionary (`translit.py`):
- `translit_name` / `translit_addr`: char TF-IDF on a Latin field where
  non-Latin source tokens (S2/S3 raw name/address) are mapped through a
  dictionary learned from Half-A train positives (native_token ->
  latin_token, Dice-filtered), falling back to `anyascii` for tokens with
  no dictionary entry. S1 rows are 100% Latin already (EDA E10), so their
  translit_name/addr collapses to a cleaned+folded form of norm_name/addr.

No dense channel (still infeasible at this record count -- memory.md SS5,
SS9: ~400 rec/s on 2xT4 -> the pool alone is many hours; see also the STEP1c
dry-run result in progress.md for today's fresh measurement).

Outputs in /kaggle/working:
- translit_dict.tsv         learned dictionary (native_token, latin_token)
- cands_{A,B,test}.parquet  s1_id, cand_id, bitmask, channel scores/ranks,
  translit_name/addr scores+ranks, cheap_score (the pruned candidate set)
- pruner.txt                LightGBM pre-ranker retrained with the 2 new channels
- blocking_report.json      SS12.3 metrics for A and B (per country, per channel)
- missed_B.tsv              <= 200 true B pairs missing from the candidates
- metrics.json
"""

import gc
import json
import os
import time

import numpy as np
import pandas as pd

RESUME = globals().get("RESUME", False)  # see nb05v7_translit_resume/: reuses this driver, skips the train phase
WORK = kaggle_env.WORK_DIR
N_JOBS = os.cpu_count() or 1
BC = dict(CONFIG["blocking"])
# Time-box override (this notebook only, not the shared config -- v6's already-shipped
# blocking_report/candidates are unaffected): the real v2 Kaggle run measured ~51 min for
# the 11-channel pass at 500k combined train queries, which alone was most of a ~1h budget
# before test (1.73M queries, no way to shrink) still has to run. Cutting the train query
# sample 250k/half -> 60k/half trades some pruner-training statistical power (still ample:
# train_pruner further subsamples to <= 3M pairs regardless) for a shot at finishing before
# the 20:45 IST V7 gate. See memory.md.
BC["train_sample_per_half"] = 60_000
CHUNK_S1 = int(os.environ.get("ER_CHUNK_S1", 200_000))  # S1 rows per union/feature/prune chunk (memory bound)
COLS = ["entity_id", "country", "raw_name", "raw_addr", "norm_name", "fold_name", "norm_addr", "fold_addr", "house_number"]


def load_split(recs_dir, split: str):
    """S1 and pool (S2 then S3) frames with the channel columns (incl. raw_* for translit)."""
    s1 = pd.read_parquet(recs_dir / f"records_{split}_S1.parquet", columns=COLS)
    pool = pd.concat([pd.read_parquet(recs_dir / f"records_{split}_S{s}.parquet", columns=COLS) for s in (2, 3)], ignore_index=True)
    return s1, pool


def sparse_parts(s1: pd.DataFrame, pool: pd.DataFrame, q_rows: np.ndarray) -> dict:
    """Run the 9 v6 sparse channels + the 2 translit channels per country group."""
    channel_names = (
        "name_char", "addr_char", "name_tok", "num_key", "name_pair", "addr_pair", "cross_pair", "reverse",
        "translit_name", "translit_addr",
    )
    parts = {c: [] for c in channel_names}
    want = np.zeros(len(s1), dtype=bool)
    want[q_rows] = True
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

        # ---- translit channels: adaptive k (empty OTHER field -> bigger k here) ----
        q_name_empty = q["translit_name"].eq("").to_numpy()
        q_addr_empty = q["translit_addr"].eq("").to_numpy()
        for ch, field, empty_mask, other_k in (
            ("translit_name", "translit_name", q_addr_empty, BC["translit_boost_k"]),
            ("translit_addr", "translit_addr", q_name_empty, BC["translit_boost_k"]),
        ):
            t0 = time.time()
            base_k, max_df = BC[f"{ch}_k"], BC["translit_max_df"]
            a, b, s, r = blocking.run_sparse_channel(q[field].tolist(), p[field].tolist(), "char", k=base_k, max_df=max_df, n_jobs=N_JOBS)
            out = [(qi[a].astype(np.int32), pi[b].astype(np.int32), s, r)]
            if empty_mask.any() and other_k > base_k:
                sub = np.flatnonzero(empty_mask)
                a2, b2, s2, r2 = blocking.run_sparse_channel(
                    q[field].iloc[sub].tolist(), p[field].tolist(), "char", k=other_k, max_df=max_df, n_jobs=N_JOBS
                )
                out.append((qi[sub][a2].astype(np.int32), pi[b2].astype(np.int32), s2, r2))
            parts[ch].append(tuple(np.concatenate([o[i] for o in out]) for i in range(4)))
            kaggle_env.log(
                f"    {c!r} {ch}: {len(qi):,} q x {len(pi):,} pool -> "
                f"{sum(len(o[0]) for o in out):,} pairs, {int(empty_mask.sum()):,} boosted ({time.time() - t0:.0f}s)"
            )
            gc.collect()

        rev_k = BC.get("rev_name_k", 0)
        if rev_k:
            t0 = time.time()
            sall = np.flatnonzero(s1_cty == c)
            a, b, sc_, rk_ = blocking.run_sparse_channel(p["norm_name"].tolist(), s1.iloc[sall]["norm_name"].tolist(), "char",
                                                         k=rev_k, max_df=BC["name_char_max_df"], n_jobs=N_JOBS)
            srow = sall[b]
            keep = want[srow]
            parts["reverse"].append((srow[keep].astype(np.int32), pi[a[keep]].astype(np.int32), sc_[keep], rk_[keep]))
            kaggle_env.log(f"    {c!r} reverse name_char: {len(pi):,} pool x {len(sall):,} S1 -> {int(keep.sum()):,} kept pairs ({time.time() - t0:.0f}s)")
            gc.collect()
    out = {}
    for ch, v in parts.items():
        if v:
            out[ch] = tuple(np.concatenate([x[i] for x in v]) for i in range(4))
    return out


def _features(u, split):
    """Add gap features to a union frame, in place (no dense channel: still infeasible, memory.md SS5/SS9)."""
    u["dense_cos"] = np.float32(np.nan)
    blocking.add_gap_features(u)


def build_candidates(split, s1, pool, q_rows, model=None, chunk=CHUNK_S1):
    """All channels -> union -> cheap features; pruned by `model` if given."""
    t0 = time.time()
    parts = sparse_parts(s1, pool, q_rows)
    if model is None:
        u = blocking.union_channels(parts, len(pool))
        del parts
        gc.collect()
        _features(u, split)
        kaggle_env.log(f"  {split}: union {len(u):,} pairs for {len(q_rows):,} S1 ({time.time() - t0:.0f}s)")
        return u
    chunk_of = np.full(len(s1), -1, dtype=np.int32)
    chunk_of[q_rows] = np.arange(len(q_rows)) // chunk
    out, n_union = [], 0
    for ci in range(int(chunk_of[q_rows].max()) + 1 if len(q_rows) else 0):
        sub = {c: tuple(a[chunk_of[v[0]] == ci] for a in v) for c, v in parts.items()}
        u = blocking.union_channels(sub, len(pool))
        del sub
        _features(u, split)
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
    keep = [
        "bitmask", "n_channels", "reverse_rank", "reverse_score", "name_char_score", "addr_char_score",
        "name_tok_score", "num_key_score", "name_pair_score", "addr_pair_score", "cross_pair_score",
        "translit_name_score", "translit_name_rank", "translit_addr_score", "translit_addr_rank",
        "dense_cos", "cheap_score",
    ]
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


def _ensure_anyascii() -> None:
    """Install `anyascii` (ISC) if the base image doesn't have it; requires internet ON.

    `translit.py` resolves it lazily (memory.md pitfall: never bind an
    optional import at module exec time), so it's enough to have this
    succeed before the first real transliteration call below.
    """
    try:
        import anyascii  # noqa: F401
    except ImportError:
        import subprocess
        import sys

        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "anyascii==0.3.3"], check=True)


def _prune_union_chunked(u: pd.DataFrame, model, chunk_s1: int) -> pd.DataFrame:
    """Score + prune an already-built union frame in S1-sized slices (bounds memory for huge countries).

    `union_channels` builds `u` from `np.unique` of int64 keys `q_row * n_pool
    + p_row`, so `u` is already sorted by `q_row` (then `p_row`) -- slice
    boundaries can be found with `searchsorted` instead of a boolean mask.
    """
    q = u["q_row"].to_numpy()
    uniq_q = np.unique(q)
    out = []
    for s in range(0, len(uniq_q), chunk_s1):
        lo, hi = uniq_q[s], uniq_q[min(s + chunk_s1, len(uniq_q)) - 1]
        a, b = np.searchsorted(q, lo, side="left"), np.searchsorted(q, hi, side="right")
        sub = u.iloc[a:b]
        score = model.predict(sub[blocking.PRUNE_FEATURES].to_numpy(np.float32), num_threads=N_JOBS)
        out.append(blocking.prune(sub, score, BC["prune_top"]))
    return pd.concat(out, ignore_index=True) if out else u.iloc[:0]


def run_country_isolated(country: str, s1: pd.DataFrame, pool: pd.DataFrame, qr: np.ndarray, model, out_path) -> dict:
    """Build one test country's candidates: channels+union in a forked child, prune in the parent.

    Two distinct failures were hit here on real Kaggle runs, in order:
    1. (OOM) France->India->US died partway through India's channels even
       though each channel's own transient memory is freed (`gc.collect()`)
       after it completes -- RSS still climbed across ~10 sequential large
       sparse-matrix operations for one huge country (809,986 queries x 4.7M
       pool) before Python's allocator returned anything to the OS. Fixed by
       running the channel+union computation for each country in a forked
       child (Linux copy-on-write: no pickling of s1/pool) that exits when
       done, so the OS reclaims everything and the next country starts clean.
    2. (SIGSEGV, exit 139) Fixing #1 by forking around the WHOLE
       `build_candidates` call (channels + `model.predict` for pruning)
       crashed instead: LightGBM's `predict()` uses OpenMP internally, and
       forking a process after a native threaded library has already
       initialised its thread pool leaves the child with an invalid copy of
       it -- a well-known fork/threading hazard, not a memory issue. The
       channels themselves (pure numpy/scipy) survived the fork fine; the
       crash traced to the `model.predict` call specifically. Fixed by moving
       the fork boundary: the child does ONLY channels + union (no LightGBM
       call at all, proven safe by #1's fix), writes the unpruned union to a
       temp parquet, and the (never-forked, stable) parent reloads it and
       runs `model.predict` + prune itself, chunked by S1 to keep peak
       memory in the parent bounded too.

    Inputs: country - label (report key only); s1, pool - test record frames
            (read-only, shared via fork, never pickled); qr - this country's S1
            row ids; model - the pruner Booster (only ever used in the parent);
            out_path - Path to write the country's final `to_cands` parquet to.
    Outputs: stats dict (n_s1, union_per_s1_mean, cands_per_s1_mean/p95,
    share_s1_zero_cands).
    """
    tmp_union = out_path.with_suffix(".union.parquet")
    pid = os.fork()
    if pid == 0:  # child: channels + union ONLY (no LightGBM), write, exit -- never returns
        try:
            u = build_candidates("test", s1, pool, qr, model=None)
            u.to_parquet(tmp_union, index=False, compression="zstd")
            os._exit(0)
        except Exception:
            import traceback

            traceback.print_exc()
            os._exit(1)
    _, status = os.waitpid(pid, 0)
    if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
        raise RuntimeError(f"country {country!r} channel/union subprocess failed (status {status})")

    u = pd.read_parquet(tmp_union)
    n_union = len(u)
    pr = _prune_union_chunked(u, model, CHUNK_S1)
    del u
    gc.collect()
    tmp_union.unlink()
    cps = pr.groupby("q_row").size().reindex(qr).fillna(0)
    stats = {"n_s1": int(len(qr)), "union_per_s1_mean": round(n_union / len(qr), 2),
             "cands_per_s1_mean": round(float(cps.mean()), 2), "cands_per_s1_p95": float(np.quantile(cps, 0.95)),
             "share_s1_zero_cands": round(float((cps == 0).mean()), 5)}
    to_cands(pr, s1, pool).to_parquet(out_path, index=False, compression="zstd")
    del pr
    gc.collect()
    return stats


def main() -> None:
    """Run NB05v7 end to end: dictionary -> translit columns -> channels -> union -> prune -> report."""
    t0 = time.time()
    _ensure_anyascii()
    io_utils.set_seeds(CONFIG["seed"])
    WORK.mkdir(parents=True, exist_ok=True)
    recs_dir = kaggle_env.find_input("records_train_S1.parquet").parent
    kaggle_env.log(f"records {recs_dir}; workers {N_JOBS}")
    metrics: dict = {"blocking_config": BC, "no_dense": True, "resume": RESUME}
    report: dict = {"dense_source_B_test": "none"}

    if RESUME:
        # Reuse a prior (real) run's already-completed train phase instead of
        # rebuilding the dictionary and re-running all 11 channels over 120k
        # train queries again -- that part succeeded cleanly last time and
        # cost ~50 min; only the test phase needs the fix below.
        prior = kaggle_env.find_input("translit_dict.tsv").parent
        kaggle_env.log(f"RESUME: reusing prior train-phase output from {prior}")
        dictionary = translit.load_dictionary(str(prior / "translit_dict.tsv"))
        metrics["translit_dict_size"] = len(dictionary)
        import lightgbm as lgb

        model = lgb.Booster(model_file=str(prior / "pruner.txt"))
        model.save_model(str(WORK / "pruner.txt"))
        report.update(json.loads((prior / "blocking_report.json").read_text(encoding="utf-8")))
        for h in ("A", "B"):
            src = prior / f"cands_{h}.parquet"
            (WORK / f"cands_{h}.parquet").write_bytes(src.read_bytes())
        kaggle_env.log(f"  reused: dictionary {len(dictionary):,} entries, cands_A/B copied through, "
                       f"prior B pair_recall {report.get('B', {}).get('ALL', {}).get('pair_recall')}")
        kaggle_env.write_json(report, WORK / "blocking_report.json")
        kaggle_env.write_json(metrics, WORK / "metrics.json")
    else:
        _run_train_phase(recs_dir, metrics, report)
        dictionary = _TRAIN_STATE["dictionary"]
        model = _TRAIN_STATE["model"]

    # ---------------- test: all S1, pruned per country to bound memory ----------------
    s1, pool = load_split(recs_dir, "test")
    kaggle_env.log("STEP1a: applying translit to test S1 + pool")
    t_app = time.time()
    translit.add_translit_columns(s1, dictionary, n_jobs=N_JOBS)
    translit.add_translit_columns(pool, dictionary, n_jobs=N_JOBS)
    metrics["translit_apply_test_s"] = round(time.time() - t_app, 1)
    s1_cty = s1["country"].to_numpy()
    per_country = {}
    cand_files = []
    for c in sorted(set(s1_cty.tolist())):
        qr = np.flatnonzero(s1_cty == c).astype(np.int32)
        kaggle_env.log(f"test {c!r}: {len(qr):,} S1 queries (isolated subprocess)")
        out_path = WORK / f"cands_test_{c}.parquet"
        per_country[c] = run_country_isolated(c, s1, pool, qr, model, out_path)
        cand_files.append(out_path)
        kaggle_env.log(f"  {c!r} done: {per_country[c]}")
        gc.collect()
    cands = pd.concat([pd.read_parquet(f) for f in cand_files], ignore_index=True)
    assert cands["s1_id"].nunique() <= len(s1)
    assert not cands.duplicated(["s1_id", "cand_id"]).any()
    cands.to_parquet(WORK / "cands_test.parquet", index=False, compression="zstd")
    for f in cand_files:
        f.unlink()
    report["test"] = per_country
    kaggle_env.write_json(report, WORK / "blocking_report.json")
    metrics["test"] = per_country
    metrics["runtime_s"] = round(time.time() - t0, 1)
    kaggle_env.write_json(metrics, WORK / "metrics.json")
    kaggle_env.log(f"done in {metrics['runtime_s']}s")


_TRAIN_STATE: dict = {}


def _run_train_phase(recs_dir, metrics: dict, report: dict) -> None:
    """The original (non-RESUME) train phase: dictionary -> channels -> union -> prune -> cands_A/B.

    Leaves its results in the module-level `_TRAIN_STATE` dict (`dictionary`,
    `model`) for `main()` to pick up, since this needs to slot into the same
    place the inline code used to occupy without changing `main()`'s overall
    shape more than necessary.
    """
    t0 = time.time()
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

    kaggle_env.log("STEP1a: building the translit dictionary from Half-A train positives")
    t_dict = time.time()
    a_s1_ids = set(s1["entity_id"].to_numpy()[q["A"]].tolist())
    dictionary = translit.build_dictionary_from_pairs(
        s1, pool, pairs, a_s1_ids, min_count=BC["translit_min_count"], min_dice=BC["translit_min_dice"]
    )
    translit.save_dictionary(dictionary, str(WORK / "translit_dict.tsv"))
    metrics["translit_dict_size"] = len(dictionary)
    metrics["translit_dict_build_s"] = round(time.time() - t_dict, 1)
    kaggle_env.log(f"  dictionary: {len(dictionary):,} entries ({metrics['translit_dict_build_s']}s)")

    kaggle_env.log("STEP1a: applying translit to train S1 + pool")
    t_app = time.time()
    translit.add_translit_columns(s1, dictionary, n_jobs=N_JOBS)
    translit.add_translit_columns(pool, dictionary, n_jobs=N_JOBS)
    metrics["translit_apply_train_s"] = round(time.time() - t_app, 1)

    unions = {}
    both = np.sort(np.concatenate([q["A"], q["B"]]))
    kaggle_env.log(f"train halves A+B: {len(both):,} S1 queries, channels computed once")
    parts = sparse_parts(s1, pool, both)
    is_a = np.zeros(len(s1), dtype=bool)
    is_a[q["A"]] = True
    # Filter the raw per-channel arrays by half BEFORE unioning, instead of
    # building one combined union (58M rows) and boolean-splitting it after:
    # that split needs the original frame plus two filtered copies alive at
    # once (~2x peak) and OOM-killed the first real Kaggle run at this exact
    # point (memory.md pitfalls).
    for h in ("A", "B"):
        mask = is_a if h == "A" else ~is_a
        sub = {c: tuple(a[mask[v[0]]] for a in v) for c, v in parts.items()}
        u = blocking.union_channels(sub, len(pool))
        del sub
        gc.collect()
        _features(u, "train")
        unions[h] = u
        kaggle_env.log(f"  train {h}: union {len(u):,} pairs for {len(q[h]):,} S1")
    del parts, u
    gc.collect()

    kaggle_env.log("training the cheap pre-ranker on Half A (with translit channels)")
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
    metrics["train_phase_runtime_s"] = round(time.time() - t0, 1)
    _TRAIN_STATE["dictionary"] = dictionary
    _TRAIN_STATE["model"] = model


if __name__ == "__main__":
    main()
