"""NB06 driver — pair features + stage-1 LightGBM (architecture.md SS6.8, plan SS13-15).

Bundled with kaggle_env/io_utils/normalize/corpus_stats/blocking/explain_diff/
features/stage1 into `nb06_features_stage1.py` (CLAUDE.md SS6.3). CPU kernel,
internet OFF.

Inputs (globbed under /kaggle/input):
- NB02 output: records_*.parquet, stats_{train,test}.parquet, gt.parquet
- NB05 output: cands_{A,B,test}.parquet (top-50 per S1 from the blocking pruner)
- NB00 output: wheels/ (rapidfuzz, installed offline if the image lacks it)
No NB03/NB04 input: dense encoding of all records was infeasible (memory.md SS5),
so there are no embedding features (group D) in stage-1.

Flow:
1. Time the full feature builder on 10k A pairs; project the runtime for all
   A/B/test candidates (written to metrics.json). If the projection exceeds
   PRERANK_LIMIT_MIN, the scale-guard pre-ranker is on: cheap vectorised
   features -> LightGBM (trained on Half A) -> top-`prerank_top` per S1.
   Its output IS the candidate set scored from here on (candidate_pairs.tsv).
2. Full features on A (capped to `a_s1_cap` S1s) -> 5-fold group-k-fold OOF
   + one full-A model.
3. Full features + stage-1 predictions on B and test (test in S1 chunks).

Outputs in /kaggle/working:
- feats_A.parquet, feats_B.parquet  (s1_id, cand_id, label, FEATURES)
- feats_test/part-NNN.parquet       (s1_id, cand_id, FEATURES)
- p1_A_oof.parquet (s1_id, cand_id, label, p1), p1_B.parquet (+ label), p1_test.parquet
- stage1_model.txt, prerank.txt (if used), metrics.json
"""

import gc
import importlib
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

WORK = kaggle_env.WORK_DIR
N_JOBS = os.cpu_count() or 1
SC = CONFIG["stage1"]
SEED = CONFIG["seed"]
PRERANK_TOP = CONFIG["scale_guard"]["prerank_top"]
PRERANK_LIMIT_MIN = float(os.environ.get("ER_PRERANK_LIMIT_MIN", 90))
A_S1_CAP = int(os.environ.get("ER_A_S1_CAP", SC.get("a_s1_cap", 150_000)))
B_S1_CAP = int(os.environ.get("ER_B_S1_CAP", SC.get("b_s1_cap", 250_000)))
CHUNK_S1 = int(os.environ.get("ER_CHUNK_S1", 150_000))  # S1s per feature chunk (bounds worker memory)
REC_COLS = features.REC_COLS
CAND_COLS = ["s1_id", "cand_id", "bitmask", "n_channels", "cheap_score"] + [f"{c}_score" for c in features.SPARSE_CHANNELS]


def ensure_rapidfuzz() -> None:
    """Install rapidfuzz from NB00's offline wheels if the image lacks it (internet OFF)."""
    try:
        importlib.import_module("rapidfuzz.process").cpdist  # noqa: B018  (needs rapidfuzz >= 3.6)
        return
    except (ImportError, AttributeError):
        pass
    wheels = kaggle_env.find_input("wheels")
    kaggle_env.log(f"installing rapidfuzz from {wheels}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "--no-index", "--find-links", str(wheels), "rapidfuzz"],
                   check=True)
    importlib.invalidate_caches()
    importlib.import_module("rapidfuzz.process").cpdist  # noqa: B018


def _arrow_strings(t):
    """types_mapper: strings stay Arrow-backed (compact, no Python objects); other types as usual."""
    return pd.ArrowDtype(t) if pa.types.is_string(t) or pa.types.is_large_string(t) else None


def read_pq(path, columns=None) -> pd.DataFrame:
    """Parquet -> DataFrame with Arrow-backed string columns (10M-row record tables stay a few GB)."""
    return pq.read_table(path, columns=columns).to_pandas(types_mapper=_arrow_strings)


def load_records(recs_dir, split: str):
    """S1 and pool (S2 then S3) record frames with the feature columns."""
    s1 = read_pq(recs_dir / f"records_{split}_S1.parquet", REC_COLS)
    pool = pd.concat([read_pq(recs_dir / f"records_{split}_S{s}.parquet", REC_COLS) for s in (2, 3)], ignore_index=True)
    return s1, pool


def cand_columns(path) -> list[str]:
    """Candidate columns present in an NB05 file (older versions lack some channel scores)."""
    names = set(pq.ParquetFile(path).schema_arrow.names)
    return [c for c in CAND_COLS if c in names]


def fill_missing(c: pd.DataFrame) -> pd.DataFrame:
    """Add absent channel-score columns as NaN (channel not run in that NB05 version)."""
    for col in CAND_COLS:
        if col not in c:
            c[col] = np.float32(np.nan)
    return c


def read_cands(path) -> pd.DataFrame:
    """Whole NB05 candidate file (train halves: ~12.5M rows)."""
    return fill_missing(read_pq(path, cand_columns(path)))


def iter_s1_batches(path, n_rows: int):
    """Stream a candidate file in batches of whole S1s (NB05 writes each S1's rows contiguously).

    Keeps the 86M-row test file out of memory. Asserts the contiguity it relies on.
    """
    seen: set = set()
    carry = None
    for rb in pq.ParquetFile(path).iter_batches(batch_size=n_rows, columns=cand_columns(path)):
        df = fill_missing(rb.to_pandas(types_mapper=_arrow_strings))
        if carry is not None:
            df = pd.concat([carry, df], ignore_index=True)
        tail = (df["s1_id"] == df["s1_id"].iloc[-1]).to_numpy(dtype=bool)
        carry, df = df[tail], df[~tail]
        if len(df):
            ids = set(df["s1_id"].unique().tolist())
            assert not ids & seen, "candidate rows of one S1 are not contiguous"
            seen |= ids
            yield df.reset_index(drop=True)
    if carry is not None and len(carry):
        assert carry["s1_id"].iloc[0] not in seen
        yield carry.reset_index(drop=True)


def add_labels(c: pd.DataFrame, gt: pd.DataFrame) -> np.ndarray:
    """0/1 label per candidate row from the ground-truth pairs (s1_id, match_id)."""
    key = c["s1_id"].astype(str) + "|" + c["cand_id"].astype(str)
    true = set((gt["s1_id"].astype(str) + "|" + gt["match_id"].astype(str)).tolist())
    return np.asarray(key.isin(true), dtype=bool).astype(np.int8)


def s1_chunks(c: pd.DataFrame, size: int):
    """Yield candidate sub-frames covering `size` S1s each (rows of one S1 stay together)."""
    ids = c["s1_id"].unique()
    for s in range(0, len(ids), size):
        keep = set(ids[s : s + size].tolist())
        yield c[c["s1_id"].isin(keep)].reset_index(drop=True)


def subset_records(c: pd.DataFrame, s1: pd.DataFrame, pool: pd.DataFrame):
    """Only the S1 / pool records referenced by a candidate chunk (keeps forked workers small)."""
    return (s1[s1["entity_id"].isin(set(c["s1_id"]))].reset_index(drop=True),
            pool[pool["entity_id"].isin(set(c["cand_id"]))].reset_index(drop=True))


def top_per_s1(c: pd.DataFrame, score: np.ndarray, top: int) -> pd.DataFrame:
    """Keep the `top` best rows per S1 by `score` (stored as the new cheap_score)."""
    c = c.assign(cheap_score=score.astype(np.float32))
    c = c.sort_values(["s1_id", "cheap_score", "cand_id"], ascending=[True, False, True], kind="stable")
    return c[c.groupby("s1_id").cumcount() < top].reset_index(drop=True)


def run_chunks(c: pd.DataFrame, s1: pd.DataFrame, pool: pd.DataFrame, lookups: dict, pre_model, name: str):
    """Pre-rank (optional) and build full features chunk by chunk.

    Outputs: (pruned candidate frame, full feature frame) aligned row by row.
    """
    cs, fs = [], []
    n_chunks = (c["s1_id"].nunique() + CHUNK_S1 - 1) // CHUNK_S1
    for ci, ch in enumerate(s1_chunks(c, CHUNK_S1)):
        t0 = time.time()
        s1s, ps = subset_records(ch, s1, pool)
        if pre_model is not None:
            xp = features.build_features(ch, s1s, ps, lookups, n_jobs=N_JOBS, with_py=False, log=kaggle_env.log)
            ch = top_per_s1(ch, stage1.predict(pre_model, xp.to_numpy(np.float32), N_JOBS), PRERANK_TOP)
            del xp
        f = features.build_features(ch, s1s, ps, lookups, n_jobs=N_JOBS, log=kaggle_env.log)
        cs.append(ch)
        fs.append(f)
        kaggle_env.log(f"  {name} chunk {ci + 1}/{n_chunks}: {len(ch):,} pairs ({time.time() - t0:.0f}s)")
        del s1s, ps
        gc.collect()
    return pd.concat(cs, ignore_index=True), pd.concat(fs, ignore_index=True)


def pair_recall(c: pd.DataFrame, gt: pd.DataFrame, s1_ids) -> float:
    """Share of true pairs of the given S1s present in candidate frame `c`."""
    ids = set(s1_ids)
    t = gt[gt["s1_id"].isin(ids)]
    got = set((c["s1_id"] + "|" + c["cand_id"]).tolist())
    return round(float(np.mean([(a + "|" + b) in got for a, b in zip(t["s1_id"], t["match_id"])])), 5) if len(t) else float("nan")


def save_feats(path, c: pd.DataFrame, f: pd.DataFrame, label=None) -> None:
    """Write s1_id, cand_id, [label,] features to parquet (zstd)."""
    out = f.copy()
    if label is not None:
        out.insert(0, "label", label)
    out.insert(0, "cand_id", c["cand_id"].to_numpy())
    out.insert(0, "s1_id", c["s1_id"].to_numpy())
    out.to_parquet(path, index=False, compression="zstd")


def main() -> None:
    """Run NB06 end to end."""
    t_start = time.time()
    io_utils.set_seeds(SEED)
    WORK.mkdir(parents=True, exist_ok=True)
    ensure_rapidfuzz()
    recs_dir = kaggle_env.find_input("records_train_S1.parquet").parent
    cand_dir = kaggle_env.find_input("cands_test.parquet").parent
    kaggle_env.log(f"records {recs_dir}; cands {cand_dir}; workers {N_JOBS}")
    metrics: dict = {"config": {"stage1": SC, "prerank_top": PRERANK_TOP, "a_s1_cap": A_S1_CAP, "b_s1_cap": B_S1_CAP,
                                "prerank_limit_min": PRERANK_LIMIT_MIN},
                     "features": {"n": len(features.FEATURES), "names": features.FEATURES}}

    # ---------------- train records, lookups, candidates ----------------
    s1, pool = load_records(recs_dir, "train")
    gt = pd.read_parquet(recs_dir / "gt.parquet")
    lookups = features.token_lookups(pd.read_parquet(recs_dir / "stats_train.parquet"))
    kaggle_env.log(f"train records S1 {len(s1):,} pool {len(pool):,}; generic name {len(lookups['generic_name'])}, "
                   f"addr {len(lookups['generic_addr'])}")
    metrics["generic_tokens_sample"] = {"name": sorted(lookups["generic_name"])[:40], "addr": sorted(lookups["generic_addr"])[:40]}
    cands = {}
    rng = np.random.default_rng(SEED)
    nb05_metrics = cand_dir / "metrics.json"
    if nb05_metrics.exists():  # records which NB05 version (channel set) these candidates come from
        import json

        metrics["nb05_blocking_config"] = json.loads(nb05_metrics.read_text()).get("blocking_config")
    metrics["nb05_has_cross_pair"] = "cross_pair_score" in cand_columns(cand_dir / "cands_test.parquet")
    for h, cap in (("A", A_S1_CAP), ("B", B_S1_CAP)):
        c = read_cands(cand_dir / f"cands_{h}.parquet")
        ids = c["s1_id"].unique()
        if len(ids) > cap:
            keep = set(rng.choice(ids, cap, replace=False).tolist())
            c = c[c["s1_id"].isin(keep)].reset_index(drop=True)
        cands[h] = c
        kaggle_env.log(f"cands {h}: {len(c):,} pairs, {c['s1_id'].nunique():,} S1")
    n_test = int(pq.ParquetFile(cand_dir / "cands_test.parquet").metadata.num_rows)

    # ---------------- 1. timing -> scale-guard decision ----------------
    ca = cands["A"]
    t_sub = ca[ca["s1_id"].isin(set(ca["s1_id"].unique()[:250].tolist()))]
    s1s, ps = subset_records(t_sub, s1, pool)
    sec_full = features.time_per_pair(t_sub, s1s, ps, lookups, N_JOBS, n=10_000)
    sec_cheap = features.time_per_pair(t_sub, s1s, ps, lookups, N_JOBS, n=10_000, with_py=False)
    n_all = len(cands["A"]) + len(cands["B"]) + n_test
    proj_full = sec_full * n_all / 60
    proj_pre = (sec_cheap * n_all + sec_full * n_all * PRERANK_TOP / 50) / 60
    use_pre = proj_full > PRERANK_LIMIT_MIN
    metrics["timing"] = {"sec_per_pair_full": sec_full, "sec_per_pair_cheap": sec_cheap, "n_pairs_all": n_all,
                         "projected_min_full": round(proj_full, 1), "projected_min_with_prerank": round(proj_pre, 1),
                         "prerank_enabled": bool(use_pre)}
    kaggle_env.log(f"timing: {metrics['timing']}")
    kaggle_env.write_json(metrics, WORK / "metrics.json")

    # ---------------- 2. pre-ranker on A (scale guard) ----------------
    pre_model = None
    if use_pre:
        kaggle_env.log("training the pre-ranker on Half A (cheap features)")
        xs, ys = [], []
        for ch in s1_chunks(cands["A"], CHUNK_S1):
            s1s, ps = subset_records(ch, s1, pool)
            xs.append(features.build_features(ch, s1s, ps, lookups, n_jobs=N_JOBS, with_py=False, log=kaggle_env.log))
            ys.append(add_labels(ch, gt))
        # rows of cands["A"] are grouped by S1 in the same order as the chunks
        xa = pd.concat(xs, ignore_index=True)
        ya = np.concatenate(ys)
        pp = stage1.lgb_params({"lr": 0.1, "num_leaves": 31, "min_data_in_leaf": 100}, SEED)
        pre_model = stage1.train_full(xa.to_numpy(np.float32), ya, pp, 200, features.PRERANK_FEATURES)
        pre_model.save_model(str(WORK / "prerank.txt"))
        metrics["prerank_gain"] = dict(zip(features.PRERANK_FEATURES, pre_model.feature_importance("gain").round(1).tolist()))
        del xs, ys, xa, ya
        gc.collect()

    # ---------------- 3. features on A and B ----------------
    feats, labels, pruned = {}, {}, {}
    for h in ("A", "B"):
        kaggle_env.log(f"features for {h}")
        c0 = cands[h]
        c, f = run_chunks(c0, s1, pool, lookups, pre_model, h)
        y = add_labels(c, gt)
        pruned[h], feats[h], labels[h] = c, f, y
        metrics[f"cands_{h}"] = {"n_s1": int(c["s1_id"].nunique()), "n_pairs": int(len(c)),
                                 "pair_recall_before": pair_recall(c0, gt, c0["s1_id"].unique()),
                                 "pair_recall_after": pair_recall(c, gt, c0["s1_id"].unique()),
                                 "pos_rate": round(float(y.mean()), 5)}
        kaggle_env.log(f"  {h}: {metrics[f'cands_{h}']}")
        save_feats(WORK / f"feats_{h}.parquet", c, f, y)
    cty = pd.Series(s1["country"].to_numpy(), index=s1["entity_id"].to_numpy())
    del cands, s1, pool
    gc.collect()

    # ---------------- 4. stage-1: OOF on A, full-A model -> B ----------------
    params = stage1.lgb_params(SC, SEED)
    XA = feats["A"].to_numpy(np.float32)
    kaggle_env.log(f"stage-1 OOF on A: {XA.shape}")
    r = stage1.train_oof(XA, labels["A"], pruned["A"]["s1_id"].to_numpy(), params, folds=SC.get("folds", 5), seed=SEED,
                         max_rounds=SC.get("max_rounds", 1500), feature_names=features.FEATURES, log=kaggle_env.log)
    n_rounds = int(np.mean(r["best_iters"]) * 1.1) + 1
    model = stage1.train_full(XA, labels["A"], params, n_rounds, features.FEATURES)
    model.save_model(str(WORK / "stage1_model.txt"))
    cA = pruned["A"]["s1_id"].map(cty).to_numpy()
    pd.DataFrame({"s1_id": pruned["A"]["s1_id"], "cand_id": pruned["A"]["cand_id"], "label": labels["A"],
                  "p1": r["oof"]}).to_parquet(WORK / "p1_A_oof.parquet", index=False)
    pB = stage1.predict(model, feats["B"].to_numpy(np.float32), N_JOBS)
    cB = pruned["B"]["s1_id"].map(cty).to_numpy()
    pd.DataFrame({"s1_id": pruned["B"]["s1_id"], "cand_id": pruned["B"]["cand_id"], "label": labels["B"],
                  "p1": pB}).to_parquet(WORK / "p1_B.parquet", index=False)
    gain = sorted(zip(features.FEATURES, model.feature_importance("gain")), key=lambda t: -t[1])
    metrics["stage1"] = {"best_iters": r["best_iters"], "fold_auc": r["fold_auc"], "n_rounds_full": n_rounds,
                         "A_oof": stage1.eval_by_group(labels["A"], r["oof"], cA),
                         "B": stage1.eval_by_group(labels["B"], pB, cB),
                         "top_gain": [(k, round(float(v), 1)) for k, v in gain[:30]],
                         "zero_gain": [k for k, v in gain if v == 0]}
    kaggle_env.log(f"stage-1 A OOF {metrics['stage1']['A_oof']['ALL']} | B {metrics['stage1']['B']['ALL']}")
    kaggle_env.write_json(metrics, WORK / "metrics.json")
    del feats, labels, pruned, XA, r
    gc.collect()

    # ---------------- 5. test: pre-rank + features + stage-1, in S1 chunks ----------------
    s1, pool = load_records(recs_dir, "test")
    lookups = features.token_lookups(pd.read_parquet(recs_dir / "stats_test.parquet"))
    (WORK / "feats_test").mkdir(exist_ok=True)
    outs, n_parts = [], 0
    kaggle_env.log(f"test: {n_test:,} candidate pairs for {len(s1):,} S1, streamed in batches of whole S1s")
    for batch in iter_s1_batches(cand_dir / "cands_test.parquet", CHUNK_S1 * 50):
        c, f = run_chunks(batch, s1, pool, lookups, pre_model, f"test[{n_parts}]")
        p = stage1.predict(model, f.to_numpy(np.float32), N_JOBS)
        save_feats(WORK / "feats_test" / f"part-{n_parts:03d}.parquet", c, f)
        outs.append(pd.DataFrame({"s1_id": c["s1_id"], "cand_id": c["cand_id"], "p1": p, "cheap_score": c["cheap_score"]}))
        n_parts += 1
        del c, f, p
        gc.collect()
    p1t = pd.concat(outs, ignore_index=True)
    assert not p1t.duplicated(["s1_id", "cand_id"]).any()
    p1t.to_parquet(WORK / "p1_test.parquet", index=False)
    tc = pd.Series(s1["country"].to_numpy(), index=s1["entity_id"].to_numpy())
    per = p1t.assign(country=p1t["s1_id"].map(tc)).groupby("country")
    metrics["test"] = {"n_pairs": int(len(p1t)), "n_s1_with_cands": int(p1t["s1_id"].nunique()), "n_s1": int(len(s1)),
                       "per_country": {k: {"pairs_per_s1": round(float(len(g) / g["s1_id"].nunique()), 2),
                                           "p1_mean": round(float(g["p1"].mean()), 4),
                                           "share_p1_gt_0.5": round(float((g["p1"] > 0.5).mean()), 4)}
                                       for k, g in per}}
    metrics["runtime_s"] = round(time.time() - t_start, 1)
    kaggle_env.write_json(metrics, WORK / "metrics.json")
    kaggle_env.log(f"done in {metrics['runtime_s']}s")


if __name__ == "__main__":
    main()
